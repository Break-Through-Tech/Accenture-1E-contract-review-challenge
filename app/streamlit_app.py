# Streamlit app: paste or upload a contract and see which clause types the model finds.
#
# The model is loaded from app/model/, or from MODEL_SOURCE if it is set
# (a folder path or a Hugging Face repo name).
# That folder needs the model files and thresholds.json from the same run.
#
# Run from the repo root:  streamlit run app/streamlit_app.py

import io
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

WINDOW, OVERLAP = 510, 256     # same chunk settings as main.ipynb
MAX_LENGTH = 512
BATCH_SIZE = 16
DEFAULT_DIR = Path(__file__).parent / "model"

st.set_page_config(page_title="Contract Clause Finder", page_icon="📄", layout="wide")

# These imports are slow, so show a message while they load.
with st.spinner("Starting up (loading PyTorch, first run takes ~20 seconds)..."):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer


# ---------------------------------------------------------------- model loading
def model_source() -> str:
    try:
        if "MODEL_SOURCE" in st.secrets:
            return st.secrets["MODEL_SOURCE"]
    except Exception:          # no secrets.toml at all is fine
        pass
    return os.environ.get("MODEL_SOURCE", str(DEFAULT_DIR))


@st.cache_resource(show_spinner="Loading model (first run only)...")
def load_model(source: str):
    path = Path(source)
    if not path.exists():
        # Not a local folder -> treat it as a Hugging Face repo id and download it once
        from huggingface_hub import snapshot_download
        token = os.environ.get("HF_TOKEN")
        try:
            token = token or st.secrets.get("HF_TOKEN")
        except Exception:
            pass
        path = Path(snapshot_download(source, token=token))

    thresholds = json.loads((path / "thresholds.json").read_text())
    categories = list(thresholds)          # saved as dict(zip(CATS, thresholds)), order kept
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path).eval()
    assert model.config.num_labels == len(categories), \
        f"model has {model.config.num_labels} outputs but thresholds.json has {len(categories)}"
    return tokenizer, model, categories, np.array([thresholds[c] for c in categories])


# ---------------------------------------------------------------- inference
def chunk_text(text: str, tokenizer):
    """Same sliding window as chunk_contract() in main.ipynb."""
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True,
                    truncation=False, verbose=False)
    offsets = enc["offset_mapping"]
    chunks, step = [], WINDOW - OVERLAP
    for start in range(0, len(offsets), step):
        end = min(start + WINDOW, len(offsets))
        a, b = offsets[start][0], offsets[end - 1][1]
        chunks.append({"char_start": a, "char_end": b, "text": text[a:b]})
        if end == len(offsets):
            break
    return chunks


@torch.no_grad()
def score_chunks(texts, tokenizer, model, progress):
    out = []
    for i in range(0, len(texts), BATCH_SIZE):
        enc = tokenizer(texts[i:i + BATCH_SIZE], truncation=True, max_length=MAX_LENGTH,
                        padding=True, return_tensors="pt")
        out.append(torch.sigmoid(model(**enc).logits).numpy())
        progress.progress(min(1.0, (i + BATCH_SIZE) / len(texts)),
                          text=f"Scoring chunk {min(i + BATCH_SIZE, len(texts))} of {len(texts)}")
    return np.vstack(out)


def read_upload(file) -> str:
    if file.name.lower().endswith(".pdf"):
        from pypdf import PdfReader
        return "\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(file.read())).pages)
    return file.read().decode("utf-8", errors="ignore")


# ---------------------------------------------------------------- page
st.title("📄 Contract Clause Finder")
st.caption("Finds which of the CUAD clause types appear in a contract, using our fine-tuned "
           "transformer. A research demo, not legal advice.")

try:
    tokenizer, model, CATS, THRESH = load_model(model_source())
except Exception as e:
    st.error(f"Could not load the model from `{model_source()}`.\n\n{e}\n\n"
             "See the 'Web demo' section of the README for setup.")
    st.stop()

tab_paste, tab_upload = st.tabs(["Paste text", "Upload file"])
with tab_paste:
    pasted = st.text_area("Contract text", height=280,
                          placeholder="Paste a full contract or just a few clauses...")
with tab_upload:
    upload = st.file_uploader("Contract (.txt or .pdf)", type=["txt", "pdf"])

with st.sidebar:
    st.header("Settings")
    sensitivity = st.slider(
        "Threshold multiplier", 0.5, 1.5, 1.0, 0.05,
        help="1.0 = the per-category cutoffs tuned on validation. "
             "Lower finds more clauses (more false alarms); higher finds fewer.")
    show_all = st.checkbox("Show every category, not just detected ones")
    st.caption(f"Model: `{model_source()}`  \n{len(CATS)} clause categories")

text = read_upload(upload) if upload else pasted
if not st.button("Analyze", type="primary", disabled=not text.strip()):
    st.stop()

chunks = chunk_text(text, tokenizer)
bar = st.progress(0.0, text="Scoring...")
probs = score_chunks([c["text"] for c in chunks], tokenizer, model, bar)   # [n_chunks, n_cats]
bar.empty()

cutoffs = np.clip(THRESH * sensitivity, 0, 1)
best_chunk = probs.argmax(0)
max_prob = probs.max(0)
found = max_prob >= cutoffs

st.subheader(f"Found {found.sum()} of {len(CATS)} clause types "
             f"({len(chunks)} chunk{'s' if len(chunks) != 1 else ''} scored)")

table = pd.DataFrame({
    "Clause type": CATS,
    "Found": found,
    "Confidence": max_prob,
    "Cutoff": cutoffs,
}).sort_values(["Found", "Confidence"], ascending=False)
if not show_all:
    table = table[table["Found"]]

st.dataframe(table, hide_index=True, width="stretch", column_config={
    "Confidence": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f"),
    "Cutoff": st.column_config.NumberColumn(format="%.2f"),
})

if found.any():
    st.subheader("Where in the contract")
    st.caption("The highest-scoring chunk for each detected clause type.")
    for k in np.argsort(-max_prob):
        if found[k]:
            with st.expander(f"{CATS[k]}  ·  {max_prob[k]:.2f}"):
                st.text(chunks[best_chunk[k]]["text"])
