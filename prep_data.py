# =============================================================================
# prep_data.py  --  STEP 1 of Task 1
#
# GOAL: CUAD comes as a question-answering dataset. We need a classification
# dataset instead: "here is a piece of a contract -> which of the 41 clause
# types does it contain?" This script does that conversion.
#
# BEFORE RUNNING: upload CUAD_v1.json to the same folder as this file.
# OUTPUT: a "data" folder with train.jsonl, val.jsonl, test.jsonl, categories.json
# =============================================================================

import json      # reading/writing JSON files
import os        # making folders
import random    # shuffling contracts for the split
from transformers import AutoTokenizer  # turns text into tokens (word pieces)

# ---------------------------- SETTINGS (edit here) ---------------------------
CUAD_PATH = "data/CUADv1.json"                   # the CUAD file you uploaded
MODEL_NAME = "nlpaueb/legal-bert-base-uncased"  # must match the model in train.py
WINDOW = 254     # each chunk holds up to 254 tokens (BERT's limit is 512; we
                 # use 256 total once the 2 special start/end tokens are added)
STRIDE = 192     # how far we slide forward for the next chunk. 254 - 192 = 62
                 # tokens of overlap, so a clause cut at a boundary still
                 # appears whole in the neighboring chunk
SEED = 42        # fixed random seed so everyone gets the same split
OVERLAP_FRAC = 0.5  # a clause "counts" for a chunk only if at least half of
                    # that clause is inside the chunk (avoids labeling a chunk
                    # because it caught one stray word of a clause)
# -----------------------------------------------------------------------------

random.seed(SEED)
tok = AutoTokenizer.from_pretrained(MODEL_NAME)  # downloads tokenizer 1st time

# --- 1. LOAD CUAD ------------------------------------------------------------
with open(CUAD_PATH) as f:
    raw = json.load(f)["data"]   # a list with one entry per contract

contracts = []        # will hold one dict per contract
categories = set()    # will collect the 41 clause category names

for doc in raw:
    # In CUAD, each contract has exactly one "paragraph" = the full contract text
    para = doc["paragraphs"][0]
    spans = {}   # category name -> list of (start, end) character positions
    for qa in para["qas"]:
        # Each qa is "find clause type X in this contract". Its id looks like
        # "ContractTitle__Category Name", so we take the part after "__".
        cat = qa["id"].split("__")[-1]
        categories.add(cat)
        # "answers" are the highlighted clause texts. We only need where each
        # one starts and ends (character positions in the contract text).
        # If a category is absent from this contract, the list is empty.
        spans[cat] = [(a["answer_start"], a["answer_start"] + len(a["text"]))
                      for a in qa["answers"]]
    contracts.append({"title": doc["title"], "text": para["context"], "spans": spans})

categories = sorted(categories)  # sort so the order is always the same
cat_idx = {c: i for i, c in enumerate(categories)}  # name -> position 0..40
print(f"{len(contracts)} contracts, {len(categories)} categories")

# --- 2. SPLIT BY CONTRACT ----------------------------------------------------
# IMPORTANT: we split whole contracts BEFORE chunking. If we chunked first and
# split after, chunks from the same contract could land in both train and test,
# and the model would look better than it really is (data leakage).
random.shuffle(contracts)
n = len(contracts)
splits = {
    "train": contracts[: int(0.8 * n)],              # 80% for learning
    "val":   contracts[int(0.8 * n): int(0.9 * n)],  # 10% for tuning thresholds
    "test":  contracts[int(0.9 * n):],               # 10% for the final score
}

# --- 3. CHUNK + LABEL --------------------------------------------------------
def chunk_contract(c):
    """Cut one contract into overlapping chunks and label each chunk."""
    # Tokenize the whole contract. "offset_mapping" tells us, for every token,
    # its start/end character position in the original text. We need that to
    # compare token windows against the clause positions from CUAD.
    enc = tok(c["text"], add_special_tokens=False, return_offsets_mapping=True,
              truncation=False, verbose=False)
    offs = enc["offset_mapping"]

    out = []
    start = 0   # index of the first token in the current chunk
    while start < len(offs):
        end = min(start + WINDOW, len(offs))   # one past the last token
        # Convert the token window into character positions in the contract
        chunk_start_char = offs[start][0]
        chunk_end_char = offs[end - 1][1]

        # Build the label vector: 41 zeros, flip to 1 for categories present
        labels = [0] * len(categories)
        for cat, spans in c["spans"].items():
            for s, e in spans:
                # How many characters of this clause fall inside the chunk?
                overlap = min(chunk_end_char, e) - max(chunk_start_char, s)
                if overlap > 0:
                    # Compare overlap to the smaller of (clause length, chunk
                    # length) so very long clauses can still match a chunk
                    smaller = min(e - s, chunk_end_char - chunk_start_char)
                    if overlap / smaller >= OVERLAP_FRAC:
                        labels[cat_idx[cat]] = 1

        out.append({
            "contract": c["title"],                              # for tracing
            "text": c["text"][chunk_start_char:chunk_end_char],  # chunk text
            "labels": labels,                                    # 41 x 0/1
        })
        if end == len(offs):   # reached the end of the contract
            break
        start += STRIDE        # slide the window forward
    return out

# --- 4. SAVE EACH SPLIT ------------------------------------------------------
os.makedirs("data", exist_ok=True)
for name, items in splits.items():
    # Chunk every contract in this split and flatten into one list
    rows = [r for c in items for r in chunk_contract(c)]

    # JSONL = one JSON object per line (easy for Hugging Face to load)
    with open(f"data/{name}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    # Sanity check numbers: screenshot these for your writeup
    # pos[k] = how many chunks in this split contain category k
    pos = [sum(r["labels"][k] for r in rows) for k in range(len(categories))]
    all_zero = sum(1 for r in rows if not any(r["labels"]))
    print(f"{name}: {len(items)} contracts, {len(rows)} chunks, "
          f"{all_zero} chunks with no clause at all (that's normal)")
    # Categories with very few examples will give noisy scores later
    print("   categories with <10 positive chunks:",
          [categories[k] for k, p in enumerate(pos) if p < 10])

# Save the category list so train.py uses the same order
json.dump(categories, open("data/categories.json", "w"), indent=2)
print("Done. Files are in the 'data' folder.")
