# =============================================================================
# tfidf_baseline.py  --  Issue #12, Task A: TF-IDF baseline
#
# GOAL: a simple baseline the transformer (main.ipynb section 6) has to beat.
# TF-IDF features per chunk -> one logistic regression per clause category
# (OneVsRestClassifier) -> per-category thresholds tuned on validation ->
# precision / recall / F1 on test.
#
# Uses the same chunks, split and categories as main.ipynb so the two models
# are directly comparable:
#   - chunks from section 5 (510 tokens, 256 overlap)
#   - fit / val split by contract from section 2.5 (the "split" column)
#   - 40 predicted categories, 38 scored (section 4.5)
#   - threshold method from section 6.4
#
# BEFORE RUNNING: run main.ipynb through section 5.4. It writes
#   data/processed/train_chunks.csv, test_chunks.csv, train_labels.csv, test_labels.csv
#
# RUN FROM THE REPO ROOT:  python tfidf_baseline.py      (about 5 min on CPU)
# OUTPUT: outputs/tfidf_baseline/
#   per_category_test_metrics.csv, summary.json, thresholds.json, config.json,
#   top_terms_per_category.csv, {val,test}_{probs,labels}.npy
# =============================================================================

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             precision_recall_fscore_support, roc_auc_score)

#  SETTINGS
PROCESSED = Path("data/processed")
OUT = Path("outputs/tfidf_baseline")

CFG = dict(
    ngram_range=(1, 2),     # single words + two-word phrases ("shall not", "governing law")
    min_df=2,               # ignore terms that appear in only one chunk (typos, names)
    max_df=0.9,             # ignore terms in >90% of chunks (no signal)
    max_features=200_000,
    sublinear_tf=True,      # 1 + log(tf): a word repeated 20x is not 20x more important
    class_weight="balanced",  # rare categories: a missed positive costs more
    C_grid=[4, 16, 64],     # regularisation strength, picked on validation
    seed=42,
)
TRIVIAL = ["Document Name", "Parties"]  # in ~every contract; excluded from macro-F1


OUT.mkdir(parents=True, exist_ok=True)

# 1. LOAD CHUNKS
train_chunks = pd.read_csv(PROCESSED / "train_chunks.csv")
test_chunks = pd.read_csv(PROCESSED / "test_chunks.csv")
for df in (train_chunks, test_chunks):
    df["labels"] = df["labels"].apply(json.loads)   # stored as JSON strings

# 2. SAME CATEGORY RULES AS defined in main.ipy
train_labels = pd.read_csv(PROCESSED / "train_labels.csv").set_index("contract_id")
test_labels = pd.read_csv(PROCESSED / "test_labels.csv").set_index("contract_id")
test_pos = test_labels.sum().reindex(train_labels.columns, fill_value=0)

categories = [c for c in train_labels.columns if test_pos[c] > 0]   # 40 predicted
EVAL_CATEGORIES = [c for c in categories if c not in TRIVIAL]       # 38 averaged
assert len(categories) == 40 and len(EVAL_CATEGORIES) == 38, \
    "categories differ from main.ipynb section 4.5 - re-run it first"
cat_to_idx = {c: i for i, c in enumerate(categories)}


def build_label_matrix(label_lists):
    """List of category names per chunk -> 0/1 matrix [n_chunks, 40]."""
    Y = np.zeros((len(label_lists), len(categories)), dtype=np.int8)
    for row, labels in enumerate(label_lists):
        for name in labels:
            j = cat_to_idx.get(name)   # None for the category removed in 2.3
            if j is not None:
                Y[row, j] = 1
    return Y


# 3. SPLIT (by contract as agreed upon)
fit_df = train_chunks[train_chunks["split"] == "train"].reset_index(drop=True)
val_df = train_chunks[train_chunks["split"] == "val"].reset_index(drop=True)
test_df = test_chunks.reset_index(drop=True)

y_train = build_label_matrix(fit_df["labels"])
y_val = build_label_matrix(val_df["labels"])
y_test = build_label_matrix(test_df["labels"])

# No contract may appear on two sides of the split (data leakage)
assert not set(fit_df["contract_id"]) & set(val_df["contract_id"])
assert not set(train_chunks["contract_id"]) & set(test_df["contract_id"])

for name, d, Y in [("fit", fit_df, y_train), ("val", val_df, y_val), ("test", test_df, y_test)]:
    print(f"{name:5s} {len(d):>6,} chunks  {d['contract_id'].nunique():>3} contracts  "
          f"{Y.mean():.2%} positive")

# 4. TF-IDF 
# Vocabulary and IDF weights learned from the fit chunks ONLY (no leakage).
# No stop-word removal: standard lists delete "not", "no", "any", which change
# what a legal clause means ("shall NOT assign").
tfidf = TfidfVectorizer(ngram_range=CFG["ngram_range"], min_df=CFG["min_df"],
                        max_df=CFG["max_df"], max_features=CFG["max_features"],
                        sublinear_tf=CFG["sublinear_tf"])
X_train = tfidf.fit_transform(fit_df["text"])
X_val = tfidf.transform(val_df["text"])
X_test = tfidf.transform(test_df["text"])
feature_names = tfidf.get_feature_names_out()
print(f"\nTF-IDF: {X_train.shape[1]:,} features")

# 5. TRAIN: one logistic regression per category, pick C on validation
# C is chosen by macro average precision on validation (threshold-free).
# n_jobs is left unset: n_jobs=-1 crashes on sparse input in some versions.


def macro_ap(Y, P):
    ks = [k for k in range(Y.shape[1]) if Y[:, k].sum() > 0]
    return float(np.mean([average_precision_score(Y[:, k], P[:, k]) for k in ks]))


search, candidates = [], {}
for C in CFG["C_grid"]:
    t0 = time.time()
    clf = OneVsRestClassifier(LogisticRegression(
        C=C, class_weight=CFG["class_weight"], solver="liblinear",
        max_iter=2000, random_state=CFG["seed"]))
    clf.fit(X_train, y_train)
    candidates[C] = clf
    search.append(dict(C=C, val_macro_ap=round(macro_ap(y_val, clf.predict_proba(X_val)), 4),
                       seconds=round(time.time() - t0)))
    print(search[-1])

best_C = max(search, key=lambda s: s["val_macro_ap"])["C"]
model = candidates[best_C]
print(f"best C = {best_C}")

# 6. PREDICT + TUNE ONE THRESHOLD PER CATEGORY ON VALIDATION
# A single 0.5 cutoff is wrong for 40 categories with different base rates.
# Try every cutoff where the confusion matrix changes; keep the best-F1 one.
val_probs = model.predict_proba(X_val)
test_probs = model.predict_proba(X_test)

thresholds = []
for k in range(len(categories)):
    if y_val[:, k].sum() == 0:          # nothing to tune on
        thresholds.append(0.5)
        continue
    prec, rec, cuts = precision_recall_curve(y_val[:, k], val_probs[:, k])
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.nan_to_num(2 * prec * rec / (prec + rec))
    thresholds.append(float(cuts[int(np.argmax(f1[:-1]))]))

# 7. SCORE ON TEST (once) 
rows = []
for k, cat in enumerate(categories):
    p, r, f1, _ = precision_recall_fscore_support(
        y_test[:, k], test_probs[:, k] >= thresholds[k], average="binary", zero_division=0)
    has_pos = y_test[:, k].sum() > 0
    rows.append(dict(
        category=cat, precision=p, recall=r, f1=f1,
        auc=roc_auc_score(y_test[:, k], test_probs[:, k]) if has_pos else np.nan,
        ap=average_precision_score(y_test[:, k], test_probs[:, k]) if has_pos else np.nan,
        n_pos_test=int(y_test[:, k].sum()), threshold=thresholds[k],
        scored=cat in EVAL_CATEGORIES))

metrics = pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)
scored = metrics[metrics["scored"] & (metrics["n_pos_test"] > 0)]
summary = {m: round(float(scored[m].mean()), 4) for m in ["f1", "precision", "recall", "auc", "ap"]}

print(f"\nTF-IDF BASELINE (C={best_C}) - macro over {len(scored)} scored categories")
for m, v in summary.items():
    print(f"  macro-{m:9s}: {v:.3f}")
print("\ntrivial categories (not in macro):")
print(metrics[metrics["category"].isin(TRIVIAL)][
    ["category", "precision", "recall", "f1", "n_pos_test"]].round(3).to_string(index=False))
print("\nper category:")
print(metrics[["category", "n_pos_test", "precision", "recall", "f1", "auc"]]
      .round(3).to_string(index=False))

# 8. TOP TERMS PER CATEGORY
top_terms = pd.DataFrame({
    cat: feature_names[np.argsort(model.estimators_[k].coef_.ravel())[::-1][:10]]
    for k, cat in enumerate(categories)
}).T

# 9. SAVE
metrics.to_csv(OUT / "per_category_test_metrics.csv", index=False)
top_terms.to_csv(OUT / "top_terms_per_category.csv")
json.dump(summary, open(OUT / "summary.json", "w"), indent=2)
json.dump(dict(zip(categories, thresholds)), open(OUT / "thresholds.json", "w"), indent=2)
json.dump({**CFG, "C": best_C, "categories": categories,
           "eval_categories": EVAL_CATEGORIES, "c_search": search},
          open(OUT / "config.json", "w"), indent=2)
for name, arr in [("val_probs", val_probs), ("test_probs", test_probs),
                  ("val_labels", y_val), ("test_labels", y_test)]:
    np.save(OUT / f"{name}.npy", arr)
print(f"\nSaved to {OUT}/")
