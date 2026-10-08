# =============================================================================
# train.py  --  STEP 2 of Task 1
#
# GOAL: Fine-tune a transformer (Legal-BERT) so that, given a contract chunk,
# it predicts which of the 41 clause types are present (multi-label: a chunk
# can have several types at once, or none).
#
# BEFORE RUNNING: run prep_data.py first (it creates the "data" folder).
# OUTPUT (in the "outputs" folder):
#   config.json                      settings used (for Task 5 comparisons)
#   model/                           the trained model
#   val_probs.npy, test_probs.npy    predicted probabilities (for Task 4)
#   val_labels.npy, test_labels.npy  the true labels
#   thresholds.json                  best cutoff per category
#   per_category_test_metrics.csv    precision/recall/F1 per category (Task 2)
# =============================================================================

import json, os
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset                       # loads our .jsonl files
from sklearn.metrics import precision_recall_fscore_support  # P / R / F1
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          Trainer, TrainingArguments, set_seed)

# ---------------------------- SETTINGS (edit here) ---------------------------
CFG = dict(
    model_name="nlpaueb/legal-bert-base-uncased",  # pretrained on legal text
                                                   # (if slow/CPU: "distilbert-base-uncased")
    max_length=256,     # max tokens per chunk fed to the model
    lr=2e-5,            # learning rate: how big each update step is
    batch_size=16,      # chunks per step (lower to 8 or 4 if out of memory)
    epochs=4,           # full passes over the training data (use 2 if slow)
    seed=42,            # fixed randomness so runs are repeatable
    weight_decay=0.01,  # mild regularization to reduce overfitting
)
OUT = "outputs"
# -----------------------------------------------------------------------------

os.makedirs(OUT, exist_ok=True)
set_seed(CFG["seed"])
json.dump(CFG, open(f"{OUT}/config.json", "w"), indent=2)  # save settings

categories = json.load(open("data/categories.json"))  # the 41 names, in order
K = len(categories)                                    # K = 41

# --- 1. LOAD AND TOKENIZE DATA ----------------------------------------------
ds = load_dataset("json", data_files={s: f"data/{s}.jsonl"
                                      for s in ["train", "val", "test"]})
tok = AutoTokenizer.from_pretrained(CFG["model_name"])

def encode(batch):
    # Turn text into token IDs the model understands (cut off at max_length)
    enc = tok(batch["text"], truncation=True, max_length=CFG["max_length"])
    # The loss function needs labels as floats (0.0 / 1.0), not ints
    enc["labels"] = [[float(x) for x in l] for l in batch["labels"]]
    return enc

# Apply to every split; drop the raw text columns since the model doesn't use them
ds = ds.map(encode, batched=True, remove_columns=["contract", "text"])

# --- 2. BUILD THE MODEL ------------------------------------------------------
# Pretrained BERT + a fresh output layer with 41 numbers (one per category).
# problem_type="multi_label_classification" makes it use BCE-with-logits loss:
# each category is treated as its own yes/no question.
model = AutoModelForSequenceClassification.from_pretrained(
    CFG["model_name"], num_labels=K, problem_type="multi_label_classification")

class MultiLabelTrainer(Trainer):
    """Same as the normal Trainer, but with a place to plug in class weights.

    TASK 3 HOOK (class imbalance): your teammates can set
        trainer.pos_weight = torch.tensor([...41 numbers...])
    Bigger weight for a rare category = mistakes on it are punished more.
    Left as None here, which means no reweighting (a clean baseline).
    """
    pos_weight = None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")      # pull the true labels out
        outputs = model(**inputs)          # run the model -> raw scores (logits)
        pw = (self.pos_weight.to(outputs.logits.device)
              if self.pos_weight is not None else None)
        loss = torch.nn.BCEWithLogitsLoss(pos_weight=pw)(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    # Runs after each epoch on the validation set so we can pick the best epoch
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))      # sigmoid: scores -> probabilities 0..1
    preds = (probs >= 0.5).astype(int)     # simple 0.5 cutoff just for monitoring
    # macro = average F1 across categories, each category counts equally
    _, _, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0)
    return {"macro_f1": f1}

# --- 3. TRAIN ----------------------------------------------------------------
args = TrainingArguments(
    output_dir=f"{OUT}/checkpoints",
    learning_rate=CFG["lr"],
    weight_decay=CFG["weight_decay"],
    per_device_train_batch_size=CFG["batch_size"],
    per_device_eval_batch_size=64,
    num_train_epochs=CFG["epochs"],
    seed=CFG["seed"],
    eval_strategy="epoch",        # evaluate on val after every epoch
                                  # (older transformers versions: evaluation_strategy)
    save_strategy="epoch",        # save a checkpoint every epoch
    load_best_model_at_end=True,  # at the end, keep the best epoch, not the last
    metric_for_best_model="macro_f1",
    save_total_limit=1,           # keep only one checkpoint to save disk space
    fp16=torch.cuda.is_available(),  # faster math if there's a GPU
    report_to="none",             # don't try to log to external services
)
trainer = MultiLabelTrainer(
    model=model, args=args,
    train_dataset=ds["train"], eval_dataset=ds["val"],
    tokenizer=tok,   # if you get an error here, change to processing_class=tok
    compute_metrics=compute_metrics,
)
trainer.train()                      # this is the slow part
trainer.save_model(f"{OUT}/model")   # save the final model

# --- 4. PREDICT AND SAVE RAW PROBABILITIES -----------------------------------
# Saved so teammates (Tasks 2, 4, 5) can reuse them without retraining.
sigmoid = lambda x: 1 / (1 + np.exp(-x))
val_p = sigmoid(trainer.predict(ds["val"]).predictions)    # shape [n_chunks, 41]
test_p = sigmoid(trainer.predict(ds["test"]).predictions)
val_y = np.array(ds["val"]["labels"])                      # true labels
test_y = np.array(ds["test"]["labels"])
for name, arr in [("val_probs", val_p), ("test_probs", test_p),
                  ("val_labels", val_y), ("test_labels", test_y)]:
    np.save(f"{OUT}/{name}.npy", arr)

# --- 5. PICK ONE THRESHOLD PER CATEGORY (using VAL only) ---------------------
# Instead of always saying "present if probability >= 0.5", find the cutoff
# that gives the best F1 for each category. Done on val, never on test,
# so the test score stays honest.
grid = np.arange(0.05, 0.96, 0.05)   # candidate cutoffs: 0.05, 0.10, ... 0.95
thresholds = []
for k in range(K):
    if val_y[:, k].sum() == 0:       # no positives in val -> can't tune, use 0.5
        thresholds.append(0.5)
        continue
    f1s = [precision_recall_fscore_support(
               val_y[:, k], val_p[:, k] >= t, average="binary", zero_division=0)[2]
           for t in grid]
    thresholds.append(float(grid[int(np.argmax(f1s))]))  # cutoff with best F1
json.dump(dict(zip(categories, thresholds)), open(f"{OUT}/thresholds.json", "w"), indent=2)

# --- 6. FINAL REPORT ON TEST (run once, don't tune on this) ------------------
rows = []
for k, cat in enumerate(categories):
    p, r, f1, _ = precision_recall_fscore_support(
        test_y[:, k], test_p[:, k] >= thresholds[k],
        average="binary", zero_division=0)
    rows.append(dict(category=cat, precision=p, recall=r, f1=f1,
                     n_pos_test=int(test_y[:, k].sum()),  # how many real examples
                     threshold=thresholds[k]))
df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/per_category_test_metrics.csv", index=False)  # hand this to Task 2/4
print(df.round(3).to_string(index=False))
# Only average categories that actually have test examples; others are undefined
print(f"\nMacro-F1 (categories with test positives): "
      f"{df[df.n_pos_test > 0].f1.mean():.3f}")
