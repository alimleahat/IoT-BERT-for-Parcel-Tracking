"""
export_seq32.py — re-export the already-fine-tuned bert-tiny intent model
at seq_len=32 instead of 64.

The trained weights don't need to change — bert-tiny's positional embeddings
cover [0, 512] and our queries are 5-10 tokens after WordPiece, so 32 is
ample. Cutting seq_len in half should reduce on-device latency by roughly
2× because:
  - attention (Q@K^T, scores@V) scales O(seq_len^2)
  - all FCs over the seq dim scale O(seq_len)
  - tensor arena activations halve

Outputs:
  tflite_models/bert_tiny_intent_dynamic_seq32.tflite
And reports accuracy on the held-out 73-sample test set so we can verify
no regression vs. the seq_len=64 number (91.78%).
"""
import json
import os
import time

import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification

# Reuse the same labels as finetune_and_export.py
LABEL2ID = {
    "VIEW_ALL": 0,
    "VIEW_ORDER": 1,
    "FILTER_BY_DEPOT": 2,
    "SHOW_HISTORY": 3,
    "CALCULATE_COST": 4,
    "SEARCH_ORDER": 5,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

NEW_SEQ_LEN = 32
CHECKPOINT_DIR = "finetuned_model"
OUT_DIR = "tflite_models"
OUT_PATH = os.path.join(OUT_DIR, f"bert_tiny_intent_dynamic_seq{NEW_SEQ_LEN}.tflite")

os.makedirs(OUT_DIR, exist_ok=True)

print(f"Loading {CHECKPOINT_DIR}/ ...")
tokenizer = BertTokenizer.from_pretrained(CHECKPOINT_DIR)
tf_model = TFBertForSequenceClassification.from_pretrained(
    CHECKPOINT_DIR, from_pt=True
)
print("Model loaded.")

# --- Sanity-check TF model accuracy at the new seq_len BEFORE quantization ---
with open("data/test.json") as f:
    test_data = json.load(f)

print(f"\nValidating TF model at seq_len={NEW_SEQ_LEN} on {len(test_data)} test samples...")
correct = 0
for sample in test_data:
    enc = tokenizer(
        sample["text"],
        return_tensors="tf",
        max_length=NEW_SEQ_LEN,
        padding="max_length",
        truncation=True,
    )
    logits = tf_model(enc).logits.numpy()
    pred = int(np.argmax(logits, axis=-1)[0])
    if pred == LABEL2ID[sample["label"]]:
        correct += 1
acc = correct / len(test_data)
print(f"  TF model accuracy at seq_len={NEW_SEQ_LEN}: {correct}/{len(test_data)} = {acc:.4f}")
if acc < 0.85:
    print("\n[WARN] Accuracy dropped below the 0.85 KPI target. Aborting export.")
    raise SystemExit(1)


# --- Build a concrete function pinned to seq_len=32 ---
@tf.function(input_signature=[
    tf.TensorSpec(shape=[1, NEW_SEQ_LEN], dtype=tf.int32, name="input_ids"),
    tf.TensorSpec(shape=[1, NEW_SEQ_LEN], dtype=tf.int32, name="attention_mask"),
])
def serving_fn(input_ids, attention_mask):
    out = tf_model(input_ids=input_ids, attention_mask=attention_mask)
    return {"logits": out.logits}


print(f"\nConverting to TFLite (dynamic-range INT8) at seq_len={NEW_SEQ_LEN}...")
concrete = serving_fn.get_concrete_function()
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete])
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_bytes = converter.convert()
with open(OUT_PATH, "wb") as f:
    f.write(tflite_bytes)
print(f"Saved: {OUT_PATH} ({os.path.getsize(OUT_PATH)/1024/1024:.2f} MB)")


# --- Benchmark the TFLite model on the test set ---
print(f"\nBenchmarking TFLite model at seq_len={NEW_SEQ_LEN}...")
interp = tf.lite.Interpreter(model_path=OUT_PATH)
interp.allocate_tensors()
in_details = interp.get_input_details()
out_details = interp.get_output_details()

ids_idx = mask_idx = None
for d in in_details:
    if "input_ids" in d["name"]:
        ids_idx = d["index"]
    elif "attention_mask" in d["name"]:
        mask_idx = d["index"]

correct = 0
latencies = []
for sample in test_data:
    enc = tokenizer(
        sample["text"],
        max_length=NEW_SEQ_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="np",
    )
    interp.set_tensor(ids_idx, enc["input_ids"].astype(np.int32))
    interp.set_tensor(mask_idx, enc["attention_mask"].astype(np.int32))
    t0 = time.perf_counter()
    interp.invoke()
    t1 = time.perf_counter()
    latencies.append((t1 - t0) * 1000)
    pred = int(np.argmax(interp.get_tensor(out_details[0]["index"]), axis=-1)[0])
    if pred == LABEL2ID[sample["label"]]:
        correct += 1

acc_tflite = correct / len(test_data)
print(f"  TFLite (dynamic-range, seq_len={NEW_SEQ_LEN}) accuracy: "
      f"{correct}/{len(test_data)} = {acc_tflite:.4f}")
print(f"  Latency on laptop (ms): "
      f"mean={np.mean(latencies):.2f}  "
      f"median={np.median(latencies):.2f}  "
      f"p95={np.percentile(latencies, 95):.2f}")
print(f"\nDone. Copy {OUT_PATH} → esp32_firmware/main/bert_model.tflite "
      f"and bump MAX_SEQ_LEN to {NEW_SEQ_LEN} in tokenizer.h.")
