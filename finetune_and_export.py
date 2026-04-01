"""Fine-tune bert-tiny for intent classification, export to TFLite INT8, benchmark."""

import json
import os
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import classification_report, confusion_matrix

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
MODEL_NAME = "prajjwal1/bert-tiny"
SEQ_LEN = 64  # short utterances don't need 128
BATCH_SIZE = 8   # smaller batches = more updates per epoch for small datasets
EPOCHS = 50
LR = 3e-5
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

LABEL2ID = {
    "VIEW_ALL": 0,
    "VIEW_ORDER": 1,
    "FILTER_BY_DEPOT": 2,
    "SHOW_HISTORY": 3,
    "CALCULATE_COST": 4,
    "SEARCH_ORDER": 5,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = len(LABEL2ID)

OUTPUT_DIR = "finetuned_model"
TFLITE_DIR = "tflite_models"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TFLITE_DIR, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────
class IntentDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=SEQ_LEN):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        enc = self.tokenizer(
            item["text"],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(LABEL2ID[item["label"]], dtype=torch.long),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────
def train_model():
    print(f"Device: {DEVICE}")
    print(f"Model:  {MODEL_NAME}")
    print(f"Labels: {LABEL2ID}\n")

    # Load data
    with open("data/train.json") as f:
        train_data = json.load(f)
    with open("data/test.json") as f:
        test_data = json.load(f)

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS, id2label=ID2LABEL, label2id=LABEL2ID
    )
    model.to(DEVICE)

    train_ds = IntentDataset(train_data, tokenizer)
    test_ds = IntentDataset(test_data, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )

    # Train
    print(f"Training for {EPOCHS} epochs ({len(train_data)} samples)...")
    print("-" * 60)
    best_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch in train_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()

            preds = torch.argmax(outputs.logits, dim=-1)
            correct += (preds == batch["labels"]).sum().item()
            total += batch["labels"].size(0)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        train_acc = correct / total
        avg_loss = total_loss / len(train_loader)

        # Evaluate
        model.eval()
        eval_correct = 0
        eval_total = 0
        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                outputs = model(**batch)
                preds = torch.argmax(outputs.logits, dim=-1)
                eval_correct += (preds == batch["labels"]).sum().item()
                eval_total += batch["labels"].size(0)
        eval_acc = eval_correct / eval_total

        print(
            f"Epoch {epoch+1:2d}/{EPOCHS} | "
            f"Loss: {avg_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Test Acc: {eval_acc:.4f}"
        )

        if eval_acc > best_acc:
            best_acc = eval_acc
            model.save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"\nBest test accuracy: {best_acc:.4f}")
    return tokenizer, test_data


# ──────────────────────────────────────────────────────────────────────────────
# Export to TFLite INT8
# ──────────────────────────────────────────────────────────────────────────────
def export_tflite_int8(tokenizer):
    import tensorflow as tf
    from transformers import TFBertForSequenceClassification

    print("\n" + "=" * 60)
    print("Exporting fine-tuned model to TFLite INT8")
    print("=" * 60)

    # Load fine-tuned PyTorch model into TF
    tf_model = TFBertForSequenceClassification.from_pretrained(OUTPUT_DIR, from_pt=True)

    # ── Sanity-check TF model BEFORE TFLite conversion ──
    with open("data/test.json") as f:
        _test = json.load(f)
    tf_correct = 0
    for sample in _test:
        enc = tokenizer(sample["text"], return_tensors="tf",
                        max_length=SEQ_LEN, padding="max_length", truncation=True)
        logits = tf_model(enc).logits.numpy()
        pred = np.argmax(logits, axis=-1)[0]
        if pred == LABEL2ID[sample["label"]]:
            tf_correct += 1
    print(f"TF model accuracy (pre-TFLite): {tf_correct}/{len(_test)} = {tf_correct/len(_test):.4f}")

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[1, SEQ_LEN], dtype=tf.int32, name="input_ids"),
        tf.TensorSpec(shape=[1, SEQ_LEN], dtype=tf.int32, name="attention_mask"),
    ])
    def serving_fn(input_ids, attention_mask):
        output = tf_model(input_ids=input_ids, attention_mask=attention_mask)
        return {"logits": output.logits}

    concrete_func = serving_fn.get_concrete_function()

    # Use FULL training set as calibration data for accurate quantization ranges
    with open("data/train.json") as f:
        calib_data = json.load(f)

    def representative_dataset():
        for sample in calib_data:
            inputs = tokenizer(
                sample["text"],
                return_tensors="np",
                max_length=SEQ_LEN,
                padding="max_length",
                truncation=True,
            )
            yield [
                inputs["input_ids"].astype(np.int32),
                inputs["attention_mask"].astype(np.int32),
            ]

    # ── Strategy 1: Dynamic range quantization (weights-only INT8) ──
    print("\n--- Strategy 1: Dynamic Range (weights-only INT8) ---")
    converter1 = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    converter1.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_dynamic = converter1.convert()
    path_dynamic = os.path.join(TFLITE_DIR, "bert_tiny_intent_dynamic.tflite")
    with open(path_dynamic, "wb") as f:
        f.write(tflite_dynamic)
    print(f"  Saved: {path_dynamic} ({os.path.getsize(path_dynamic)/1024/1024:.2f} MB)")

    # ── Strategy 2: Full INT8 with real calibration data ──
    print("--- Strategy 2: Full INT8 (weights + activations) ---")
    converter2 = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    converter2.optimizations = [tf.lite.Optimize.DEFAULT]
    converter2.representative_dataset = representative_dataset
    converter2.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
        tf.lite.OpsSet.TFLITE_BUILTINS,
    ]
    tflite_int8 = converter2.convert()
    path_int8 = os.path.join(TFLITE_DIR, "bert_tiny_intent_int8.tflite")
    with open(path_int8, "wb") as f:
        f.write(tflite_int8)
    print(f"  Saved: {path_int8} ({os.path.getsize(path_int8)/1024/1024:.2f} MB)")

    # ── Strategy 3: Float16 quantization ──
    print("--- Strategy 3: Float16 ---")
    converter3 = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    converter3.optimizations = [tf.lite.Optimize.DEFAULT]
    converter3.target_spec.supported_types = [tf.float16]
    tflite_fp16 = converter3.convert()
    path_fp16 = os.path.join(TFLITE_DIR, "bert_tiny_intent_fp16.tflite")
    with open(path_fp16, "wb") as f:
        f.write(tflite_fp16)
    print(f"  Saved: {path_fp16} ({os.path.getsize(path_fp16)/1024/1024:.2f} MB)")

    return {
        "dynamic": path_dynamic,
        "int8": path_int8,
        "fp16": path_fp16,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Benchmark TFLite model
# ──────────────────────────────────────────────────────────────────────────────
def benchmark_tflite(tflite_path, tokenizer, test_data):
    import tensorflow as tf

    print("\n" + "=" * 60)
    print("Benchmarking TFLite INT8 Model")
    print("=" * 60)

    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"\nInput details:")
    for d in input_details:
        print(f"  {d['name']:30s} shape={d['shape']} dtype={d['dtype']}")
    print(f"Output details:")
    for d in output_details:
        print(f"  {d['name']:30s} shape={d['shape']} dtype={d['dtype']}")

    # Find input tensor indices
    input_ids_detail = None
    attention_mask_detail = None
    for d in input_details:
        if "input_ids" in d["name"]:
            input_ids_detail = d
        elif "attention_mask" in d["name"]:
            attention_mask_detail = d

    y_true = []
    y_pred = []
    latencies = []

    for sample in test_data:
        enc = tokenizer(
            sample["text"],
            max_length=SEQ_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )

        input_ids = enc["input_ids"].astype(np.int32)
        attention_mask = enc["attention_mask"].astype(np.int32)

        interpreter.set_tensor(input_ids_detail["index"], input_ids)
        interpreter.set_tensor(attention_mask_detail["index"], attention_mask)

        t0 = time.perf_counter()
        interpreter.invoke()
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)

        output = interpreter.get_tensor(output_details[0]["index"])
        pred_id = np.argmax(output, axis=-1)[0]
        y_true.append(LABEL2ID[sample["label"]])
        y_pred.append(pred_id)

    # Results
    accuracy = sum(1 for a, b in zip(y_true, y_pred) if a == b) / len(y_true)

    print(f"\n{'='*60}")
    print(f"RESULTS  ({len(test_data)} test samples)")
    print(f"{'='*60}")
    print(f"\nOverall Accuracy: {accuracy:.4f} ({sum(1 for a, b in zip(y_true, y_pred) if a == b)}/{len(y_true)})")
    print(f"\nLatency (ms): mean={np.mean(latencies):.2f}  "
          f"median={np.median(latencies):.2f}  "
          f"p95={np.percentile(latencies, 95):.2f}  "
          f"p99={np.percentile(latencies, 99):.2f}")

    target_names = [ID2LABEL[i] for i in range(NUM_LABELS)]
    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=target_names, digits=4))

    print("Confusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    # Header
    header = "            " + "".join(f"{ID2LABEL[i][:8]:>10s}" for i in range(NUM_LABELS))
    print(header)
    for i in range(NUM_LABELS):
        row = f"{ID2LABEL[i]:12s}" + "".join(f"{cm[i][j]:10d}" for j in range(NUM_LABELS))
        print(row)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Step 1: Fine-tune
    tokenizer, test_data = train_model()

    # Step 2: Export (multiple strategies)
    tflite_paths = export_tflite_int8(tokenizer)

    # Step 3: Benchmark each strategy
    for name, path in tflite_paths.items():
        print(f"\n{'#'*60}")
        print(f"# Benchmarking: {name.upper()}")
        print(f"{'#'*60}")
        benchmark_tflite(path, tokenizer, test_data)

    # Final comparison table
    print(f"\n{'='*60}")
    print("FINAL COMPARISON")
    print(f"{'='*60}")
    print(f"  {'Strategy':<20s} {'Size (MB)':>10s}")
    print(f"  {'-'*20} {'-'*10}")
    for name, path in tflite_paths.items():
        sz = os.path.getsize(path) / 1024 / 1024
        print(f"  {name:<20s} {sz:>10.2f}")
