#!/usr/bin/env python3
"""
diag_model_accuracy.py — sanity-check intent classification at three levels:

  1. HuggingFace fine-tuned model (ground truth)
  2. TFLite FP32 (round-trip to TFLite)
  3. TFLite INT8 (what actually runs on the ESP32)

If (1) is right and (3) is wrong, quantization ate the accuracy.
If (1) is already wrong, the fine-tune or dataset is the problem.
"""

import numpy as np
import os

LABELS = ["VIEW_ALL", "VIEW_ORDER", "FILTER_BY_DEPOT",
          "SHOW_HISTORY", "CALCULATE_COST", "SEARCH_ORDER"]

TESTS = [
    ("show me all orders",         "VIEW_ALL"),
    ("view order 101",             "VIEW_ORDER"),
    ("filter by birmingham depot", "FILTER_BY_DEPOT"),
    ("show history",               "SHOW_HISTORY"),
    ("how much did I spend",       "CALCULATE_COST"),
    ("find order for John Smith",  "SEARCH_ORDER"),
]

SEQ_LEN = 64


def run_hf():
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    tok = AutoTokenizer.from_pretrained("finetuned_model")
    mdl = AutoModelForSequenceClassification.from_pretrained("finetuned_model")
    mdl.eval()

    print("\n=== (1) HuggingFace fine-tuned model ===")
    for text, exp in TESTS:
        enc = tok(text, padding="max_length", truncation=True,
                  max_length=SEQ_LEN, return_tensors="pt")
        with torch.no_grad():
            out = mdl(**enc).logits[0].numpy()
        idx = int(np.argmax(out))
        mark = "✓" if LABELS[idx] == exp else "✗"
        print(f"  {mark} {text!r:40s} pred={LABELS[idx]:<15s} exp={exp:<15s} "
              f"scores={[f'{x:.2f}' for x in out]}")
    return tok


def tokenize_for_tflite(tok, text):
    enc = tok(text, padding="max_length", truncation=True,
              max_length=SEQ_LEN, return_tensors="np")
    ids  = enc["input_ids"].astype(np.int32)
    mask = enc["attention_mask"].astype(np.int32)
    return ids, mask


def run_tflite(path, tok, title):
    import tensorflow as tf
    if not os.path.exists(path):
        print(f"\n=== {title}: skipped (not found at {path})")
        return
    interp = tf.lite.Interpreter(model_path=path)
    interp.allocate_tensors()
    inp = interp.get_input_details()
    out = interp.get_output_details()
    # Figure out which input is ids vs mask by name (order varies).
    name_to_idx = {d["name"]: d["index"] for d in inp}
    ids_idx  = next(i["index"] for i in inp if "input_ids" in i["name"])
    mask_idx = next(i["index"] for i in inp if "attention_mask" in i["name"])

    print(f"\n=== {title} ({path}) ===")
    for text, exp in TESTS:
        ids, mask = tokenize_for_tflite(tok, text)
        interp.set_tensor(ids_idx,  ids)
        interp.set_tensor(mask_idx, mask)
        interp.invoke()
        logits = interp.get_tensor(out[0]["index"])[0]
        idx = int(np.argmax(logits))
        mark = "✓" if LABELS[idx] == exp else "✗"
        print(f"  {mark} {text!r:40s} pred={LABELS[idx]:<15s} exp={exp:<15s} "
              f"scores={[f'{x:.2f}' for x in logits]}")


if __name__ == "__main__":
    tok = run_hf()
    run_tflite("tflite_models/bert_tiny_intent_fp32.tflite",    tok, "(2) TFLite FP32")
    run_tflite("tflite_models/bert_tiny_intent_dynamic.tflite", tok, "(3) TFLite dynamic-range (INT8 weights / FP32 acts)")
    run_tflite("tflite_models/bert_tiny_intent_fp16.tflite",    tok, "(4) TFLite FP16")
    run_tflite("tflite_models/bert_tiny_intent_int8.tflite",    tok, "(5) TFLite INT8 full (currently on-device)")
