#!/usr/bin/env python3
"""Hunt for a strong SEARCH_ORDER phrase."""
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

LABELS = ["VIEW_ALL", "VIEW_ORDER", "FILTER_BY_DEPOT",
          "SHOW_HISTORY", "CALCULATE_COST", "SEARCH_ORDER"]

CANDIDATES = [
    "search history for package 123",
    "search orders by order ID 515",
    "find package 123 in history",
    "locate order 123",
    "search for package 123 across active and history",
    "search for order with ID 100",  # straight from training set
    "find overnight shipping orders",  # straight from training set
    "find orders with complaints",  # straight from training set
    "search orders by warehouse",  # straight from training set
    "look up orders with complaints",  # straight from training set
]

SEQ_LEN = 64
tok = AutoTokenizer.from_pretrained("finetuned_model")
mdl = AutoModelForSequenceClassification.from_pretrained("finetuned_model")
mdl.eval()

print(f"{'Pred':<16} {'Margin':>7}  Phrase")
for text in CANDIDATES:
    enc = tok(text, padding="max_length", truncation=True,
              max_length=SEQ_LEN, return_tensors="pt")
    with torch.no_grad():
        logits = mdl(**enc).logits[0].numpy()
    idx = int(np.argmax(logits))
    order = np.argsort(-logits)
    pred = LABELS[idx]
    margin = logits[idx] - logits[order[1]]
    mark = "✓" if pred == "SEARCH_ORDER" else " "
    print(f"{mark} {pred:<14} {margin:+7.2f}  {text}")
