#!/usr/bin/env python3
"""Find demo-robust phrases per intent: test many candidates against the
fine-tuned HF model and keep only the ones with high confidence correct."""

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

LABELS = ["VIEW_ALL", "VIEW_ORDER", "FILTER_BY_DEPOT",
          "SHOW_HISTORY", "CALCULATE_COST", "SEARCH_ORDER"]

CANDIDATES = {
    "VIEW_ALL": [
        "show me all orders",
        "list all active orders",
        "show everything",
        "give me the full list",
        "display all orders",
    ],
    "VIEW_ORDER": [
        "view order 101",
        "show order 515",
        "display details for order 101",
        "open order 515",
        "look up order 101",
    ],
    "FILTER_BY_DEPOT": [
        "filter by depot FadEx",
        "show orders from depot USP",
        "orders from depot DLH",
        "filter orders by RoyalMile depot",
        "list orders at PDP depot",
    ],
    "SHOW_HISTORY": [
        "show history",
        "show my order history",
        "display delivered orders",
        "list past orders",
        "view previous orders",
    ],
    "CALCULATE_COST": [
        "calculate the cost of order 101",
        "cost of order 515",
        "how much does order 101 cost",
        "what is the cost for a 5kg package",
        "tell me the cost of order 101",
    ],
    "SEARCH_ORDER": [
        "search for order 101",
        "find order 515",
        "look up order 101 in history",
        "search orders by order ID 515",
        "find order with ID 101",
    ],
}

SEQ_LEN = 64

tok = AutoTokenizer.from_pretrained("finetuned_model")
mdl = AutoModelForSequenceClassification.from_pretrained("finetuned_model")
mdl.eval()

print(f"{'Intent':<17} {'Score':>6}  Phrase")
print("-" * 80)
for intent, phrases in CANDIDATES.items():
    expected = LABELS.index(intent)
    results = []
    for text in phrases:
        enc = tok(text, padding="max_length", truncation=True,
                  max_length=SEQ_LEN, return_tensors="pt")
        with torch.no_grad():
            logits = mdl(**enc).logits[0].numpy()
        idx = int(np.argmax(logits))
        correct = idx == expected
        # margin = score_expected - score_second_highest
        order = np.argsort(-logits)
        margin = logits[expected] - logits[order[1] if order[0] == expected else order[0]]
        results.append((correct, margin, text, logits))
    # Sort by correctness then by margin
    results.sort(key=lambda r: (-int(r[0]), -r[1]))
    for correct, margin, text, logits in results:
        mark = "✓" if correct else "✗"
        print(f"  {mark} {intent:<14} {margin:+6.2f}  {text}")
    print()
