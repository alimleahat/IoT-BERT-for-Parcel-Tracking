# Training and reproduction notes

Run root-level scripts from the repository root: their data and checkpoint paths are relative to the current directory.

1. `python generate_dataset.py` writes the synthetic phrase splits.
2. `python finetune_and_export.py` trains BERT-tiny and exports model variants.
3. `python export_seq32.py` loads the fine-tuned checkpoint and exports a sequence-length-32 dynamic-range model.
4. Copy `tflite_models/bert_tiny_intent_dynamic_seq32.tflite` to `esp32_firmware/main/bert_model.tflite` before rebuilding firmware.

Training uses PyTorch, Transformers, NumPy, and scikit-learn. Export additionally requires TensorFlow and the Transformers TensorFlow BERT classes. The original repository did not lock a complete training/export environment, so package compatibility must be established before claiming a reproducible fresh training run. The lightweight host requirements file intentionally does not install the ML toolchain.

`finetuned_model/` contains configuration and tokenizer artifacts, but the weight file is ignored. A model checkpoint must exist before `export_seq32.py` can run.

## Evaluation interpretation

`finetune_and_export.py` evaluates on `data/test.json` every epoch and saves the checkpoint when that score improves. Despite the filename, this split therefore functions as validation data. The historical 67/73 result should not be described as an untouched final-test result. A stronger experiment would add a separately collected final test set, compare model-only and rule-assisted predictions, and log per-class results and end-to-end latency.

`kpi_latency.csv` records 30 final inference measurements. The original project notes include historical claims and intended extensions; the root README distinguishes those from directly committed evidence.
