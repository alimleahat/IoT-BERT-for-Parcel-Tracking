# IoT-BERT for Parcel Tracking

> Natural-language parcel tracking running entirely on a $10 microcontroller.

This project deploys a fine-tuned BERT-tiny transformer onto an ESP32-S3 microcontroller for **fully on-device intent classification** of natural-language parcel-tracking commands. No cloud, no GPU, no WiFi at runtime — the chip classifies queries like *"show me all orders"* or *"calculate the cost of order 101"* in **510 ms** with **91.78 % accuracy**, then queries a local Flask server (running on a laptop) over USB serial.

A live demo UI ships with the Flask server: open `http://127.0.0.1:5001/` after starting the app and click **▶ DEMO** to walk through the system.

---

## Final results

| Metric | Value | Notes |
|---|---|---|
| On-device inference latency (p50) | **510 ms** | n=30 runs, σ=0 ms (perfectly deterministic) |
| Test accuracy | **91.78 %** | 67/73 held-out, identical to the FP32 baseline |
| Model size | **4.22 MB** | dynamic-range INT8 (FP32 was 16.5 MB) |
| Tensor arena | **197 KB / 256 KB** | placed in internal DRAM for speed |
| Hardware | **ESP32-S3-WROOM-1** | 8 MB octal PSRAM, 8 MB flash, ~$10 module |
| Privacy footprint | **0 bytes leave home** | NLP fully on-chip, no internet calls |

The latency journey was **3.7 s → 510 ms (86 % reduction)** without retraining the model — five reproducible config flips:

1. Compiler flag `-Og` (debug) → `-O2` (perf)
2. CPU clock 160 MHz → 240 MHz (rated speed)
3. Data cache 32 KB / 32 B lines → 64 KB / 64 B lines
4. Tensor arena moved from PSRAM (slow external) → internal DRAM (fast on-chip)
5. Model re-exported at sequence length 32 instead of 64 (real queries are 5–10 tokens; positional embeddings cover up to 512)

---

## Architecture

```
   ┌─────────────────────────────────┐
   │  USER  →  types natural-language commands
   └─────────────────────────────────┘
                  ↓ USB serial
   ┌─────────────────────────────────┐
   │  ESP32-S3 (edge inference)      │
   │  • WordPiece tokenizer          │
   │  • TFLite Micro + bert-tiny     │
   │    (with 2 custom kernels)      │
   │  • argmax + entity extraction   │
   │  • emits TX:{intent, params}    │
   └─────────────────────────────────┘
                  ↓ USB serial
   ┌─────────────────────────────────┐
   │  Laptop (Flask + flat-file db)  │
   │  • intent dispatcher            │
   │  • mirror of the original C     │
   │    parcel tracker logic         │
   │  • returns formatted response   │
   └─────────────────────────────────┘
```

### Two custom TFLite Micro kernels (in `esp32_firmware/main/`)

The stock TFLite Micro framework didn't ship two operations bert-tiny needs:

- **`custom_gelu.cc`** — the GELU activation used inside transformer feed-forward blocks. Implemented from the standard tanh approximation.
- **`custom_fc.cc`** — a hybrid FullyConnected kernel that handles **FP32 input × INT8 filter** with per-channel scales. This is the configuration dynamic-range quantization produces, and stock TFLM rejects it outright.

### Hybrid NLP guard layer

After BERT's argmax, a small rule layer in firmware (`post_classify` in `main.c`) handles two cases the model alone can't:
- **`SEARCH_ORDER` suppression** — the model has 6 fixed classes, but `SEARCH_ORDER` is a duplicate of `VIEW_ORDER` semantically. The rule layer suppresses it post-argmax.
- **High-precision keywords** — words like `delivered`, `upcoming`, `find`, `cost` are unambiguous in this domain. If they appear, they override the model. Out-of-distribution phrasings like *"show upcoming orders"* (the word *upcoming* never appeared during training) are caught here without retraining.

### Product-name search

Extending the system without touching the trained model: queries like *"find airpods"* or *"where's my macbook"* go through the rule layer (`find` keyword + no order-ID number) → the entity extractor pulls a candidate name → the server does case-insensitive substring matching across active orders + history.

---

## Repository structure

```
.
├── esp32_firmware/                # ESP-IDF v5.x project
│   ├── main/
│   │   ├── main.c                 # main loop, post-classify guard, JSON building
│   │   ├── inference.cc           # TFLite Micro setup, custom-op registration
│   │   ├── tokenizer.c/h          # WordPiece tokenizer (FNV-1a hash table)
│   │   ├── entity.c/h             # rule-based entity extraction
│   │   ├── custom_gelu.cc/h       # custom GELU kernel
│   │   ├── custom_fc.cc/h         # custom hybrid FP32/INT8 FullyConnected kernel
│   │   ├── bert_model.tflite      # quantized model (4.22 MB, embedded in flash)
│   │   └── vocab.txt              # WordPiece vocabulary (30,522 entries)
│   ├── partitions.csv             # custom flash layout (7.5 MB app partition)
│   ├── sdkconfig.defaults         # ESP-IDF build flags (incl. perf optimizations)
│   └── CMakeLists.txt
│
├── server/                        # Flask laptop server
│   ├── app.py                     # routes: /, /api/run, /api/status, /api/intent
│   ├── handlers.py                # 6 intent handlers — mirrors of original C functions
│   ├── data_layer.py              # flat-file I/O matching the C parcel-tracker format
│   ├── cost.py                    # shipping cost: base + distance·rate + weight·rate
│   ├── time_utils.py              # "3 days left" / "10 hours ago" helpers
│   ├── serial_runner.py           # USB serial bridge — owns /dev/cu.usbmodem*
│   ├── static/index.html          # the demo UI (single-file ~2,000 lines)
│   ├── test_server.py             # functional tests (7/7 passing)
│   ├── test_intent_e2e.py         # end-to-end round-trip test
│   ├── bench_latency.py           # latency benchmarking (n=30)
│   └── data/                      # parcel database flat files
│
├── tflite_models/                 # all quantization variants
├── data/                          # generated training/test JSON
├── generate_dataset.py            # synthetic training-phrase generator (~60/intent)
├── finetune_and_export.py         # PyTorch fine-tuning + TFLite export pipeline
├── export_seq32.py                # re-export at seq_len=32 from existing checkpoint
├── PRESENTATION_PREP.md           # demo-day prep doc (sections 1–15)
└── kpi_latency.csv                # raw bench results (30 runs × 6 phrases)
```

---

## Quick start

### 1. Run the Flask server + UI

```bash
python3 -m venv venv
source venv/bin/activate
pip install flask pyserial
cd server
python app.py
# open http://127.0.0.1:5001/ in your browser
```

If macOS gives the chip a different `usbmodem*` number after a replug:

```bash
SERIAL_PORT=$(ls /dev/cu.usbmodem* | head -1) python server/app.py
```

### 2. Build + flash the firmware (optional — pre-built model is already in the repo)

Requires ESP-IDF v5.x.

```bash
cd esp32_firmware
idf.py set-target esp32s3
idf.py build
idf.py -p /dev/cu.usbmodem* flash monitor
```

### 3. Fine-tune from scratch (optional)

```bash
pip install torch transformers datasets scikit-learn tensorflow
python generate_dataset.py        # produces data/train.json + data/test.json
python finetune_and_export.py     # 50 epochs on MPS, exports 4 TFLite variants
python export_seq32.py            # re-export at seq_len=32 (used in firmware)
cp tflite_models/bert_tiny_intent_dynamic_seq32.tflite \
   esp32_firmware/main/bert_model.tflite
```

---

## Demo Mode

The UI ships with a guided 9-chapter tour. Click **▶ DEMO** in the header. Chapters:

1. Motivation — why on-device NLP
2. Architecture — 3-tier breakdown
3. The build — pipeline from C parcel tracker → fine-tuned BERT → flashed firmware
4. First query · interface tour — fire `show me all orders`, walk through every panel
5. Classifier output — 6 logits, argmax wins
6. Latency optimization — 3.7 s → 510 ms in 5 flips
7. Rule-based override — `show upcoming orders` triggers the hybrid guard
8. Extending without retraining — `find airpods` → name-search via the rule layer
9. Results — KPIs in the Stats drawer

Keyboard: `←` / `→` to navigate, `Esc` to exit.

---

## Notes on training

- Base model: [`prajjwal1/bert-tiny`](https://huggingface.co/prajjwal1/bert-tiny) (4.4 M parameters, 2 layers, hidden=128).
- Training data: 361 hand-written template phrases (≈60 per intent), shuffled with seed `42`, 80/20 split → 288 train + 73 test.
- Recipe: AdamW, lr=`3e-5`, weight_decay=`0.01`, batch=8, 50 epochs, linear warmup over 10 % of steps.
- Hardware: Apple MPS (M-series GPU). Total training time ~5 minutes.
- Final test accuracy: 91.78 % (67/73). Identical to the FP32 baseline after dynamic-range INT8 quantization — quantization preserved every correct prediction.

The fine-tuned PyTorch checkpoint (~2.78 GB) is **not** in the repo; reproduce by running the steps in *Quick start §3*.

---

## Limitations & future work

- **Synthetic dataset only** — 361 hand-typed phrases is small. Real user data would help. The hybrid guard layer mitigates this for known OOD patterns but isn't a permanent fix.
- **No live courier API integration** — the laptop's parcel database is fixed test data. Real deployment would sync against shipper APIs.
- **TFLite Micro dispatch overhead** — about 100 ms of the 510 ms is per-op framework overhead across the model's many small ops. Layer fusion (e.g. combining LayerNorm's six ops into one) would close this gap further.
- **Single-user, single-laptop** — the system is designed for a smart home, not multi-tenant deployment.

Possible extensions:
- BatchMatMul custom kernel for attention (saves a few ms each call).
- WiFi/MQTT bridge as an alternative to USB serial (for actual smart-home deployment).
- Larger model on a chip with more PSRAM (e.g. ESP32-P4 with 32 MB) — bert-mini would fit there.

---

## Acknowledgements

- [`prajjwal1/bert-tiny`](https://huggingface.co/prajjwal1/bert-tiny) — pretrained base model.
- [TensorFlow Lite for Microcontrollers](https://github.com/tensorflow/tflite-micro).
- [Espressif ESP-IDF](https://github.com/espressif/esp-idf) and the [esp-tflite-micro](https://github.com/espressif/esp-tflite-micro) component.
- The original C parcel tracker (CW1, ELEC2302) — its flat-file format is preserved here for compatibility.
