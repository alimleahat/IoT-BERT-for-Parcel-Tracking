# TinyBERT MCU-Based NLP for Smart Home Parcel Tracking

An edge-AI system that runs a quantised BERT model on an ESP32-S3 microcontroller to understand natural-language parcel tracking queries. The device connects to a Flask server that maintains order, depot and delivery history data, enabling voice/text-based parcel status lookups without cloud-based NLP.

## Hardware

- **MCU:** ESP32-S3 (8 MB PSRAM, 8 MB flash)
- **Framework:** ESP-IDF with TensorFlow Lite Micro

## Repository Structure

```
├── esp32_firmware/          # ESP-IDF project
│   ├── main/                # Firmware source (inference, tokenizer, entity extraction)
│   │   ├── main.c           # Entry point — WiFi, HTTP, orchestration
│   │   ├── inference.cc     # TFLite Micro inference runner
│   │   ├── tokenizer.c/h    # WordPiece tokenizer (on-device)
│   │   ├── entity.c/h       # Lightweight entity extraction
│   │   ├── custom_gelu.cc/h # Custom GELU op for TFLite
│   │   ├── bert_model.tflite# Quantised model embedded in firmware
│   │   └── vocab.txt        # BERT vocabulary
│   ├── partitions.csv
│   ├── sdkconfig.defaults
│   └── CMakeLists.txt
├── server/                  # Python/Flask back-end
│   ├── app.py               # Flask application entry point
│   ├── handlers.py          # Request handlers / business logic
│   ├── data_layer.py        # File-based data access
│   ├── cost.py              # Shipping cost calculations
│   ├── time_utils.py        # Date/time helpers
│   ├── test_server.py       # Server tests
│   └── data/                # Flat-file data (orders, depots, history)
├── tflite_models/           # Exported TFLite models (various quantisations)
├── data/                    # Training / test datasets (JSON)
├── finetune_and_export.py   # Fine-tune TinyBERT and export to TFLite
├── convert_to_tflite_int8.py# Standalone int8 quantisation script
└── generate_dataset.py      # Synthetic training data generator
```

## Model

The fine-tuned PyTorch checkpoint (~2.78 GB) is **not** included in this repository.

To obtain the model, fine-tune from scratch using `finetune_and_export.py` (see [Fine-Tuning](#2-fine-tuning-optional) below).

The quantised `.tflite` model files **are** included under `tflite_models/` and embedded in `esp32_firmware/main/bert_model.tflite`.

## Setup

### 1. Python / Flask Server

```bash
python3 -m venv venv
source venv/bin/activate
pip install flask
cd server
python app.py
```

The server listens on the local network for requests from the ESP32.

### 2. Fine-Tuning (optional)

```bash
pip install torch transformers datasets scikit-learn tensorflow
python generate_dataset.py      # create synthetic training data
python finetune_and_export.py   # fine-tune and export to TFLite
```

### 3. ESP-IDF Firmware Build

Requires [ESP-IDF v5.x](https://docs.espressif.com/projects/esp-idf/en/stable/esp32s3/get-started/).

```bash
cd esp32_firmware
# Set your WiFi credentials
cp sdkconfig.defaults sdkconfig
# Edit sdkconfig or use: idf.py menuconfig

idf.py set-target esp32s3
idf.py build
idf.py -p /dev/ttyUSB0 flash monitor
```

## License

TBD
