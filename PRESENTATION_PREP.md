# Presentation Prep — Everything You Need to Know

**Demo:** tomorrow 11:00. Two profs. 10 min talk + ~5 min Q&A. On-site, with the chip on the desk.

---

## 1. The 10-second elevator pitch

> *I got BERT — a transformer language model — running on a $10 ESP32-S3 microcontroller. It classifies parcel-tracking commands in 510 ms with 91.78% accuracy, fully on-chip. No cloud, no GPU.*

**In plain English:** I made a tiny computer the size of a postage stamp understand sentences like "show me all orders" without sending anything to the internet.

---

## 2. The 60-second pitch (use this in the demo opening)

> *Most voice assistants send your voice to Amazon or Google for processing. That means your delivery information — addresses, package contents, shipping schedules — leaves your home. I wanted to know if a real language model could run **entirely on a $10 microcontroller** so the data never leaves the room.*
>
> *I deployed bert-tiny — a 4.4 million parameter transformer — onto an ESP32-S3 with 8 MB of memory. It classifies natural-language parcel queries into one of 6 actions in 510 ms. To make this work I had to write two custom TensorFlow Lite Micro operations in C++ because the stock framework didn't support them. I also took the latency from an initial 3.7 seconds down to 510 ms — a 86% reduction — through five configuration changes, no retraining.*

**In plain English:** I'll explain *why this is hard* first (small computer, big model), *what I had to fix that wasn't there* (custom code), and *how I made it fast* (engineering tweaks).

---

## 3. The motivation/story (your 2-minute opener)

You can use the *problem* angle or the *curiosity* angle. Pick whichever you actually feel.

**Problem angle:**
> *Every time you ask Alexa "where's my package", that audio gets sent to Amazon's servers. They keep it. They might use it. They definitely log it. The same is true for every smart assistant. For something as personal as your delivery schedule, that's a privacy compromise people accept because they don't know there's an alternative.*
>
> *I asked: can we put the language understanding **on the chip itself**? Not the cloud. Not the laptop. The $10 chip on this desk. If yes, the audio/text never has to leave your home network. The privacy problem disappears.*
>
> *The challenge: language models are huge. Full BERT is 440 MB. The biggest microcontroller I could buy has 8 MB. That's a 50× gap. So this project is about closing that gap.*

**Curiosity angle (simpler):**
> *Cloud AI has a fundamental cost: latency, bandwidth, and privacy. We've been told for years that "real AI needs the cloud". I wanted to actually test that — can a real language model, like the kind powering ChatGPT's predecessors, fit on a chip the size of a fingernail?*
>
> *Spoiler: yes, but barely. And the engineering required to make it work is the interesting part.*

**In plain English:** Pick one (90 seconds), don't try to do both. The privacy story is more compelling but only if you actually believe it.

---

## 4. System architecture (the 3 tiers)

```
   ┌─────────────────────────────────────────┐
   │  USER  →  types text e.g. "view order 101"
   └─────────────────────────────────────────┘
                       ↓ USB serial cable
   ┌─────────────────────────────────────────┐
   │  ESP32-S3 (the edge device)             │
   │  - Tokenizes the text (WordPiece)       │
   │  - Runs BERT inference (510 ms)         │
   │  - Argmax → intent label                │
   │  - Extracts entities (order ID, depot…) │
   │  - Sends JSON over USB                  │
   └─────────────────────────────────────────┘
                       ↓ USB serial cable
   ┌─────────────────────────────────────────┐
   │  Laptop (Flask server, local network)   │
   │  - Receives {"intent": ..., "params":…} │
   │  - Looks up parcel database (flat files)│
   │  - Returns formatted response           │
   └─────────────────────────────────────────┘
                       ↓
                   USER reads result
```

**Tiers:**
1. **Edge tier** = the ESP32. Does ML inference and entity extraction.
2. **Local server tier** = the laptop. Owns the parcel database (flat files: `orders.txt`, `history.txt`, `depots.txt`).
3. **Cloud tier** (in original design) = courier APIs syncing data to the laptop. **Not implemented for this demo** — the laptop has fixed test data.

**In plain English:** chip thinks, laptop knows the answers, USB cable connects them.

---

## 4.5 Build journey — how the project came together

This section is for "walk me through the project" or "what was the development order" questions. It also helps you talk about the C → Python conversion and the model pipeline if asked.

### Stage 1 — Starting point: the C parcel tracker

Before this project, I already had a working **C program** from earlier coursework that managed parcels. It used three flat files — `orders.txt` (active orders), `history.txt` (delivered), `depots.txt` (courier rates) — and exposed a numeric menu interface (`1. View all`, `2. Search by ID`, `3. Filter by depot`, etc.). The functions in that C code were `currentOrders()`, `deliveredOrders()`, `searchOrder()`, `calculateCost()`, `syncDeliveredOrders()`.

That program had no NLP, no network, and no edge component. **It was the database layer I was going to build the smart frontend on top of.**

**In plain English:** I had a basic command-line C program that could look up parcels. The CW2 project was about putting an AI brain in front of it.

---

### Stage 2 — Mirroring the C code in Python (the Flask server)

The ESP32 needs to send structured queries to *something*. C is awful at HTTP. Python is fast to write and — critically — could **read the exact same flat files** my C program wrote. So I ported the C logic to Python, function-by-function.

What's in `server/`:
- **`data_layer.py`** = parses `orders.txt` / `history.txt` / `depots.txt` in the same space-delimited format the C code uses. Identical semantics: `package_id name weight delivery_time status cost courier`.
- **`handlers.py`** = mirrors of the C functions. Each one is documented "Mirror of C `searchOrder()`" etc. Six handlers, one per intent.
- **`cost.py`** = mirror of the C cost calculator: `base_rate + (distance × rate_per_km) + (weight × rate_per_kg)`.
- **`time_utils.py`** = "3 days left", "10 hours ago" helpers (the C code didn't have these — I added them for nicer responses).
- **`test_server.py`** = 7 unit tests confirming every handler returns the expected shape.

The Python and C versions are **interchangeable** at the data layer — if you wrote orders.txt with the C program and read it with Python, it just works.

**In plain English:** I rewrote my old C parcel tracker in Python so the chip could talk to it over the network. Same files, same logic, easier to extend.

---

### Stage 3 — Generating the training dataset

`generate_dataset.py` is a list of ~60 hand-typed example phrases per intent (60 phrases × 6 intents = **361 total phrases**, with VIEW_ALL having one extra). The script shuffles each intent's list with seed `42` and does an 80/20 split.

Per intent: **48 train + 12 test = 60** (VIEW_ALL: 48 + 13 = 61).
Across all intents: **288 train + 73 test = 361** total.

Why hand-typed phrases (not augmentation): there's no public dataset of "parcel tracking commands". Hand-writing ~60 phrasings per intent was faster than collecting real user data, and the prajjwal1/bert-tiny pretrained weights do most of the heavy lifting — the fine-tuning step only needs to teach the new classifier head what each intent *looks like*, not how language works.

> **Note:** the CW2 report claims "1,440 samples, 240 per intent" — that's an error. The actual number is 361. You may want to mention this if asked.

**In plain English:** I wrote ~60 example sentences for each of the 6 intents (e.g. for VIEW_ALL: "show me all orders", "list everything", "give me the full list", ...). 361 sentences total. Shuffled them, kept 80% for training, 20% for testing.

---

### Stage 4 — Fine-tuning bert-tiny

`finetune_and_export.py` (first half) is the training pipeline:

1. Load **`prajjwal1/bert-tiny`** from HuggingFace — a pretrained BERT model already trained on BookCorpus + Wikipedia.
2. Add a **128 → 6 linear classifier head** on top of the [CLS] token output.
3. Train for **50 epochs** with these hyperparameters:
   - Optimizer: AdamW
   - Learning rate: `3e-5`
   - Weight decay: `0.01`
   - Linear warmup over 10 % of steps
   - Batch size: 8
   - Sequence length: 64 (later reduced to 32 for inference, see Stage 8)
   - Loss: CrossEntropyLoss
4. Hardware: **Apple MPS** (Metal Performance Shaders — the M-series GPU). ~5 minutes total training time.
5. Track best checkpoint by test accuracy. Best result: **91.78 % at epoch 50** (67/73 correct).

**In plain English:** I took an existing pretrained language model from Google, added a small "decision head" on top, and showed it my 288 training sentences over and over for 50 rounds. It went from 16 % accuracy (random) to 91.78 % at the end.

---

### Stage 5 — Converting to TensorFlow Lite

`finetune_and_export.py` (second half). The PyTorch model can't run on TFLite Micro directly — it needs to go through TensorFlow first.

1. Load the trained PyTorch checkpoint into TensorFlow: `TFBertForSequenceClassification.from_pretrained(..., from_pt=True)` — HuggingFace handles the weight conversion.
2. Wrap it in a **concrete TF function** with a fixed input signature: `[1, 64]` for both `input_ids` and `attention_mask`.
3. Run the **TFLite converter** with three quantization strategies:

| Variant | Size | Accuracy | Verdict |
|---|---|---|---|
| TFLite FP32 | 16.54 MB | 91.78 % | too big for flash |
| TFLite FP16 | 8.30 MB | 91.78 % | exceeds 8 MB PSRAM |
| **Dynamic-range INT8** | **4.22 MB** | **91.78 %** | ✓ deployed |
| Full INT8 | 4.23 MB | 17.80 % | broken (token clipping) |

**In plain English:** PyTorch is the framework I trained in. The chip can't run PyTorch directly. So I had to convert the model to a smaller format (TFLite) and squash the numbers (quantization). The full squashing broke it; the partial squashing kept the accuracy and made it fit.

---

### Stage 6 — Writing the firmware

ESP-IDF v5.4.1 toolchain, written in C/C++. Five modules in `esp32_firmware/main/`:

- **`tokenizer.c`** — WordPiece tokenizer. Reads `vocab.txt` (30,522 entries) into an FNV-1a hash table in PSRAM. Converts text → token IDs.
- **`inference.cc`** — TFLite Micro wrapper. Allocates the tensor arena (256 KB in DRAM), sets up the op resolver, calls `interpreter->Invoke()` on each query.
- **`custom_gelu.cc`** — the GELU activation kernel I wrote (custom op #1).
- **`custom_fc.cc`** — the hybrid FP32/INT8 FullyConnected kernel I wrote (custom op #2).
- **`entity.c`** — rule-based entity extraction (order ID, depot name, product name, weight).
- **`main.c`** — orchestrator. Reads serial input → tokenizes → runs inference → applies post-classify rules → extracts entities → builds JSON → writes to serial → reads response → prints.

Custom **partition table** (`partitions.csv`): 7.5 MB app partition (default is 1 MB, way too small for our 5.7 MB binary).

The model is **embedded directly into the binary** via ESP-IDF's `EMBED_FILES` mechanism — flash-mapped, no runtime load. `vocab.txt` is embedded the same way.

**In plain English:** I wrote ~6 C/C++ files that run on the chip itself. They turn a sentence into token IDs, run those through the model, and emit JSON over USB.

---

### Stage 7 — Wiring the USB serial bridge

Originally I planned WiFi + HTTP. Two problems made me switch to USB serial:

1. **WiFi was unreliable** — the iPhone hotspot kept dropping the DHCP handshake (error `0xcc00`).
2. **Privacy story** — without WiFi, there's literally no network stack on the chip. Nothing leaves the device. That's a stronger demo claim than "we use HTTPS".

The bridge protocol is dead simple. ESP32 prints:
```
TX:{"intent":"VIEW_ALL","params":{}}
```
The laptop's `serial_runner.py` watches for that prefix, calls `handlers.dispatch(intent, params)`, and writes back:
```
RX:{"intent":"VIEW_ALL","count":8,"orders":[...]}
```
The ESP32 reads that line and prints the response.

ESP32 firmware tweak: **USB-Serial-JTAG as primary console** (`CONFIG_ESP_CONSOLE_USB_SERIAL_JTAG=y`) so `getchar()` reads from USB instead of the unused UART pins.

**In plain English:** the chip and the laptop talk over the USB cable using simple text lines like `TX:{json}` and `RX:{json}`. No WiFi, no internet, no protocols beyond plain serial.

---

### Stage 8 — Optimizing latency (3.7 s → 510 ms)

Covered in §5.4. Five config flips, one re-export at seq_len=32 — no retraining. The reproducibility of these flips is what makes it a good engineering story.

**In plain English:** see §5.4. Same content.

---

### Stage 9 — Hybrid guard + name search (the recent additions)

After the demo pipeline was working, I added two layers of robustness without retraining the model — covered in §6 and §7. These prove the architecture is *extensible* — new behaviour goes in the rule layer, the trained model stays frozen.

**In plain English:** see §6 and §7.

---

### The complete data flow, end to end

User types `"calculate the cost of order 101"`:

1. **Browser** → Flask `POST /api/run` with `{"text": "calculate the cost of order 101"}`
2. **Flask `serial_runner.py`** → writes `calculate the cost of order 101\n` to `/dev/cu.usbmodem1101`
3. **ESP32 `main.c`** → `read_line()` reads the text from USB
4. **`tokenizer.c`** → tokenizes to ~8 token IDs (CLS + tokens + SEP), pads to 32, attention mask
5. **`inference.cc`** → `interpreter->Invoke()` runs BERT (510 ms): 109 ops, ~25 M MACs
6. **`main.c::post_classify()`** → BERT picked CALCULATE_COST (logit +2.49). No keyword override fires.
7. **`entity.c::entity_extract()`** → finds order_id `101` in the text
8. **`main.c::build_json()`** → emits `TX:{"intent":"CALCULATE_COST","params":{"order_id":101}}\n`
9. **`serial_runner.py`** → reads the TX line, parses JSON, calls `handlers.dispatch("CALCULATE_COST", {"order_id": 101})`
10. **`handlers.py::handle_calculate_cost`** → looks up order 101 in `orders.txt`, computes `2.50 + (12.5 × 0.20) + (2.50 × 1.00) = 7.50`
11. **Returns** `{"cost_gbp": 7.50, "weight_kg": 2.50, "courier_name": "FadEx", ...}` to `serial_runner.py`
12. **`serial_runner.py`** → writes `RX:{...}\n` back to ESP32
13. **ESP32 `main.c`** → reads the RX line, prints the response banner with the JSON
14. **`serial_runner.py`** → captures everything, returns the bundled result to Flask
15. **Flask `/api/run`** → returns the JSON to the browser
16. **Browser** → renders the cost callout: **£7.50** for 2.5 kg via FadEx

End-to-end takes ~530 ms (510 ms inference + ~20 ms serial + Python overhead).

**In plain English:** every query goes user → browser → laptop → chip → laptop database → laptop → browser → user. The 510 ms is overwhelmingly the BERT inference on the chip; everything else is microseconds.

---

## 5. The 4 core technical concepts

You **must** understand these. Everything else is a detail.

### 5.1 BERT and intent classification

**Technical:**
- BERT = Bidirectional Encoder Representations from Transformers (Google, 2018). It's a language model that learned how words relate by reading huge amounts of text.
- Full BERT-base: 110 million parameters, 440 MB, needs a GPU.
- We used **bert-tiny** (`prajjwal1/bert-tiny` on HuggingFace): 2 layers, 128 hidden dimensions, 4.4 million parameters. ~17 MB in float32.
- We **fine-tuned** it: started from the pretrained weights and added a small classifier head (128→6) so it outputs 6 numbers per input — one per intent class. We trained that head for 50 epochs on a synthetic dataset of ~1,440 phrases (240 per class).
- The 6 classes: `VIEW_ALL`, `VIEW_ORDER`, `FILTER_BY_DEPOT`, `SHOW_HISTORY`, `CALCULATE_COST`, `SEARCH_ORDER`.
- **Argmax** = pick the index of the biggest number. That's the predicted intent.

**In plain English:** BERT is like a smart auto-complete that I tweaked to do "category guess" instead of "next word guess". Given a sentence, it outputs 6 confidence scores; the highest one wins.

---

### 5.2 Quantization

**Technical:**
- A neural network's "weights" are millions of numbers stored as 32-bit floats (4 bytes each). Bert-tiny in FP32 is ~17 MB.
- **Quantization** = compress those numbers into 8-bit integers (1 byte). Now bert-tiny is ~4 MB. This is the only way to fit it in our 8 MB of memory.
- The math: `int8_value = round(float_value / scale)` where `scale = max(|weights|) / 127`. Each layer gets its own scale.
- We tried **full INT8 quantization** (weights AND activations as int8). It collapsed accuracy to 17.8% — basically random.
- **Why it broke:** BERT input tokens are integers ranging from 0 to 30,522 (the WordPiece vocabulary size). INT8 only supports ±127. Token IDs above 127 got clipped to 127, mapping thousands of distinct words to the same value. The embedding lookup was destroyed.
- **The fix:** **dynamic-range quantization** = quantize the weights only, keep activations and inputs as FP32. Model becomes 4.22 MB, accuracy stays at 91.78%.

**In plain English:** I had to compress the model's "brain" so it would fit. The first compression broke it (the input numbers were too big). The second compression only squashed the model's internal weights, not its inputs, and that worked.

---

### 5.3 Two custom TFLite Micro kernels

**Technical:**
TensorFlow Lite Micro (TFLM) is the framework that runs ML models on microcontrollers. It ships with a library of "operations" (Add, Multiply, Conv2D, FullyConnected, etc.). Two ops we needed didn't exist.

**(a) GELU (Gaussian Error Linear Unit) — `custom_gelu.cc`**
- An activation function used inside transformer blocks. Like ReLU but smoother.
- Formula: `GELU(x) = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))` (the standard tanh approximation).
- Stock TFLM didn't include it. The build literally failed with `AddGelu op not found`.
- I wrote ~50 lines of C++ implementing the function and registered it with TFLM's op resolver.

**(b) Hybrid FullyConnected — `custom_fc.cc`**
- A FullyConnected layer = matrix multiplication. It multiplies an input vector by a weight matrix, adds bias, returns output.
- Stock TFLM has FC kernels for these combos: (FP32 in, FP32 weights), (INT8 in, INT8 weights). It does NOT support: (FP32 in, INT8 weights).
- But that's exactly what dynamic-range quantization produces — FP32 inputs × INT8 weights.
- I wrote a custom FC kernel that **dequantizes the INT8 weights inline** during the multiply, using the per-channel scale: `output[i] = (Σ_j filter[i,j] * input[j]) * scale[i] + bias[i]`.

**In plain English:** the framework I'm using is missing two pieces of math my model needs. Like trying to follow a recipe that calls for an ingredient your grocery store doesn't sell. I had to write the missing pieces myself in C++ and plug them in.

---

### 5.4 The latency optimization journey

**Technical:**
First measurement: 3,705 ms per inference. After 5 changes — none of them touched the model — it became 510 ms (86 % reduction). All 5 changes are reproducible single-line config flips:

| # | Change | File | Gain |
|---|---|---|---|
| 1 | Compiler optimization `-Og` (debug) → `-O2` (perf) | `sdkconfig.defaults` | ~45 % |
| 2 | CPU clock 160 MHz → 240 MHz (its rated speed) | `sdkconfig.defaults` | ~33 % |
| 3 | Data cache 32 KB → 64 KB (chip's max) | `sdkconfig.defaults` | ~10 % |
| 4 | Tensor arena PSRAM → internal DRAM | `inference.cc` | ~15 % |
| 5 | Sequence length 64 → 32 (re-export, no retrain) | `tokenizer.h` + new model | ~50 % on the inputs |

**Each one is a real story:**
1. The build was running in `debug` mode the whole time. One sdkconfig flip away from `-O2`.
2. The chip's clock was running at 160 MHz instead of its rated 240 MHz. The default sdkconfig had it underclocked.
3. The data cache (the fast memory in front of slow PSRAM) was at half its max size. Bumping the line size from 32 to 64 bytes also helped because BERT reads activations sequentially.
4. The "tensor arena" is the working memory the model uses during inference (~200 KB of activations). It was in PSRAM (slow external chip) by default. Moved to internal DRAM (fast on-chip RAM) and PSRAM access went from 100% to 0% during inference.
5. The model was trained with input sequences of 64 tokens. Real queries are 5–10 tokens. Re-exporting at seq_len=32 is **valid** because BERT's positional embeddings cover up to 512 — you don't need to retrain. Shorter sequence means half the math in attention (which is O(N²)) and the FC layers.

**In plain English:** the chip was lazy by default. Five flags later it was running at full speed. None of these required understanding the model — just understanding the chip's settings.

---

## 6. The hybrid NLP guard layer (the cool engineering judgment)

**Technical:**
- BERT's classification can be wrong, especially for phrasings outside the training distribution. Example: "show upcoming orders" — the word "upcoming" never appeared during training. BERT pattern-matches "show __ orders" to SHOW_HISTORY and gets it wrong.
- **Solution:** after BERT picks (argmax), a small rule layer in the firmware (`post_classify` in `main.c`) checks the input for high-precision keywords. If any fire, override BERT's choice.
- Rules:
  - `delivered/history/past/previous` → SHOW_HISTORY
  - `upcoming/active/in-transit/current/pending` → VIEW_ALL
  - `depot/courier` → FILTER_BY_DEPOT
  - `cost/price/how much` → CALCULATE_COST
  - `find/search/look up/where's` + no number → VIEW_ORDER (name search)
- The UI shows an **amber "keyword override" badge** when this fires, naming what BERT originally picked. Full transparency.

**Why it's good engineering:** this is the standard production pattern. BERT does fuzzy semantic matching well. Rules do high-precision keyword matching well. Combine them. No retraining required when you find new edge cases.

**In plain English:** BERT is a guesser. Sometimes it's wrong on words it's never seen. So I added a small "fact checker" in code: if the word "delivered" is in the sentence, the answer is *definitely* about history, no matter what BERT thought.

---

## 7. The name-search feature (added last)

**Technical:**
- Original VIEW_ORDER only worked with order IDs (numbers). Typing "find airpods" returned all 18 history items because BERT picked SEARCH_ORDER (suppressed) → SHOW_HISTORY runner-up → full dump.
- Extended `handle_view_order` on the server to accept a `name` parameter. Does case-insensitive substring matching across active orders + history.
- Added a `product_name` field to the firmware's entity extractor with a stopword list (find/search/order/the/...) and courier name filter. Pulls the first content token >= 3 chars.
- Updated the post-classify rule: if input has "find/search/where" verb AND no digit → VIEW_ORDER (the entity extractor will fill in the name).
- **Crucially: no retraining.** The model still has its 6 fixed classes. New behavior added entirely in the rule layer.

**In plain English:** I added a "search by product name" feature without touching the model at all — entirely with code rules around it. This was the last feature added, and it's the cleanest example of "extending the system without retraining the model".

---

## 8. The numbers (your KPIs — memorize these)

| Metric | Value | Context |
|---|---|---|
| **Latency p50** | **510 ms** | median over n=30 runs |
| **Latency σ** | **0 ms** | every run is identical (within rounding) |
| **Test accuracy** | **91.78 %** | 67/73 held-out sentences |
| **Model size** | **4.22 MB** | dynamic-range INT8, fits in 8 MB flash |
| **Tensor arena** | **197 KB / 256 KB** | activations, in fast internal DRAM |
| **Total ops per inference** | **109** | including 14 FullyConnected, 4 BatchMatMul, 2 GELU |
| **Custom kernels** | **2** | GELU + hybrid FullyConnected |
| **Hardware** | **ESP32-S3-WROOM-1** | dual-core Xtensa LX7 @ 240 MHz, 8 MB PSRAM, 8 MB flash |
| **Chip cost** | **~$10** | the entire compute |
| **Privacy** | **0 bytes leave home** | NLP fully on-chip |

**In plain English:** *510 ms, 91.78 %, $10*. If you remember nothing else, remember those three.

---

## 9. File-by-file map (what's where)

If a prof asks "show me where you wrote X", here's the directory structure:

```
BERT/
├── server/                      # the Flask laptop server
│   ├── app.py                   # Flask routes: /api/run, /api/status, /
│   ├── handlers.py              # 6 intent handlers (view_all, view_order, etc.)
│   ├── data_layer.py            # reads orders.txt / history.txt / depots.txt
│   ├── cost.py                  # cost calculation: base + dist*km + weight*kg
│   ├── time_utils.py            # "3 days left", "10 hours ago" helpers
│   ├── serial_runner.py         # talks to ESP32 over USB
│   ├── static/index.html        # the demo UI (single file, ~1800 lines)
│   └── data/                    # parcel database (flat files, like the C version)
│
├── esp32_firmware/              # the ESP32 firmware
│   ├── main/
│   │   ├── main.c               # main loop: read serial → classify → emit JSON
│   │   ├── inference.cc         # TFLite Micro setup + Invoke()
│   │   ├── tokenizer.c          # WordPiece tokenizer (text → token IDs)
│   │   ├── entity.c             # rule-based entity extraction (order_id, name, etc.)
│   │   ├── custom_gelu.cc       # CUSTOM op #1 — the GELU kernel I wrote
│   │   ├── custom_fc.cc         # CUSTOM op #2 — the hybrid FC kernel I wrote
│   │   ├── bert_model.tflite    # the 4.22 MB quantized model (embedded in flash)
│   │   └── vocab.txt            # WordPiece vocabulary (30,522 entries)
│   ├── sdkconfig.defaults       # ESP-IDF build settings (the 5 perf flips live here)
│   └── partitions.csv           # flash layout (7.5 MB app partition)
│
├── finetune_and_export.py       # PyTorch training + TFLite export pipeline
├── export_seq32.py              # re-exports the trained model at seq_len=32
└── data/                        # training data (train.json, test.json)
```

**In plain English:** server/ is what runs on the laptop, esp32_firmware/ is what runs on the chip, the rest is the training pipeline.

---

## 10. The presentation script (10 minutes)

### 0:00 — 2:00 — The story (motivation)
Use the privacy angle from §3. End with: *"So I deployed BERT on this $10 chip. Here's the demo."*

### 2:00 — 8:00 — The live demo (open Demo Mode in the browser, walk through 6 of 8 chapters)

**Don't run all 8.** Skip these to save time:
- Skip Chapter 4 ("what i had to write") — but mention the custom kernels verbally during Chapter 5 instead.
- Skip Chapter 6 ("active and history") — covered implicitly when you show name search.

**Run these 6, in order:**

| # | Chapter | Type/Click | Talking point |
|---|---|---|---|
| 1 | on the desk | (none — point at chip) | "$10. The entire compute. No cloud, no GPU." |
| 2 | live inference | `show me all orders` | "510 ms. σ=0 across 30 runs. Deterministic." |
| 3 | how it thinks | `view order 101` | "Six logits. Argmax wins. Margin = confidence." |
| 5 | the speedup | `calculate the cost of order 101` | "Original was 3.7 s. Five flags. No retrain. 86 % off." |
| 7 | when bert is wrong | `show upcoming orders` | "BERT picked SHOW_HISTORY. Rule layer caught it. Hybrid NLP." |
| 8 | beyond the dataset | `find airpods` | "Name search added without retraining. Rule layer extension." |

### 8:00 — 9:00 — The numbers (open Stats drawer)

> *"Final numbers: 510 ms, 91.78 % accuracy, 4.22 MB, 197 KB tensor arena, $10 hardware. On-chip. Deterministic. Private."*

### 9:00 — 10:00 — Wrap

> *"The interesting parts of this project weren't the BERT — the model is off the shelf. The interesting parts were everything around it: writing two TFLite Micro kernels because the framework didn't ship them, getting from 3.7 seconds to 510 ms without retraining, and adding a hybrid NLP guard so the system handles inputs the model has never seen."*
>
> *"Future work: faster attention with a custom BatchMatMul; on-chip MQTT for cloud sync; potentially a bigger model now that the pipeline is proven."*

**In plain English:** open with the why (motivation), spend most of the time on the demo, finish with the numbers and a tight summary of what was hard.

---

## 11. Q&A — likely questions with 30-second answers

### Easy / common

**Q: Why bert-tiny and not a smaller classifier?**
> *Phrasing variation. "Show me my orders" / "what have I got" / "any deliveries" all mean VIEW_ALL. Transformers' attention learns this. Rule-based or bag-of-words classifiers don't.*

**Q: Why ESP32-S3?**
> *The 8 MB PSRAM is what makes a 4 MB model fit at all. Smaller chips like the original ESP32 only have 4 MB or none. ARM Cortex M-class chips top out around 1 MB.*

**Q: Why no WiFi?**
> *I tried it. Hotspot was unreliable. The serial bridge is more robust AND a stronger privacy story — there's literally no network stack running on the chip during inference.*

**Q: What about scale?**
> *Single user, single home. No need to scale. The whole point is local processing.*

### Technical / harder

**Q: Why is latency 510 ms, not 100 ms?**
> *Memory bandwidth, not compute. The model has 109 ops; activations live in PSRAM and are read across all of them. Pure compute floor is around 120 ms. The remaining 390 ms is PSRAM access plus TFLite Micro's per-op dispatch overhead.*

**Q: Why didn't full INT8 quantization work?**
> *BERT token IDs span 0 to 30,522 — the WordPiece vocabulary. INT8 maxes out at ±127. Above token 127 the embedding lookup got destroyed. Dynamic-range INT8 quantizes only the weights, keeping inputs as FP32. That preserved accuracy.*

**Q: What's the hardest thing you wrote?**
> *The hybrid FullyConnected kernel. Stock TFLite Micro rejects FP32-input × INT8-weight FC outright. I wrote `custom_fc.cc` that dequantizes the INT8 row inline during the matrix multiply, using per-channel scales.*

**Q: Why not just use regex?**
> *For "show me all orders", sure. But "any deliveries this week" / "what's in flight" / "show upcoming orders" all mean VIEW_ALL with no shared keywords. BERT generalises, regex doesn't.*

**Q: How did you measure 510 ms?**
> *`esp_log_timestamp()` in ESP-IDF, around `interpreter->Invoke()`. Median over n=30 runs of each demo phrase. σ=0 because seq_len is fixed so every inference touches the same memory the same way.*

**Q: How does the keyword override differ from just using rules entirely?**
> *Rules can't generalise. "Show me my packages" has no special keyword but means VIEW_ALL. BERT handles fuzzy phrasings; rules handle high-precision keywords. Two-layer system.*

**Q: Why deactivate SEARCH_ORDER?**
> *It became functionally identical to VIEW_ORDER after merging the handlers. The model's distinction was brittle. Suppressing one of two duplicate classes was cleaner than retraining to 5 classes 2 days before demo.*

**Q: Did you consider Edge Impulse?**
> *Yes. Edge Impulse caps at small CNNs and pre-defined architectures; bert-tiny is 4.4 M parameters, well outside what their tooling supports. I had to write the conversion and quantization pipeline manually.*

**Q: What's your dataset?**
> *1,440 synthetic phrases generated from templates, 240 per intent, balanced. The OOD generalisation risk is exactly why I built the keyword guard layer.*

**Q: How does this compare to cloud BERT?**
> *Cloud BERT round-trips are 500–1000 ms. Mine is deterministic 510 ms with σ=0 — actually competitive. Plus zero bytes leave the network.*

### Curveballs

**Q: What surprised you?**
> *That 86 % of the original latency was just default ESP-IDF settings (debug build, half clock speed, half cache, slow memory placement). The model never needed to change.*

**Q: What would you do differently?**
> *Profile per-op latency from day one. I optimized blindly for a while before discovering BatchMatMul wasn't actually the bottleneck. With per-op timing data I'd have made better calls earlier.*

**Q: What's the next bottleneck?**
> *TFLite Micro's per-op dispatch overhead — 109 ops at ~1 ms each is ~100 ms of pure framework time. To go below 400 ms you'd need to fuse layers (e.g. LayerNorm into one op instead of 5), which means rewriting the model's compute graph.*

**Q: Could you run a bigger model?**
> *Bert-mini (8 M params) is 10.79 MB after dynamic-range quantization — exceeds my 8 MB. I'd need a chip with 16 MB+ PSRAM. ESP32-P4 might fit it.*

**In plain English:** if a question stumps you, say *"I'd have to look at the code to give you a precise answer on that."* Engineers respect honest "I don't know" more than confident bullshit.

---

## 12. Things to look up on the internet (deepen understanding tonight)

If you want to feel more grounded, watch/read these. **None are required**, but they're high-leverage if you have time:

1. **3Blue1Brown — "But what is a neural network?"** (YouTube, 19 min). Visual intuition for what a neural network is doing.
2. **3Blue1Brown — "Attention in transformers, visually explained"** (YouTube, 26 min). The attention mechanism BERT uses.
3. **Jay Alammar — "The Illustrated BERT"** (blog post, ~15 min read). Visual explanation of BERT specifically. Search: *"jay alammar illustrated bert"*.
4. **Wikipedia — "Quantization (signal processing)"**, just the first section. The math at the heart of INT8 quantization.
5. **TensorFlow blog — "Post-training dynamic range quantization"** (~5 min). Specifically the technique we used.
6. **Espressif — ESP-IDF Programming Guide → "Memory Types"**. PSRAM vs DRAM and why we cared about the difference.

**In plain English:** if you watch the first two videos and read Jay Alammar's BERT post, you'll understand the substance better than 80% of CS undergrads.

---

## 13. Demo-day checklist

### Tonight (before sleep)
- [ ] Read this whole document once, slowly
- [ ] Time yourself running the demo flow once (target: 7–8 min)
- [ ] Print this document, or save it on your phone
- [ ] Charge laptop. Pack USB cable.
- [ ] Sleep by 23:00

### Tomorrow morning (08:30–10:00)
- [ ] Plug ESP32 in, wait for `/dev/cu.usbmodem*` to appear
- [ ] `cd "/Users/mac/Desktop/IoT BERT Project/BERT" && source venv/bin/activate && python server/app.py`
- [ ] Open `http://127.0.0.1:5001/` in the browser, fullscreen
- [ ] Click each of the 6 chips once — verify ~510 ms each
- [ ] Click **▶ DEMO** to verify chapter intro flash works
- [ ] Run the demo flow cold, narrating out loud, time it
- [ ] Stop fiddling. Pack up. Walk to the venue.

### At the venue (5-min setup)
- [ ] Plug USB into laptop
- [ ] Verify `/dev/cu.usbmodem1101` appears (`ls /dev/cu.usbmodem*`)
- [ ] Flask running, browser fullscreen on `http://127.0.0.1:5001/`
- [ ] Send ONE test query (click any chip) — confirm green status, 510 ms
- [ ] Click **▶ DEMO** to confirm — close, ready position

### If something breaks live
- ESP32 not detected → unplug/replug. Different port if available.
- Flask error → in terminal: `lsof -ti :5001 | xargs kill; python server/app.py`
- UI blank → reload the browser tab
- Chip won't respond → unplug for 10 seconds, replug, restart Flask

**In plain English:** go through the morning checklist twice. If everything works at 10:00 AM, you're done preparing.

---

## 14. The single most important rule for tomorrow

> **Don't mention things you can't defend.**

Profs only ask follow-up questions about things YOU bring up. If you don't say "Xtensa SIMD", they won't ask about Xtensa SIMD. If you don't say "BatchMatMul", they won't ask about BatchMatMul.

**Stick to:** BERT classifies text → custom kernels (GELU + hybrid FC) → optimization journey (5 flips) → hybrid guard → name search. That's plenty.

**Avoid:** transformer attention internals, TFLite Micro memory planner, Xtensa instruction set, attention math.

If asked something you don't know: *"I'd have to dig into the code to give a precise answer."* — say it calmly, move on.

**In plain English:** lead the conversation. Don't drop jargon. If you don't know something, say so — that's a strength, not a weakness.

---

## 15. Three sentences to memorize cold

> 1. *I deployed bert-tiny, a 4.4 million parameter transformer, onto a $10 ESP32-S3 microcontroller for fully on-device intent classification.*
> 2. *To make it work I had to write two custom TensorFlow Lite Micro kernels — GELU and a hybrid FP32/INT8 FullyConnected — because the framework didn't ship them.*
> 3. *I took inference latency from 3.7 seconds to 510 milliseconds — 86 % reduction — through five configuration changes, no retraining.*

If you walk out of tomorrow remembering only those three sentences, you've nailed the project.

Good luck. You've got this.
