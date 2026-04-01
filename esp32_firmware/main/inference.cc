/**
 * inference.cc — TFLite Micro wrapper for BERT-tiny intent classification.
 *
 * Loads the .tflite model (embedded in flash) and allocates the tensor arena
 * in PSRAM.  Exposes a plain-C API via inference.h.
 */

#include "inference.h"
#include "custom_gelu.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_log.h"

#include "esp_log.h"
#include "esp_heap_caps.h"
#include <cstring>

static const char *TAG = "INFERENCE";

/* ── Intent labels (must match LABEL2ID in finetune_and_export.py) ──── */
static const char *INTENT_LABELS[NUM_INTENTS] = {
    "VIEW_ALL",         /* 0 */
    "VIEW_ORDER",       /* 1 */
    "FILTER_BY_DEPOT",  /* 2 */
    "SHOW_HISTORY",     /* 3 */
    "CALCULATE_COST",   /* 4 */
    "SEARCH_ORDER",     /* 5 */
};

/* ── Tensor arena in PSRAM ─────────────────────────────────────────────
 * bert-tiny (2 layers, 128 hidden, seq_len=64) needs roughly 1-2 MB
 * for activations.  We allocate 4 MB to be safe.
 */
#define TENSOR_ARENA_SIZE  (4 * 1024 * 1024)

static uint8_t *tensor_arena = nullptr;
static tflite::MicroInterpreter *interpreter = nullptr;
static const tflite::Model *model = nullptr;

/* ── Op resolver: every op used by the dynamic-range bert-tiny model ── */
static tflite::MicroMutableOpResolver<18> resolver;

/* ── Public API ────────────────────────────────────────────────────────── */

extern "C" int inference_init(const uint8_t *model_data, int model_len)
{
    ESP_LOGI(TAG, "Loading TFLite model (%d bytes)...", model_len);

    model = tflite::GetModel(model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Model schema version %lu != expected %d",
                 (unsigned long)model->version(), TFLITE_SCHEMA_VERSION);
        return -1;
    }

    /* Register the 18 ops found in the model */
    resolver.AddAdd();
    resolver.AddBatchMatMul();
    resolver.AddCast();
    resolver.AddDequantize();
    resolver.AddFullyConnected();
    resolver.AddGather();
    resolver.AddGelu(Register_GELU(), ParseGelu);
    resolver.AddMean();
    resolver.AddMul();
    resolver.AddReshape();
    resolver.AddRsqrt();
    resolver.AddSoftmax();
    resolver.AddSquaredDifference();
    resolver.AddStridedSlice();
    resolver.AddSub();
    resolver.AddTanh();
    resolver.AddTranspose();

    /* Allocate tensor arena in PSRAM */
    tensor_arena = (uint8_t *)heap_caps_malloc(TENSOR_ARENA_SIZE, MALLOC_CAP_SPIRAM);
    if (!tensor_arena) {
        ESP_LOGE(TAG, "Failed to allocate %d bytes in PSRAM for tensor arena",
                 TENSOR_ARENA_SIZE);
        return -1;
    }
    ESP_LOGI(TAG, "Tensor arena: %d bytes in PSRAM", TENSOR_ARENA_SIZE);

    /* Create interpreter */
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, TENSOR_ARENA_SIZE);
    interpreter = &static_interpreter;

    TfLiteStatus status = interpreter->AllocateTensors();
    if (status != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors() failed");
        return -1;
    }

    ESP_LOGI(TAG, "Model loaded. Arena used: %zu / %d bytes",
             interpreter->arena_used_bytes(), TENSOR_ARENA_SIZE);

    /* Log input/output tensor info */
    for (size_t i = 0; i < interpreter->inputs_size(); i++) {
        TfLiteTensor *t = interpreter->input(i);
        ESP_LOGI(TAG, "Input[%d]: dims=%d×%d type=%d",
                 (int)i, t->dims->data[0], t->dims->data[1], t->type);
    }
    TfLiteTensor *out = interpreter->output(0);
    ESP_LOGI(TAG, "Output[0]: dims=%d×%d type=%d",
             out->dims->data[0], out->dims->data[1], out->type);

    return 0;
}

extern "C" int inference_run(const int32_t *input_ids, const int32_t *attention_mask,
                             float *out_scores)
{
    if (!interpreter) return -1;

    /* Copy inputs into the model's input tensors */
    TfLiteTensor *in_ids  = interpreter->input(0);
    TfLiteTensor *in_mask = interpreter->input(1);

    memcpy(in_ids->data.i32,  input_ids,      64 * sizeof(int32_t));
    memcpy(in_mask->data.i32, attention_mask,  64 * sizeof(int32_t));

    /* Run inference */
    TfLiteStatus status = interpreter->Invoke();
    if (status != kTfLiteOk) {
        ESP_LOGE(TAG, "Invoke() failed");
        return -1;
    }

    /* Read output logits */
    TfLiteTensor *output = interpreter->output(0);
    memcpy(out_scores, output->data.f, NUM_INTENTS * sizeof(float));

    /* Find argmax */
    int best = 0;
    for (int i = 1; i < NUM_INTENTS; i++) {
        if (out_scores[i] > out_scores[best]) best = i;
    }

    return best;
}

extern "C" const char *inference_intent_label(int idx)
{
    if (idx < 0 || idx >= NUM_INTENTS) return "UNKNOWN";
    return INTENT_LABELS[idx];
}
