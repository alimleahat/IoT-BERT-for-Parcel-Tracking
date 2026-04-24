/**
 * main.c — ESP32-S3 Parcel Tracker with on-device BERT inference.
 *
 * USB Serial Bridge mode (no WiFi).
 *
 * Pipeline:
 *   1. Load vocab.txt into WordPiece tokenizer (PSRAM)
 *   2. Load bert_model.tflite into TFLite Micro (PSRAM)
 *   3. Loop:
 *      a. Read command line from serial (sent by laptop bridge)
 *      b. Tokenize → run BERT inference → get intent
 *      c. Extract entities (order ID, depot, weight, etc.)
 *      d. Build JSON payload and print as "TX:<json>\n"
 *      e. Wait for "RX:<response>\n" line from bridge
 *      f. Print the response back to the user terminal
 *
 * The laptop-side bridge (server/serial_bridge.py) mediates all I/O:
 *   - forwards user keystrokes from its terminal → ESP32 stdin
 *   - intercepts TX: lines, POSTs to Flask, writes RX: back
 *   - forwards everything else (ESP_LOG etc.) straight to the user terminal
 */

#include <stdio.h>
#include <string.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "esp_system.h"
#include "esp_log.h"

#include "tokenizer.h"
#include "inference.h"
#include "entity.h"

/* ── Configuration ─────────────────────────────────────────────────────── */

#define MAX_RESPONSE_LEN  4096
#define INPUT_BUF_LEN      256

static const char *TAG = "PARCEL";

/* ── Embedded binary files (linked by CMakeLists.txt) ─────────────────── */

extern const uint8_t  model_start[] asm("_binary_bert_model_tflite_start");
extern const uint8_t  model_end[]   asm("_binary_bert_model_tflite_end");
extern const uint8_t  vocab_start[] asm("_binary_vocab_txt_start");
extern const uint8_t  vocab_end[]   asm("_binary_vocab_txt_end");

/* ── Build JSON payload from intent + entities ─────────────────────────── */

static void build_json(const char *intent, const entities_t *ent, char *buf, int buf_size)
{
    if (strcmp(intent, "VIEW_ALL") == 0) {
        snprintf(buf, buf_size, "{\"intent\":\"VIEW_ALL\"}");
    }
    else if (strcmp(intent, "VIEW_ORDER") == 0) {
        snprintf(buf, buf_size,
                 "{\"intent\":\"VIEW_ORDER\",\"params\":{\"order_id\":%d}}",
                 ent->order_id);
    }
    else if (strcmp(intent, "SEARCH_ORDER") == 0) {
        snprintf(buf, buf_size,
                 "{\"intent\":\"SEARCH_ORDER\",\"params\":{\"order_id\":%d}}",
                 ent->order_id);
    }
    else if (strcmp(intent, "FILTER_BY_DEPOT") == 0) {
        if (ent->depot_name[0] != '\0') {
            snprintf(buf, buf_size,
                     "{\"intent\":\"FILTER_BY_DEPOT\",\"params\":{\"depot_name\":\"%s\"}}",
                     ent->depot_name);
        } else {
            snprintf(buf, buf_size,
                     "{\"intent\":\"FILTER_BY_DEPOT\",\"params\":{\"depot_id\":%d}}",
                     ent->depot_id);
        }
    }
    else if (strcmp(intent, "SHOW_HISTORY") == 0) {
        snprintf(buf, buf_size, "{\"intent\":\"SHOW_HISTORY\"}");
    }
    else if (strcmp(intent, "CALCULATE_COST") == 0) {
        /* Two phrasings:
         *   "cost of order 101"    → server looks up weight+courier by order_id
         *   "cost of 5kg at FadEx" → send explicit weight (+ optional courier) */
        if (ent->order_id >= 0) {
            snprintf(buf, buf_size,
                     "{\"intent\":\"CALCULATE_COST\",\"params\":{\"order_id\":%d}}",
                     ent->order_id);
        } else {
            char courier_part[64] = "";
            if (ent->courier_name[0] != '\0') {
                snprintf(courier_part, sizeof(courier_part),
                         ",\"courier_name\":\"%s\"", ent->courier_name);
            } else if (ent->courier_id > 0) {
                snprintf(courier_part, sizeof(courier_part),
                         ",\"courier_id\":%d", ent->courier_id);
            }
            snprintf(buf, buf_size,
                     "{\"intent\":\"CALCULATE_COST\",\"params\":{\"weight\":%.1f%s}}",
                     ent->weight, courier_part);
        }
    }
    else {
        snprintf(buf, buf_size, "{\"intent\":\"%s\"}", intent);
    }
}

/* ── Read a line from serial (blocking, no echo) ───────────────────────── */
/* The bridge terminal handles echo on the host side, so we don't need to
 * echo characters back — echoing would duplicate every keystroke. */

static int read_line(char *buf, int max_len)
{
    int pos = 0;
    while (pos < max_len - 1) {
        int c = getchar();
        if (c == EOF) {
            vTaskDelay(pdMS_TO_TICKS(10));
            continue;
        }
        if (c == '\n' || c == '\r') {
            break;
        }
        if (c == '\b' || c == 127) {
            if (pos > 0) pos--;
            continue;
        }
        buf[pos++] = (char)c;
    }
    buf[pos] = '\0';
    return pos;
}

/* ── Send JSON over serial; wait for RX: response ──────────────────────── */

static void send_intent_serial(const char *json_body, char *response_buf, int resp_buf_len)
{
    /* Emit the request marker. The bridge script watches for "TX:" at the
     * start of a line and parses everything after it as the JSON body. */
    printf("TX:%s\n", json_body);
    fflush(stdout);

    /* Wait for a line starting with "RX:". The bridge will POST to Flask
     * and then write a single line of the form "RX:<response-json>\n".
     * Ignore any other lines that arrive in between (shouldn't happen,
     * but be defensive). */
    char line[MAX_RESPONSE_LEN];
    while (1) {
        int n = read_line(line, sizeof(line));
        if (n >= 3 && line[0] == 'R' && line[1] == 'X' && line[2] == ':') {
            strncpy(response_buf, line + 3, resp_buf_len - 1);
            response_buf[resp_buf_len - 1] = '\0';
            return;
        }
        /* Unexpected line; drop it and keep waiting. */
    }
}

/* ── Main Application ──────────────────────────────────────────────────── */

void app_main(void)
{
    ESP_LOGI(TAG, "========================================");
    ESP_LOGI(TAG, "  Parcel Tracker — BERT on ESP32-S3");
    ESP_LOGI(TAG, "  (USB Serial Bridge mode)");
    ESP_LOGI(TAG, "========================================");

    /* 1. Init tokenizer from embedded vocab.txt */
    int vocab_len = vocab_end - vocab_start;
    ESP_LOGI(TAG, "Loading vocabulary (%d bytes)...", vocab_len);
    tokenizer_init((const char *)vocab_start, vocab_len);

    /* 2. Init TFLite Micro model from embedded .tflite */
    int model_len = model_end - model_start;
    ESP_LOGI(TAG, "Loading BERT model (%d bytes)...", model_len);
    if (inference_init(model_start, model_len) != 0) {
        ESP_LOGE(TAG, "Model init failed! Halting.");
        while (1) vTaskDelay(pdMS_TO_TICKS(1000));
    }

    ESP_LOGI(TAG, "========================================");
    ESP_LOGI(TAG, "  Ready! Send a command via the bridge.");
    ESP_LOGI(TAG, "========================================");

    /* Announce readiness to the bridge with a dedicated marker so the
     * bridge can show a prompt only after the model is loaded. */
    printf("READY\n");
    fflush(stdout);

    /* 3. Main loop: read command → infer → send → print */
    char input[INPUT_BUF_LEN];
    int32_t input_ids[MAX_SEQ_LEN];
    int32_t attention_mask[MAX_SEQ_LEN];
    float scores[NUM_INTENTS];
    char json_buf[512];
    char response_buf[MAX_RESPONSE_LEN];

    while (1) {
        int len = read_line(input, sizeof(input));
        if (len == 0) continue;

        ESP_LOGI(TAG, "Input: \"%s\"", input);

        /* Tokenize */
        int n_tokens = tokenizer_encode(input, input_ids, attention_mask);
        ESP_LOGI(TAG, "Tokens: %d", n_tokens);

        /* Run BERT inference */
        int64_t t0 = esp_log_timestamp();
        int intent_idx = inference_run(input_ids, attention_mask, scores);
        int64_t t1 = esp_log_timestamp();
        if (intent_idx < 0) {
            ESP_LOGE(TAG, "Inference failed");
            continue;
        }
        const char *intent = inference_intent_label(intent_idx);

        /* Log scores + timing */
        printf("Intent: %s (score=%.3f) — %lld ms\n",
               intent, scores[intent_idx], (long long)(t1 - t0));
        printf("  Scores: ");
        for (int i = 0; i < NUM_INTENTS; i++) {
            printf("%s=%.2f ", inference_intent_label(i), scores[i]);
        }
        printf("\n");

        /* Extract entities */
        entities_t ent;
        entity_extract(input, intent, &ent);

        if (ent.order_id >= 0)
            ESP_LOGI(TAG, "Entity: order_id=%d", ent.order_id);
        if (ent.depot_name[0])
            ESP_LOGI(TAG, "Entity: depot=%s (id=%d)", ent.depot_name, ent.depot_id);
        if (ent.weight >= 0)
            ESP_LOGI(TAG, "Entity: weight=%.1f", ent.weight);
        if (ent.courier_name[0])
            ESP_LOGI(TAG, "Entity: courier=%s (id=%d)", ent.courier_name, ent.courier_id);

        /* Build JSON, send over serial, wait for response */
        build_json(intent, &ent, json_buf, sizeof(json_buf));
        send_intent_serial(json_buf, response_buf, sizeof(response_buf));

        printf("\n─── SERVER RESPONSE ───\n%s\n───────────────────────\n\n",
               response_buf);
        fflush(stdout);
    }
}
