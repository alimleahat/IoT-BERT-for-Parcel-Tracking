/**
 * main.c — ESP32-S3 Parcel Tracker with on-device BERT inference.
 *
 * Pipeline:
 *   1. Connect to WiFi
 *   2. Load vocab.txt into WordPiece tokenizer (PSRAM)
 *   3. Load bert_model.tflite into TFLite Micro (PSRAM)
 *   4. Loop:
 *      a. Read command from serial (UART)
 *      b. Tokenize → run BERT inference → get intent
 *      c. Extract entities (order ID, depot, weight, etc.)
 *      d. Build JSON payload
 *      e. HTTP POST to Flask server
 *      f. Print response
 */

#include <stdio.h>
#include <string.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/event_groups.h"

#include "esp_system.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "nvs_flash.h"
#include "esp_http_client.h"

#include "tokenizer.h"
#include "inference.h"
#include "entity.h"

/* ── Configuration ─────────────────────────────────────────────────────── */

#define WIFI_SSID       CONFIG_WIFI_SSID
#define WIFI_PASS       CONFIG_WIFI_PASSWORD
#define SERVER_URL      CONFIG_SERVER_URL

#define MAX_RESPONSE_LEN  4096
#define WIFI_CONNECTED_BIT BIT0
#define WIFI_FAIL_BIT      BIT1
#define MAX_RETRY          10
#define INPUT_BUF_LEN      256

static const char *TAG = "PARCEL";

/* ── Embedded binary files (linked by CMakeLists.txt) ─────────────────── */

extern const uint8_t  model_start[] asm("_binary_bert_model_tflite_start");
extern const uint8_t  model_end[]   asm("_binary_bert_model_tflite_end");
extern const uint8_t  vocab_start[] asm("_binary_vocab_txt_start");
extern const uint8_t  vocab_end[]   asm("_binary_vocab_txt_end");

/* ── Globals ───────────────────────────────────────────────────────────── */

static EventGroupHandle_t s_wifi_event_group;
static int s_retry_num = 0;

static char response_buffer[MAX_RESPONSE_LEN];
static int  response_len = 0;

/* ── WiFi Event Handler ───────────────────────────────────────────────── */

static void wifi_event_handler(void *arg, esp_event_base_t event_base,
                               int32_t event_id, void *event_data)
{
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    }
    else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {
        if (s_retry_num < MAX_RETRY) {
            esp_wifi_connect();
            s_retry_num++;
            ESP_LOGW(TAG, "Retrying WiFi connection (%d/%d)...", s_retry_num, MAX_RETRY);
        } else {
            xEventGroupSetBits(s_wifi_event_group, WIFI_FAIL_BIT);
            ESP_LOGE(TAG, "WiFi connection failed after %d retries", MAX_RETRY);
        }
    }
    else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t *event = (ip_event_got_ip_t *)event_data;
        ESP_LOGI(TAG, "Connected! IP: " IPSTR, IP2STR(&event->ip_info.ip));
        s_retry_num = 0;
        xEventGroupSetBits(s_wifi_event_group, WIFI_CONNECTED_BIT);
    }
}

/* ── WiFi Init ─────────────────────────────────────────────────────────── */

static void wifi_init_sta(void)
{
    s_wifi_event_group = xEventGroupCreate();

    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    esp_event_handler_instance_t instance_any_id;
    esp_event_handler_instance_t instance_got_ip;

    ESP_ERROR_CHECK(esp_event_handler_instance_register(
        WIFI_EVENT, ESP_EVENT_ANY_ID, &wifi_event_handler, NULL, &instance_any_id));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(
        IP_EVENT, IP_EVENT_STA_GOT_IP, &wifi_event_handler, NULL, &instance_got_ip));

    wifi_config_t wifi_config = {
        .sta = {
            .ssid = WIFI_SSID,
            .password = WIFI_PASS,
            .threshold.authmode = WIFI_AUTH_OPEN,
        },
    };

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_start());

    ESP_LOGI(TAG, "Connecting to WiFi SSID: [%s]", WIFI_SSID);

    EventBits_t bits = xEventGroupWaitBits(
        s_wifi_event_group,
        WIFI_CONNECTED_BIT | WIFI_FAIL_BIT,
        pdFALSE, pdFALSE, portMAX_DELAY);

    if (bits & WIFI_CONNECTED_BIT) {
        ESP_LOGI(TAG, "WiFi connected to %s", WIFI_SSID);
    } else {
        ESP_LOGE(TAG, "WiFi FAILED to connect to %s", WIFI_SSID);
    }
}

/* ── HTTP Event Handler ────────────────────────────────────────────────── */

static esp_err_t http_event_handler(esp_http_client_event_t *evt)
{
    switch (evt->event_id) {
    case HTTP_EVENT_ON_DATA:
        if (response_len + evt->data_len < MAX_RESPONSE_LEN) {
            memcpy(response_buffer + response_len, evt->data, evt->data_len);
            response_len += evt->data_len;
            response_buffer[response_len] = '\0';
        }
        break;
    default:
        break;
    }
    return ESP_OK;
}

/* ── Send JSON to Flask Server ─────────────────────────────────────────── */

static void send_intent_request(const char *json_body)
{
    response_len = 0;
    memset(response_buffer, 0, sizeof(response_buffer));

    ESP_LOGI(TAG, "POST %s", SERVER_URL);
    ESP_LOGI(TAG, "Body: %s", json_body);

    esp_http_client_config_t config = {
        .url = SERVER_URL,
        .event_handler = http_event_handler,
        .timeout_ms = 10000,
    };

    esp_http_client_handle_t client = esp_http_client_init(&config);
    esp_http_client_set_method(client, HTTP_METHOD_POST);
    esp_http_client_set_header(client, "Content-Type", "application/json");
    esp_http_client_set_post_field(client, json_body, strlen(json_body));

    esp_err_t err = esp_http_client_perform(client);

    if (err == ESP_OK) {
        int status = esp_http_client_get_status_code(client);
        ESP_LOGI(TAG, "HTTP %d — %d bytes", status, response_len);
        printf("\n─── SERVER RESPONSE ───\n%s\n───────────────────────\n\n", response_buffer);
    } else {
        ESP_LOGE(TAG, "HTTP POST failed: %s", esp_err_to_name(err));
    }

    esp_http_client_cleanup(client);
}

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
    else {
        snprintf(buf, buf_size, "{\"intent\":\"%s\"}", intent);
    }
}

/* ── Read a line from serial (blocking) ────────────────────────────────── */

static int read_line(char *buf, int max_len)
{
    int pos = 0;
    while (pos < max_len - 1) {
        int c = getchar();
        if (c == EOF) {
            vTaskDelay(pdMS_TO_TICKS(10));
            continue;
        }
        /* Echo character back */
        putchar(c);
        if (c == '\n' || c == '\r') {
            putchar('\n');
            break;
        }
        if (c == '\b' || c == 127) {
            if (pos > 0) {
                pos--;
                printf("\b \b");
            }
            continue;
        }
        buf[pos++] = (char)c;
    }
    buf[pos] = '\0';
    return pos;
}

/* ── Main Application ──────────────────────────────────────────────────── */

void app_main(void)
{
    ESP_LOGI(TAG, "========================================");
    ESP_LOGI(TAG, "  Parcel Tracker — BERT on ESP32-S3");
    ESP_LOGI(TAG, "========================================");

    /* Init NVS (required by WiFi) */
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    /* 1. Connect to WiFi */
    wifi_init_sta();
    vTaskDelay(pdMS_TO_TICKS(500));

    /* 2. Init tokenizer from embedded vocab.txt */
    int vocab_len = vocab_end - vocab_start;
    ESP_LOGI(TAG, "Loading vocabulary (%d bytes)...", vocab_len);
    tokenizer_init((const char *)vocab_start, vocab_len);

    /* 3. Init TFLite Micro model from embedded .tflite */
    int model_len = model_end - model_start;
    ESP_LOGI(TAG, "Loading BERT model (%d bytes)...", model_len);
    if (inference_init(model_start, model_len) != 0) {
        ESP_LOGE(TAG, "Model init failed! Halting.");
        while (1) vTaskDelay(pdMS_TO_TICKS(1000));
    }

    ESP_LOGI(TAG, "========================================");
    ESP_LOGI(TAG, "  Ready! Type a command below:");
    ESP_LOGI(TAG, "========================================");

    /* 4. Main loop: read serial → infer → send → print */
    char input[INPUT_BUF_LEN];
    int32_t input_ids[MAX_SEQ_LEN];
    int32_t attention_mask[MAX_SEQ_LEN];
    float scores[NUM_INTENTS];
    char json_buf[512];

    while (1) {
        printf("parcel> ");
        fflush(stdout);

        int len = read_line(input, sizeof(input));
        if (len == 0) continue;

        ESP_LOGI(TAG, "Input: \"%s\"", input);

        /* Tokenize */
        int n_tokens = tokenizer_encode(input, input_ids, attention_mask);
        ESP_LOGI(TAG, "Tokens: %d", n_tokens);

        /* Run BERT inference */
        int intent_idx = inference_run(input_ids, attention_mask, scores);
        if (intent_idx < 0) {
            ESP_LOGE(TAG, "Inference failed");
            continue;
        }
        const char *intent = inference_intent_label(intent_idx);

        /* Log scores */
        printf("Intent: %s (score=%.3f)\n", intent, scores[intent_idx]);
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

        /* Build JSON and send to server */
        build_json(intent, &ent, json_buf, sizeof(json_buf));
        send_intent_request(json_buf);
    }
}
