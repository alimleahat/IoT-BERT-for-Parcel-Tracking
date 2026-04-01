/**
 * tokenizer.c — Minimal WordPiece tokenizer for BERT (uncased).
 *
 * Stores the 30 522-entry vocabulary in a hash table allocated in PSRAM.
 * At encode time: lowercases, splits on whitespace/punctuation, then applies
 * the greedy longest-match WordPiece algorithm with "##" continuation prefixes.
 */

#include <string.h>
#include <ctype.h>
#include <stdlib.h>
#include "tokenizer.h"
#include "esp_log.h"
#include "esp_heap_caps.h"

static const char *TAG = "TOKENIZER";

/* ── Hash table (open-addressing, power-of-2 size) ───────────────────── */

#define HT_SIZE  65536          /* 2^16, ~47 % load for 30 522 entries */
#define HT_MASK  (HT_SIZE - 1)

typedef struct {
    char *key;     /* token string (allocated in PSRAM) */
    int   value;   /* token ID */
} ht_entry_t;

static ht_entry_t *ht = NULL;  /* hash table array in PSRAM */

static uint32_t fnv1a(const char *s, int len)
{
    uint32_t h = 2166136261u;
    for (int i = 0; i < len; i++) {
        h ^= (uint8_t)s[i];
        h *= 16777619u;
    }
    return h;
}

static void ht_insert(const char *key, int key_len, int value)
{
    uint32_t idx = fnv1a(key, key_len) & HT_MASK;
    while (ht[idx].key != NULL) {
        idx = (idx + 1) & HT_MASK;
    }
    char *k = (char *)heap_caps_malloc(key_len + 1, MALLOC_CAP_SPIRAM);
    memcpy(k, key, key_len);
    k[key_len] = '\0';
    ht[idx].key   = k;
    ht[idx].value = value;
}

static int ht_lookup(const char *key, int key_len)
{
    uint32_t idx = fnv1a(key, key_len) & HT_MASK;
    while (ht[idx].key != NULL) {
        if ((int)strlen(ht[idx].key) == key_len &&
            memcmp(ht[idx].key, key, key_len) == 0) {
            return ht[idx].value;
        }
        idx = (idx + 1) & HT_MASK;
    }
    return -1;   /* not found */
}

/* ── Vocab loading ────────────────────────────────────────────────────── */

void tokenizer_init(const char *vocab_data, int vocab_len)
{
    /* Allocate hash table in PSRAM */
    ht = (ht_entry_t *)heap_caps_calloc(HT_SIZE, sizeof(ht_entry_t), MALLOC_CAP_SPIRAM);
    if (!ht) {
        ESP_LOGE(TAG, "Failed to allocate hash table in PSRAM");
        return;
    }

    /* Parse vocab.txt: one token per line, line number = token ID */
    int id = 0;
    const char *p   = vocab_data;
    const char *end = vocab_data + vocab_len;

    while (p < end) {
        const char *line_start = p;
        while (p < end && *p != '\n' && *p != '\r') p++;
        int line_len = p - line_start;

        if (line_len > 0) {
            ht_insert(line_start, line_len, id);
        }
        id++;

        /* skip newline chars */
        while (p < end && (*p == '\n' || *p == '\r')) p++;
    }

    ESP_LOGI(TAG, "Loaded %d vocab entries", id);
}

/* ── WordPiece encoding ───────────────────────────────────────────────── */

static int is_punct(char c)
{
    return (c >= '!' && c <= '/') || (c >= ':' && c <= '@') ||
           (c >= '[' && c <= '`') || (c >= '{' && c <= '~');
}

/**
 * WordPiece-encode a single lowercase word and append IDs to output.
 * Returns number of tokens added.
 */
static int wordpiece(const char *word, int word_len, int32_t *out, int max_tokens)
{
    int count = 0;
    int start = 0;
    char buf[128];

    while (start < word_len && count < max_tokens) {
        int end = word_len;
        int found = 0;

        while (start < end) {
            int sub_len;
            const char *sub;

            if (start == 0) {
                sub = word + start;
                sub_len = end - start;
            } else {
                /* prepend ## for continuation */
                buf[0] = '#'; buf[1] = '#';
                sub_len = end - start;
                if (sub_len + 2 >= (int)sizeof(buf)) sub_len = sizeof(buf) - 3;
                memcpy(buf + 2, word + start, sub_len);
                sub_len += 2;
                sub = buf;
            }

            int id = ht_lookup(sub, sub_len);
            if (id >= 0) {
                out[count++] = id;
                found = 1;
                start = end;
                break;
            }
            end--;
        }

        if (!found) {
            out[count++] = UNK_ID;
            break;
        }
    }
    return count;
}

int tokenizer_encode(const char *text, int32_t *input_ids, int32_t *attention_mask)
{
    /* Lowercase copy */
    int text_len = strlen(text);
    if (text_len > 512) text_len = 512;
    char lower[513];
    for (int i = 0; i < text_len; i++) {
        lower[i] = tolower((unsigned char)text[i]);
    }
    lower[text_len] = '\0';

    /* Tokenise: [CLS] tokens... [SEP] [PAD]... */
    int pos = 0;
    input_ids[pos++] = CLS_ID;

    int i = 0;
    while (i < text_len && pos < MAX_SEQ_LEN - 1) {  /* leave room for SEP */
        /* skip whitespace */
        while (i < text_len && isspace((unsigned char)lower[i])) i++;
        if (i >= text_len) break;

        /* handle punctuation as single token */
        if (is_punct(lower[i])) {
            int id = ht_lookup(&lower[i], 1);
            input_ids[pos++] = (id >= 0) ? id : UNK_ID;
            i++;
            continue;
        }

        /* extract word */
        int word_start = i;
        while (i < text_len && !isspace((unsigned char)lower[i]) && !is_punct(lower[i])) i++;
        int word_len = i - word_start;

        /* WordPiece-encode the word */
        int added = wordpiece(lower + word_start, word_len, input_ids + pos, MAX_SEQ_LEN - 1 - pos);
        pos += added;
    }

    input_ids[pos++] = SEP_ID;

    int token_count = pos;

    /* attention mask: 1 for real tokens, 0 for padding */
    for (int j = 0; j < MAX_SEQ_LEN; j++) {
        attention_mask[j] = (j < pos) ? 1 : 0;
    }

    /* pad the rest */
    while (pos < MAX_SEQ_LEN) {
        input_ids[pos++] = PAD_ID;
    }

    return token_count;
}
