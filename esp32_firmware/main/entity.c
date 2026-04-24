/**
 * entity.c — Rule-based entity extraction for parcel tracker commands.
 *
 * Since our commands are short and structured, regex-level parsing is
 * overkill.  We scan for:
 *   - Numbers          → order_id
 *   - Known depot names → depot_name / depot_id
 *   - Weight patterns   → weight  (e.g. "3kg", "2.5 kg")
 */

#include <string.h>
#include <ctype.h>
#include <stdlib.h>
#include "entity.h"

/* Known couriers — must match depots.txt */
static const struct { int id; const char *name; } COURIERS[] = {
    {1, "fadex"},
    {2, "usp"},
    {3, "dlh"},
    {4, "royalmile"},
    {5, "pdp"},
};
#define NUM_COURIERS 5

/* ── Helpers ──────────────────────────────────────────────────────────── */

/** Find first integer in string. Returns -1 if none found. */
static int find_number(const char *text)
{
    const char *p = text;
    while (*p) {
        if (isdigit((unsigned char)*p)) {
            return atoi(p);
        }
        p++;
    }
    return -1;
}

/** Find a weight pattern like "3kg", "2.5 kg", "4 kilos". Returns -1.0 if not found. */
static float find_weight(const char *text)
{
    /* Make lowercase copy */
    char lower[256];
    int len = strlen(text);
    if (len > 255) len = 255;
    for (int i = 0; i < len; i++) lower[i] = tolower((unsigned char)text[i]);
    lower[len] = '\0';

    /* Look for "kg" and scan backwards for number */
    const char *kg = strstr(lower, "kg");
    if (!kg) kg = strstr(lower, "kilo");
    if (kg) {
        /* Scan backwards from kg position to find the number */
        const char *p = kg - 1;
        while (p >= lower && (*p == ' ')) p--;
        if (p >= lower) {
            /* Find start of number */
            const char *num_end = p + 1;
            while (p >= lower && (isdigit((unsigned char)*p) || *p == '.')) p--;
            p++;
            if (p < num_end) {
                return (float)atof(p);
            }
        }
    }
    return -1.0f;
}

/** Find a known courier/depot name in text. */
static int find_courier(const char *text, char *name_out)
{
    char lower[256];
    int len = strlen(text);
    if (len > 255) len = 255;
    for (int i = 0; i < len; i++) lower[i] = tolower((unsigned char)text[i]);
    lower[len] = '\0';

    for (int i = 0; i < NUM_COURIERS; i++) {
        if (strstr(lower, COURIERS[i].name)) {
            strcpy(name_out, COURIERS[i].name);
            /* Capitalize first letter for the server */
            name_out[0] = toupper((unsigned char)name_out[0]);
            /* Special cases */
            if (COURIERS[i].id == 1) strcpy(name_out, "FadEx");
            if (COURIERS[i].id == 2) strcpy(name_out, "USP");
            if (COURIERS[i].id == 3) strcpy(name_out, "DLH");
            if (COURIERS[i].id == 4) strcpy(name_out, "RoyalMile");
            if (COURIERS[i].id == 5) strcpy(name_out, "PDP");
            return COURIERS[i].id;
        }
    }

    /* Also check for "depot X" or "courier X" with a number */
    const char *depot_str = strstr(lower, "depot");
    if (!depot_str) depot_str = strstr(lower, "courier");
    if (depot_str) {
        int id = find_number(depot_str);
        if (id >= 1 && id <= 5) {
            name_out[0] = '\0';
            return id;
        }
    }

    name_out[0] = '\0';
    return -1;
}

/* ── Main extraction ──────────────────────────────────────────────────── */

void entity_extract(const char *text, const char *intent, entities_t *out)
{
    out->order_id     = -1;
    out->depot_id     = -1;
    out->depot_name[0] = '\0';
    out->weight       = -1.0f;
    out->courier_id   = -1;
    out->courier_name[0] = '\0';

    if (strcmp(intent, "VIEW_ORDER") == 0 || strcmp(intent, "SEARCH_ORDER") == 0) {
        out->order_id = find_number(text);
    }
    else if (strcmp(intent, "FILTER_BY_DEPOT") == 0) {
        out->depot_id = find_courier(text, out->depot_name);
    }
    else if (strcmp(intent, "CALCULATE_COST") == 0) {
        out->weight = find_weight(text);
        out->courier_id = find_courier(text, out->courier_name);

        /* Decide if the bare number in the text is an order_id or a weight.
         *   "cost of order 101"       → order_id = 101  (server looks it up)
         *   "cost of a 5kg package"   → weight = 5.0    (explicit kg)
         *   "cost for 4 at FadEx"     → weight = 4.0    (fallback)
         * When no "kg"/"kilo" unit was found, prefer order_id if the phrase
         * literally mentions "order"; otherwise fall back to treating the
         * number as a weight. */
        if (out->weight < 0) {
            /* lowercase search for "order" keyword */
            char lower[256];
            int len = (int)strlen(text);
            if (len > 255) len = 255;
            for (int i = 0; i < len; i++) lower[i] = (char)tolower((unsigned char)text[i]);
            lower[len] = '\0';

            if (strstr(lower, "order") != NULL) {
                out->order_id = find_number(text);
            } else {
                int n = find_number(text);
                if (n >= 0) out->weight = (float)n;
            }
        }
    }
    /* VIEW_ALL and SHOW_HISTORY need no entities */
}
