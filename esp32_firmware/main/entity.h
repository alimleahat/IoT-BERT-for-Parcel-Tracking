#ifndef ENTITY_H
#define ENTITY_H

/**
 * Extracted entities from user text, populated based on the classified intent.
 */
typedef struct {
    int  order_id;          /* -1 if not found */
    int  depot_id;          /* -1 if not found */
    char depot_name[32];    /* empty string if not found */
    float weight;           /* -1.0 if not found */
    int  courier_id;        /* -1 if not found */
    char courier_name[32];  /* empty string if not found */
} entities_t;

/**
 * Extract relevant entities from user text based on the classified intent.
 * Simple rule-based extraction: numbers for IDs, known depot names, weight patterns.
 */
void entity_extract(const char *text, const char *intent, entities_t *out);

#endif
