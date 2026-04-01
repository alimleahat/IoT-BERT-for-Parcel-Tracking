#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <stdint.h>

#define MAX_SEQ_LEN   64
#define VOCAB_SIZE    30522
#define PAD_ID        0
#define UNK_ID        100
#define CLS_ID        101
#define SEP_ID        102

/**
 * Initialise the WordPiece tokenizer.
 * Parses the embedded vocab.txt into a hash table stored in PSRAM.
 * Must be called once before tokenizer_encode().
 */
void tokenizer_init(const char *vocab_data, int vocab_len);

/**
 * Encode a text string into BERT input_ids and attention_mask.
 *
 * Produces:  [CLS] token1 token2 ... tokenN [SEP] [PAD] ...
 *
 * @param text           Input text (will be lowercased internally)
 * @param input_ids      Output array of size MAX_SEQ_LEN
 * @param attention_mask Output array of size MAX_SEQ_LEN
 * @return               Number of tokens (including CLS/SEP)
 */
int tokenizer_encode(const char *text, int32_t *input_ids, int32_t *attention_mask);

#endif
