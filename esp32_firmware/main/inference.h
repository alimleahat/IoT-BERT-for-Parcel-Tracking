#ifndef INFERENCE_H
#define INFERENCE_H

#include <stdint.h>

#define NUM_INTENTS 6

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Initialise TFLite Micro: load model from flash, allocate tensor arena in PSRAM.
 * Returns 0 on success, -1 on failure.
 */
int inference_init(const uint8_t *model_data, int model_len);

/**
 * Run intent classification on tokenized input.
 *
 * @param input_ids       Token IDs array of length seq_len (64)
 * @param attention_mask  Mask array of length seq_len (64)
 * @param out_scores      Output logits/scores for each intent (6 floats)
 * @return                Predicted intent index (0-5), or -1 on error
 */
int inference_run(const int32_t *input_ids, const int32_t *attention_mask,
                  float *out_scores);

/**
 * Map intent index to string label.
 */
const char *inference_intent_label(int idx);

#ifdef __cplusplus
}
#endif

#endif
