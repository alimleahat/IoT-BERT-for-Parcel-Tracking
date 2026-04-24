/**
 * custom_fc.h — custom FullyConnected kernel for TFLite Micro.
 *
 * Extends the default FC kernel (which only handles pure-float or pure-int8)
 * to also support the HYBRID case produced by dynamic-range quantization:
 *   FP32 input  ×  INT8 filter (per-channel scales)  →  FP32 output
 *
 * This is what weights-only INT8 quantization produces, and it's the only
 * regime where BERT keeps its accuracy on-device.  Register with:
 *
 *     resolver.AddFullyConnected(Register_HYBRID_FC());
 */

#pragma once
#include "tensorflow/lite/micro/micro_common.h"

TFLMRegistration Register_HYBRID_FC(void);
