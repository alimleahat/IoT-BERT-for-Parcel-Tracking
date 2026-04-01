#ifndef CUSTOM_GELU_H
#define CUSTOM_GELU_H

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

/**
 * GELU kernel for TFLite Micro (not provided by esp-tflite-micro).
 * Uses tanh approximation: GELU(x) ≈ 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))
 */
const TFLMRegistration& Register_GELU(void);

/**
 * Parse function for GELU op (reads GeluOptions from flatbuffer).
 */
TfLiteStatus ParseGelu(const tflite::Operator* op,
                        tflite::ErrorReporter* error_reporter,
                        tflite::BuiltinDataAllocator* allocator,
                        void** builtin_data);

#endif
