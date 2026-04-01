/**
 * custom_gelu.cc — GELU op implementation for TFLite Micro.
 *
 * GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
 */

#include "custom_gelu.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include <cmath>

namespace {

constexpr float kSqrt2OverPi = 0.7978845608f;  // sqrt(2/pi)
constexpr float kCoeff = 0.044715f;

TfLiteStatus GeluPrepare(TfLiteContext* context, TfLiteNode* node) {
    TF_LITE_ENSURE_EQ(context, tflite::NumInputs(node), 1);
    TF_LITE_ENSURE_EQ(context, tflite::NumOutputs(node), 1);

    const TfLiteTensor* input = tflite::GetInput(context, node, 0);
    TfLiteTensor* output = tflite::GetOutput(context, node, 0);

    TF_LITE_ENSURE_TYPES_EQ(context, input->type, kTfLiteFloat32);
    TF_LITE_ENSURE_TYPES_EQ(context, output->type, kTfLiteFloat32);

    return kTfLiteOk;
}

TfLiteStatus GeluEval(TfLiteContext* context, TfLiteNode* node) {
    const TfLiteTensor* input = tflite::GetInput(context, node, 0);
    TfLiteTensor* output = tflite::GetOutput(context, node, 0);

    const float* in_data = tflite::GetTensorData<float>(input);
    float* out_data = tflite::GetTensorData<float>(output);

    /* Check if approximate mode was requested (from GeluOptions) */
    bool approximate = false;
    if (node->builtin_data != nullptr) {
        auto* params = reinterpret_cast<TfLiteGeluParams*>(node->builtin_data);
        approximate = params->approximate;
    }

    int num_elements = 1;
    for (int i = 0; i < input->dims->size; i++) {
        num_elements *= input->dims->data[i];
    }

    if (approximate) {
        /* Tanh approximation */
        for (int i = 0; i < num_elements; i++) {
            float x = in_data[i];
            float cdf = 0.5f * (1.0f + tanhf(kSqrt2OverPi * (x + kCoeff * x * x * x)));
            out_data[i] = x * cdf;
        }
    } else {
        /* Exact GELU using erf */
        for (int i = 0; i < num_elements; i++) {
            float x = in_data[i];
            out_data[i] = 0.5f * x * (1.0f + erff(x * 0.7071067811865475f));
        }
    }

    return kTfLiteOk;
}

}  // namespace

const TFLMRegistration& Register_GELU(void) {
    static TFLMRegistration r = tflite::micro::RegisterOp(
        /*init=*/nullptr,
        /*prepare=*/GeluPrepare,
        /*invoke=*/GeluEval);
    return r;
}

TfLiteStatus ParseGelu(const tflite::Operator* op,
                        tflite::ErrorReporter* error_reporter,
                        tflite::BuiltinDataAllocator* allocator,
                        void** builtin_data) {
    auto params = allocator->AllocatePOD<TfLiteGeluParams>();
    if (params == nullptr) return kTfLiteError;

    if (const auto* gelu_params = op->builtin_options_as_GeluOptions()) {
        params->approximate = gelu_params->approximate();
    } else {
        params->approximate = false;
    }

    *builtin_data = params;
    return kTfLiteOk;
}
