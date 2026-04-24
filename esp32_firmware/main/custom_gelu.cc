/**
 * custom_gelu.cc — GELU op implementation for TFLite Micro.
 *
 * Supports both float32 and int8 (quantized) tensors.
 *
 * Float path:
 *   GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))                  (exact)
 *   GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))  (approximate)
 *
 * Int8 path:
 *   Precomputes a 256-entry LUT in Prepare that maps every possible input
 *   INT8 value to its GELU output INT8 value using the input/output tensor
 *   quantization parameters. Eval is just a byte-level table lookup.
 */

#include "custom_gelu.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_context.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace {

constexpr float kSqrt2OverPi = 0.7978845608f;  // sqrt(2/pi)
constexpr float kCoeff = 0.044715f;
constexpr float kInvSqrt2 = 0.7071067811865475f;

struct GeluOpData {
    bool is_int8;
    // INT8 LUT: indexed by (input_byte + 128) → output INT8 byte.
    int8_t lut[256];
};

static inline float gelu_exact(float x) {
    return 0.5f * x * (1.0f + erff(x * kInvSqrt2));
}

static inline float gelu_approx(float x) {
    float cdf = 0.5f * (1.0f + tanhf(kSqrt2OverPi * (x + kCoeff * x * x * x)));
    return x * cdf;
}

void* GeluInit(TfLiteContext* context, const char* /*buffer*/, size_t /*length*/) {
    void* p = context->AllocatePersistentBuffer(context, sizeof(GeluOpData));
    return p;
}

TfLiteStatus GeluPrepare(TfLiteContext* context, TfLiteNode* node) {
    TF_LITE_ENSURE_EQ(context, tflite::NumInputs(node), 1);
    TF_LITE_ENSURE_EQ(context, tflite::NumOutputs(node), 1);

    auto* op_data = static_cast<GeluOpData*>(node->user_data);

    tflite::MicroContext* micro_ctx = tflite::GetMicroContext(context);
    TfLiteTensor* input = micro_ctx->AllocateTempInputTensor(node, 0);
    TfLiteTensor* output = micro_ctx->AllocateTempOutputTensor(node, 0);

    TF_LITE_ENSURE(context, input != nullptr);
    TF_LITE_ENSURE(context, output != nullptr);
    TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);

    // Determine which mode from GELU options.
    bool approximate = false;
    if (node->builtin_data != nullptr) {
        auto* params = reinterpret_cast<TfLiteGeluParams*>(node->builtin_data);
        approximate = params->approximate;
    }

    if (input->type == kTfLiteFloat32) {
        op_data->is_int8 = false;
        micro_ctx->DeallocateTempTfLiteTensor(input);
        micro_ctx->DeallocateTempTfLiteTensor(output);
        return kTfLiteOk;
    }

    if (input->type != kTfLiteInt8) {
        MicroPrintf("GELU: unsupported type %d (only f32 and int8 supported)",
                    (int)input->type);
        micro_ctx->DeallocateTempTfLiteTensor(input);
        micro_ctx->DeallocateTempTfLiteTensor(output);
        return kTfLiteError;
    }

    // INT8 path: build 256-entry LUT using tensor quantization params.
    op_data->is_int8 = true;
    const float in_scale  = input->params.scale;
    const int32_t in_zp   = input->params.zero_point;
    const float out_scale = output->params.scale;
    const int32_t out_zp  = output->params.zero_point;

    // Capture everything we need; we're done with the temps.
    micro_ctx->DeallocateTempTfLiteTensor(input);
    micro_ctx->DeallocateTempTfLiteTensor(output);

    for (int i = 0; i < 256; i++) {
        int q = i - 128;                          // INT8 value
        float x = (q - in_zp) * in_scale;         // dequantize
        float y = approximate ? gelu_approx(x) : gelu_exact(x);
        int32_t q_out = static_cast<int32_t>(std::lround(y / out_scale)) + out_zp;
        if (q_out < -128) q_out = -128;
        if (q_out >  127) q_out =  127;
        op_data->lut[i] = static_cast<int8_t>(q_out);
    }

    return kTfLiteOk;
}

TfLiteStatus GeluEval(TfLiteContext* context, TfLiteNode* node) {
    const TfLiteEvalTensor* input =
        tflite::micro::GetEvalInput(context, node, 0);
    TfLiteEvalTensor* output =
        tflite::micro::GetEvalOutput(context, node, 0);

    auto* op_data = static_cast<const GeluOpData*>(node->user_data);

    int num_elements = 1;
    for (int i = 0; i < input->dims->size; i++) {
        num_elements *= input->dims->data[i];
    }

    if (!op_data->is_int8) {
        // Float path.
        const float* in_data = tflite::micro::GetTensorData<float>(input);
        float* out_data = tflite::micro::GetTensorData<float>(output);

        bool approximate = false;
        if (node->builtin_data != nullptr) {
            auto* params = reinterpret_cast<TfLiteGeluParams*>(node->builtin_data);
            approximate = params->approximate;
        }
        if (approximate) {
            for (int i = 0; i < num_elements; i++) out_data[i] = gelu_approx(in_data[i]);
        } else {
            for (int i = 0; i < num_elements; i++) out_data[i] = gelu_exact(in_data[i]);
        }
        return kTfLiteOk;
    }

    // INT8 LUT path.
    const int8_t* in_data = tflite::micro::GetTensorData<int8_t>(input);
    int8_t* out_data = tflite::micro::GetTensorData<int8_t>(output);
    const int8_t* lut = op_data->lut;
    for (int i = 0; i < num_elements; i++) {
        // Index by (in + 128) so -128..127 maps to 0..255.
        out_data[i] = lut[static_cast<uint8_t>(in_data[i] + 128)];
    }
    return kTfLiteOk;
}

}  // namespace

const TFLMRegistration& Register_GELU(void) {
    static TFLMRegistration r = tflite::micro::RegisterOp(
        /*init=*/GeluInit,
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
