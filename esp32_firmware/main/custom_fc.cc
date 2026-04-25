/**
 * custom_fc.cc — FullyConnected kernel with hybrid (FP32/INT8) support.
 *
 * Modes:
 *   FC_FLOAT   : FP32 input, FP32 filter, FP32 output   → reference float FC
 *   FC_HYBRID  : FP32 input, INT8 filter, FP32 output   → our own loop
 *   (pure INT8 is not used by dynamic-range models and intentionally errors)
 *
 * Hybrid math, per output row i (per-channel scale[i]):
 *     output[i] = (sum_j filter[i,j] * input[j]) * scale[i] + bias[i]
 */

#include "custom_fc.h"

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/reference/fully_connected.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/fully_connected.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_context.h"
#include "tensorflow/lite/micro/micro_log.h"

#include <cstdint>
#include <cmath>

namespace {

enum FcMode : uint8_t { FC_FLOAT, FC_HYBRID, FC_UNSUPPORTED };

struct FcOpData {
    FcMode mode;
    bool has_bias;
    bool is_per_channel;
    int in_features;
    int out_features;
    const float* scales;       // persistent pointer to filter scales
    TfLiteFusedActivation activation;
};

/* Standard input-tensor indices for FullyConnected. */
constexpr int kInputTensor  = 0;
constexpr int kFilterTensor = 1;
constexpr int kBiasTensor   = 2;
constexpr int kOutputTensor = 0;

void* FcInit(TfLiteContext* context, const char* /*buffer*/, size_t /*length*/) {
    return context->AllocatePersistentBuffer(context, sizeof(FcOpData));
}

TfLiteStatus FcPrepare(TfLiteContext* context, TfLiteNode* node) {
    auto* op_data = static_cast<FcOpData*>(node->user_data);
    const auto* params =
        static_cast<const TfLiteFullyConnectedParams*>(node->builtin_data);
    op_data->activation = params ? params->activation : kTfLiteActNone;

    tflite::MicroContext* micro_ctx = tflite::GetMicroContext(context);

    TfLiteTensor* input  = micro_ctx->AllocateTempInputTensor(node, kInputTensor);
    TfLiteTensor* filter = micro_ctx->AllocateTempInputTensor(node, kFilterTensor);
    TfLiteTensor* bias   = nullptr;
    if (tflite::NumInputs(node) >= 3) {
        bias = micro_ctx->AllocateTempInputTensor(node, kBiasTensor);
    }
    TfLiteTensor* output = micro_ctx->AllocateTempOutputTensor(node, kOutputTensor);

    TfLiteStatus status = kTfLiteOk;

    if (input == nullptr || filter == nullptr || output == nullptr) {
        MicroPrintf("Custom FC: missing required tensors");
        status = kTfLiteError;
        op_data->mode = FC_UNSUPPORTED;
        goto done;
    }

    op_data->has_bias     = (bias != nullptr);
    op_data->in_features  = filter->dims->data[1];
    op_data->out_features = filter->dims->data[0];

    if (input->type == kTfLiteFloat32 && filter->type == kTfLiteFloat32 &&
        output->type == kTfLiteFloat32) {
        op_data->mode = FC_FLOAT;
        op_data->scales = nullptr;
        op_data->is_per_channel = false;
    }
    else if (input->type == kTfLiteFloat32 && filter->type == kTfLiteInt8 &&
             output->type == kTfLiteFloat32) {
        op_data->mode = FC_HYBRID;

        auto* q = reinterpret_cast<const TfLiteAffineQuantization*>(
                      filter->quantization.params);
        if (q == nullptr || q->scale == nullptr || q->scale->size < 1) {
            MicroPrintf("Hybrid FC: INT8 filter has no scale");
            status = kTfLiteError;
            op_data->mode = FC_UNSUPPORTED;
            goto done;
        }
        const int n_scales = q->scale->size;
        op_data->is_per_channel = (n_scales > 1);

        /* Copy scales into a persistent buffer.  The TfLiteAffineQuantization
         * struct lives with the TfLiteTensor which is a temp here, so we can't
         * rely on it surviving past Prepare. */
        float* scales_buf = static_cast<float*>(
            context->AllocatePersistentBuffer(context, n_scales * sizeof(float)));
        for (int i = 0; i < n_scales; i++) {
            scales_buf[i] = q->scale->data[i];
        }
        op_data->scales = scales_buf;
    }
    else {
        MicroPrintf("Custom FC: unsupported types input=%d filter=%d output=%d",
                    (int)input->type, (int)filter->type, (int)output->type);
        op_data->mode = FC_UNSUPPORTED;
        status = kTfLiteError;
    }

done:
    if (input)  micro_ctx->DeallocateTempTfLiteTensor(input);
    if (filter) micro_ctx->DeallocateTempTfLiteTensor(filter);
    if (bias)   micro_ctx->DeallocateTempTfLiteTensor(bias);
    if (output) micro_ctx->DeallocateTempTfLiteTensor(output);
    return status;
}

static inline float apply_activation(float x, TfLiteFusedActivation act) {
    switch (act) {
        case kTfLiteActRelu:   return x < 0.0f ? 0.0f : x;
        case kTfLiteActRelu6:  return x < 0.0f ? 0.0f : (x > 6.0f ? 6.0f : x);
        case kTfLiteActReluN1To1: return x < -1.0f ? -1.0f : (x > 1.0f ? 1.0f : x);
        case kTfLiteActTanh:   return std::tanh(x);
        case kTfLiteActNone:
        default:               return x;
    }
}

TfLiteStatus FcEval(TfLiteContext* context, TfLiteNode* node) {
    const auto* op_data = static_cast<const FcOpData*>(node->user_data);
    const TfLiteEvalTensor* input  = tflite::micro::GetEvalInput(context, node, kInputTensor);
    const TfLiteEvalTensor* filter = tflite::micro::GetEvalInput(context, node, kFilterTensor);
    const TfLiteEvalTensor* bias   =
        (tflite::NumInputs(node) >= 3)
            ? tflite::micro::GetEvalInput(context, node, kBiasTensor)
            : nullptr;
    TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, kOutputTensor);

    if (op_data->mode == FC_FLOAT) {
        tflite::reference_ops::FullyConnected(
            tflite::FullyConnectedParamsFloat(op_data->activation),
            tflite::micro::GetTensorShape(input),
            tflite::micro::GetTensorData<float>(input),
            tflite::micro::GetTensorShape(filter),
            tflite::micro::GetTensorData<float>(filter),
            tflite::micro::GetTensorShape(bias),
            op_data->has_bias ? tflite::micro::GetTensorData<float>(bias) : nullptr,
            tflite::micro::GetTensorShape(output),
            tflite::micro::GetTensorData<float>(output));
        return kTfLiteOk;
    }

    if (op_data->mode == FC_HYBRID) {
        const float*  in_data     = tflite::micro::GetTensorData<float>(input);
        const int8_t* filter_data = tflite::micro::GetTensorData<int8_t>(filter);
        const float*  bias_data   =
            op_data->has_bias ? tflite::micro::GetTensorData<float>(bias) : nullptr;
        float*        out_data    = tflite::micro::GetTensorData<float>(output);

        const int N = op_data->in_features;
        const int M = op_data->out_features;
        const bool per_ch = op_data->is_per_channel;
        const float* scales = op_data->scales;
        const TfLiteFusedActivation act = op_data->activation;

        /* Figure out batch count from the input tensor's flat size. */
        int total = 1;
        for (int i = 0; i < input->dims->size; i++) total *= input->dims->data[i];
        const int batches = (N > 0) ? (total / N) : 1;

        /* Single-pass scalar fused-multiply-accumulate. Under -O2 the
         * compiler unrolls/pipelines this loop into something close to
         * 1 FMAC per cycle on Xtensa LX7 — empirically faster than calling
         * a separate dequantize-pass + dsps_dotprod_f32 (the two-pass
         * version doubles memory traffic and adds function-call overhead
         * that isn't amortized for N=128). */
        for (int b = 0; b < batches; b++) {
            const float* x = in_data  + b * N;
            float*       y = out_data + b * M;
            for (int i = 0; i < M; i++) {
                const int8_t* row = filter_data + i * N;
                float acc = 0.0f;
                for (int j = 0; j < N; j++) {
                    acc += (float)row[j] * x[j];
                }
                acc *= per_ch ? scales[i] : scales[0];
                if (bias_data) acc += bias_data[i];
                y[i] = apply_activation(acc, act);
            }
        }
        return kTfLiteOk;
    }

    return kTfLiteError;
}

}  // namespace

TFLMRegistration Register_HYBRID_FC(void) {
    static TFLMRegistration r = tflite::micro::RegisterOp(
        /*init=*/FcInit, /*prepare=*/FcPrepare, /*invoke=*/FcEval);
    return r;
}
