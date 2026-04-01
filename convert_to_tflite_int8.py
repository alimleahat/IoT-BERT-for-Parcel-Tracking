"""Convert HuggingFace BERT models to TFLite INT8 format."""

import os
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel

MODELS = ["prajjwal1/bert-tiny", "prajjwal1/bert-mini"]
SEQ_LEN = 128
NUM_CALIBRATION = 100
OUTPUT_DIR = "tflite_models"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def representative_dataset(tokenizer):
    """Generate calibration data for INT8 quantization."""
    for _ in range(NUM_CALIBRATION):
        dummy_text = "This is a sample sentence for calibration purposes."
        inputs = tokenizer(
            dummy_text,
            return_tensors="np",
            max_length=SEQ_LEN,
            padding="max_length",
            truncation=True,
        )
        yield [
            inputs["input_ids"].astype(np.int32),
            inputs["attention_mask"].astype(np.int32),
        ]


def convert_model(model_name):
    print(f"\n{'='*60}")
    print(f"Processing: {model_name}")
    print(f"{'='*60}")

    # Load model and tokenizer
    print("Loading tokenizer and model...")
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = TFBertModel.from_pretrained(model_name, from_pt=True)

    # Build concrete function with fixed input shapes
    print("Building concrete function...")

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[1, SEQ_LEN], dtype=tf.int32, name="input_ids"),
        tf.TensorSpec(shape=[1, SEQ_LEN], dtype=tf.int32, name="attention_mask"),
    ])
    def serving_fn(input_ids, attention_mask):
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        return {"last_hidden_state": output.last_hidden_state}

    concrete_func = serving_fn.get_concrete_function()

    # Convert to TFLite with INT8 quantization
    print("Converting to TFLite INT8...")
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset(tokenizer)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    # Save
    safe_name = model_name.replace("/", "_")
    output_path = os.path.join(OUTPUT_DIR, f"{safe_name}_int8.tflite")
    with open(output_path, "wb") as f:
        f.write(tflite_model)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Saved: {output_path}")
    print(f"Size:  {size_mb:.2f} MB ({os.path.getsize(output_path):,} bytes)")
    return output_path, size_mb


if __name__ == "__main__":
    results = []
    for model_name in MODELS:
        path, size = convert_model(model_name)
        results.append((model_name, path, size))

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    for name, path, size in results:
        print(f"  {name:25s} -> {size:.2f} MB  ({path})")
