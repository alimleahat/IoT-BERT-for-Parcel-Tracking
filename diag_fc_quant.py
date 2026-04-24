"""Inspect FC filter quantization in the dynamic-range model."""
import tensorflow as tf
interp = tf.lite.Interpreter(model_path="tflite_models/bert_tiny_intent_dynamic.tflite")
interp.allocate_tensors()
for d in interp.get_tensor_details():
    if d["dtype"].__name__ == "int8":
        q = d["quantization_parameters"]
        scales = q["scales"]
        print(f"name={d['name']!s:80s} shape={d['shape']} scales_len={len(scales)} first_scales={scales[:3]}")
