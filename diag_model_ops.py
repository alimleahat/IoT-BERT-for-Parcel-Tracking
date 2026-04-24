#!/usr/bin/env python3
"""Print the set of ops used by a .tflite file."""
import sys
import tensorflow as tf

path = sys.argv[1]
interp = tf.lite.Interpreter(model_path=path)
interp.allocate_tensors()
ops = sorted({d["op_name"] for d in interp._get_ops_details()})
print(f"\n{path} ({len(ops)} ops):")
for op in ops:
    print(f"  {op}")
