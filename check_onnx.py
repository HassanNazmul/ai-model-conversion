# check_onnx.py

import onnx
import onnxruntime as ort
import numpy as np

onnx_model = onnx.load("Path to your converted ONNX model") # Replace with the actual path to your ONNX model
onnx.checker.check_model(onnx_model)
print("ONNX model is valid.")

# Quick inference check
session = ort.InferenceSession("Path to your converted ONNX model") # Replace with the actual path to your ONNX model
dummy_input = np.random.randn(1, 224, 224, 3).astype(np.float32)
outputs = session.run(None, {"input": dummy_input})
print("ONNX inference successful, output shape:", outputs[0].shape)
