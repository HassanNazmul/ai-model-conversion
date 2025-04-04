# ViTModel to ONNX Conversion Guide

This guide provides step-by-step instructions for converting a Vision Transformer (ViT) model to ONNX format.

## Prerequisites

Ensure you have the necessary packages installed:

```bash
pip install transformers tensorflow tf2onnx onnxruntime
```

## Conversion Process

### Step 1: Environment Setup
Create and activate a new Python environment to avoid dependency conflicts.

### Step 2: Model Conversion
Run the conversion script to transform the Keras model to ONNX format:

```bash
python keras_to_onnx.py
```

### Step 3: Verification
Verify the converted model works correctly:

```bash
python check_onnx.py
```