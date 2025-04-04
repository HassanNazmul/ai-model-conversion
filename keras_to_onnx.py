# keras_to_onnx.py

import tensorflow as tf
import tf2onnx
from transformers import TFViTModel

# Recreate base_model EXACTLY as in your original training
base_model = TFViTModel.from_pretrained("google/vit-large-patch16-224")

# Define the ViT layer exactly as in original training code
def vit_layer(x):
    return base_model({"pixel_values": x}).last_hidden_state[:, 0, :]

# Load the saved Keras model with custom objects provided
custom_objects = {
    'base_model': base_model,
    'vit_layer': vit_layer,
    'tf': tf
}

model = tf.keras.models.load_model(
    'your_model_path.h5',   # Replace with your actual model path
    custom_objects=custom_objects,
    compile=False
)

# Conversion spec exactly matches the input shape from original model
spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)

output_path = "your_model.onnx"  # Replace with your desired output path

# Convert to ONNX
model_proto, _ = tf2onnx.convert.from_keras(
    model,
    input_signature=spec,
    opset=14,
    output_path=output_path
)

print("Model converted successfully to ONNX:", output_path)
