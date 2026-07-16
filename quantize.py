import os
import shutil
import glob
import onnx
import onnxruntime as ort
from onnxruntime.transformers import optimizer

def quantize_models():
    models_dir = "./models"
    pattern = os.path.join(models_dir, "*.onnx")
    onnx_files = glob.glob(pattern)

    # Filter out already quantized models
    base_models = []
    for f in onnx_files:
        name = os.path.basename(f)
        if "_fp16.onnx" not in name:
            base_models.append(f)

    # Determine GPU availability for the Transformer optimizer
    use_gpu = 'CUDAExecutionProvider' in ort.get_available_providers()
    print(f"Found {len(base_models)} base ONNX models to quantize.")
    print(f"CUDA/GPU available for FP16 optimization: {use_gpu}")
    print("-" * 50)

    for model_path in sorted(base_models):
        base_name = os.path.splitext(model_path)[0]
        fp16_path = f"{base_name}_fp16.onnx"

        # Vocab paths
        vocab_path = f"{base_name}.vocab"
        fp16_vocab_path = f"{base_name}_fp16.vocab"

        # 1. Copy vocab files if they exist
        if os.path.exists(vocab_path):
            shutil.copy2(vocab_path, fp16_vocab_path)
            print(f"Copied vocabulary for {os.path.basename(model_path)}")
        else:
            print(f"WARNING: Vocab file not found at {vocab_path}")

        # 2. FP16 Quantization via Transformer Optimizer
        print(f"Quantizing {os.path.basename(model_path)} to FP16 (with Transformer Optimizer)...")
        try:
            # First, optimize the model graph using ONNX Runtime's BERT optimizer.
            # This fuses LayerNorm, Attention, and GeLU blocks, avoiding subsequent fusion/type mismatch bugs in ORT.
            opt_model = optimizer.optimize_model(
                model_path,
                model_type="bert",
                use_gpu=use_gpu
            )
            # Convert internal weights/ops to float16, keeping input/output boundaries float32 for maximum compatibility
            opt_model.convert_float_to_float16(keep_io_types=True)
            opt_model.save_model_to_file(fp16_path)
            print(f"Successfully saved FP16 model to {fp16_path}")
        except Exception as e:
            print(f"ERROR quantizing {model_path} to FP16: {e}")

        print("-" * 50)

    print("Quantization complete.")

if __name__ == "__main__":
    quantize_models()
