import os
import glob
import numpy as np
import onnxruntime as ort

def verify_models():
    models_dir = "./models"
    pattern = os.path.join(models_dir, "*.onnx")
    onnx_files = glob.glob(pattern)

    # Filter base models
    base_models = []
    for f in onnx_files:
        name = os.path.basename(f)
        if "_fp16.onnx" not in name:
            base_models.append(f)

    print("=" * 60)
    print("Verifying Quantized Models")
    print("=" * 60)

    # Dummy inputs: batch=1, seq_len=8
    dummy_input_ids = np.random.randint(1, 1000, (1, 8), dtype=np.int64)
    dummy_attention_mask = np.ones((1, 8), dtype=np.int64)

    for base_model_path in sorted(base_models):
        base_name = os.path.splitext(base_model_path)[0]
        base_size = os.path.getsize(base_model_path) / (1024 * 1024)
        print(f"Base model: {os.path.basename(base_model_path)} ({base_size:.2f} MB)")

        # Run base model to get ground truth
        try:
            base_session = ort.InferenceSession(base_model_path, providers=['CPUExecutionProvider'])
            base_outputs = base_session.run(
                ["cws_logits", "pos_logits", "dep_arc_scores", "dep_rel_scores"],
                {"input_ids": dummy_input_ids, "attention_mask": dummy_attention_mask}
            )
        except Exception as e:
            print(f"  Error running base model: {e}")
            continue

        q_type = "fp16"
        q_model_path = f"{base_name}_{q_type}.onnx"
        if not os.path.exists(q_model_path):
            print(f"  ERROR: Quantized model {q_model_path} does not exist!")
            continue

        q_size = os.path.getsize(q_model_path) / (1024 * 1024)
        reduction = (1 - (q_size / base_size)) * 100

        try:
            # Load and run quantized model
            q_session = ort.InferenceSession(q_model_path, providers=['CPUExecutionProvider'])
            q_outputs = q_session.run(
                ["cws_logits", "pos_logits", "dep_arc_scores", "dep_rel_scores"],
                {"input_ids": dummy_input_ids, "attention_mask": dummy_attention_mask}
            )

            # Validate output types and shapes
            assert len(base_outputs) == len(q_outputs), "Output count mismatch"
            for idx, (b_out, q_out) in enumerate(zip(base_outputs, q_outputs)):
                assert b_out.shape == q_out.shape, f"Shape mismatch at output {idx}"
                assert q_out.dtype == np.float32, f"Output {idx} is not float32 (got {q_out.dtype})"

            # Calculate differences (Mean Absolute Error) for CWS logits
            mae = np.mean(np.abs(base_outputs[0] - q_outputs[0]))
            print(f"  -> {q_type.upper()}: {q_size:.2f} MB ({reduction:.1f}% smaller). OK. CWS logit MAE: {mae:.6f}")
            
        except Exception as e:
            print(f"  -> {q_type.upper()}: FAILED verification: {e}")

        print("-" * 60)

if __name__ == "__main__":
    verify_models()
