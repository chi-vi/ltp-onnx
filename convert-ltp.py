import torch
import torch.nn as nn
from ltp import LTP

# 1. Load your standard LTP model (small, base, or tiny)
ltp = LTP("LTP/small")
export_path = "./models/ltp_small.onnx"
vocab_path = export_path.replace(".onnx", ".vocab")

# 2. Define the simplified wrapper
class LtpSimplifiedONNX(nn.Module):
    def __init__(self, raw_model):
        super().__init__()
        # In LTP4, the shared encoder is stored in raw_model.backbone
        self.backbone = raw_model.backbone

        # Extract the specific task-specific heads you need from raw_model.task_heads
        self.cws_head = raw_model.task_heads.cws  # Tokenization / Word segmentation head
        self.pos_head = raw_model.task_heads.pos  # Part of Speech head
        self.dep_head = raw_model.task_heads.dep  # Dependency parsing head

    def forward(self, input_ids, attention_mask):
        # 1. Forward pass through the Transformer backbone
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]  # Shape: [batch, seq_len, hidden_size]

        # 2. Extract Tokenization (CWS) logits
        cws_logits = self.cws_head(sequence_output).logits

        # 3. Extract Part-of-Speech (POS) logits
        pos_logits = self.pos_head(sequence_output).logits

        # 4. Extract Dependency parsing (DEP) arc and relation scores
        dep_result = self.dep_head(sequence_output)
        dep_arc_scores = dep_result.arc_logits
        dep_rel_scores = dep_result.rel_logits

        return cws_logits, pos_logits, dep_arc_scores, dep_rel_scores


# Create dummy inputs for tracing (batch size = 1, sequence length = 16)
dummy_input_ids = torch.randint(1, 1000, (1, 16))
dummy_attention_mask = torch.ones((1, 16), dtype=torch.long)

raw_model = ltp.model  # Access the underlying PyTorch model
simplified_model = LtpSimplifiedONNX(raw_model)
simplified_model.eval()


torch.onnx.export(
    simplified_model,
    (dummy_input_ids, dummy_attention_mask),
    export_path,
    input_names=["input_ids", "attention_mask"],
    output_names=["cws_logits", "pos_logits", "dep_arc_scores", "dep_rel_scores"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "cws_logits": {0: "batch_size", 1: "sequence_length"},
        "pos_logits": {0: "batch_size", 1: "sequence_length"},
        # Dependency matrices are sequence_length x sequence_length
        "dep_arc_scores": {0: "batch_size", 1: "sequence_length", 2: "sequence_length"},
        "dep_rel_scores": {0: "batch_size", 1: "sequence_length", 2: "sequence_length"}
    },
    opset_version=13  # Highly compatible across ONNX Runtimes
)

# Export vocabulary mapping for the Rust parser

with open(vocab_path, "w", encoding="utf-8") as f:
    f.write(",".join(ltp.cws_vocab) + "\n")
    f.write(",".join(ltp.pos_vocab) + "\n")
    f.write(",".join(ltp.dep_vocab) + "\n")
