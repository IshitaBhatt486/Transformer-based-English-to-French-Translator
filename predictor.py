from pathlib import Path
from typing import Optional

import torch
from tokenizers import Tokenizer

from config import get_config, latest_weights_file_path
from dataset import causal_mask
from model import build_Transformer


class Translator:
    """Load model artifacts once and serve repeatable translations."""

    def __init__(self, checkpoint: Optional[str] = None, device: Optional[str] = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        checkpoint = checkpoint or latest_weights_file_path(get_config())
        if checkpoint is None:
            raise FileNotFoundError("No checkpoint found. Run train.py first.")
        state = torch.load(checkpoint, map_location=self.device)
        self.config = state.get("config", get_config())
        src_path = Path(self.config["tokenizer_file"].format(self.config["src_lang"]))
        tgt_path = Path(self.config["tokenizer_file"].format(self.config["tgt_lang"]))
        if not src_path.exists() or not tgt_path.exists():
            raise FileNotFoundError("Tokenizer artifacts are missing beside the application.")
        self.tokenizer_src = Tokenizer.from_file(str(src_path))
        self.tokenizer_tgt = Tokenizer.from_file(str(tgt_path))
        self.model = build_Transformer(
            self.tokenizer_src.get_vocab_size(), self.tokenizer_tgt.get_vocab_size(),
            self.config["seq_length"], self.config["seq_length"], self.config["d_model"],
            self.config["num_layers"], self.config["num_heads"],
            self.config["dropout_rate"], self.config["d_ff"],
        ).to(self.device)
        self.model.load_state_dict(state["model_state_dict"])
        self.model.eval()
        self.checkpoint = checkpoint

    @torch.inference_mode()
    def translate(self, sentence: str) -> str:
        if not sentence or not sentence.strip():
            raise ValueError("sentence must not be empty")
        seq_len = self.config["seq_length"]
        source_ids = self.tokenizer_src.encode(sentence).ids
        if len(source_ids) > seq_len - 2:
            raise ValueError(f"Input exceeds the {seq_len - 2}-token limit")
        special = lambda name: self.tokenizer_src.token_to_id(name)
        tokens = [special("[SOS]"), *source_ids, special("[EOS]")]
        tokens += [special("[PAD]")] * (seq_len - len(tokens))
        source = torch.tensor(tokens, dtype=torch.long, device=self.device).unsqueeze(0)
        source_mask = (source != special("[PAD]")).unsqueeze(1).unsqueeze(1)
        encoder_output = self.model.encode(source, source_mask)
        sos = self.tokenizer_tgt.token_to_id("[SOS]")
        eos = self.tokenizer_tgt.token_to_id("[EOS]")
        decoder_input = torch.tensor([[sos]], dtype=torch.long, device=self.device)
        while decoder_input.size(1) < seq_len:
            decoder_mask = causal_mask(decoder_input.size(1)).to(self.device)
            output = self.model.decode(decoder_input, encoder_output, source_mask, decoder_mask)
            next_token = self.model.project(output[:, -1]).argmax(dim=-1)
            decoder_input = torch.cat([decoder_input, next_token.unsqueeze(0)], dim=1)
            if next_token.item() == eos:
                break
        ids = decoder_input.squeeze(0).tolist()[1:]
        if eos in ids:
            ids = ids[:ids.index(eos)]
        return self.tokenizer_tgt.decode(ids, skip_special_tokens=True)
