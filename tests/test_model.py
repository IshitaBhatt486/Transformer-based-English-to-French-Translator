import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from dataset import BilingualDataset
from model import build_Transformer


def tokenizer(tokens):
    vocab = {token: index for index, token in enumerate(tokens)}
    return Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))


def test_dataset_uses_target_special_token_ids():
    src = tokenizer(["[UNK]", "[PAD]", "[SOS]", "[EOS]", "hello"])
    tgt = tokenizer(["[UNK]", "bonjour", "[EOS]", "[SOS]", "[PAD]"])
    raw = [{"translation": {"en": "hello", "fr": "bonjour"}}]
    item = BilingualDataset(raw, src, tgt, "en", "fr", 6)[0]
    assert item["decoder_input"].tolist() == [3, 1, 4, 4, 4, 4]
    assert item["label"].tolist() == [1, 2, 4, 4, 4, 4]


def test_transformer_forward_shapes_and_finite_values():
    model = build_Transformer(20, 25, 8, 8, dmodel=16, N=2, h=4, dff=32)
    src = torch.randint(0, 20, (2, 8))
    tgt = torch.randint(0, 25, (2, 8))
    src_mask = torch.ones((2, 1, 1, 8), dtype=torch.bool)
    tgt_mask = torch.ones((2, 1, 8, 8), dtype=torch.bool)
    logits = model.project(model.decode(tgt, model.encode(src, src_mask), src_mask, tgt_mask))
    assert logits.shape == (2, 8, 25)
    assert torch.isfinite(logits).all()
