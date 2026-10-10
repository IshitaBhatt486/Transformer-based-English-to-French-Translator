"""Generate real attention heatmaps from one validation example and checkpoint."""

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset import causal_mask
from predictor import Translator
from train import get_ds


def token_labels(tokenizer, ids):
    return [tokenizer.id_to_token(int(token_id)) for token_id in ids]


def decode_with_scores(translator, source_text):
    cfg, device = translator.config, translator.device
    src_tokenizer, tgt_tokenizer = translator.tokenizer_src, translator.tokenizer_tgt
    src_ids = src_tokenizer.encode(source_text).ids
    src_tokens = [src_tokenizer.token_to_id("[SOS]"), *src_ids, src_tokenizer.token_to_id("[EOS]")]
    padded = src_tokens + [src_tokenizer.token_to_id("[PAD]")] * (cfg["seq_length"] - len(src_tokens))
    source = torch.tensor([padded], dtype=torch.long, device=device)
    source_mask = (source != src_tokenizer.token_to_id("[PAD]")).unsqueeze(1).unsqueeze(1)
    encoder_output = translator.model.encode(source, source_mask)

    sos, eos = tgt_tokenizer.token_to_id("[SOS]"), tgt_tokenizer.token_to_id("[EOS]")
    decoder_input = torch.tensor([[sos]], dtype=torch.long, device=device)
    while decoder_input.size(1) < cfg["seq_length"]:
        decoder_mask = causal_mask(decoder_input.size(1)).to(device)
        output = translator.model.decode(decoder_input, encoder_output, source_mask, decoder_mask)
        next_token = translator.model.project(output[:, -1]).argmax(dim=-1)
        decoder_input = torch.cat([decoder_input, next_token.unsqueeze(0)], dim=1)
        if next_token.item() == eos:
            break
    return src_tokens, decoder_input.squeeze(0).tolist()


def save_heatmap(scores, row_labels, col_labels, title, output):
    heads = scores.shape[0]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True)
    for head, ax in enumerate(axes.flat):
        image = ax.imshow(scores[head], aspect="auto", cmap="viridis")
        ax.set_title(f"Head {head}")
        ax.set_xticks(range(len(col_labels)), col_labels, rotation=70, ha="right", fontsize=7)
        ax.set_yticks(range(len(row_labels)), row_labels, fontsize=7)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(title, fontsize=14)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"Wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render attention maps from a real checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sample-index", type=int, default=0, help="Index in deterministic validation split")
    parser.add_argument("--layer", type=int, default=0, help="Attention layer, zero indexed")
    parser.add_argument("--max-tokens", type=int, default=20)
    parser.add_argument("--device")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for heatmaps (defaults to docs/assets/attention in the repository)",
    )
    args = parser.parse_args()

    translator = Translator(checkpoint=args.checkpoint, device=args.device)
    if not 0 <= args.layer < translator.config["num_layers"]:
        raise ValueError(f"layer must be 0 through {translator.config['num_layers'] - 1}")
    _, validation_loader, _, _ = get_ds(translator.config)
    if not 0 <= args.sample_index < len(validation_loader.dataset):
        raise ValueError(f"sample-index must be 0 through {len(validation_loader.dataset) - 1}")
    source_text = validation_loader.dataset[args.sample_index]["src_text"]
    src_ids, tgt_ids = decode_with_scores(translator, source_text)
    max_src, max_tgt = min(len(src_ids), args.max_tokens), min(len(tgt_ids), args.max_tokens)
    src_labels = token_labels(translator.tokenizer_src, src_ids[:max_src])
    tgt_labels = token_labels(translator.tokenizer_tgt, tgt_ids[:max_tgt])
    out = args.output_dir or ROOT / "docs" / "assets" / "attention"

    encoder = translator.model.encoder.layers[args.layer].self_attention_block.attention_scores[0, :, :max_src, :max_src].detach().cpu().numpy()
    decoder = translator.model.decoder.layers[args.layer].self_attention_block.attention_scores[0, :, :max_tgt, :max_tgt].detach().cpu().numpy()
    cross = translator.model.decoder.layers[args.layer].cross_attention_block.attention_scores[0, :, :max_tgt, :max_src].detach().cpu().numpy()
    save_heatmap(encoder, src_labels, src_labels, f"Encoder self-attention - layer {args.layer}", out / f"encoder_layer_{args.layer}.png")
    save_heatmap(decoder, tgt_labels, tgt_labels, f"Decoder causal self-attention - layer {args.layer}", out / f"decoder_layer_{args.layer}.png")
    save_heatmap(cross, tgt_labels, src_labels, f"Decoder cross-attention - layer {args.layer}", out / f"cross_layer_{args.layer}.png")


if __name__ == "__main__":
    main()
