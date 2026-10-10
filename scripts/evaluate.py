"""Evaluate a saved checkpoint on the deterministic validation split.

Requires the checkpoint plus the exact tokenizer_en.json and tokenizer_fr.json
from the training run. The script performs real greedy decoding; its BLEU and
chrF values must not be compared to beam-search or differently tokenized runs.
"""

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import sacrebleu
import torch
import torch.nn as nn
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from predictor import Translator
from train import get_ds


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute validation BLEU and chrF for a checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path to a model checkpoint")
    parser.add_argument("--max-samples", type=int, default=1000, help="Validation examples to decode; 0 means all")
    parser.add_argument("--device", help="cpu or cuda; defaults to available device")
    parser.add_argument(
        "--output",
        type=Path,
        help="Metrics JSON path (defaults to docs/assets/validation_metrics.json in the repository)",
    )
    args = parser.parse_args()

    translator = Translator(checkpoint=args.checkpoint, device=args.device)
    _, validation_loader, _, _ = get_ds(translator.config)
    predictions, references, examples = [], [], []
    limit = args.max_samples if args.max_samples > 0 else len(validation_loader)
    loss_fn = nn.CrossEntropyLoss(
        ignore_index=translator.tokenizer_tgt.token_to_id("[PAD]"),
        label_smoothing=0.1,
        reduction="sum",
    ).to(translator.device)
    total_loss, total_tokens = 0.0, 0

    for batch in tqdm(validation_loader, total=min(limit, len(validation_loader)), desc="Decoding validation"):
        source = batch["src_text"][0]
        reference = batch["tgt_text"][0]
        prediction = translator.translate(source)
        with torch.inference_mode():
            encoder_input = batch["encoder_input"].to(translator.device)
            decoder_input = batch["decoder_input"].to(translator.device)
            encoder_mask = batch["encoder_mask"].to(translator.device)
            decoder_mask = batch["decoder_mask"].to(translator.device)
            label = batch["label"].to(translator.device)
            logits = translator.model.project(
                translator.model.decode(
                    decoder_input,
                    translator.model.encode(encoder_input, encoder_mask),
                    encoder_mask,
                    decoder_mask,
                )
            )
            total_loss += loss_fn(logits.view(-1, translator.tokenizer_tgt.get_vocab_size()), label.view(-1)).item()
            total_tokens += (label != translator.tokenizer_tgt.token_to_id("[PAD]")).sum().item()
        predictions.append(prediction)
        references.append(reference)
        if len(examples) < 3:
            examples.append({"source": source, "reference": reference, "prediction": prediction})
        if len(predictions) >= limit:
            break

    metrics = {
        "checkpoint": str(Path(args.checkpoint)),
        "decoded_validation_examples": len(predictions),
        "decoding": "greedy argmax",
        "validation_token_cross_entropy": round(total_loss / total_tokens, 4),
        "bleu": round(sacrebleu.corpus_bleu(predictions, [references]).score, 3),
        "chrf": round(sacrebleu.corpus_chrf(predictions, [references]).score, 3),
        "examples": examples,
    }
    output = args.output or ROOT / "docs" / "assets" / "validation_metrics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    chart_output = output.with_suffix(".png")
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4), layout="constrained")
    loss_bar = axes[0].bar(["Cross entropy"], [metrics["validation_token_cross_entropy"]], color="#173f5f")
    axes[0].set(title="Teacher Forced Validation Loss", ylabel="Loss", ylim=(0, max(3, metrics["validation_token_cross_entropy"] * 1.25)))
    axes[0].bar_label(loss_bar, fmt="%.4f", padding=3)
    score_bars = axes[1].bar(["BLEU", "chrF"], [metrics["bleu"], metrics["chrf"]], color=["#20639b", "#3caea3"])
    axes[1].set(title="Autoregressive Translation Scores", ylabel="Score", ylim=(0, 100))
    axes[1].bar_label(score_bars, fmt="%.3f", padding=3)
    fig.text(0.5, -0.03, f"Greedy decoding; {len(predictions)} validation examples", ha="center", fontsize=9)
    fig.savefig(chart_output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Wrote {output}")
    print(f"Wrote {chart_output}")


if __name__ == "__main__":
    main()
