import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer
from datasets import load_dataset

from config import get_config, latest_weights_file_path
from model import build_Transformer
from dataset import BilingualDataset, causal_mask


def translate(sentence: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = get_config()

    tokenizer_src = Tokenizer.from_file(str(Path(config["tokenizer_file"].format(config["src_lang"]))))
    tokenizer_tgt = Tokenizer.from_file(str(Path(config["tokenizer_file"].format(config["tgt_lang"]))))

    model = build_Transformer(
        tokenizer_src.get_vocab_size(),
        tokenizer_tgt.get_vocab_size(),
        config["seq_length"],
        config["seq_length"],
        config["d_model"],
        config["num_layers"],
        config["num_heads"],
        config["dropout_rate"],
        config["d_ff"],
    ).to(device)

    model_filename = latest_weights_file_path(config)
    if model_filename is None:
        raise FileNotFoundError(
            f"No trained weights found in '{config['model_folder']}'. Train the model first."
        )
    state = torch.load(model_filename, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    # If the input is an integer (or a numeric string), treat it as an index into the dataset
    label = ""
    if isinstance(sentence, int) or (isinstance(sentence, str) and sentence.isdigit()):
        idx = int(sentence)
        ds_raw = load_dataset("opus_books", f'{config["src_lang"]}-{config["tgt_lang"]}', split="train")
        ds = BilingualDataset(
            ds_raw, tokenizer_src, tokenizer_tgt, config["src_lang"], config["tgt_lang"], config["seq_length"]
        )
        sentence = ds[idx]["src_text"]
        label = ds[idx]["tgt_text"]

    seq_len = config["seq_length"]

    with torch.no_grad():
        source_ids = tokenizer_src.encode(sentence).ids
        if len(source_ids) > seq_len - 2:
            raise ValueError("Sentence is too long for the configured sequence length.")

        source = torch.cat(
            [
                torch.tensor([tokenizer_src.token_to_id("[SOS]")], dtype=torch.int64),
                torch.tensor(source_ids, dtype=torch.int64),
                torch.tensor([tokenizer_src.token_to_id("[EOS]")], dtype=torch.int64),
                torch.tensor(
                    [tokenizer_src.token_to_id("[PAD]")] * (seq_len - len(source_ids) - 2), dtype=torch.int64
                ),
            ],
            dim=0,
        ).unsqueeze(0).to(device)  # (1, seq_len)

        source_mask = (source != tokenizer_src.token_to_id("[PAD]")).unsqueeze(1).unsqueeze(1).int().to(device)

        encoder_output = model.encode(source, source_mask)
        decoder_input = torch.empty(1, 1).fill_(tokenizer_tgt.token_to_id("[SOS]")).type_as(source).to(device)

        if label:
            print(f"{'SOURCE: ':>12}{sentence}")
            print(f"{'TARGET: ':>12}{label}")
        else:
            print(f"{'SOURCE: ':>12}{sentence}")
        print(f"{'PREDICTED: ':>12}", end="")

        while decoder_input.size(1) < seq_len:
            decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
            out = model.decode(decoder_input, encoder_output, source_mask, decoder_mask)

            prob = model.project(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            decoder_input = torch.cat(
                [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
            )

            print(f"{tokenizer_tgt.decode([next_word.item()])}", end=" ")

            if next_word.item() == tokenizer_tgt.token_to_id("[EOS]"):
                break

    print()
    return tokenizer_tgt.decode(decoder_input[0].tolist())


if __name__ == "__main__":
    text = sys.argv[1] if len(sys.argv) > 1 else "I am not a very good student."
    translate(text)
