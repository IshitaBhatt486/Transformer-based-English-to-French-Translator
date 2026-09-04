import torch
import torch.nn as nn
import warnings
import argparse
import random
import numpy as np

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from torch.utils.data import DataLoader, Subset, random_split
from torch.utils.tensorboard import SummaryWriter

from datasets import load_dataset
from dataset import BilingualDataset, causal_mask
from model import build_Transformer
from config import get_config, get_weights_file_path
from tqdm import tqdm

from pathlib import Path

# run encoder only once- precompute the encoder output once and use it for all decoder steps
def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_length, device):
    sos_index = tokenizer_tgt.token_to_id("[SOS]")
    eos_index = tokenizer_tgt.token_to_id("[EOS]")

    encoder_output = model.encode(source, source_mask)
    decoder_input = torch.empty(1, 1).fill_(sos_index).type_as(source).to(device)

    while True:
        if decoder_input.size(1) >= max_length:
            break

        # target mask(decoder mask)
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        output = model.decode(decoder_input, encoder_output, source_mask, decoder_mask)

        prob = model.project(output[:, -1]) #(last_token, vocab_size)
        _, next_word = torch.max(prob, dim=1) # greedy search
        decoder_input = torch.cat([decoder_input, torch.empty(1,1).type_as(source).fill_(next_word.item()).to(device)], dim=1)

        if next_word == eos_index:
            break

    return decoder_input.squeeze(0)

def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_length, device, print_msg, global_state, writer, num_examples=2):
    model.eval()
    count = 0

    source_texts = []
    expected_texts = []
    predicted_texts = []

    console_width = 80

    with torch.no_grad(): 
        for batch in validation_ds:
            count += 1

            encoder_input = batch["encoder_input"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            decoder_mask = batch["decoder_mask"].to(device)

            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            model_output = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_length, device)

            source_text = batch["src_text"][0]
            expected_text = batch["tgt_text"][0]
            model_output_text = tokenizer_tgt.decode(model_output.detach().cpu().numpy())

            source_texts.append(source_text)
            expected_texts.append(expected_text)
            predicted_texts.append(model_output_text)

            print_msg('-' *console_width)# dont prit directly to console when tqdm is running, use print_msg to redirect to tensorboard  
            print_msg(f"Source: {source_text}")
            print_msg(f"Expected: {expected_text}")
            print_msg(f"Predicted: {model_output_text}")

            if count == num_examples:
                break

def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config["tokenizer_file"].format(lang))

    tokenizer_path.parent.mkdir(parents=True, exist_ok=True)

    if not tokenizer_path.exists():
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]")) #unknown
        tokenizer.pre_tokenizer = Whitespace()

        trainer = WordLevelTrainer(
            vocab_size=config["vocab_size"],
            #special tokens: unknown, padding, start of sentence, end of sentence
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
        )

        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    ds_raw = load_dataset(
        "Helsinki-NLP/opus_books",
        f'{config["src_lang"]}-{config["tgt_lang"]}',
        split="train",
    )

    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config["src_lang"])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config["tgt_lang"])

    # Remove examples that cannot fit instead of failing unpredictably mid-epoch.
    valid_indices = []
    for index, item in enumerate(ds_raw):
        src_len = len(tokenizer_src.encode(item['translation'][config["src_lang"]]).ids)
        tgt_len = len(tokenizer_tgt.encode(item['translation'][config["tgt_lang"]]).ids)
        if src_len <= config["seq_length"] - 2 and tgt_len <= config["seq_length"] - 1:
            valid_indices.append(index)
    ds_filtered = Subset(ds_raw, valid_indices)
    print(f"Using {len(ds_filtered):,}/{len(ds_raw):,} examples that fit seq_length={config['seq_length']}")

    ds_train_size = int(0.9 * len(ds_filtered))
    ds_validation_size = len(ds_filtered) - ds_train_size
    generator = torch.Generator().manual_seed(config["seed"])
    ds_train_raw, ds_validation_raw = random_split(
        ds_filtered, [ds_train_size, ds_validation_size], generator=generator
    )

    train_ds = BilingualDataset(ds_train_raw, tokenizer_src, tokenizer_tgt, config["src_lang"], config["tgt_lang"], config["seq_length"])
    validation_ds = BilingualDataset(ds_validation_raw, tokenizer_src, tokenizer_tgt, config["src_lang"], config["tgt_lang"], config["seq_length"])

    max_len_src = 0
    max_len_tgt = 0

    for item in ds_filtered:
        src_ids = tokenizer_src.encode(item['translation'][config["src_lang"]]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config["tgt_lang"]]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f"Max length of source sentences: {max_len_src}")
    print(f"Max length of target sentences: {max_len_tgt}")

    loader_args = {"num_workers": config["num_workers"], "pin_memory": torch.cuda.is_available()}
    train_dataloader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, **loader_args)
    validation_dataloader = DataLoader(validation_ds, batch_size=1, shuffle=False, **loader_args)

    return train_dataloader, validation_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_length, vocab_tgt_length):
    model = build_Transformer(
        vocab_src_length,
        vocab_tgt_length,
        config["seq_length"],
        config["seq_length"],
        config["d_model"],
        config["num_layers"],
        config["num_heads"],
        config["dropout_rate"],
        config["d_ff"]
    )
    return model

def train_model(config):
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)

    train_dataloader, validation_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    # Tensorboard
    writer = SummaryWriter(log_dir=config["experiment_name"])

    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], eps=1e-9)

    initial_epoch = 0
    global_step = 0

    if config["preload"] and config["preload"] != "none":
        model_filename = get_weights_file_path(config, config["preload"])
        print(f"Loading model weights from {model_filename}")
        state = torch.load(model_filename, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        initial_epoch = state["epoch"] + 1
        optimizer.load_state_dict(state["optimizer_state_dict"])
        global_step = state["global_step"]

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_tgt.token_to_id("[PAD]"), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config["num_epochs"]):
        batch_iterator = tqdm(train_dataloader, desc=f'Processing epoch {epoch:02d}')

        for batch in batch_iterator:
            model.train()

            encoder_input = batch["encoder_input"].to(device) #(batch_size, seq_length)
            decoder_input = batch["decoder_input"].to(device) #(batch_size, seq_length)

            encoder_mask = batch["encoder_mask"].to(device) #(0, 1, 1, seq_length)
            decoder_mask = batch["decoder_mask"].to(device) #(0, 1, seq_length, seq_length)

            encoder_output = model.encode(encoder_input, encoder_mask) #(batch_size, seq_length, d_model)
            decoder_output = model.decode(decoder_input, encoder_output, encoder_mask, decoder_mask) #(batch_size, seq_length, d_model)

            proj_output = model.project(decoder_output) #(batch_size, seq_length, vocab_tgt_length)

            label =batch["label"].to(device) #(batch_size, seq_length)
            # (batch_size, seq_length, vocab_tgt_length) -> (batch_size * seq_length, vocab_tgt_length)
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}"})

            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            # backpropagation
            loss.backward()

            # weight update
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1

        run_validation(model, validation_dataloader, tokenizer_src, tokenizer_tgt, config["seq_length"], device, lambda msg:batch_iterator.write(msg), global_step, writer)

        model_filename = get_weights_file_path(config, f'{epoch:02d}')
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "global_step": global_step,
            "config": config,
        }, model_filename)
    writer.close()

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser(description="Train the English-to-French Transformer")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--preload", help="Checkpoint epoch (for example 03)")
    parser.add_argument("--num-workers", type=int)
    args = parser.parse_args()
    config = get_config()
    if args.epochs is not None:
        config["num_epochs"] = args.epochs
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.preload is not None:
        config["preload"] = args.preload
    if args.num_workers is not None:
        config["num_workers"] = args.num_workers
    train_model(config)
