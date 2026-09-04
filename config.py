import os
from pathlib import Path

def get_config():
    return {
        "src_lang": "en",
        "tgt_lang": "fr",
        "vocab_size": 10000,
        "seq_length": 128,

        "tokenizer_file": "tokenizer_{}.json",
        "model_folder": os.getenv("MODEL_FOLDER", "weights"),
        "model_filename": "tmodel_",

        "batch_size": int(os.getenv("BATCH_SIZE", "8")),
        "num_epochs": int(os.getenv("NUM_EPOCHS", "20")),
        "learning_rate": float(os.getenv("LEARNING_RATE", "0.0001")),
        "seed": int(os.getenv("SEED", "42")),
        "num_workers": int(os.getenv("NUM_WORKERS", "2")),

        "num_layers": 4,
        "num_heads": 8,

        "d_model": 512,
        "d_ff": 2048,
        "dropout_rate": 0.1,

        "preload": "none",
        "experiment_name": "runs/tmodel"

    }

def get_weights_file_path(config, epoch: str):
    model_folder = Path(config["model_folder"])
    model_basename = config["model_filename"]
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

def latest_weights_file_path(config):
    model_folder = Path(config["model_folder"])
    model_filename = f"{config['model_filename']}*"
    weights_files = list(model_folder.glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])
