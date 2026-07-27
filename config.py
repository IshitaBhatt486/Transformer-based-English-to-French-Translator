from pathlib import Path

def get_config():
    return {
        "src_lang": "en",
        "tgt_lang": "fr",
        "vocab_size": 10000,
        "seq_length": 128,

        "tokenizer_file": "tokenizer_{}.json",
        "model_folder": "weights",
        "model_filename": "tmodel_",

        "batch_size": 8,
        "num_epochs": 20,
        "learning_rate": 0.001,

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
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('_') / model_folder / model_filename)