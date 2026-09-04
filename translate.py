import argparse

from predictor import Translator


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate English text to French")
    parser.add_argument("text", nargs="?", default="I am not a very good student.")
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", choices=["cpu", "cuda"])
    args = parser.parse_args()
    translator = Translator(checkpoint=args.checkpoint, device=args.device)
    print(translator.translate(args.text))
