# Transformer using PyTorch

A complete PyTorch implementation of the Transformer architecture proposed in the paper **"Attention Is All You Need" (Vaswani et al., 2017)**. This project focuses on understanding the mathematical and engineering component of the original architecture by implementing it from scratch.

A translation model

Libraries used:
- PyTorch

## Quick start

```bash
python -m pip install -r requirements.txt
python train.py --epochs 20 --batch-size 8
python translate.py "Hello world"
```

Training writes versioned checkpoints to `weights/` and tokenizer artifacts to
the project root. Keep the checkpoint and both tokenizer JSON files together.
Use `--preload 03` to resume after checkpoint `tmodel_03.pt`.

## Google Colab

Open [`colab.ipynb`](colab.ipynb), select a T4 GPU, and run all cells. The
notebook trains, saves artifacts to Google Drive, runs inference, and smoke-tests
the API. The notebook clones the default GitHub branch, so push your current
changes before running it.

## API and container deployment

Start the API locally:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
curl -X POST http://localhost:8000/translate \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello world"}'
```

Or build the CPU container after placing trained artifacts in the repository:

```bash
docker build -t transformer-translator .
docker run --rm -p 8000:8000 transformer-translator
```

Set `MODEL_CHECKPOINT` to select a specific checkpoint and `DEVICE=cpu` or
`DEVICE=cuda` to select the inference device. The service exposes `GET /health`,
`POST /translate`, and interactive documentation at `/docs`.

# Project Motivation

The Transformer architecture fundamentally changed deep learning by replacing recurrence with self-attention, enabling highly parallel training and significantly improving performance on sequence-to-sequence tasks.

The objectives of this project are:

- Understand the mathematical foundations of self-attention
- Implementing the complete model from first principles
- Learn how encoder-decoder architectures work internally
- Train the model on a machine translation task
- Reproduce results similar to those reported in the original paper
- Visualize what the model learns during training
- Build a production-quality implementation suitable for research and experimentation

## References

- Vaswani et al., *Attention Is All You Need*, NeurIPS 2017
- PyTorch Documentation
- Harvard NLP Annotated Transformer
- The Illustrated Transformer (Jay Alammar)
