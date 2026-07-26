# Attention is all you need: Paper Summary

The paper introduces the Transformer, a neural network architecture that relies entirely on attention mechanisms without using recurrent or convolutional networks.



Major contributions include:

- Self-attention replaces recurrence
- Multi-head attention captures multiple relationships simultaneously
- Positional encoding provides sequence order information
- Parallel computation significantly speeds up training
- State-of-the-art translation accuracy on WMT datasets

The Transformer is the foundation of modern large language models including BERT, GPT, T5, LLaMA, Gemini, Claude, and many others.

## The Problem
Neural Networks are ideal for sequential tasks as statistical ML models cannot do this
For example, the words in a sentence cannot be processed independently

Previous models(such as GRU, RNN and LSTM) used to do this by updating an internal memory as it processes each word and passes it to the next word.
The problems with approach are:
- Slow, Processes data sequentially, token by token ($t_1 \rightarrow t_2 \rightarrow t_3$). This creates a computational bottleneck because step $t_n$ cannot run until step $t_{n-1}$ finishes, making it impossible to fully parallelize training on GPUs.
- By the time the network reached the end of the sentence, much of the early information was lost

## How the Transformer solves these problems
It adds a special layer called attention that lets all tokens in a sequence talk to each other directly
Each token looks at every other token and decides which ones are important and thus it captures context efficiently

Previously, The best performing models also connect the encoder and decoder through an attention mechanism. The new proposes architecture, the Transformer, was based solely on attention mechanisms, and got rid of recurrence and convolutions entirely.

Initially, the parameters are random and during training, the attention layer learns meaningful patterns(Ex: verbs-subject and pronoun-noun relations)

The variants of attention such as masked, multi-head and cross-attention modify how attention is calculated.

The original paper uses this architecture as an encoder-decoder setup for translation, but the strongest point of a transformer is that it's design is very general and can thus be used in various applications-translation, summarization and text generation(LLM models). Anything that can be represented as a sequence of elements that interact with each other, the transformer architecture can be used and thus it can also be extended beyond language to images, audio and code.
Decoder only design is used in text generation (Ex: GPT)