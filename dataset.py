import torch
import torch.nn as nn

class BilingualDataset(torch.utils.data.Dataset):
    def __init__(self, ds, src_tokenizer, tgt_tokenizer, src_lang, tgt_lang, seq_length)-> None:
        super().__init__()

        self.seq_length = seq_length
        self.ds =ds
        self.src_lang =src_lang
        self.tgt_lang =tgt_lang
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer

        self.sos_token= torch.tensor([src_tokenizer.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token= torch.tensor([src_tokenizer.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token= torch.tensor([src_tokenizer.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index: any) -> any:
        src_target_pair = self.ds[index]

        src_sentence = src_target_pair['translation'][self.src_lang]
        tgt_sentence = src_target_pair['translation'][self.tgt_lang]

        encode_input_tokens = self.src_tokenizer.encode(src_sentence).ids
        decode_input_tokens = self.tgt_tokenizer.encode(tgt_sentence).ids

        encoder_num_padding_tokens = self.seq_length - len(encode_input_tokens) -2 # 2 for SOS and EOS
        decoder_num_padding_tokens = self.seq_length - len(decode_input_tokens) -1 # only SOS in training and only EOS in inference

        if encoder_num_padding_tokens < 0 or decoder_num_padding_tokens < 0:
            raise ValueError("Sentence is too long.")

        encoder_input =torch.cat(
            [
                self.sos_token,
                torch.tensor(encode_input_tokens, dtype=torch.int64), 
                self.eos_token,
                self.pad_token.repeat(encoder_num_padding_tokens)
            ]
        )

        # add SOS to decoder input
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(decode_input_tokens, dtype=torch.int64),
                self.pad_token.repeat(decoder_num_padding_tokens)
            ]
        )

        # Add EOS to label (expected output)
        label=torch.cat(
            [
                torch.tensor(decode_input_tokens, dtype=torch.int64),
                self.eos_token,
                self.pad_token.repeat(decoder_num_padding_tokens)
            ]
        )

        assert encoder_input.shape[0] == self.seq_length
        assert decoder_input.shape[0] == self.seq_length
        assert label.shape[0] == self.seq_length

        return {
            
            "encoder_input": encoder_input, # seq length
            "decoder_input": decoder_input, # seq length

            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1,1,seq_length) mask for padding tokens
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), #  we only need to mask padding tokens here

            "label": label, # seq length

            "src_text": src_sentence, # original source sentence
            "tgt_text": tgt_sentence # original target sentence
        }

def causal_mask(size):
    mask = torch.triu(torch.ones((1,size , size), diagonal=1).type(torch.int64)) # upper triangular matrix with 1s above the diagonal
    return mask==0
