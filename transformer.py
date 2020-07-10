import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerModel(nn.Module):

    def __init__(self, ntoken, tgt_ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        from torch.nn import TransformerDecoder, TransformerDecoderLayer
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)

        self.encoder_embed = nn.Embedding(ntoken, ninp)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        self.decoder_embed = nn.Embedding(tgt_ntoken, ninp)
        decoder_layers = TransformerDecoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_decoder = TransformerDecoder(decoder_layers, nlayers)
        self.decoder_out = nn.Linear(ninp, tgt_ntoken)

        self.ninp = ninp

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder_embed.weight.data.uniform_(-initrange, initrange)
        self.decoder_embed.weight.data.uniform_(-initrange, initrange)
        self.decoder_out.bias.data.zero_()
        self.decoder_out.weight.data.uniform_(-initrange, initrange)

    def train_step(self, src, src_key_pad_mask, tgt, tgt_key_pad_mask):
        # src: [T, B]
        # src_key_pad_mask: [B, T]
        # tgt: [T, B]
        # tgt_key_pad_mask: [B, T]
        src = self.encoder_embed(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        enc_output = self.transformer_encoder(src, src_key_padding_mask=src_key_pad_mask)

        device = src.device
        tgt_mask = self._generate_square_subsequent_mask(len(tgt)).to(device)

        tgt = self.decoder_embed(tgt) * math.sqrt(self.ninp)
        tgt = self.pos_encoder(tgt)
        dec_out = self.transformer_decoder(tgt, enc_output, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_pad_mask, memory_key_padding_mask=src_key_pad_mask)
        logits = self.decoder_out(dec_out)
        return logits

    def generate(self, src, src_key_pad_mask):
        # src: [T, B]
        # src_key_pad_mask: [B, T]
        batch_size = src.shape[1]
        src = self.encoder_embed(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        enc_output = self.transformer_encoder(src, src_key_padding_mask=src_key_pad_mask)

        device = src.device
        max_len = 80
        # <sos> is 2 and <eos> is 3
        eos = 3
        sos = 2
        dec_inp = sos * torch.ones((max_len, batch_size)).long().to(device)
        dec_inp_mask = self._generate_square_subsequent_mask(len(dec_inp)).to(device)

        output = eos * torch.ones((max_len, batch_size)).long().to(device)
        stop = torch.zeros((batch_size)).bool().to(device)

        for t in range(max_len):
            x = dec_inp
            x = self.decoder_embed(x) * math.sqrt(self.ninp)
            x = self.pos_encoder(x)
            dec_out = self.transformer_decoder(x, enc_output, tgt_mask=dec_inp_mask, memory_key_padding_mask=src_key_pad_mask)
            logits = self.decoder_out(dec_out)
            #logits = dec_out
            _, topi = logits.topk(1)
            topi_t = topi[t].squeeze().long()
            output[t] = topi_t

            # Stop if all sentences reache eos
            stop_t = (topi_t == eos)
            stop = stop | stop_t
            if torch.all(stop):
                break

            if t == max_len - 1:
                break
            dec_inp[t+1] = topi_t.detach()
        return output

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

