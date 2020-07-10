import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderRNN(nn.Module):
    def __init__(self, src_voc_size, emsize, nhid):
        super(EncoderRNN, self).__init__()
        self.emsize = emsize
        self.nhid = nhid
        self.src_voc_size = src_voc_size
        self.embedding = nn.Embedding(src_voc_size, emsize)
        self.rnn = nn.LSTM(emsize, nhid, num_layers=2)

    def forward(self, src, lengths):
        # x: [T, B]
        x = self.embedding(src)
        x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, enforce_sorted=False)
        x, hidden = self.rnn(x)
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x)

        return x, hidden


class DecoderRNN(nn.Module):
    def __init__(self, tgt_voc_size, emsize, nhid):
        super(DecoderRNN, self).__init__()
        self.nhid = nhid
        self.tgt_voc_size = tgt_voc_size
        self.emsize = emsize
        self.embedding = nn.Embedding(tgt_voc_size, emsize)


        self.rnn = nn.LSTM(emsize, nhid, num_layers=2)
        self.out = nn.Linear(nhid, tgt_voc_size)


    def forward(self, x, hidden):
        x = x.unsqueeze(dim=0) # [1, B]
        x = self.embedding(x)
        output, hidden = self.rnn(x, hidden)
        output = output.squeeze(0)
        output = self.out(output)
        return output, hidden # output: [B,O]

class AttentionDecoderRNN(nn.Module):
    def __init__(self, tgt_voc_size, emsize, nhid):
        super(AttentionDecoderRNN, self).__init__()
        self.nhid = nhid
        self.tgt_voc_size = tgt_voc_size
        self.emsize = emsize
        self.max_length = 80

        self.embedding = nn.Embedding(tgt_voc_size, emsize)

        self.attn = nn.Linear(nhid * 2, self.max_length)
        self.attn_combine = nn.Linear(nhid * 2, nhid)


        self.rnn = nn.LSTM(emsize, nhid, num_layers=2)
        self.out = nn.Linear(nhid, tgt_voc_size)


    def forward(self, x, hidden, encoder_outputs):
        x = x.unsqueeze(dim=0) # [1, B]
        emb = self.embedding(x) # [1. B, D]
        attn_weights = F.softmax(
            self.attn(torch.cat((emb[0], hidden[0][-1]), 1)), dim=1)

        T = min(encoder_outputs.shape[0], self.max_length)
        encoder_outputs = encoder_outputs[:T, :, :]
        attn_weights = attn_weights[:, :T] # [B, T]

        # attn_applied = [B, 1, T] (bmm) [B, T, D] = [B, 1, D]
        attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs.permute(1, 0, 2))
        x = torch.cat((emb[0], attn_applied.squeeze(1)), 1)
        x = self.attn_combine(x)
        output, hidden = self.rnn(x.unsqueeze(0), hidden)
        output = output.squeeze(0)
        output = self.out(output)
        return output, hidden # output: [B,O]


class Seq2Seq(nn.Module):
    def __init__(self, src_voc_size, tgt_voc_size, emsize, nhid, device):
        super(Seq2Seq, self).__init__()
        self.encoder = EncoderRNN(src_voc_size, emsize, nhid)
        #self.decoder = DecoderRNN(tgt_voc_size, emsize, nhid)
        self.decoder = AttentionDecoderRNN(tgt_voc_size, emsize, nhid)
        self.device = device
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=1)

    def train_step(self, src, src_lengths, tgt_for_inp, tgt_for_loss):
        enc_out, hidden = self.encoder(src, src_lengths)


        loss = 0
        for t in range(len(tgt_for_inp)):
            inp = tgt_for_inp[t]
            dec_output, hidden = self.decoder(inp, hidden, enc_out)
            loss += self.criterion(dec_output, tgt_for_loss[t])
        return loss

    def generate(self, src, src_lengths):
        batch_size = src.shape[1]
        device = src.device
        enc_out, hidden = self.encoder(src, src_lengths)
        max_len = 80

        eos = 3
        sos = 2
        dec_inp = sos * torch.ones((batch_size)).long().to(device)

        output = eos * torch.ones((max_len, batch_size)).long().to(device)
        stop = torch.zeros((batch_size)).bool().to(device)

        x = dec_inp
        for t in range(max_len):
            dec_out, hidden = self.decoder(x, hidden, enc_out)
            _, topi = dec_out.topk(1)
            topi_t = topi.squeeze().long()
            output[t] = topi_t

            # Stop if all sentences reache eos
            stop_t = (topi_t == eos)
            stop = stop | stop_t
            if torch.all(stop):
                break

            if t == max_len - 1:
                break
            x = topi_t.detach()
        return output




