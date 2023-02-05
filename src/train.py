import spacy
from nltk.translate.bleu_score import sentence_bleu
from torchtext.legacy.data import Field, TabularDataset
from torchtext.legacy import data
import yaml
# The script is based on https://towardsdatascience.com/how-to-use-torchtext-for-neural-machine-translation-plus-hack-to-make-it-5x-faster-77f3884d95


with open('src/config/config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
src_lang = config['src_lang']
tgt_lang = config['tgt_lang']
datasets_prefix = config['dataset_prefix']

with open('src/config/spacy_model_config.yaml') as f:
    spacy_model_config = yaml.load(f, Loader=yaml.FullLoader)

spacy_src = spacy.load(spacy_model_config[src_lang])
spacy_tgt = spacy.load(spacy_model_config[tgt_lang])

def tokenize_src(sentence):
    return [tok.text for tok in spacy_src.tokenizer(sentence)]


def tokenize_tgt(sentence):
    return [tok.text for tok in spacy_tgt.tokenizer(sentence)]


SRC_TEXT = Field(tokenize=tokenize_src)
TGT_TEXT = Field(tokenize=tokenize_tgt, init_token ='<sos>', eos_token ='<eos>')

# associate the text in the $src_lang_name column with the SRC_TEXT field, # and $tgt_lang_name with TGT_TEXT
data_fields = [('SRC', SRC_TEXT), ('TGT', TGT_TEXT)]
train,val = TabularDataset.splits(path=datasets_prefix, train='train.csv', validation='val.csv', format='csv', fields=data_fields)

TGT_TEXT.build_vocab(train, val)
SRC_TEXT.build_vocab(train, val)

max_src_in_batch, max_tgt_in_batch = 100, 100


def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch,  len(new.SRC))
    max_tgt_in_batch = max(max_tgt_in_batch,  len(new.TGT) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)

class MyIterator(data.Iterator):
    def create_batches(self):
        if self.train:
            def pool(d, random_shuffler):
                for p in data.batch(d, self.batch_size * 100):
                    p_batch = data.batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b
            self.batches = pool(self.data(), self.random_shuffler)

        else:
            self.batches = []
            for b in data.batch(self.data(), self.batch_size,
                                          self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))

import torch
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
tokens_per_batch = 1000


train_iter = MyIterator(train, batch_size=tokens_per_batch, device=device,
                        repeat=False, sort_key= lambda x:
                        (len(x.SRC), len(x.TGT)),
                        batch_size_fn=batch_size_fn, train=True,
                        shuffle=True)



src_ntokens = len(SRC_TEXT.vocab.stoi) # the size of vocabulary
tgt_ntokens = len(TGT_TEXT.vocab.stoi) # the size of vocabulary
emsize = 200 # embedding dimension
nhid = 200 # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2 # the number of heads in the multiheadattention models
dropout = 0.2 # the dropout value
from model.transformer import TransformerModel
from model.lstm_seq2seq import Seq2Seq

model_type = 'Transformer' # LSTM or Transformer
if model_type == 'LSTM':
    model = Seq2Seq(src_ntokens, tgt_ntokens, emsize, nhid, device).to(device)
else:
    model = TransformerModel(src_ntokens, tgt_ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)

import numpy as np
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print('# parameters: {:e}'.format(params))

criterion = torch.nn.CrossEntropyLoss(ignore_index=1)
optimizer = torch.optim.Adam(model.parameters())
model.train()

log_interval = 200
step = 0
while step < 100000:
    for batch in iter(train_iter):
        optimizer.zero_grad()
        # 1. is the padding index
        src = batch.SRC.to(device)
        tgt = batch.TGT.to(device)
        tgt_for_inp = tgt[:-1]
        tgt_for_loss = tgt[1:]

        if model_type == 'Transformer':

            src_mask = (batch.SRC == 1.).permute(1, 0).to(device)
            tgt_mask = (tgt_for_inp == 1.).permute(1, 0).to(device)


            logits = model.train_step(src, src_mask, tgt_for_inp, tgt_mask)
            loss = criterion(logits.view(-1, tgt_ntokens), tgt_for_loss.view(-1))
        else:
            # Weird bug for RNN. Has to make src_mask to cpu instead of cuda
            src_mask = (batch.SRC == 1.).permute(1, 0).to(torch.device('cpu'))
            src_lengths = src.shape[0] - src_mask.int().sum(axis=1)
            loss = model.train_step(src, src_lengths, tgt_for_inp, tgt_for_loss)


        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        step += 1
        if step > 100000:
            break

        if step % log_interval == 0:
            print(f'Step {step}: loss: {loss.item()}')

            model.eval()
            if model_type == 'Transformer':
                src_mask = (batch.SRC == 1.).permute(1, 0).to(device)
                #output = model.generate(src, src_mask)
                output = []
                for i in range(src.shape[1]):
                    output.append(model.generate(src[:, i:i+1], src_mask[i:i+1]))
                output = torch.stack(output).squeeze().permute(1, 0)
            else:
                # Weird bug for RNN. Has to make src_mask to cpu instead of cuda
                src_mask = (batch.SRC == 1.).permute(1, 0).to(torch.device('cpu'))
                src_lengths = src.shape[0] - src_mask.int().sum(axis=1)
                output = model.generate(src, src_lengths)

            output = output.permute(1, 0) # [B, T]
            output = output.cpu().detach().numpy()
            B, T = output.shape
            ref = tgt.permute(1, 0).detach().cpu().numpy()
            scores = []
            for i in range(B):
                sent = []
                for t in range(T):
                    if TGT_TEXT.vocab.itos[output[i][t]] != '<eos>':
                        sent.append(TGT_TEXT.vocab.itos[output[i][t]])
                    else:
                        break
                sent_ref = []
                for t in range(len(ref[i])):
                    if TGT_TEXT.vocab.itos[ref[i][t]] != '<pad>':
                        sent_ref.append(TGT_TEXT.vocab.itos[ref[i][t]])
                    else:
                        break
                # Remove <sos> and <eos>
                sent_ref = sent_ref[1:-1]
                scores.append(sentence_bleu([sent_ref], sent))
            print(' '.join(sent))
            print('Average BLEU: {:.4f}'.format(sum(scores) / len(scores)))
            model.train()

