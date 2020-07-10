import spacy
import torchtext
from torchtext.data import Field, BucketIterator, TabularDataset
import pandas as pd
from sklearn.model_selection import train_test_split


# The script is based on https://towardsdatascience.com/how-to-use-torchtext-for-neural-machine-translation-plus-hack-to-make-it-5x-faster-77f3884d95

europarl_en = open('en.txt', encoding='utf-8').read().split('\n')
europarl_fr = open('fr.txt', encoding='utf-8').read().split('\n')

en = spacy.load('en_core_web_sm')
fr = spacy.load('fr_core_news_sm')

def tokenize_en(sentence):
    return [tok.text for tok in en.tokenizer(sentence)]

def tokenize_fr(sentence):
    return [tok.text for tok in fr.tokenizer(sentence)]

EN_TEXT = Field(tokenize=tokenize_en)
FR_TEXT = Field(tokenize=tokenize_fr, init_token = '<sos>', eos_token = '<eos>')

raw_data = {'English' : [line for line in europarl_en], 'French': [line for line in europarl_fr]}
df = pd.DataFrame(raw_data, columns=["English", "French"])
# remove very long sentences and sentences where translations are
# not of roughly equal length
df['eng_len'] = df['English'].str.count(' ')
df['fr_len'] = df['French'].str.count(' ')
df = df.query('fr_len < 80 & eng_len < 80')
df = df.query('fr_len < eng_len * 1.5 & fr_len * 1.5 > eng_len')

# create train and validation set
train, val = train_test_split(df, test_size=0.1)
train.to_csv("data/train.csv", index=False)
val.to_csv("data/val.csv", index=False)
