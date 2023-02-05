import spacy
from torchtext.legacy.data import Field
import pandas as pd
from sklearn.model_selection import train_test_split
import jieba


# The script is based on https://towardsdatascience.com/how-to-use-torchtext-for-neural-machine-translation-plus-hack-to-make-it-5x-faster-77f3884d95
# Config
src_lang = "en"
tgt_lang = "zh"
datasets_prefix = 'datasets/en_zh'

spacy_model_name_dict = {
    'en': 'en_core_web_sm',
    'fr': 'fr_core_news_sm',
    'zh': 'zh_core_web_sm'
}

src_sent_list = open(f'{datasets_prefix}/{src_lang}.txt', encoding='utf-8').read().split('\n')
tgt_sent_list = open(f'{datasets_prefix}/{tgt_lang}.txt', encoding='utf-8').read().split('\n')


def perform_word_segmentation_for_zh(sent_list):
    new_sent_list = []
    for sent in sent_list:
        new_sent_list.append(' '.join(jieba.cut(sent)))
    return new_sent_list


if src_lang == 'zh':
    src_sent_list = perform_word_segmentation_for_zh(src_sent_list)
if tgt_lang == 'zh':
    tgt_sent_list = perform_word_segmentation_for_zh(tgt_sent_list)


spacy_src = spacy.load(spacy_model_name_dict[src_lang])
spacy_tgt = spacy.load(spacy_model_name_dict[tgt_lang])


def tokenize_src(sentence):
    return [tok.text for tok in spacy_src.tokenizer(sentence)]


def tokenize_tgt(sentence):
    return [tok.text for tok in spacy_tgt.tokenizer(sentence)]


SRC_TEXT = Field(tokenize=tokenize_src)
TGT_TEXT = Field(tokenize=tokenize_tgt, init_token ='<sos>', eos_token ='<eos>')

raw_data = {'SRC' : [line for line in src_sent_list], 'TGT': [line for line in tgt_sent_list]}
df = pd.DataFrame(raw_data, columns=['SRC', 'TGT'])
# remove very long sentences and sentences where translations are
# not of roughly equal length
df[f'src_len'] = df['SRC'].str.count(' ')
df[f'tgt_len'] = df['TGT'].str.count(' ')
df = df.query(f'tgt_len < 80 & src_len < 80')
df = df.query(f'tgt_len < src_len * 1.5 & tgt_len * 1.5 > src_len')

# create train and validation set
train, val = train_test_split(df, test_size=0.1)
train.to_csv("datasets/train.csv", index=False)
val.to_csv("datasets/val.csv", index=False)
