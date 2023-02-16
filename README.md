# Transformer on Machine Translation

The repository is a PyTorch implementation of transformer.
We also implement attentional LSTM and use machine translation to compare the two models.
The MT data and preprocessing code is based on the [Link](https://towardsdatascience.com/how-to-use-torchtext-for-neural-machine-translation-plus-hack-to-make-it-5x-faster-77f3884d95).
The transformer code is based on the [Link](https://pytorch.org/tutorials/beginner/transformer_tutorial.html)

# Datasets
The repository contains two translation pairs: en-fr, en-zh.
We provide some example files for illustrating the proper paths of each file.
Please follow the steps in setup to download and put the files in their corresponding paths.

- [French-English corpus](http://www.statmt.org/europarl/)
- [Chinese-English corpus](https://statmt.org/wmt20/translation-task.html)

# Setup (French-English)
1. Please download the French-English corpus.
2. Unzip the corpus and rename europarl-v7.fr-en.en, europarl-v7.fr-en.fr to en.txt, fr.txt, respectively.
3. Put en.txt, fr.txt in datasets/en_fr
4. Download the SpaCy models we need for preprocessing. The full list of languages of SpaCy is [here](https://spacy.io/usage/models).
```
python -m spacy download en_core_web_sm
python -m spacy download fr_core_news_sm
```

# Run the code
Use `bash run.sh` to run the code, it would generate the following items in directory `results`.
- model.cpu.pt : The model which can be loaded with CPU environment.
- model.pt: The model which can be loaded with GPU or CPU environment, based on if the model is trained with GPU.
- src.vocab: Source language vocabulary.
- tgt.vocab: Target language vocabulary.

# Use different models
Specify the model_type variable to be Transformer or LSTM.

# English-Chinese Translation
We use Back-translated news released by WMT20 for our English-Chinese Translation dataset.
The dataset could be found [here](https://statmt.org/wmt20/translation-task.html)

# Configs
- Python: 3.8.10
- torch: 1.9.0
- torchtext: 0.10.0
- CUDA: 11.3
