# Transformer on Machine Translation

The repository is a PyTorch implementation of transformer. We also implement attentional LSTM and use machine translation to compare the two models. The MT data and preprocessing code is based on the [Link](https://towardsdatascience.com/how-to-use-torchtext-for-neural-machine-translation-plus-hack-to-make-it-5x-faster-77f3884d95). The transformer code is based on the [Link](https://pytorch.org/tutorials/beginner/transformer_tutorial.html)

# Set up
1. Please download the French-English corpus in the [Link](http://www.statmt.org/europarl/)
2. Unzip the corpus and rename europarl-v7.fr-en.en, europarl-v7.fr-en.fr to en.txt, fr.txt, respectively.
3. Download the SpaCy models we need for preprocessing.
```
python -m spacy download en_core_web_sm
python -m spacy download fr_core_news_sm
```

# Run the code
```bash run.sh```

# Use different models
Specify the model_type variable to be Transformer or LSTM.

# Configs
Python: 3.8.10
