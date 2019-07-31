# aivivn-tone

Submission for AIviVN Vietnamese diacritics restoration contest https://www.aivivn.com/contests/3.

### Requirements

Python > 3.6, PyTorch 1.0.1, torchtext, unidecode, dill, visdom, tqdm, kenlm.

`visdom` is mainly used for visualizing training loss and accuracy.

`kenlm` can be found [here](https://github.com/kpu/kenlm). I had some troubles with the version on the master branch, so the stable release may be better.

### Overview

#### Character-level BiLSTM seq2seq model

[<p align="center"><img width="510" src="https://i.imgur.com/67CCUEU.png"></p>]()

The embedding layer and encoder are standard. The model consists of 3 decoders:
- a left-to-right decoder
- a right-to-left decoder
- a combined decoder constructed by concatenating output LSTM states of two previous decoders

The final loss is a sum of 3 component losses:
`L = L_ltr + L_rtl + L_combined`

Since only a certain set of characters requires diacritics restoration (`a, d, e, i, o, u, y`), we can apply teacher forcing at both training time and test time. 

In addition, since each character only has a fixed set of targets (e.g., for `i` it's `i, í, ì, ỉ, ĩ, ị`), masked softmax can also be applied. 

#### Beam search

[<p align="center"><img width="780" src="https://i.imgur.com/MultVmF.png"></p>]()

We run a standard beam search in 2 directions, left-to-right and right-to-left, and combine results. For any disagreements that may appear between the two searches, repeat the procedure until there are no disagreements left. We fall back on exhaustive search after a number of recursive calls in case of infinite recursion.

A 4-gram word-level language model is used to score candidates during beam search.

The beam search component is separated from the seq2seq model (not jointly trained during training time), so it can be used with any other models.

### Replicating submission results

I filtered out sentences longer than 300 characters, and divided the training data into smaller splits so they could fit in my computer's limited RAM. The data I used can be found [here](https://drive.google.com/file/d/1NqrYfs1cK63ZRlfl__6-p64NEfgsMLFD/view?usp=sharing).

I trained the seq2seq model until the accuracy on validation set stopped increasing. The final model can be found [here](https://drive.google.com/file/d/1cWp0P2Uj6rcXZzqpfQt9kY8a9SlLEsNU/view?usp=sharing).

The n-gram language model can be found [here](https://drive.google.com/file/d/14RmQSYgijeSVzZNZ2mPGL0lCLg_guXGE/view?usp=sharing).

The `main` function in [`train.py`](./train.py) and [`predict.py`](./predict.py) has examples of how to train the model from scratch and run predictions on test data. I set beam size to a very large number so it may take **very long** to run predictions.

### Credits

Some of the code is taken from [IBM's PyTorch seq2seq implementation](https://github.com/IBM/pytorch-seq2seq).

The code to produce cleaned test data is [written by Khoi Nguyen](https://gist.github.com/suicao/5fd9e27bfb00a147998035730ca224d7?fbclid=IwAR13ufFJUTjTeyMO3KuN8dZTBACkC9ix-_XxN9Z6lshDdD8Eyn3KGPJri6o).

The data used to train n-gram language model are taken from [this repo by @binhvq](https://github.com/binhvq/news-corpus).
