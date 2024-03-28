
# Enwik8 dataset, character-level

Dataset from the Hutter prize, contains 100M chars from wikipedia.
We follow the convention of splitting train, val, test according to 90M, 5M, 5M as described [here](https://arxiv.org/abs/1808.04444).

After running `prepare.py`:

- train.bin has 90,000,000 tokens
- val.bin has 5,000,000 tokens
- test.bin has 5,000,000 tokens



"""

https://arxiv.org/pdf/1502.02367.pdf

Closely following the protocols in (Mikolov et al., 2012; Graves, 2013),
we used the first 90 MBytes of characters to train a model,
the next 5 MBytes as a validation set, and the remaining
as a test set, with the vocabulary of 205 characters including a token for an unknown character. We used the average
number of bits-per-character (BPC, E[− log2 P(xt+1|ht)])
to measure the performance of each model on the Hutter
dataset"""