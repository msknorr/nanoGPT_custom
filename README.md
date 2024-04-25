
# nanoGPT

This repo includes some architectural changes for the nanoGPT model and implements the enwik8 dataset.

To prepare the dataset, run:
```
$ python data/enwik8_char/prepare.py
```


To run experiments, run:

```
$ python train.py
```

The configuration can be found in train.py.


Other branches include:
- Spacebyte for character level modeling (proprietary, I could not get good results yet.)
- Encoder-decoder with cross-attention visualization


