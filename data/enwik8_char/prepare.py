import os
import pickle
import requests
import numpy as np
import os
import urllib.request
import zipfile
import sys
      
zip_path = os.path.join(os.path.dirname(__file__), 'enwik8.zip')
if not os.path.exists(zip_path):
    print("Downloading...")
    data_url = 'http://mattmahoney.net/dc/enwik8.zip'
    urllib.request.urlretrieve(data_url, zip_path)
    
    
data = zipfile.ZipFile(zip_path).read('enwik8')

num_test_chars = 5000000
train_data = data[: -2 * num_test_chars]
valid_data = data[-2 * num_test_chars: -num_test_chars]
test_data = data[-num_test_chars:]

train_ascii = " ".join([str(c) if c != ord('\n') else '\n' for c in train_data])
val_ascii = " ".join([str(c) if c != ord('\n') else '\n' for c in valid_data])
test_ascii = " ".join([str(c) if c != ord('\n') else '\n' for c in test_data])

chars = list(set(train_data)) + ['<eos>']
chars.remove(10)  # 10 is \n in ascii
assert len(chars) == 205

# create dictionaries
stoi = {str(ch): i for i, ch in enumerate(chars)}
itos = {i: str(ch) for i, ch in enumerate(chars)}


def encode(ascii):
    ids = []
    for line in ascii.split('\n'):
        words = line.split() + ['<eos>']
        for word in words:
            ids.append(stoi[word])
    return np.array(ids, dtype=np.uint16)


train_ids = encode(train_ascii)
val_ids = encode(val_ascii)
test_ids = encode(test_ascii)

train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))
test_ids.tofile(os.path.join(os.path.dirname(__file__), 'test.bin'))


meta = {
    'vocab_size': 205,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)