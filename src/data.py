import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import wordpunct_tokenize
from tqdm import tqdm

lemma = WordNetLemmatizer()


def process_text(text):
    words = wordpunct_tokenize(str(text).lower())
    words = [lemma.lemmatize(word) for word in words if word.isalnum()]

    return words


def get_word2vec(exist_words=None, max_words=1000000, file_path='data/cc.en.300.vec', verbose=True, pad_token='PAD'):
    word2index = {pad_token: 0}
    embeddings = []

    word2vec_file = open(file_path)

    n_words, embedding_dim = word2vec_file.readline().split()
    n_words, embedding_dim = int(n_words), int(embedding_dim)

    # Zero vector for PAD
    embeddings.append(np.zeros(embedding_dim))

    progress_bar = tqdm(desc='Read word2vec',
                        total=(n_words if max_words < 1 else max_words) if exist_words is None else len(exist_words),
                        disable=not verbose)

    while True:

        line = word2vec_file.readline().strip()

        if not line:
            break

        current_parts = line.split()

        current_word = ' '.join(current_parts[:-embedding_dim])

        if exist_words is not None and current_word not in exist_words \
                or max_words != -1 and len(word2index) >= max_words \
                or current_word == pad_token:
            continue

        word2index[current_word] = len(word2index)

        current_embeddings = current_parts[-embedding_dim:]
        current_embeddings = np.array(list(map(float, current_embeddings)))

        embeddings.append(current_embeddings)

        progress_bar.update()

    progress_bar.close()

    word2vec_file.close()

    embeddings = np.stack(embeddings)

    return word2index, embeddings


def get_data(data_path):
    data = pd.read_csv(data_path, sep='\t', header=None)

    text_a = list(data[0].map(process_text))
    text_b = list(data[1].map(process_text))
    target = list(data[2].map(int))

    return text_a, text_b, target


def get_logits(logits_data_path):
    logits = pd.read_csv(logits_data_path, sep='\t', header=None)
    logits = torch.tensor(list(zip(logits[0], logits[1])))
    return logits


class PairedData(Dataset):

    def __init__(self, word2index, texts_a, texts_b, targets, logits=None, pad_index=0, max_length=64):

        self.word2index = word2index
        self.pad_index = pad_index
        self.max_length = max_length

        self.texts_a = [self.padding(self.indexing(sample)) for sample in texts_a]
        self.texts_b = [self.padding(self.indexing(sample)) for sample in texts_b]
        self.targets = targets
        self.logits = logits

    def indexing(self, tokens):
        return [self.word2index[tok] for tok in tokens if tok in self.word2index]

    def padding(self, sequence, max_length=None):

        max_length = min(self.max_length, max_length) if max_length is not None else self.max_length

        sequence = sequence[:max_length]

        sequence += [self.pad_index] * (max_length - len(sequence))

        return sequence

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        text_a = torch.tensor(self.texts_a[index])
        text_b = torch.tensor(self.texts_b[index])
        target = self.targets[index]

        if self.logits is not None:
            logits = self.logits[index]
            return text_a, text_b, target, logits
        else:
            return text_a, text_b, target
