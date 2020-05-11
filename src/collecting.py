import os
import json
import numpy as np
import pandas as pd
import torch
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import wordpunct_tokenize
from tqdm import tqdm

lemma = WordNetLemmatizer()


def process_text(text):
    words = wordpunct_tokenize(str(text).lower())
    words = [lemma.lemmatize(word) for word in words if word.isalnum()]

    return words


def get_word2vec(exist_words=None, max_words=1000000,
                 file_path='data/cc.en.300.vec',
                 verbose=True, pad_token='PAD'):
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


def collecting(data_dir):

    if os.path.isfile(os.path.join(data_dir, 'word2index.json')) \
            or os.path.isfile(os.path.join(data_dir, 'embeddings.npy')):
        return None

    train_path = os.path.join(data_dir, 'train.tsv')
    validation_path = os.path.join(data_dir, 'validation.tsv')
    test_path = os.path.join(data_dir, 'test.tsv')

    train_a, train_b, train_targets = get_data(train_path)
    validation_a, validation_b, validation_targets = get_data(validation_path)
    test_a, test_b, test_targets = get_data(test_path)

    vocab = dict()

    for dataset in (train_a, train_b, validation_a, validation_b, test_a, test_b):
        for text in dataset:
            for word in text:
                vocab[word] = vocab.get(word, 0) + 1

    exist_words = set(vocab.keys())

    word2index, embeddings = get_word2vec(exist_words=exist_words)

    with open(os.path.join(data_dir, 'word2index.json'), 'w') as f:
        json.dump(word2index, f)

    np.save(os.path.join(data_dir, 'embeddings.npy'), embeddings)
