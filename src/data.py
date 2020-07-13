import json
import math
import os
from abc import ABC
from argparse import Namespace
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
import torch
import transformers
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class BaseConfig(Namespace):

    def __init__(self,
                 data_dir: str = 'data/convai2/',
                 model_type: str = 'bert-base-uncased',
                 data_type: str = 'basic',
                 max_length: int = 32,
                 max_candidates: int = 20,
                 question_token_type_id: int = 1,
                 response_token_type_id: int = 2,
                 batch_size: int = 256,
                 candidates_batch_size: int = 8,
                 tokenize_batch_size: int = 2048,
                 verbose: bool = True):
        super().__init__(data_dir=data_dir,
                         model_type=model_type,
                         data_type=data_type,
                         max_length=max_length,
                         max_candidates=max_candidates,
                         question_token_type_id=question_token_type_id,
                         response_token_type_id=response_token_type_id,
                         batch_size=batch_size,
                         candidates_batch_size=candidates_batch_size,
                         tokenize_batch_size=tokenize_batch_size,
                         verbose=verbose)


class DatasetPreparer:

    TRAIN_FILE = 'train.json'
    TRAIN_WITH_CANDIDATES_FILE = 'train_with_candidates.json'
    VALID_FILE = 'valid.json'
    VALID_WITH_CANDIDATES_FILE = 'valid_with_candidates.json'

    INDEX_TO_TEXT_FILE = 'index_to_text.json'

    def __init__(self, config: Namespace):

        self.config = config

        self.data_dir = os.path.join(os.getcwd(), self.config.data_dir)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.config.model_type)

        setattr(self.config, 'pad_index', self.tokenizer.pad_token_id)

        self.index_to_text = self.build_index()

    def load_json_file(self, file: str):

        with open(os.path.join(self.data_dir, file)) as file_object:
            data = json.load(file_object)

        return data

    def build_index(self):

        index_to_text = dict()

        for key, value in self.load_json_file(self.INDEX_TO_TEXT_FILE).items():
            index_to_text[int(key)] = value

        if self.config.data_type in ('text',):
            return index_to_text

        index_to_tokenized_text = list()

        indices = list(range(len(index_to_text)))

        for i_batch in tqdm(range(math.ceil(len(index_to_text) / self.config.tokenize_batch_size)),
                            desc='Building index',
                            disable=not self.config.verbose):

            start = i_batch * self.config.tokenize_batch_size
            stop = (i_batch + 1) * self.config.tokenize_batch_size

            batch = [index_to_text[i] for i in indices[start:stop]]

            tokenized_batch = self.tokenizer.batch_encode_plus(batch,
                                                               truncation=True,
                                                               max_length=self.config.max_length)['input_ids']

            index_to_tokenized_text.extend(tokenized_batch)

        return index_to_tokenized_text

    def build_data(self, file, with_candidates: bool):

        if self.config.data_type in ('basic', 'common'):
            dataset = ConvAI2Dataset(data=self.load_json_file(file=file),
                                     index_to_text=self.index_to_text,
                                     with_candidates=with_candidates,
                                     max_candidates=self.config.max_candidates,
                                     question_token_type_id=self.config.question_token_type_id,
                                     response_token_type_id=self.config.response_token_type_id,
                                     max_length=self.config.max_length,
                                     pad_index=self.config.pad_index)
        elif self.config.data_type in ('text',):
            dataset = ConvAI2TextDataset(data=self.load_json_file(file=file),
                                         index_to_text=self.index_to_text,
                                         with_candidates=with_candidates,
                                         max_candidates=self.config.max_candidates)
        else:
            raise ValueError('Not available data_type')

        return dataset

    def load_data(self, as_data_loader: bool = False):

        train_data = self.build_data(file=self.TRAIN_FILE, with_candidates=False)
        train_with_candidates_data = self.build_data(file=self.TRAIN_WITH_CANDIDATES_FILE, with_candidates=True)

        valid_data = self.build_data(file=self.VALID_FILE, with_candidates=False)
        valid_with_candidates_data = self.build_data(file=self.VALID_WITH_CANDIDATES_FILE, with_candidates=True)

        if as_data_loader:
            train_loader = DataLoader(dataset=train_data,
                                      batch_size=self.config.batch_size,
                                      shuffle=True,
                                      collate_fn=train_data.collate,
                                      drop_last=True)

            valid_loader = DataLoader(dataset=valid_data,
                                      batch_size=self.config.batch_size,
                                      collate_fn=valid_data.collate,
                                      drop_last=True)

            train_with_candidates_loader = DataLoader(dataset=train_with_candidates_data,
                                                      batch_size=self.config.candidates_batch_size,
                                                      shuffle=True,
                                                      collate_fn=train_with_candidates_data.collate)

            valid_with_candidates_loader = DataLoader(dataset=valid_with_candidates_data,
                                                      batch_size=self.config.candidates_batch_size,
                                                      collate_fn=valid_with_candidates_data.collate)

            data = (train_loader, valid_loader)
            data_with_candidates = (train_with_candidates_loader, valid_with_candidates_loader)

        else:
            data = (train_data, valid_data)
            data_with_candidates = (train_with_candidates_data, valid_with_candidates_data)

        return data, data_with_candidates


class Recall:

    def __init__(self, k_variants: Tuple[int] = (1, 3, 5), c_variants: Tuple[int] = (2, 5, 10, 15, 20)):

        self.k_variants = k_variants
        self.c_variants = c_variants

        self.matrices = None
        self._messages = list()
        self.step = 0

        self.reset()

    def add(self, similarity_matrix: np.array):
        self.matrices.append(similarity_matrix)

    def reset(self):
        self.matrices = list()

    def calculate(self, k: int, c: int):
        similarity_matrix = torch.cat(self.matrices)[:, :c]

        ranked = similarity_matrix.argsort(descending=True)
        ranked = ranked[:, :k] == 0

        ranked = ranked.sum(dim=-1).float()

        recall = (ranked.sum() / ranked.shape[0]).item()

        return recall

    @property
    def metrics(self):

        metrics = list()
        self.step += 1

        if len(self._messages) > 0:
            self._messages.append(30 * '=')

        for k in self.k_variants:

            metrics.append(list())

            for c in self.c_variants:

                if k >= c:
                    metrics[-1].append(np.NaN)
                else:
                    current_metric = round(self.calculate(k, c), 3)
                    self._messages.append(f'Step {self.step} | Recall @ {k}/{c}: {current_metric:.3f}')
                    metrics[-1].append(current_metric)

        metrics = pd.DataFrame(data=metrics)

        metrics.index = [f'@ {i}' for i in self.k_variants]
        metrics.columns = [f'n_candidates {i}' for i in self.c_variants]

        return metrics

    @property
    def messages(self):
        return '\n'.join(self._messages)


class BaseConvAI2Dataset(Dataset, ABC):

    def __init__(self,
                 data: Any,
                 index_to_text: Any,
                 with_candidates: bool = False,
                 max_candidates: int = 20,
                 question_token_type_id: int = 1,
                 response_token_type_id: int = 2,
                 max_length: int = 256,
                 pad_index: int = 0):

        self.data = data
        self.index_to_text = index_to_text

        self.with_candidates = with_candidates
        self.max_candidates = max_candidates

        self.question_token_type_id = question_token_type_id
        self.response_token_type_id = response_token_type_id
        self.max_length = max_length
        self.pad_index = pad_index

        self.positions = list(range(self.pad_index)) + list(range(self.pad_index + 1, self.max_length + 1))

    def get_pad_sequence(self, sequence, max_length=None):

        max_length = max_length if max_length is not None else self.max_length

        pad_sequence = [self.pad_index] * (max_length - len(sequence))

        return pad_sequence

    def get_positions(self, sequence):

        position_ids = self.positions[:len(sequence)]

        return position_ids

    def get_token_types(self, sequence):

        token_types = [self.question_token_type_id] * len(sequence)

        return token_types

    def __len__(self):
        return len(self.data)

    def collate(self, batch):
        raise NotImplementedError


class ConvAI2TextDataset(BaseConvAI2Dataset):

    def __init__(self,
                 data: Any,
                 index_to_text: List[List[str]],
                 with_candidates: bool = False,
                 max_candidates: int = 20,
                 question_token_type_id: int = 1,
                 response_token_type_id: int = 2,
                 max_length: int = 256,
                 pad_index: int = 0):
        super().__init__(data=data,
                         index_to_text=index_to_text,
                         with_candidates=with_candidates,
                         max_candidates=max_candidates,
                         question_token_type_id=question_token_type_id,
                         response_token_type_id=response_token_type_id,
                         max_length=max_length,
                         pad_index=pad_index)

    def collate(self, batch):
        question_batch, response_batch = list(), list()

        for question, response in batch:
            question_batch.append(question)

            if self.with_candidates:
                response_batch.extend(response)
            else:
                response_batch.append(response)

        return question_batch, response_batch

    def __getitem__(self, index):

        if self.with_candidates:

            question_id, response_id, _, candidates = self.data[index]
            candidates = candidates[:self.max_candidates]

            question = self.index_to_text[question_id]

            candidates = [self.index_to_text[index] for index in candidates]
            response = [self.index_to_text[response_id]] + candidates

        else:

            question_id, response_id, _ = self.data[index]

            question = self.index_to_text[question_id]
            response = self.index_to_text[response_id]

        return question, response


class ConvAI2Dataset(BaseConvAI2Dataset):

    def __init__(self,
                 data: Any,
                 index_to_text: List[List[int]],
                 with_candidates: bool = False,
                 max_candidates: int = 20,
                 question_token_type_id: int = 1,
                 response_token_type_id: int = 2,
                 max_length: int = 256,
                 pad_index: int = 0):
        super().__init__(data=data,
                         index_to_text=index_to_text,
                         with_candidates=with_candidates,
                         max_candidates=max_candidates,
                         question_token_type_id=question_token_type_id,
                         response_token_type_id=response_token_type_id,
                         max_length=max_length,
                         pad_index=pad_index)

    def prepare_batch(self, batch):

        max_length = max([len(sample) for sample in batch])

        token_ids, positions, token_types = list(), list(), list()

        for sample in batch:

            pad_sequence = self.get_pad_sequence(sequence=sample, max_length=max_length)

            token_ids.append(sample + pad_sequence)
            positions.append(self.get_positions(sequence=sample) + pad_sequence)
            token_types.append(self.get_token_types(sequence=sample) + pad_sequence)

        token_ids = torch.tensor(token_ids)
        positions = torch.tensor(positions)
        token_types = torch.tensor(token_types)

        return token_ids, positions, token_types

    def collate(self, batch):

        question_batch, response_batch = list(), list()

        for question, response in batch:
            question_batch.append(question)

            if self.with_candidates:
                response_batch.extend(response)
            else:
                response_batch.append(response)

        question_token_ids, question_positions, question_token_types = self.prepare_batch(question_batch)
        response_token_ids, response_positions, response_token_types = self.prepare_batch(response_batch)

        token_ids = (question_token_ids, response_token_ids)
        positions = (question_positions, response_positions)
        token_types = (question_token_types, response_token_types)

        return token_ids, positions, token_types

    def __getitem__(self, index):

        if self.with_candidates:

            question_id, response_id, _, candidates = self.data[index]
            candidates = candidates[:self.max_candidates]

            question = self.index_to_text[question_id]

            candidates = [self.index_to_text[index] for index in candidates]
            response = [self.index_to_text[response_id]] + candidates

        else:

            question_id, response_id, _ = self.data[index]

            question = self.index_to_text[question_id]
            response = self.index_to_text[response_id]

        return question, response


# class ConvAI2Dataset(Dataset):
#
#     def __init__(self,
#                  data,
#                  index_to_tokenized_text,
#                  context_length=5,
#                  input_phrase_token_type_id=1,
#                  response_token_type_id=2,
#                  persons_type_ids=(3, 4),
#                  max_length=256,
#                  pad_index=0,
#                  context_pad_index=0):
#
#         self.data = data
#         self.index_to_tokenized_text = index_to_tokenized_text
#
#         self.context_length = context_length
#
#         self.input_phrase_token_type_id = input_phrase_token_type_id
#         self.response_token_type_id = response_token_type_id
#         self.persons_type_ids = persons_type_ids
#
#         self.max_length = max_length
#         self.pad_index = pad_index
#         self.context_pad_index = context_pad_index
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, index):
#         input_phrase_id, response_id, context_ids = self.data[index]
#
#         input_phrase = self.index_to_tokenized_text[input_phrase_id]
#         response = self.index_to_tokenized_text[response_id]
#
#         context = [self.index_to_tokenized_text[phrase_id]
#                    for phrase_id in context_ids[len(context_ids) - self.context_length:]]
#
#         persons_type_ids = self.persons_type_ids * math.ceil(len(context) / 2)
#         persons_type_ids = persons_type_ids[:len(context)]
#
#         token_type_ids = list()
#
#         for n in range(len(context)):
#             token_type_ids.extend([persons_type_ids[n]] * len(context[n]))
#
#         return input_phrase, response, context, token_type_ids
