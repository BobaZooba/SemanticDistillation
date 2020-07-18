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
