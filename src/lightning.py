import json
import os
from argparse import ArgumentParser

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.collecting import get_data, get_logits
from src.data import PairedData
from src.model import RNNEncoder, DistillationLoss, SimilarityWrapper


class LightningDistillation(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams

        self.train_path = os.path.join(self.hparams.data_dir, 'train.tsv')
        self.validation_path = os.path.join(self.hparams.data_dir, 'validation.tsv')
        self.test_path = os.path.join(self.hparams.data_dir, 'test.tsv')
        self.logits_path = os.path.join(self.hparams.data_dir, 'logits.tsv')

        with open(os.path.join(self.hparams.data_dir, 'word2index.json')) as file_object:
            self.word2index = json.load(file_object)

        embeddings = np.load(os.path.join(self.hparams.data_dir, 'embeddings.npy'))

        self.encoder = RNNEncoder(pretrained_embeddings=torch.tensor(embeddings).float(),
                                  rnn_dim=self.hparams.model_dim,
                                  rnn_layers=self.hparams.rnn_layers,
                                  dropout=self.hparams.dropout)

        self.wrapper = SimilarityWrapper(encoder=self.encoder)

        self.distillation_criterion = DistillationLoss(temperature=self.hparams.temperature)
        self.classification_criterion = nn.CrossEntropyLoss()

        self.alpha = self.hparams.alpha

    def forward(self, texts_a, texts_b):
        prediction = self.wrapper(texts_a, texts_b)
        return prediction

    def training_step(self, batch, batch_idx):
        texts_a, texts_b, targets, teacher_logits = batch

        lengths_a = (texts_a != 0).sum(-1)
        lengths_b = (texts_b != 0).sum(-1)
        mask = (lengths_a > 0) & (lengths_b > 0)
        texts_a = texts_a[mask]
        texts_b = texts_b[mask]
        targets = targets[mask]
        teacher_logits = teacher_logits[mask]

        prediction = self.forward(texts_a, texts_b)

        distillation_loss = self.distillation_criterion(prediction, teacher_logits)
        classification_loss = self.classification_criterion(prediction, targets)

        loss = self.alpha * distillation_loss + (1. - self.alpha) * classification_loss

        binary_predictions = prediction.argmax(dim=-1)
        accuracy = (binary_predictions == targets).sum().float() / targets.shape[0]

        log = {
            'train_loss': loss.item(),
            'train_distillation_loss': distillation_loss.item(),
            'train_classification_loss': classification_loss.item(),
            'train_accuracy': accuracy.item(),
            'alpha': self.alpha,
        }

        return {'loss': loss, 'log': log}

    def validation_step(self, batch, batch_idx):
        texts_a, texts_b, targets = batch

        lengths_a = (texts_a != 0).sum(-1)
        lengths_b = (texts_b != 0).sum(-1)
        mask = (lengths_a > 0) & (lengths_b > 0)
        texts_a = texts_a[mask]
        texts_b = texts_b[mask]
        targets = targets[mask]

        prediction = self.forward(texts_a, texts_b)

        loss = self.classification_criterion(prediction, targets)

        binary_predictions = prediction.argmax(dim=-1)
        accuracy = (binary_predictions == targets).sum().float() / targets.shape[0]

        return {'val_loss': loss.item(), 'val_accuracy': accuracy.item()}

    def validation_epoch_end(self, outputs: list):
        mean_loss = torch.stack([batch['val_loss'] for batch in outputs]).mean()
        accuracy = torch.stack([batch['val_accuracy'] for batch in outputs]).mean()

        log = {'val_loss': mean_loss, 'val_accuracy': accuracy}

        return {'val_loss': mean_loss, 'log': log}

    def test_step(self, batch, batch_idx):
        texts_a, texts_b, targets = batch

        lengths_a = (texts_a != 0).sum(-1)
        lengths_b = (texts_b != 0).sum(-1)
        mask = (lengths_a > 0) & (lengths_b > 0)
        texts_a = texts_a[mask]
        texts_b = texts_b[mask]
        targets = targets[mask]

        prediction = self.forward(texts_a, texts_b)

        loss = self.classification_criterion(prediction, targets)

        accuracy = (prediction.argmax(dim=-1) == targets).sum().float() / targets.shape[0]

        return {'test_loss': loss.item(), 'test_accuracy': accuracy.item()}

    def test_epoch_end(self, outputs):
        mean_loss = torch.stack([batch['test_loss'] for batch in outputs]).mean()
        accuracy = torch.stack([batch['test_accuracy'] for batch in outputs]).mean()

        log = {'test_loss': mean_loss, 'test_accuracy': accuracy}

        return {'test_loss': mean_loss, 'log': log}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(params=self.wrapper.parameters(),
                                      lr=self.hparams.learning_rate,
                                      weight_decay=self.hparams.weight_decay)

        return optimizer

    def train_dataloader(self):
        texts_a, texts_b, targets = get_data(self.train_path)
        logits = get_logits(self.logits_path)

        dataset = PairedData(self.word2index, texts_a, texts_b, targets, logits=logits,
                             max_length=self.hparams.max_length)
        loader = DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=True)

        return loader

    def val_dataloader(self):
        texts_a, texts_b, targets = get_data(self.validation_path)

        dataset = PairedData(self.word2index, texts_a, texts_b, targets,
                             max_length=self.hparams.max_length)
        loader = DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=False)

        return loader

    def test_dataloader(self):
        texts_a, texts_b, targets = get_data(self.test_path)

        dataset = PairedData(self.word2index, texts_a, texts_b, targets,
                             max_length=self.hparams.max_length)
        loader = DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=False)

        return loader

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser])

        # model
        parser.add_argument('--model_dim', type=int, default=256)
        parser.add_argument('--rnn_layers', type=int, default=2)
        parser.add_argument('--dropout', type=float, default=0.3)

        # loss
        parser.add_argument('--alpha', type=float, default=0.5)
        parser.add_argument('--temperature', type=float, default=1.)

        # optimizers & schedulers
        parser.add_argument('--learning_rate', type=float, default=0.001)
        parser.add_argument('--weight_decay', type=float, default=0.)

        return parser
