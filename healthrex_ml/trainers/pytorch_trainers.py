"""
Definition of SequenceTrainer
"""
import os
import json
import numpy as np
from sklearn.metrics import roc_auc_score
import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from healthrex_ml.featurizers import DEFAULT_LAB_COMPONENT_IDS
from healthrex_ml.featurizers import DEFAULT_FLOWSHEET_FEATURES

import pdb

class SequenceTrainer():
    """
    Used to train models that leverage SequenceFeaturizer.  Example model
    classes include GRUs, GRUs with Attention, Transformers. 
    """

    def __init__(self, outpath, model, criterion, optimizer, device,
                 train_dataloader, val_dataloader, test_dataloader,
                 stopping_metric, num_epochs=100, scheduler=None,
                 stopping_tolerance=20):
        self.outpath = outpath
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.stopping_metric = stopping_metric
        self.stopping_tolerance = stopping_tolerance
        self.num_epochs = num_epochs
        self.scheduler = scheduler
        self.writer = SummaryWriter(self.outpath)

    # Loop through epochs, train, validate, stop
    def __call__(self):
        best_stopping_metric = 0
        tolerance_counter = 0
        for epoch in range(self.num_epochs):
            train_metrics = self.train()
            print(f"Training Loss: {train_metrics['loss']} | "
                  f"Training AUC: {train_metrics['auc']}")

            val_metrics = self.evaluate()
            print(f"Val Loss: {val_metrics['loss']} | "
                  f"Val AUC: {val_metrics['auc']}")

            self.writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
            self.writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
            self.writer.add_scalar('AUC/train', train_metrics['auc'], epoch)
            self.writer.add_scalar('AUC/val', val_metrics['auc'], epoch)

            if tolerance_counter == self.stopping_tolerance:
                break

            if epoch == 0:
                best_stopping_metric = val_metrics[self.stopping_metric]
                torch.save(self.model.state_dict(),
                           os.path.join(self.outpath, f'model_{epoch}.pt'))
            elif val_metrics[self.stopping_metric] > best_stopping_metric:
                best_stopping_metric = val_metrics[self.stopping_metric]
                tolerance_counter = 0
                torch.save(self.model.state_dict(),
                           os.path.join(self.outpath, f'model_{epoch}.pt'))
            else:
                tolerance_counter += 1
    
    # Enter train round
    def train(self):
        self.model.train()
        train_loss = 0
        predictions, targets = [], []
        for batch in tqdm(self.train_dataloader):
            self.model.zero_grad()
            sequence = batch['sequence'].to(self.device)
            seq_lengths = batch['lengths']
            labels = batch['labels'].to(self.device)
            output = self.model(sequence.int(), seq_lengths)
            loss = self.criterion(output, labels.float())
            train_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            predictions.append(output.cpu().detach().numpy())
            targets.append(labels.cpu().detach().numpy())
        train_loss /= len(self.train_dataloader)
        predictions = np.concatenate(predictions)
        targets = np.concatenate(targets)
        auc = roc_auc_score(targets, predictions)

        train_metrics = {
            'loss': train_loss,
            'auc': auc
        }
        return train_metrics

    # Enter eval round
    def evaluate(self):
        self.model.eval()
        total_loss = 0
        predictions, targets = [], []
        for batch in tqdm(self.val_dataloader):
            sequence = batch['sequence'].to(self.device)
            seq_lengths = batch['lengths']
            labels = batch['labels'].to(self.device)
            output = self.model(sequence.int(), seq_lengths)
            loss = self.criterion(output, labels.float())
            total_loss += loss.item()
            predictions.append(output.cpu().detach().numpy())
            targets.append(labels.cpu().detach().numpy())
        total_loss /= len(self.val_dataloader)
        predictions = np.concatenate(predictions)
        targets = np.concatenate(targets)
        auc = roc_auc_score(targets, predictions)
        metrics = {
            'loss': total_loss,
            'auc': auc
        }
        return metrics
