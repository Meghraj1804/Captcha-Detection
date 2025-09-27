
import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import os
import sys



from src.exception.exception import CustomException
from src.logging.logger import logging
from src.utils.main_utils import decode_prediction, save_test_samples

class TrainModel:
    def __init__(self, model:Module, criterion:_Loss, optimizer:Optimizer):
        try:
            self.device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = model.to(self.device)
            self.criterion = criterion
            self.optimizer = optimizer
        except Exception as e:
            raise CustomException(e,sys)

    def train(self, dataloader:DataLoader):
        try:
            self.model.train()
            running_loss = 0.0

            for images, labels in dataloader:
                images = images.to(self.device)
                # labels shape: (batch_size, num_chars)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(images)  # outputs is a list: [char1_logits, char2_logits, ..., charN_logits]

                # Compute loss for each character position
                total_loss = 0.0
                for i, output in enumerate(outputs):
                    total_loss += self.criterion(output, labels[:, i])  # labels[:, i] → ground truth for char at pos i

                total_loss.backward()
                self.optimizer.step()

                running_loss += total_loss.item()

            epoch_loss = running_loss / len(dataloader)
            return epoch_loss
        except Exception as e:
            raise CustomException(e,sys)

    def validate(self, dataloader:DataLoader):
        try:
            self.model.eval()
            running_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for images, labels in dataloader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    outputs = self.model(images)

                    total_loss = 0.0
                    predictions = []

                    for i, output in enumerate(outputs):
                        total_loss += self.criterion(output, labels[:, i])
                        preds = output.argmax(dim=1)
                        predictions.append(preds)

                    # predictions: list of [batch_size], stacked = [batch_size, num_chars]
                    predictions = torch.stack(predictions, dim=1)
                    correct += (predictions == labels).all(dim=1).sum().item()
                    total += labels.size(0)
                    running_loss += total_loss.item()

            val_loss = running_loss / len(dataloader)
            val_accuracy = correct / total
            return val_loss, val_accuracy
        except Exception as e:
            raise CustomException(e,sys)
        

    def save_model(self, state_dict, save_path):
        try:
            torch.save(state_dict, save_path)
        except Exception as e:
            raise CustomException(e,sys)
        
    def testing(self, data:DataLoader, idx2char:dict, path:str):
        self.model.eval()
        self.model.to(self.device)
        results = []

        with torch.no_grad():
            for images, labels in data:
                images = images.to(self.device)
                outputs = self.model(images)  # list of logits per character

                predictions = []
                for output in outputs:
                    preds = output.argmax(dim=1)
                    predictions.append(preds)
                predictions = torch.stack(predictions, dim=1)  # (batch_size, sequence_length)

                decoded_preds = decode_prediction(predictions, idx2char)

                for img, pred in zip(images.cpu(), decoded_preds):
                    results.append((img, pred))

        save_test_samples(results, path)
                
                

        

    