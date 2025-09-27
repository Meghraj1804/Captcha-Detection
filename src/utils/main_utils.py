import pandas as pd
from src.exception.exception import CustomException
import sys
import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2

def extract_csv(path:str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def plot_training_summary(train_losses, val_losses, val_accuracies, save_path):
        try:
            epochs = list(range(1, len(train_losses) + 1))

            fig, ax1 = plt.subplots()

            color = 'tab:red'
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss', color=color)
            ax1.plot(epochs, train_losses, label='Train Loss', color='red')
            ax1.plot(epochs, val_losses, label='Val Loss', color='orange')
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.legend(loc='upper left')

            ax2 = ax1.twinx()
            color = 'tab:blue'
            ax2.set_ylabel('Validation Accuracy', color=color)
            ax2.plot(epochs, val_accuracies, label='Val Accuracy', color=color)
            ax2.tick_params(axis='y', labelcolor=color)
            ax2.legend(loc='upper right')

            fig.tight_layout()
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            plt.close()
        except Exception as e:
            raise CustomException(e,sys)
        
def decode_prediction(tensor_batch, idx2char):

    results = []
    for row in tensor_batch:
        chars = [idx2char[int(idx)] for idx in row]
        results.append("".join(chars))
    return results

def save_test_samples(results: list, save_path: str):
    annotated_images = []

    for img_tensor, pred_text in results:
        img_np = img_tensor.numpy().transpose(1, 2, 0)  # CHW -> HWC
        img_np = (img_np * 255).astype(np.uint8)  # Assuming the tensor is normalized between [0,1]

        # Resize to ensure text fits
        img_np = cv2.resize(img_np, (img_np.shape[1] * 2, img_np.shape[0] * 2))

        # Put text
        cv2.putText(img_np, pred_text, (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
        annotated_images.append(img_np)

    # Stack all images vertically (or horizontally)
    stacked_img = np.vstack(annotated_images)

    cv2.imwrite(save_path, stacked_img)