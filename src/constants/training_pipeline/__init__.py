import os 
import sys
import torch
from typing import Tuple


'''
Defining comman constant variable for training pipeline
'''
ARTIFACT_DIR: str = 'Artifacts'
NUMBER_OF_CHARACTERS = 5


'''
Data Ingestion related constants start with DATA_INGESTION VAR NAME
'''
DATA_INGESTION_IMAGE_DATABASE_FOLDER: str = 'raw_data'
DATA_INGESTION_DIR_NAME: str = 'data_ingestion'
DATA_INGESTION_FEATURE_STORE_DIR: str =  "feature_score"
DATA_INGESTION_FEATURE_STORE_FILE_NAME: str =  "feature_score.csv"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_FILE_NAME: str = 'train.csv'
DATA_INGESTION_TEST_FILE_NAME: str = 'test.csv'
DATA_INGESTION_VAL_FILE_NAME: str = 'val.csv'
DATA_INGESTION_TEST_SIZE: int = 10
DATA_INGESTION_VAL_SIZE: float = 0.1

'''
Data AUGMENTATION related constants starts with DATA_AUGMENTATION var name
'''
DATA_AUGMENTATION_IMAGE_SIZE: Tuple[int, int] = (50, 200)
DATA_AUGMENTATION_IMAGE_BRIGHTNESS: float = 0.3
DATA_AUGMENTATION_IMAGE_CONTRAST: float = 0.3
DATA_AUGMENTATION_IMAGE_ROTATION: int = 5
DATA_AUGMENTATION_IMAGE_DISTORTION_SCALE: float = 0.3
DATA_AUGMENTATION_IMAGE_P: float = 0.5



DATA_LOADER_BATCH_SIZE: int = 32
DATA_LOADER_SHUFFLE: bool = True

DATA_TRAINER_CLASS: str = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

DATA_TRAINER_IMAGE_HEIGHT: int = 50
DATA_TRAINER_IMAGE_WIDTH: int = 200
DATA_TRAINER_EPOCHS: int = 5
DATA_TRAINER_LEARNING_RATE: float = 0.01
DATA_TTRAINER_MODEL_DIR: str = 'model'
DATA_TRAINER_MODEL_NAME:str = 'best_model.pth'





