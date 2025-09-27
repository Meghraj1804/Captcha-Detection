import os
from datetime import datetime
import torch
from src.constants import training_pipeline
from torchvision import transforms

class TrainingPipelineConfig:
    def __init__(self, timestamp=datetime.now()):
        timestamp = timestamp.strftime("%m_%d_%Y_%H_%M_%S")
        self.artifact_name = training_pipeline.ARTIFACT_DIR
        self.artifact_dir = os.path.join(self.artifact_name,timestamp)
        self.timestamp: str = timestamp
        self.num_chars = training_pipeline.NUMBER_OF_CHARACTERS
        
class DataIngestionConfig:
    def __init__(self, training_pipeline_config:TrainingPipelineConfig):
        self.data_ingestion_dir: str = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.DATA_INGESTION_DIR_NAME
        )
        
        self.feature_store_file_path = os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_FEATURE_STORE_DIR,
            training_pipeline.DATA_INGESTION_FEATURE_STORE_FILE_NAME
        )
        
        self.train_file_path: str = os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_INGESTED_DIR,
            training_pipeline.DATA_INGESTION_TRAIN_FILE_NAME
        )
        
        self.test_file_path: str = os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_INGESTED_DIR,
            training_pipeline.DATA_INGESTION_TEST_FILE_NAME
        )
        
        self.val_file_path: str = os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_INGESTED_DIR,
            training_pipeline.DATA_INGESTION_VAL_FILE_NAME
        )
        
        self.train_test_split_ratio: float = training_pipeline.DATA_INGESTION_TEST_SIZE
        self.train_val_split_ratio: float = training_pipeline.DATA_INGESTION_VAL_SIZE
        self.database_folder: str = training_pipeline.DATA_INGESTION_IMAGE_DATABASE_FOLDER
        self.num_chars = training_pipeline_config.num_chars
        
class DataLoaderConfig:
    def __init__(self, training_pipeline_config:TrainingPipelineConfig):
        
        self.transform = transforms.Compose([
                        transforms.Resize(training_pipeline.DATA_AUGMENTATION_IMAGE_SIZE),     
                        transforms.ColorJitter(training_pipeline.DATA_AUGMENTATION_IMAGE_BRIGHTNESS,
                                               training_pipeline.DATA_AUGMENTATION_IMAGE_CONTRAST),     
                        transforms.RandomRotation(training_pipeline.DATA_AUGMENTATION_IMAGE_ROTATION),
                        transforms.RandomPerspective(training_pipeline.DATA_AUGMENTATION_IMAGE_DISTORTION_SCALE,
                                                    training_pipeline.DATA_AUGMENTATION_IMAGE_P),
                        transforms.ToTensor()
            
                                    ])
    
        self.batch_size: int = training_pipeline.DATA_LOADER_BATCH_SIZE
        self.is_shuffle: bool = training_pipeline.DATA_LOADER_SHUFFLE
        self.char2idx: dict = {ch:i for i, ch in enumerate(training_pipeline.DATA_TRAINER_CLASS)}
        
        
        
class DataTrainerConfig:
    def __init__(self, training_pipeline_config:TrainingPipelineConfig):
        
        self.model_dir: str = os.path.join(training_pipeline_config.artifact_dir,
                                           training_pipeline.DATA_TTRAINER_MODEL_DIR,
                                           training_pipeline.DATA_TRAINER_MODEL_NAME)
        
        self.model_summery:str = os.path.join(training_pipeline_config.artifact_dir,
                                           training_pipeline.DATA_TTRAINER_MODEL_DIR)
        
        self.num_class: int = len(training_pipeline.DATA_TRAINER_CLASS)
        
        self.image_height: int = training_pipeline.DATA_TRAINER_IMAGE_HEIGHT
        self.image_width: int = training_pipeline.DATA_TRAINER_IMAGE_WIDTH
        
        
        self.epoch = training_pipeline.DATA_TRAINER_EPOCHS
        self.learning_rate = training_pipeline.DATA_TRAINER_LEARNING_RATE
        self.num_chars = training_pipeline_config.num_chars
        self.idx2char = {i: ch for i, ch in enumerate(training_pipeline.DATA_TRAINER_CLASS)}
        
        