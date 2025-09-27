
import sys
from torch.utils.data import DataLoader

from src.exception.exception import CustomException
from src.logging.logger import logging

from src.entity.artifact_entity import DataIngestionArtifact, DataLoaderArtifact
from src.entity.config_entity import DataLoaderConfig
from src.model_components.dataloaders import ImageDataset
from src.utils.main_utils import extract_csv



class LoadData:
    def __init__(self,data_ingestion_artifact:DataIngestionArtifact,
                 data_loader_config:DataLoaderConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_loader_config = data_loader_config
            
        except Exception as e:
            raise CustomException(e, sys)
            
    def data_loader(self) -> DataLoaderArtifact:
        try:
            trained_images_path = extract_csv(self.data_ingestion_artifact.trained_file_path)
            train_image_dataset = ImageDataset(images_path=trained_images_path,
                                               class_ids=self.data_loader_config.char2idx,
                                                    transform=self.data_loader_config.transform)
            logging.info("train image augmentation is done")
            
            train_image_loader = DataLoader(dataset = train_image_dataset, 
                                      batch_size = self.data_loader_config.batch_size,
                                      shuffle = self.data_loader_config.is_shuffle)
            logging.info("train data loaded successfully")
            
            tested_images_path = extract_csv(self.data_ingestion_artifact.test_file_path)
            test_image_dataset = ImageDataset(images_path=tested_images_path,
                                              class_ids=self.data_loader_config.char2idx,
                                                    transform=self.data_loader_config.transform) 
            logging.info("test image augmentation is done")
               
            test_image_loader = DataLoader(dataset = test_image_dataset, 
                                      batch_size = self.data_loader_config.batch_size,
                                      shuffle = self.data_loader_config.is_shuffle)
            logging.info("test data loaded successfully")
            
            valed_images_path = extract_csv(self.data_ingestion_artifact.val_file_path)
            val_image_dataset = ImageDataset(images_path=valed_images_path,
                                             class_ids=self.data_loader_config.char2idx,
                                                    transform=self.data_loader_config.transform) 
            logging.info("val image augmentation is done")
               
            val_image_loader = DataLoader(dataset = val_image_dataset, 
                                      batch_size = self.data_loader_config.batch_size,
                                      shuffle = self.data_loader_config.is_shuffle)
            logging.info("val data loaded successfully")
            

            
            data_loader_artifact = DataLoaderArtifact(
                train_image_loader = train_image_loader,
                test_image_loader = test_image_loader,
                val_image_loader = val_image_loader,
                image_transformation = self.data_loader_config.transform)
            
            return data_loader_artifact
        except Exception as e:
            raise CustomException(e, sys)
