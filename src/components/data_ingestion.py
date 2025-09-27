import os
import sys
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split

from src.exception.exception import CustomException
from src.logging.logger import logging

from src.entity.config_entity import DataIngestionConfig, TrainingPipelineConfig
from src.entity.artifact_entity import DataIngestionArtifact


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig,):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise CustomException(e, sys)
        
    def export_collection_as_dataframe(self):
        try:
            database_folder = self.data_ingestion_config.database_folder
            dataset = {}
            
            for i in (os.listdir(database_folder)):
                if len(i[:-4]) == self.data_ingestion_config.num_chars and re.fullmatch(r'[A-Za-z0-9]+',i[:-4]) :
                    dataset[os.path.join(database_folder,i)] = i[:-4]
                
            df = pd.DataFrame(list(dataset.items()), columns=['Path', 'Label'])

            # df.replace({'na':np.nan}, inplace=True)
            
            return df     
    
        except Exception as e:
            raise CustomException(e, sys)
        
    def export_data_into_feature_store(self, dataframe:pd.DataFrame):
        try:
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path,exist_ok=True)
            dataframe.to_csv(feature_store_file_path, index=False, header=True)
            return dataframe
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def split_data_as_train_test(self,dataframe:pd.DataFrame):
        try:            
            train_set, val_set = train_test_split(dataframe,
                                                   test_size=self.data_ingestion_config.train_test_split_ratio)
            
            test_set = train_set.sample(n=10, random_state=42)
            
            train_set = train_set.drop(test_set.index)
            
            logging.info('train test split done')
            
            dir_path = os.path.dirname(self.data_ingestion_config.train_file_path)
            os.makedirs(dir_path, exist_ok=True)
            train_set.to_csv(self.data_ingestion_config.train_file_path, index=False, header=True)
            logging.info('train set saved successfully')
            
            dir_path = os.path.dirname(self.data_ingestion_config.test_file_path)
            os.makedirs(dir_path, exist_ok=True)
            test_set.to_csv(self.data_ingestion_config.test_file_path, index=False, header=True)
            logging.info('test set saved successfully')
            
            dir_path = os.path.dirname(self.data_ingestion_config.val_file_path)
            os.makedirs(dir_path, exist_ok=True)
            val_set.to_csv(self.data_ingestion_config.val_file_path, index=False, header=True)
            logging.info('test set saved successfully')
            
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_ingestion(self):
        try:
            dataframe = self.export_collection_as_dataframe()
            dataframe = self.export_data_into_feature_store(dataframe)
            logging.info("feature store has been saved")
            
            self.split_data_as_train_test(dataframe)
            
            dataingestionartifact = DataIngestionArtifact(
                trained_file_path = self.data_ingestion_config.train_file_path,
                test_file_path = self.data_ingestion_config.test_file_path,
                val_file_path = self.data_ingestion_config.val_file_path
            )
            
            return dataingestionartifact
            
        except Exception as e:
            raise CustomException(e,sys)
    