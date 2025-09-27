
from src.logging.logger import logging
from src.exception.exception import CustomException

from src.entity.config_entity import TrainingPipelineConfig, DataIngestionConfig, DataLoaderConfig, DataTrainerConfig

from src.components.data_ingestion import DataIngestion
from src.components.data_load import LoadData
from src.components.training_pipeline import TrainData

import sys


if __name__ == '__main__':
    try:
        train_pipeline_config = TrainingPipelineConfig()
        data_ingestion_config = DataIngestionConfig(train_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config)
        logging.info('data ingestion initiate')
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        
        data_loader_config = DataLoaderConfig(train_pipeline_config)
        load_data = LoadData(data_ingestion_artifact,data_loader_config)
        data_loader_artifact = load_data.data_loader()
        logging.info("data loader complete")
        
        data_trainer_config = DataTrainerConfig(train_pipeline_config)
        train_data = TrainData(data_loader_artifact, data_trainer_config)
        train_data.initialize_model()
        data_trainer_artifact = train_data.train_model()
        
        
        
    except Exception as e:
        raise CustomException(e,sys)