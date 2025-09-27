import sys
import os


from src.exception.exception import CustomException
from src.logging.logger import logging

from src.entity.config_entity import  DataTrainerConfig
from src.entity.artifact_entity import  DataLoaderArtifact, DataTrainerArtifact
from src.model_components.model_architecture import LoadModel
from src.model_components.training import TrainModel
from src.utils.main_utils import plot_training_summary

class TrainData:
    def __init__(self, data_loader_artifact: DataLoaderArtifact,
                 data_trainer_config: DataTrainerConfig,):
        
        try:
            self.data_loader_artifact = data_loader_artifact
            self.data_trainer_config = data_trainer_config
            
            
        except Exception as e:
            raise CustomException(e, sys)
        
    def initialize_model(self):
        try:
            components = LoadModel(input_height=self.data_trainer_config.image_height,
                                        input_width=self.data_trainer_config.image_width,
                                        num_classes=self.data_trainer_config.num_class,
                                        num_char=self.data_trainer_config.num_chars,
                                        learning_rate=self.data_trainer_config.learning_rate)
            
            self.model = components.model
            self.criterion = components.criterion
            self.optimizer = components.optimizer
            
            logging.info("Model intialize successfully")
            
        except Exception as e:
            raise CustomException(e, sys)
        
    def train_model(self) -> DataTrainerArtifact:
        try:
            num_epochs = self.data_trainer_config.epoch
            train_losses = []
            val_losses = []
            val_accuracies = []

            best_val_loss = float('inf')
            best_model_state = None
            
            model_train = TrainModel(model=self.model,
                                     criterion=self.criterion,
                                     optimizer=self.optimizer)
            
            logging.info("Training started")
            for epoch in range(num_epochs):
                print('epoch = ',epoch)
                train_loss = model_train.train(dataloader=self.data_loader_artifact.train_image_loader)
                val_loss, val_accuracy = model_train.validate(dataloader=self.data_loader_artifact.test_image_loader)
                
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                val_accuracies.append(val_accuracy)
            
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = self.model.state_dict()
                    
            
            dir_path = os.path.dirname(self.data_trainer_config.model_dir)
            os.makedirs(dir_path, exist_ok=True)      
            model_train.save_model(best_model_state, self.data_trainer_config.model_dir)
            plot_training_summary(
                train_losses, val_losses, val_accuracies, 
                os.path.join(self.data_trainer_config.model_summery, "training_summary.png")
            )
            
            data_trainer_artifact = DataTrainerArtifact(
                model_path = self.data_trainer_config.model_dir
            )
            
            logging.info('Model Training done ')
            logging.info(f"Model has been saved at {self.data_trainer_config.model_dir}")
            
            model_train.testing(data=self.data_loader_artifact.test_image_loader,
                                idx2char=self.data_trainer_config.idx2char,
                                path=os.path.join(self.data_trainer_config.model_summery, "test_samples.png"))
            
            
            
            return data_trainer_artifact
        except Exception as e:
            raise CustomException(e, sys)
        