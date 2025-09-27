import sys
import os
import torch
from torchvision import transforms
from PIL import Image
import cv2

from src.exception.exception import CustomException
from src.logging.logger import logging

from src.constants import training_pipeline
from src.model_components.model_architecture import CustomCNNModel

class ModelInference:
    def __init__(self,path):
        self.model_path = path
        self.classes = training_pipeline.DATA_TRAINER_CLASS
        self.idx2char = {i: ch for i, ch in enumerate(self.classes)}
        self.char2idx: dict = {ch:i for i, ch in enumerate(self.classes)}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        
        
        
        self.model = CustomCNNModel(input_height=training_pipeline.DATA_TRAINER_IMAGE_HEIGHT,
                                    input_width=training_pipeline.DATA_TRAINER_IMAGE_WIDTH,
                                    num_classes=len(self.classes),
                                    num_char=training_pipeline.NUMBER_OF_CHARACTERS)
        
        self.model.load_state_dict(torch.load(self.model_path,map_location=self.device))
        
        self.transform = transforms.Compose([
                        transforms.Resize(training_pipeline.DATA_AUGMENTATION_IMAGE_SIZE),     
                        transforms.ColorJitter(training_pipeline.DATA_AUGMENTATION_IMAGE_BRIGHTNESS,
                                               training_pipeline.DATA_AUGMENTATION_IMAGE_CONTRAST),     
                        transforms.RandomRotation(training_pipeline.DATA_AUGMENTATION_IMAGE_ROTATION),
                        transforms.RandomPerspective(training_pipeline.DATA_AUGMENTATION_IMAGE_DISTORTION_SCALE,
                                                    training_pipeline.DATA_AUGMENTATION_IMAGE_P),
                        transforms.ToTensor()
            
                                    ])
        
    def predict(self, image_path):
        image = Image.open(image_path)
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(image_tensor)
            predicted_indices = [torch.argmax(logit, dim=1).item() for logit in output]
            predicted_text = ''.join([self.idx2char[i] for i in predicted_indices])
    
            print('output = ',predicted_text)
            
        # label = self.class_name[predicted.item()]
        
        img = cv2.imread(image_path)
        
        # cv2.putText(img, predicted_text,(10,30), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0),2)
        output_path = 'outputs/labeled_image.jpg'
        cv2.imwrite(output_path, img)
        cwd = os.getcwd()
        output_path = os.path.join(cwd, output_path)
        
        return predicted_text, output_path
        