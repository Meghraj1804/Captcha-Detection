from dataclasses import dataclass
from torch.utils.data import DataLoader
from torchvision import transforms

@dataclass
class DataIngestionArtifact:
    trained_file_path: str
    test_file_path: str
    val_file_path: str
    
@dataclass
class DataLoaderArtifact:
    train_image_loader : DataLoader
    test_image_loader : DataLoader
    val_image_loader : DataLoader
    image_transformation : transforms.Compose
    
@dataclass
class DataTrainerArtifact:
    model_path: str

    