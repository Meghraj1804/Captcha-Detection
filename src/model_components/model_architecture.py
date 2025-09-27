import torch 
import torch.nn as nn
import torch.optim as optim



class CustomCNNModel(nn.Module):
    def __init__(self,input_height:int, input_width:int, num_classes:int, num_char:int):
        super(CustomCNNModel,self).__init__()
        self.input_height = input_height
        self.input_width = input_width
        self.num_classes = num_classes 
        self.num_char = num_char

        self.conv_layers = nn.Sequential(
            
            # Convolution Layer 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Convolution Layer 2
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Convolution Layer 3
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Convolution Layer 4
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten()
            
        )
        
        self.to_linear = None
        self._get_conv_output(self.input_height, self.input_width)
        
        self.classifiers = nn.ModuleList([
                        nn.Sequential(
                            nn.Linear(self.to_linear, 512),
                            nn.ReLU(),
                            nn.Linear(512, 128),
                            nn.ReLU(),
                            nn.Linear(128, self.num_classes)
                        ) for _ in range(self.num_char)
                            ])
        
    def _get_conv_output(self, input_height, input_width):
        with torch.no_grad():
            dummy_input = torch.zeros(1,3,input_height, input_width)
            output = self.conv_layers(dummy_input)
            self.to_linear = output.view(1,-1).size(1)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return [classifier(x) for classifier in self.classifiers]


class LoadModel:
    def __init__(self,input_height, input_width, num_classes, num_char, learning_rate):
        self.model = CustomCNNModel(input_height, input_width, num_classes, num_char)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        

