
import torch
import torch.nn as nn
from torchvision import models

try:
    from efficientnet_pytorch import EfficientNet
except ImportError:
    EfficientNet = None


class SimpleCNN(nn.Module):
    """
    Simple CNN architecture for Gram bacteria classification.
    Used in Scenario 1 (from scratch) and Scenario 2 (with augmentation).
    
    Architecture:
    - 5 Convolutional blocks (Conv2d → BatchNorm2d → ReLU → MaxPool2d)
    - Adaptive Average Pooling
    - 3 Fully Connected layers with Dropout
    
    Parameters: ~27.8 million
    Input size: 224×224×3
    Output: 2 classes
    """
    
    def __init__(self, num_classes: int = 2, dropout_p: float = 0.5):

        super(SimpleCNN, self).__init__()
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # Block 1: 3 → 32 channels
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 224 → 112
            
            # Block 2: 32 → 64 channels
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 112 → 56
            
            # Block 3: 64 → 128 channels
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 56 → 28
            
            # Block 4: 128 → 256 channels
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 28 → 14
            
            # Block 5: 256 → 512 channels
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 14 → 7
        )
        
        # Classifier layers
        self.classifier = nn.Sequential(
            nn.Flatten(),  # Index 0
            nn.Linear(512 * 7 * 7, 1024),  # Index 1: 25088 → 1024
            nn.ReLU(inplace=True),  # Index 2
            nn.Dropout(p=dropout_p),  # Index 3
            nn.Linear(1024, 512),  # Index 4
            nn.ReLU(inplace=True),  # Index 5
            nn.Dropout(p=dropout_p),  # Index 6
            nn.Linear(512, num_classes),  # Index 7
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        x = self.features(x)
        x = self.classifier(x)
        return x


class GramResNet50Classifier(nn.Module):
    """
    ResNet50-based classifier for Gram bacteria classification (Scenario 4a).
    
    Architecture:
    - Backbone: ResNet50 pre-trained on ImageNet
    - Custom classifier head with Dropout layers
    - Output: 2 classes (Gram Negative, Gram Positive)
    - Total Parameters: ~24.6M
    
    This matches the training checkpoint structure with `backbone.*` keys.
    """

    def __init__(self, num_classes: int = 2) -> None:
        """
        Initialize ResNet50 classifier.
        
        Args:
            num_classes (int): Number of output classes (default: 2)
        """
        super().__init__()
        self.backbone = models.resnet50(weights=None)
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 224, 224)
        
        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        return self.backbone(x)


class GramResNet101Classifier(nn.Module):
    """
    ResNet101-based classifier for Gram bacteria classification (Scenario 4b).
    
    Architecture:
    - Backbone: ResNet101 pre-trained on ImageNet (101 layers deep)
    - Custom classifier head with BatchNorm and Dropout
    - Output: 2 classes (Gram Negative, Gram Positive)
    - Total Parameters: ~43.5M
    - Trainable Parameters (fine-tuning last 5 layers): ~18.2M
    
    Key Features:
    - Transfer learning from ImageNet weights
    - Fine-tuning: Last 5 layers of backbone + full classifier
    - Batch Normalization for training stability
    - Dropout (0.5 and 0.3) for regularization
    - Medium-depth classifier: 2048 -> 512 -> 2
    
    Training Strategy:
    - Phase 1: Freeze backbone, train only classifier head
    - Phase 2: Unfreeze last 5 layers (layer4) for fine-tuning
    - Best validation accuracy: 95.26% at epoch 7
    - Test accuracy: 93.74%
    """
    
    def __init__(self, num_classes: int = 2, dropout_p: float = 0.5):
        """
        Initialize ResNet101 classifier.
        
        Args:
            num_classes (int): Number of output classes (default: 2)
            dropout_p (float): Dropout probability for first dropout layer (default: 0.5)
        """
        super(GramResNet101Classifier, self).__init__()
        
        # Load pre-trained ResNet101 (weights will be loaded from checkpoint)
        self.backbone = models.resnet101(weights=None)
        
        # Get the number of input features to the final FC layer
        # ResNet101 outputs 2048 features after global average pooling
        in_features = self.backbone.fc.in_features  # 2048
        
        # Replace the classifier head with custom architecture
        self.backbone.fc = nn.Sequential(
            # First dropout for regularization
            nn.Dropout(p=dropout_p),
            
            # First FC layer: 2048 -> 512
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            
            # Second dropout (lighter)
            nn.Dropout(p=0.3),
            
            # Output layer: 512 -> num_classes
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 224, 224)
        
        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        return self.backbone(x)


class GramVGG16Classifier(nn.Module):
    """
    VGG16-based classifier for Gram bacteria classification (Scenario 5a).
    
    Architecture:
    - Backbone: VGG16 pre-trained on ImageNet
    - Custom classifier head with BatchNorm and Dropout layers
    - Output: 2 classes (Gram Negative, Gram Positive)
    - Total Parameters: ~134.3M
    
    Key Features:
    - Transfer learning from ImageNet weights
    - Deep convolutional architecture (16 layers)
    - Batch Normalization for training stability
    - Dropout (0.5 and 0.3) for regularization
    - Classifier: 25088 -> 4096 -> 512 -> 2
    """
    
    def __init__(self, num_classes: int = 2, dropout_p: float = 0.5):
        """
        Initialize VGG16 classifier.
        
        Args:
            num_classes (int): Number of output classes (default: 2)
            dropout_p (float): Dropout probability for first dropout layer (default: 0.5)
        """
        super(GramVGG16Classifier, self).__init__()
        
        # Load pre-trained VGG16 (weights will be loaded from checkpoint)
        self.backbone = models.vgg16(weights=None)
        
        # VGG16 features output: 7×7×512 = 25,088 features
        in_features = 25088
        
        # Replace the classifier head with custom architecture matching training
        self.backbone.classifier = nn.Sequential(
            # First FC layer: 25088 -> 4096
            nn.Linear(in_features, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            
            # Second FC layer: 4096 -> 512
            nn.Linear(4096, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            
            # Output layer: 512 -> num_classes
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 224, 224)
        
        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        return self.backbone(x)


class GramVGG19Classifier(nn.Module):
    """
    VGG19-based classifier for Gram bacteria classification (Scenario 5b).
    
    Architecture:
    - Backbone: VGG19 pre-trained on ImageNet
    - Custom classifier head with BatchNorm and Dropout layers
    - Output: 2 classes (Gram Negative, Gram Positive)
    - Total Parameters: ~139.6M
    
    Key Features:
    - Transfer learning from ImageNet weights
    - Very deep convolutional architecture (19 layers)
    - Batch Normalization for training stability
    - Dropout (0.5 and 0.3) for regularization
    - Classifier: 25088 -> 4096 -> 512 -> 2
    """
    
    def __init__(self, num_classes: int = 2, dropout_p: float = 0.5):
        """
        Initialize VGG19 classifier.
        
        Args:
            num_classes (int): Number of output classes (default: 2)
            dropout_p (float): Dropout probability for first dropout layer (default: 0.5)
        """
        super(GramVGG19Classifier, self).__init__()
        
        # Load pre-trained VGG19 (weights will be loaded from checkpoint)
        self.backbone = models.vgg19(weights=None)
        
        # VGG19 features output: 7×7×512 = 25,088 features
        in_features = 25088
        
        # Replace the classifier head with custom architecture matching training
        self.backbone.classifier = nn.Sequential(
            # First FC layer: 25088 -> 4096
            nn.Linear(in_features, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            
            # Second FC layer: 4096 -> 512
            nn.Linear(4096, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            
            # Output layer: 512 -> num_classes
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 224, 224)
        
        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        return self.backbone(x)


class GramDenseNet121Classifier(nn.Module):
    """
    DenseNet121-based classifier for Gram bacteria classification.

    Architecture:
    - Backbone: DenseNet121
    - Custom classifier: 1024 -> 512 -> 2
    - Output: 2 classes (Gram Negative, Gram Positive)
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = False) -> None:
        super().__init__()
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.densenet121(weights=weights)
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class GramEfficientNetB0Classifier(nn.Module):
    """
    EfficientNet-B0-based classifier for Gram bacteria classification.
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = False) -> None:
        super().__init__()
        if EfficientNet is None:
            raise ImportError("efficientnet_pytorch is required for GramEfficientNetB0Classifier")

        self.backbone = (
            EfficientNet.from_pretrained("efficientnet-b0")
            if pretrained
            else EfficientNet.from_name("efficientnet-b0")
        )
        in_features = self.backbone._fc.in_features
        self.backbone._fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class GramEfficientNetB3Classifier(nn.Module):
    """
    EfficientNet-B3-based classifier for Gram bacteria classification.
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = False) -> None:
        super().__init__()
        if EfficientNet is None:
            raise ImportError("efficientnet_pytorch is required for GramEfficientNetB3Classifier")

        self.backbone = (
            EfficientNet.from_pretrained("efficientnet-b3")
            if pretrained
            else EfficientNet.from_name("efficientnet-b3")
        )
        in_features = self.backbone._fc.in_features
        self.backbone._fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


# Model metadata for external reference
MODEL_INFO = {
    "SimpleCNN": {
        "parameters": 27787522,
        "architecture": "5 Conv Blocks + 3 FC Layers",
        "description": "Custom CNN trained from scratch"
    },
    "GramResNet50Classifier": {
        "parameters": 24558146,
        "architecture": "ResNet50 + Custom Head",
        "description": "Transfer learning with ResNet50"
    },
    "GramResNet101Classifier": {
        "parameters": 43550274,
        "architecture": "ResNet101 + Custom Head",
        "description": "Transfer learning with ResNet101"
    },
    "GramVGG16Classifier": {
        "parameters": 134301514,
        "architecture": "VGG16 + Custom Head",
        "description": "Transfer learning with VGG16"
    },
    "GramVGG19Classifier": {
        "parameters": 139611210,
        "architecture": "VGG19 + Custom Head",
        "description": "Transfer learning with VGG19"
    },
    "GramDenseNet121Classifier": {
        "parameters": 7713410,
        "architecture": "DenseNet121 + Custom Head",
        "description": "Transfer learning with DenseNet121"
    },
    "GramEfficientNetB0Classifier": {
        "parameters": 5288546,
        "architecture": "EfficientNet-B0 + Custom Head",
        "description": "Transfer learning with EfficientNet-B0"
    },
    "GramEfficientNetB3Classifier": {
        "parameters": 12398130,
        "architecture": "EfficientNet-B3 + Custom Head",
        "description": "Transfer learning with EfficientNet-B3"
    }
}
