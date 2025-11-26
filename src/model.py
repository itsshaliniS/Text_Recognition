"""
CRNN Model Architecture for Handwritten Text Recognition
Combines ResNet18 encoder with BiLSTM and CTC Loss
"""

import torch
import torch.nn as nn
import torchvision.models as models


class CRNN(nn.Module):
    """
    Convolutional Recurrent Neural Network for OCR
    Architecture:
        - ResNet18 as CNN backbone (feature extractor)
        - BiLSTM for sequence modeling
        - Fully connected layer for character prediction
    """
    
    def __init__(self, num_classes, input_height=32, hidden_size=256, num_lstm_layers=2):
        """
        Initialize CRNN model
        
        Args:
            num_classes (int): Number of output classes (alphabet size + blank)
            input_height (int): Height of input images (default: 32)
            hidden_size (int): Hidden size for LSTM layers (default: 256)
            num_lstm_layers (int): Number of stacked LSTM layers (default: 2)
        """
        super(CRNN, self).__init__()
        
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        
        # Use ResNet18 as backbone encoder
        resnet = models.resnet18(pretrained=True)
        
        # Remove the average pooling and fully connected layers
        # We keep only the convolutional features
        modules = list(resnet.children())[:-2]
        self.cnn = nn.Sequential(*modules)
        
        # Adaptive pooling to get fixed height features
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, None))
        
        # Calculate the number of features from ResNet18
        # ResNet18 outputs 512 channels
        self.cnn_output_channels = 512
        
        # Bidirectional LSTM for sequence modeling
        self.rnn = nn.LSTM(
            input_size=self.cnn_output_channels,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            bidirectional=True,
            batch_first=False,
            dropout=0.3 if num_lstm_layers > 1 else 0
        )
        
        # Fully connected layer for character prediction
        # Bidirectional LSTM outputs hidden_size * 2
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            log_probs: Log probabilities of shape (seq_len, batch_size, num_classes)
        """
        # CNN feature extraction
        # Input: (batch, channels, height, width)
        conv_features = self.cnn(x)
        # Output: (batch, 512, h, w)
        
        # Adaptive pooling to reduce height to 1
        # This gives us a sequence of features
        pooled = self.adaptive_pool(conv_features)
        # Output: (batch, 512, 1, w)
        
        # Squeeze height dimension and permute for LSTM
        # (batch, channels, 1, width) -> (batch, channels, width)
        pooled = pooled.squeeze(2)
        # (batch, channels, width) -> (width, batch, channels)
        pooled = pooled.permute(2, 0, 1)
        # Output: (seq_len, batch, channels)
        
        # LSTM sequence modeling
        # self.rnn.flatten_parameters()  # Optional: for optimization
        rnn_output, _ = self.rnn(pooled)
        # Output: (seq_len, batch, hidden_size * 2)
        
        # Fully connected layer for character prediction
        output = self.fc(rnn_output)
        # Output: (seq_len, batch, num_classes)
        
        # Apply log softmax for CTC Loss
        log_probs = nn.functional.log_softmax(output, dim=2)
        
        return log_probs


class CTCLoss(nn.Module):
    """
    CTC Loss wrapper for training
    """
    
    def __init__(self, blank=0, reduction='mean'):
        """
        Initialize CTC Loss
        
        Args:
            blank (int): Index of blank label (default: 0)
            reduction (str): Reduction method ('mean', 'sum', 'none')
        """
        super(CTCLoss, self).__init__()
        self.ctc_loss = nn.CTCLoss(blank=blank, reduction=reduction, zero_infinity=True)
    
    def forward(self, log_probs, targets, input_lengths, target_lengths):
        """
        Compute CTC loss
        
        Args:
            log_probs: Model output (seq_len, batch, num_classes)
            targets: Target sequences (batch, max_target_len) or flattened
            input_lengths: Length of each sequence in log_probs (batch,)
            target_lengths: Length of each target sequence (batch,)
            
        Returns:
            loss: CTC loss value
        """
        return self.ctc_loss(log_probs, targets, input_lengths, target_lengths)


def get_model(num_classes, device='cuda'):
    """
    Factory function to create and initialize CRNN model
    
    Args:
        num_classes (int): Number of output classes
        device (str): Device to load model on ('cuda' or 'cpu')
        
    Returns:
        model: Initialized CRNN model
        criterion: CTC loss function
    """
    model = CRNN(num_classes=num_classes, hidden_size=256, num_lstm_layers=2)
    model = model.to(device)
    
    criterion = CTCLoss(blank=0)
    
    return model, criterion


if __name__ == "__main__":
    # Test model architecture
    print("Testing CRNN Model Architecture...")
    
    # Parameters
    batch_size = 4
    num_classes = 80  # Example: 26 letters + 10 digits + symbols + blank
    height = 32
    width = 128
    
    # Create dummy input
    x = torch.randn(batch_size, 3, height, width)
    
    # Create model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, criterion = get_model(num_classes, device)
    
    x = x.to(device)
    
    # Forward pass
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: (seq_len, batch_size, num_classes)")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Device: {device}")
    print("\n✓ Model architecture test passed!")

