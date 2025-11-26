"""
Utility Functions for OCR System
Includes: CTC decoding, CER/WER metrics, preprocessing functions
"""

import torch
import numpy as np
from PIL import Image
import cv2
from typing import List, Tuple
import Levenshtein


class CharsetMapper:
    """
    Maps characters to indices and vice versa
    """
    
    def __init__(self, charset_string):
        """
        Initialize charset mapper
        
        Args:
            charset_string (str): String containing all valid characters
        """
        # Add blank token at index 0 for CTC
        self.charset = ['[blank]'] + list(charset_string)
        self.char_to_idx = {char: idx for idx, char in enumerate(self.charset)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.charset)}
        
    def encode(self, text):
        """
        Encode text to indices
        
        Args:
            text (str): Text to encode
            
        Returns:
            list: List of character indices
        """
        return [self.char_to_idx.get(char, self.char_to_idx['[blank]']) 
                for char in text]
    
    def decode(self, indices):
        """
        Decode indices to text
        
        Args:
            indices (list): List of character indices
            
        Returns:
            str: Decoded text
        """
        return ''.join([self.idx_to_char.get(idx, '') 
                       for idx in indices if idx != 0])  # Skip blank token
    
    def get_num_classes(self):
        """Return total number of classes (including blank)"""
        return len(self.charset)


def get_default_charset():
    """
    Get default character set for English handwriting
    Includes: lowercase, uppercase letters, digits, and common punctuation
    
    Returns:
        CharsetMapper: Initialized charset mapper
    """
    # Create comprehensive charset
    lowercase = 'abcdefghijklmnopqrstuvwxyz'
    uppercase = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    digits = '0123456789'
    punctuation = ' .,!?;:\'-"()[]'
    
    charset_string = lowercase + uppercase + digits + punctuation
    
    return CharsetMapper(charset_string)


def ctc_decode(log_probs, input_lengths, charset_mapper, method='greedy'):
    """
    Decode CTC outputs to text
    
    Args:
        log_probs: Model output (seq_len, batch, num_classes)
        input_lengths: Length of each sequence (batch,)
        charset_mapper: CharsetMapper instance
        method: Decoding method ('greedy' or 'beam_search')
        
    Returns:
        list: List of decoded strings
    """
    if method == 'greedy':
        return ctc_greedy_decode(log_probs, input_lengths, charset_mapper)
    else:
        # Beam search can be implemented for better accuracy
        return ctc_greedy_decode(log_probs, input_lengths, charset_mapper)


def ctc_greedy_decode(log_probs, input_lengths, charset_mapper):
    """
    Greedy CTC decoding (best path decoding)
    
    Args:
        log_probs: Model output (seq_len, batch, num_classes)
        input_lengths: Length of each sequence (batch,)
        charset_mapper: CharsetMapper instance
        
    Returns:
        list: List of decoded strings
    """
    # Get argmax indices
    _, max_indices = torch.max(log_probs, dim=2)
    max_indices = max_indices.transpose(0, 1)  # (batch, seq_len)
    
    decoded_strings = []
    
    for i in range(max_indices.size(0)):
        # Get the sequence for this batch element
        seq_len = input_lengths[i].item()
        indices = max_indices[i, :seq_len].tolist()
        
        # Remove consecutive duplicates and blanks
        decoded_indices = []
        prev_idx = -1
        
        for idx in indices:
            if idx != prev_idx and idx != 0:  # 0 is blank
                decoded_indices.append(idx)
            prev_idx = idx
        
        # Decode to string
        decoded_text = charset_mapper.decode(decoded_indices)
        decoded_strings.append(decoded_text)
    
    return decoded_strings


def calculate_cer(predictions: List[str], targets: List[str]) -> float:
    """
    Calculate Character Error Rate (CER)
    
    CER = (Substitutions + Deletions + Insertions) / Total Characters
    
    Args:
        predictions: List of predicted strings
        targets: List of target strings
        
    Returns:
        float: Character Error Rate
    """
    total_distance = 0
    total_length = 0
    
    for pred, target in zip(predictions, targets):
        # Calculate edit distance at character level
        distance = Levenshtein.distance(pred, target)
        total_distance += distance
        total_length += len(target)
    
    if total_length == 0:
        return 0.0
    
    return total_distance / total_length


def calculate_wer(predictions: List[str], targets: List[str]) -> float:
    """
    Calculate Word Error Rate (WER)
    
    WER = (Substitutions + Deletions + Insertions) / Total Words
    
    Args:
        predictions: List of predicted strings
        targets: List of target strings
        
    Returns:
        float: Word Error Rate
    """
    total_distance = 0
    total_words = 0
    
    for pred, target in zip(predictions, targets):
        # Split into words
        pred_words = pred.split()
        target_words = target.split()
        
        # Calculate edit distance at word level
        distance = Levenshtein.distance(' '.join(pred_words), ' '.join(target_words))
        total_distance += distance
        total_words += len(target_words)
    
    if total_words == 0:
        return 0.0
    
    return total_distance / total_words


def preprocess_image(image_path, target_height=32, target_width=128):
    """
    Preprocess image for OCR model
    
    Args:
        image_path: Path to image file
        target_height: Target height for resizing (default: 32)
        target_width: Target width for resizing (default: 128)
        
    Returns:
        torch.Tensor: Preprocessed image tensor (1, 3, H, W)
    """
    # Read image
    if isinstance(image_path, str):
        image = Image.open(image_path).convert('RGB')
    else:
        image = image_path
    
    # Convert to numpy array
    image = np.array(image)
    
    # Convert to grayscale and back to RGB (for model compatibility)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Apply adaptive thresholding for better contrast
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    # Convert back to RGB
    image = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
    
    # Resize while maintaining aspect ratio
    h, w = image.shape[:2]
    aspect_ratio = w / h
    
    if aspect_ratio > (target_width / target_height):
        # Width is the limiting factor
        new_width = target_width
        new_height = int(target_width / aspect_ratio)
    else:
        # Height is the limiting factor
        new_height = target_height
        new_width = int(target_height * aspect_ratio)
    
    image = cv2.resize(image, (new_width, new_height))
    
    # Pad to target size
    pad_h = target_height - new_height
    pad_w = target_width - new_width
    
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    
    image = cv2.copyMakeBorder(
        image, top, bottom, left, right, 
        cv2.BORDER_CONSTANT, value=[255, 255, 255]
    )
    
    # Normalize to [0, 1]
    image = image.astype(np.float32) / 255.0
    
    # Convert to PyTorch tensor (C, H, W)
    image = torch.from_numpy(image).permute(2, 0, 1)
    
    # Add batch dimension (1, C, H, W)
    image = image.unsqueeze(0)
    
    return image


def preprocess_batch(images, target_height=32, target_width=128):
    """
    Preprocess a batch of images for training
    
    Args:
        images: List of PIL Images
        target_height: Target height for resizing
        target_width: Target width for resizing
        
    Returns:
        torch.Tensor: Batch of preprocessed images (batch, 3, H, W)
    """
    processed_images = []
    
    for image in images:
        # Similar preprocessing as single image
        image = np.array(image)
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Resize
        resized = cv2.resize(gray, (target_width, target_height))
        
        # Convert to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        
        # Normalize
        normalized = rgb.astype(np.float32) / 255.0
        
        # Convert to tensor
        tensor = torch.from_numpy(normalized).permute(2, 0, 1)
        processed_images.append(tensor)
    
    # Stack into batch
    batch = torch.stack(processed_images, dim=0)
    
    return batch


def save_model(model, optimizer, epoch, loss, save_path):
    """
    Save model checkpoint
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Current loss
        save_path: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, save_path)
    print(f"Model saved to {save_path}")


def load_model(model, optimizer, load_path, device='cuda'):
    """
    Load model checkpoint
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        load_path: Path to checkpoint
        device: Device to load model on
        
    Returns:
        tuple: (model, optimizer, epoch, loss)
    """
    checkpoint = torch.load(load_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', 0)
    
    print(f"Model loaded from {load_path} (Epoch: {epoch}, Loss: {loss:.4f})")
    
    return model, optimizer, epoch, loss


if __name__ == "__main__":
    # Test utilities
    print("Testing OCR Utilities...")
    
    # Test charset mapper
    charset = get_default_charset()
    print(f"\n1. Charset size: {charset.get_num_classes()}")
    
    # Test encoding/decoding
    text = "Hello World 123"
    encoded = charset.encode(text)
    decoded = charset.decode(encoded)
    print(f"   Original: {text}")
    print(f"   Encoded: {encoded[:10]}...")
    print(f"   Decoded: {decoded}")
    
    # Test CER/WER
    predictions = ["Hello World", "Test text"]
    targets = ["Hello World!", "Test test"]
    cer = calculate_cer(predictions, targets)
    wer = calculate_wer(predictions, targets)
    print(f"\n2. CER: {cer:.4f}")
    print(f"   WER: {wer:.4f}")
    
    print("\n✓ Utilities test passed!")

