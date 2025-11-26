"""
DataLoader for IAM Handwriting Database
Handles data loading, preprocessing, and augmentation
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import xml.etree.ElementTree as ET
from typing import List, Tuple
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


class IAMDataset(Dataset):
    """
    IAM Handwriting Dataset
    
    Expected structure:
        data/
            words/
                a01/
                    a01-000u/
                        a01-000u-00-00.png
                        ...
            words.txt (or alternative annotation file)
    """
    
    def __init__(self, data_dir, annotation_file, charset_mapper, 
                 transform=None, max_samples=None, img_height=32, img_width=128):
        """
        Initialize IAM Dataset
        
        Args:
            data_dir (str): Root directory of IAM dataset
            annotation_file (str): Path to annotation file
            charset_mapper: CharsetMapper instance
            transform: Albumentations transform pipeline
            max_samples (int): Maximum number of samples to load (for debugging)
            img_height (int): Target image height
            img_width (int): Target image width
        """
        self.data_dir = data_dir
        self.charset_mapper = charset_mapper
        self.transform = transform
        self.img_height = img_height
        self.img_width = img_width
        
        # Load annotations
        self.samples = self._load_annotations(annotation_file, max_samples)
        
        print(f"Loaded {len(self.samples)} samples from dataset")
    
    def _load_annotations(self, annotation_file, max_samples):
        """
        Load annotations from file
        
        For IAM dataset, the words.txt file has format:
        # word_id status gray_level x y w h grammar_tag transcription
        """
        samples = []
        
        if not os.path.exists(annotation_file):
            print(f"Warning: Annotation file {annotation_file} not found!")
            print("Creating dummy dataset for testing purposes...")
            return self._create_dummy_dataset(max_samples or 100)
        
        with open(annotation_file, 'r', encoding='utf-8') as f:
            for line in f:
                # Skip comments and empty lines
                if line.startswith('#') or not line.strip():
                    continue
                
                parts = line.strip().split()
                if len(parts) < 9:
                    continue
                
                word_id = parts[0]
                status = parts[1]
                
                # Skip if status indicates issues (err, etc.)
                if status != 'ok':
                    continue
                
                # Get transcription (last part, may contain spaces)
                transcription = ' '.join(parts[8:])
                
                # Build image path from word_id
                # e.g., a01-000u-00-00 -> words/a01/a01-000u/a01-000u-00-00.png
                parts_id = word_id.split('-')
                img_path = os.path.join(
                    self.data_dir, 
                    'words',
                    parts_id[0],
                    f"{parts_id[0]}-{parts_id[1]}",
                    f"{word_id}.png"
                )
                
                if os.path.exists(img_path):
                    samples.append({
                        'image_path': img_path,
                        'text': transcription,
                        'word_id': word_id
                    })
                
                if max_samples and len(samples) >= max_samples:
                    break
        
        return samples
    
    def _create_dummy_dataset(self, num_samples):
        """
        Create dummy dataset for testing when real data is not available
        """
        samples = []
        dummy_texts = [
            "Hello", "World", "Test", "OCR", "System",
            "Machine", "Learning", "Deep", "Neural", "Network",
            "Python", "PyTorch", "Model", "Training", "Data"
        ]
        
        for i in range(num_samples):
            samples.append({
                'image_path': None,  # Will generate dummy image
                'text': dummy_texts[i % len(dummy_texts)],
                'word_id': f'dummy_{i:04d}'
            })
        
        return samples
    
    def _generate_dummy_image(self, text):
        """
        Generate a dummy image with text for testing
        """
        # Create white image
        img = np.ones((self.img_height, self.img_width, 3), dtype=np.uint8) * 255
        
        # Add some random noise
        noise = np.random.randint(200, 255, (self.img_height, self.img_width, 3), dtype=np.uint8)
        img = cv2.addWeighted(img, 0.7, noise, 0.3, 0)
        
        # Add text (simplified)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, text[:10], (5, 20), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        
        return Image.fromarray(img)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get item by index
        
        Returns:
            image: Preprocessed image tensor (C, H, W)
            text: Target text string
            text_encoded: Encoded text as list of indices
        """
        sample = self.samples[idx]
        
        # Load or generate image
        if sample['image_path'] and os.path.exists(sample['image_path']):
            image = Image.open(sample['image_path']).convert('RGB')
        else:
            image = self._generate_dummy_image(sample['text'])
        
        # Convert to numpy for albumentations
        image = np.array(image)
        
        # Resize to target size
        image = cv2.resize(image, (self.img_width, self.img_height))
        
        # Apply augmentations
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        else:
            # Default normalization
            image = image.astype(np.float32) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1)
        
        # Encode text
        text = sample['text']
        text_encoded = self.charset_mapper.encode(text)
        
        return image, text, text_encoded


def get_train_transform():
    """
    Get training augmentation pipeline
    """
    return A.Compose([
        # Geometric transformations
        A.ShiftScaleRotate(
            shift_limit=0.05, 
            scale_limit=0.1, 
            rotate_limit=5, 
            border_mode=cv2.BORDER_CONSTANT, 
            value=255, 
            p=0.5
        ),
        
        # Elastic transform for handwriting variation
        A.ElasticTransform(
            alpha=1, 
            sigma=20, 
            alpha_affine=10, 
            border_mode=cv2.BORDER_CONSTANT, 
            value=255, 
            p=0.3
        ),
        
        # Grid distortion
        A.GridDistortion(
            num_steps=5, 
            distort_limit=0.2, 
            border_mode=cv2.BORDER_CONSTANT, 
            value=255, 
            p=0.3
        ),
        
        # Blur
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MotionBlur(blur_limit=5, p=1.0),
        ], p=0.3),
        
        # Noise
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        
        # Brightness and contrast
        A.RandomBrightnessContrast(
            brightness_limit=0.2, 
            contrast_limit=0.2, 
            p=0.5
        ),
        
        # Normalization
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        
        # Convert to PyTorch tensor
        ToTensorV2()
    ])


def get_val_transform():
    """
    Get validation transform pipeline (no augmentation)
    """
    return A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def collate_fn(batch):
    """
    Custom collate function for DataLoader
    Handles variable length text sequences
    
    Args:
        batch: List of (image, text, text_encoded) tuples
        
    Returns:
        images: Batched images (batch, C, H, W)
        texts: List of text strings
        text_encoded: Concatenated encoded texts
        text_lengths: Length of each encoded text
    """
    images, texts, text_encoded_list = zip(*batch)
    
    # Stack images
    images = torch.stack(images, dim=0)
    
    # Concatenate text encodings for CTC loss
    text_lengths = torch.LongTensor([len(text) for text in text_encoded_list])
    text_encoded = torch.LongTensor([idx for text in text_encoded_list for idx in text])
    
    return images, list(texts), text_encoded, text_lengths


def get_dataloaders(data_dir, train_annotation, val_annotation, 
                    charset_mapper, batch_size=32, num_workers=4,
                    img_height=32, img_width=128):
    """
    Create train and validation dataloaders
    
    Args:
        data_dir: Root directory of dataset
        train_annotation: Path to training annotation file
        val_annotation: Path to validation annotation file
        charset_mapper: CharsetMapper instance
        batch_size: Batch size
        num_workers: Number of worker processes
        img_height: Target image height
        img_width: Target image width
        
    Returns:
        train_loader, val_loader: DataLoader instances
    """
    # Create datasets
    train_dataset = IAMDataset(
        data_dir=data_dir,
        annotation_file=train_annotation,
        charset_mapper=charset_mapper,
        transform=get_train_transform(),
        img_height=img_height,
        img_width=img_width
    )
    
    val_dataset = IAMDataset(
        data_dir=data_dir,
        annotation_file=val_annotation,
        charset_mapper=charset_mapper,
        transform=get_val_transform(),
        img_height=img_height,
        img_width=img_width
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test dataloader
    print("Testing DataLoader...")
    
    from utils import get_default_charset
    
    # Get charset
    charset = get_default_charset()
    
    # Create dummy dataset
    dataset = IAMDataset(
        data_dir='../data',
        annotation_file='../data/words.txt',  # Will use dummy data if not found
        charset_mapper=charset,
        transform=get_train_transform(),
        max_samples=10
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    # Test single item
    image, text, text_encoded = dataset[0]
    print(f"\nSample 0:")
    print(f"  Image shape: {image.shape}")
    print(f"  Text: {text}")
    print(f"  Encoded length: {len(text_encoded)}")
    
    # Test dataloader
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
    images, texts, text_encoded, text_lengths = next(iter(loader))
    
    print(f"\nBatch:")
    print(f"  Images shape: {images.shape}")
    print(f"  Texts: {texts}")
    print(f"  Text lengths: {text_lengths}")
    
    print("\n✓ DataLoader test passed!")

