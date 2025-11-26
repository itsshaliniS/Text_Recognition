"""
Training Script for CRNN OCR Model
Includes MLflow tracking, checkpointing, and evaluation
"""

import os
import sys
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import pickle

# Import local modules
from model import CRNN, CTCLoss
from utils import (
    get_default_charset, 
    ctc_decode, 
    calculate_cer, 
    calculate_wer,
    save_model,
    load_model
)
from dataloader import get_dataloaders
from mlflow_logger import MLflowLogger


class Trainer:
    """
    Trainer class for CRNN OCR model
    """
    
    def __init__(self, config):
        """
        Initialize trainer
        
        Args:
            config (dict): Configuration dictionary
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Using device: {self.device}")
        
        # Initialize charset
        self.charset = get_default_charset()
        self.num_classes = self.charset.get_num_classes()
        
        print(f"Number of classes: {self.num_classes}")
        
        # Initialize model
        self.model = CRNN(
            num_classes=self.num_classes,
            hidden_size=config.get('hidden_size', 256),
            num_lstm_layers=config.get('num_lstm_layers', 2)
        ).to(self.device)
        
        print(f"Model initialized with {sum(p.numel() for p in self.model.parameters()):,} parameters")
        
        # Initialize criterion
        self.criterion = CTCLoss(blank=0)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.get('learning_rate', 0.001),
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
        # Initialize scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=3, 
            verbose=True
        )
        
        # Initialize dataloaders
        self.train_loader, self.val_loader = get_dataloaders(
            data_dir=config.get('data_dir', '../data'),
            train_annotation=config.get('train_annotation', '../data/train.txt'),
            val_annotation=config.get('val_annotation', '../data/val.txt'),
            charset_mapper=self.charset,
            batch_size=config.get('batch_size', 32),
            num_workers=config.get('num_workers', 4),
            img_height=config.get('img_height', 32),
            img_width=config.get('img_width', 128)
        )
        
        # Initialize MLflow logger
        self.mlflow_logger = MLflowLogger(
            experiment_name=config.get('experiment_name', 'OCR-CRNN'),
            dagshub_username=config.get('dagshub_username'),
            dagshub_repo=config.get('dagshub_repo')
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_cers = []
        self.val_cers = []
        
        self.best_val_cer = float('inf')
        self.start_epoch = 0
    
    def train_epoch(self, epoch):
        """
        Train for one epoch
        
        Args:
            epoch (int): Current epoch number
            
        Returns:
            tuple: (average_loss, average_cer)
        """
        self.model.train()
        
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]")
        
        for batch_idx, (images, texts, text_encoded, text_lengths) in enumerate(pbar):
            # Move to device
            images = images.to(self.device)
            text_encoded = text_encoded.to(self.device)
            text_lengths = text_lengths.to(self.device)
            
            # Forward pass
            log_probs = self.model(images)
            
            # Calculate sequence lengths
            batch_size = images.size(0)
            input_lengths = torch.full(
                size=(batch_size,), 
                fill_value=log_probs.size(0), 
                dtype=torch.long
            )
            
            # Calculate CTC loss
            loss = self.criterion(log_probs, text_encoded, input_lengths, text_lengths)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            
            self.optimizer.step()
            
            # Track loss
            total_loss += loss.item()
            
            # Decode predictions for CER calculation
            with torch.no_grad():
                predictions = ctc_decode(log_probs, input_lengths, self.charset)
                all_predictions.extend(predictions)
                all_targets.extend(texts)
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Calculate average loss and CER
        avg_loss = total_loss / len(self.train_loader)
        avg_cer = calculate_cer(all_predictions, all_targets)
        
        return avg_loss, avg_cer
    
    def validate(self, epoch):
        """
        Validate the model
        
        Args:
            epoch (int): Current epoch number
            
        Returns:
            tuple: (average_loss, average_cer, predictions, targets)
        """
        self.model.eval()
        
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1} [Val]")
        
        with torch.no_grad():
            for batch_idx, (images, texts, text_encoded, text_lengths) in enumerate(pbar):
                # Move to device
                images = images.to(self.device)
                text_encoded = text_encoded.to(self.device)
                text_lengths = text_lengths.to(self.device)
                
                # Forward pass
                log_probs = self.model(images)
                
                # Calculate sequence lengths
                batch_size = images.size(0)
                input_lengths = torch.full(
                    size=(batch_size,), 
                    fill_value=log_probs.size(0), 
                    dtype=torch.long
                )
                
                # Calculate CTC loss
                loss = self.criterion(log_probs, text_encoded, input_lengths, text_lengths)
                
                # Track loss
                total_loss += loss.item()
                
                # Decode predictions
                predictions = ctc_decode(log_probs, input_lengths, self.charset)
                all_predictions.extend(predictions)
                all_targets.extend(texts)
                
                # Update progress bar
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Calculate average loss and CER
        avg_loss = total_loss / len(self.val_loader)
        avg_cer = calculate_cer(all_predictions, all_targets)
        avg_wer = calculate_wer(all_predictions, all_targets)
        
        return avg_loss, avg_cer, avg_wer, all_predictions, all_targets
    
    def train(self):
        """
        Main training loop
        """
        num_epochs = self.config.get('num_epochs', 20)
        
        # Start MLflow run
        self.mlflow_logger.start_run(run_name=self.config.get('run_name'))
        
        # Log parameters
        params = {
            'learning_rate': self.config.get('learning_rate', 0.001),
            'batch_size': self.config.get('batch_size', 32),
            'num_epochs': num_epochs,
            'hidden_size': self.config.get('hidden_size', 256),
            'num_lstm_layers': self.config.get('num_lstm_layers', 2),
            'img_height': self.config.get('img_height', 32),
            'img_width': self.config.get('img_width', 128),
            'num_classes': self.num_classes,
            'device': str(self.device)
        }
        self.mlflow_logger.log_params(params)
        
        # Set tags
        tags = {
            'model': 'CRNN',
            'framework': 'PyTorch',
            'task': 'OCR',
            'dataset': 'IAM'
        }
        self.mlflow_logger.set_tags(tags)
        
        print(f"\n{'='*60}")
        print(f"Starting training for {num_epochs} epochs")
        print(f"{'='*60}\n")
        
        for epoch in range(self.start_epoch, num_epochs):
            # Train
            train_loss, train_cer = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_cer, val_wer, predictions, targets = self.validate(epoch)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Store history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_cers.append(train_cer)
            self.val_cers.append(val_cer)
            
            # Log metrics to MLflow
            metrics = {
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_cer': train_cer,
                'val_cer': val_cer,
                'val_wer': val_wer,
                'learning_rate': self.optimizer.param_groups[0]['lr']
            }
            self.mlflow_logger.log_metrics(metrics, step=epoch)
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{num_epochs}:")
            print(f"  Train Loss: {train_loss:.4f} | Train CER: {train_cer:.4f}")
            print(f"  Val Loss:   {val_loss:.4f} | Val CER:   {val_cer:.4f} | Val WER: {val_wer:.4f}")
            
            # Save checkpoint
            checkpoint_dir = self.config.get('checkpoint_dir', '../models')
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt')
            save_model(self.model, self.optimizer, epoch, val_loss, checkpoint_path)
            
            # Save best model
            if val_cer < self.best_val_cer:
                self.best_val_cer = val_cer
                best_model_path = os.path.join(checkpoint_dir, 'best_model.pt')
                save_model(self.model, self.optimizer, epoch, val_loss, best_model_path)
                print(f"  ✓ New best model saved! (CER: {val_cer:.4f})")
                
                # Save as pickle for easy loading in Flask
                best_model_pkl = os.path.join(checkpoint_dir, 'best_model.pkl')
                with open(best_model_pkl, 'wb') as f:
                    pickle.dump({
                        'model_state_dict': self.model.state_dict(),
                        'num_classes': self.num_classes,
                        'charset': self.charset,
                        'config': self.config
                    }, f)
                print(f"  ✓ Model saved as pickle: {best_model_pkl}")
        
        # Log final training plots
        self.mlflow_logger.log_training_plots(
            self.train_losses, 
            self.val_losses, 
            self.train_cers, 
            self.val_cers
        )
        
        # Log final model
        best_model_path = os.path.join(
            self.config.get('checkpoint_dir', '../models'), 
            'best_model.pkl'
        )
        if os.path.exists(best_model_path):
            self.mlflow_logger.log_artifact(best_model_path)
        
        # End MLflow run
        self.mlflow_logger.end_run()
        
        print(f"\n{'='*60}")
        print(f"Training completed!")
        print(f"Best validation CER: {self.best_val_cer:.4f}")
        print(f"{'='*60}\n")


def main():
    """
    Main function
    """
    # Configuration
    config = {
        # Data
        'data_dir': '../data',
        'train_annotation': '../data/train.txt',
        'val_annotation': '../data/val.txt',
        
        # Model
        'hidden_size': 256,
        'num_lstm_layers': 2,
        
        # Training
        'num_epochs': 20,
        'batch_size': 32,
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        
        # Data loading
        'num_workers': 4,
        'img_height': 32,
        'img_width': 128,
        
        # Checkpointing
        'checkpoint_dir': '../models',
        
        # MLflow
        'experiment_name': 'OCR-CRNN',
        'run_name': 'crnn_resnet18_bilstm',
        
        # DAGsHub (optional - set these if you want to use DAGsHub)
        'dagshub_username': None,  # e.g., 'your_username'
        'dagshub_repo': None,      # e.g., 'ocr-project'
    }
    
    # Create trainer
    trainer = Trainer(config)
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()

