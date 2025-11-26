"""
MLflow Logger for OCR Training
Integrates with DAGsHub for remote tracking
"""

import mlflow
import mlflow.pytorch
import os
from typing import Dict, Any
import matplotlib.pyplot as plt
import numpy as np
import torch


class MLflowLogger:
    """
    MLflow experiment tracking logger
    """
    
    def __init__(self, experiment_name, tracking_uri=None, 
                 dagshub_repo=None, dagshub_username=None):
        """
        Initialize MLflow logger
        
        Args:
            experiment_name (str): Name of the MLflow experiment
            tracking_uri (str): MLflow tracking URI (optional)
            dagshub_repo (str): DAGsHub repository name (optional)
            dagshub_username (str): DAGsHub username (optional)
        """
        self.experiment_name = experiment_name
        
        # Setup DAGsHub tracking if credentials provided
        if dagshub_repo and dagshub_username:
            self._setup_dagshub(dagshub_username, dagshub_repo)
        elif tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        # Set or create experiment
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                mlflow.create_experiment(experiment_name)
            mlflow.set_experiment(experiment_name)
        except Exception as e:
            print(f"Warning: Could not set experiment: {e}")
            print("Using default experiment")
        
        self.run = None
        self.run_id = None
        
        print(f"✓ MLflow logger initialized for experiment: {experiment_name}")
    
    def _setup_dagshub(self, username, repo):
        """
        Setup DAGsHub remote tracking
        
        Args:
            username (str): DAGsHub username
            repo (str): Repository name
        """
        # Set environment variables for DAGsHub
        dagshub_url = f"https://dagshub.com/{username}/{repo}.mlflow"
        
        os.environ['MLFLOW_TRACKING_URI'] = dagshub_url
        mlflow.set_tracking_uri(dagshub_url)
        
        print(f"✓ DAGsHub tracking configured: {dagshub_url}")
        print(f"  Note: Set MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_PASSWORD")
        print(f"        environment variables with your DAGsHub credentials")
    
    def start_run(self, run_name=None):
        """
        Start a new MLflow run
        
        Args:
            run_name (str): Name for this run (optional)
        """
        self.run = mlflow.start_run(run_name=run_name)
        self.run_id = self.run.info.run_id
        print(f"✓ Started MLflow run: {self.run_id}")
        return self.run
    
    def end_run(self):
        """End the current MLflow run"""
        if self.run:
            mlflow.end_run()
            print(f"✓ Ended MLflow run: {self.run_id}")
            self.run = None
            self.run_id = None
    
    def log_params(self, params: Dict[str, Any]):
        """
        Log parameters to MLflow
        
        Args:
            params (dict): Dictionary of parameters
        """
        try:
            for key, value in params.items():
                mlflow.log_param(key, value)
            print(f"✓ Logged {len(params)} parameters")
        except Exception as e:
            print(f"Warning: Could not log parameters: {e}")
    
    def log_metric(self, key: str, value: float, step: int = None):
        """
        Log a single metric to MLflow
        
        Args:
            key (str): Metric name
            value (float): Metric value
            step (int): Step number (optional)
        """
        try:
            mlflow.log_metric(key, value, step=step)
        except Exception as e:
            print(f"Warning: Could not log metric {key}: {e}")
    
    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """
        Log multiple metrics to MLflow
        
        Args:
            metrics (dict): Dictionary of metrics
            step (int): Step number (optional)
        """
        try:
            mlflow.log_metrics(metrics, step=step)
        except Exception as e:
            print(f"Warning: Could not log metrics: {e}")
    
    def log_model(self, model, artifact_path="model"):
        """
        Log PyTorch model to MLflow
        
        Args:
            model: PyTorch model
            artifact_path (str): Path to save model in MLflow
        """
        try:
            mlflow.pytorch.log_model(model, artifact_path)
            print(f"✓ Logged model to MLflow: {artifact_path}")
        except Exception as e:
            print(f"Warning: Could not log model: {e}")
    
    def log_artifact(self, local_path, artifact_path=None):
        """
        Log a file as an artifact to MLflow
        
        Args:
            local_path (str): Local path to file
            artifact_path (str): Path in MLflow artifacts (optional)
        """
        try:
            mlflow.log_artifact(local_path, artifact_path)
            print(f"✓ Logged artifact: {local_path}")
        except Exception as e:
            print(f"Warning: Could not log artifact: {e}")
    
    def log_figure(self, figure, filename):
        """
        Log a matplotlib figure to MLflow
        
        Args:
            figure: Matplotlib figure
            filename (str): Filename to save as
        """
        try:
            mlflow.log_figure(figure, filename)
            print(f"✓ Logged figure: {filename}")
        except Exception as e:
            print(f"Warning: Could not log figure: {e}")
    
    def log_dict(self, dictionary, filename):
        """
        Log a dictionary as JSON to MLflow
        
        Args:
            dictionary (dict): Dictionary to log
            filename (str): Filename to save as
        """
        try:
            mlflow.log_dict(dictionary, filename)
            print(f"✓ Logged dictionary: {filename}")
        except Exception as e:
            print(f"Warning: Could not log dictionary: {e}")
    
    def log_text(self, text, filename):
        """
        Log text to MLflow
        
        Args:
            text (str): Text content
            filename (str): Filename to save as
        """
        try:
            mlflow.log_text(text, filename)
            print(f"✓ Logged text: {filename}")
        except Exception as e:
            print(f"Warning: Could not log text: {e}")
    
    def log_training_plots(self, train_losses, val_losses, train_cers, val_cers):
        """
        Create and log training plots
        
        Args:
            train_losses (list): Training losses per epoch
            val_losses (list): Validation losses per epoch
            train_cers (list): Training CERs per epoch
            val_cers (list): Validation CERs per epoch
        """
        try:
            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Plot losses
            epochs = range(1, len(train_losses) + 1)
            ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
            ax1.plot(epochs, val_losses, 'r-', label='Val Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training and Validation Loss')
            ax1.legend()
            ax1.grid(True)
            
            # Plot CER
            ax2.plot(epochs, train_cers, 'b-', label='Train CER')
            ax2.plot(epochs, val_cers, 'r-', label='Val CER')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('CER')
            ax2.set_title('Training and Validation CER')
            ax2.legend()
            ax2.grid(True)
            
            plt.tight_layout()
            
            # Log the figure
            self.log_figure(fig, "training_curves.png")
            
            plt.close(fig)
        except Exception as e:
            print(f"Warning: Could not log training plots: {e}")
    
    def log_prediction_samples(self, images, predictions, targets, num_samples=5):
        """
        Log sample predictions with images
        
        Args:
            images (torch.Tensor): Batch of images
            predictions (list): List of predicted strings
            targets (list): List of target strings
            num_samples (int): Number of samples to log
        """
        try:
            num_samples = min(num_samples, len(predictions))
            
            fig, axes = plt.subplots(num_samples, 1, figsize=(10, 2 * num_samples))
            if num_samples == 1:
                axes = [axes]
            
            for i in range(num_samples):
                # Convert tensor to numpy for display
                img = images[i].cpu().numpy().transpose(1, 2, 0)
                
                # Denormalize if needed
                img = (img * 255).astype(np.uint8)
                
                axes[i].imshow(img)
                axes[i].axis('off')
                axes[i].set_title(f"Pred: {predictions[i]}\nTrue: {targets[i]}", 
                                 fontsize=10)
            
            plt.tight_layout()
            self.log_figure(fig, "prediction_samples.png")
            plt.close(fig)
        except Exception as e:
            print(f"Warning: Could not log prediction samples: {e}")
    
    def set_tags(self, tags: Dict[str, Any]):
        """
        Set tags for the current run
        
        Args:
            tags (dict): Dictionary of tags
        """
        try:
            for key, value in tags.items():
                mlflow.set_tag(key, value)
            print(f"✓ Set {len(tags)} tags")
        except Exception as e:
            print(f"Warning: Could not set tags: {e}")


def setup_mlflow_dagshub(username, repo, experiment_name):
    """
    Convenience function to setup MLflow with DAGsHub
    
    Args:
        username (str): DAGsHub username
        repo (str): Repository name
        experiment_name (str): Experiment name
        
    Returns:
        MLflowLogger: Configured logger instance
    """
    logger = MLflowLogger(
        experiment_name=experiment_name,
        dagshub_repo=repo,
        dagshub_username=username
    )
    return logger


if __name__ == "__main__":
    # Test MLflow logger
    print("Testing MLflow Logger...")
    
    # Create logger (will use local tracking if DAGsHub not configured)
    logger = MLflowLogger(experiment_name="test_experiment")
    
    # Start a run
    logger.start_run(run_name="test_run")
    
    # Log some parameters
    params = {
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 10,
        "model": "CRNN"
    }
    logger.log_params(params)
    
    # Log some metrics
    for epoch in range(5):
        metrics = {
            "train_loss": 1.0 / (epoch + 1),
            "val_loss": 1.2 / (epoch + 1),
            "train_cer": 0.5 / (epoch + 1),
            "val_cer": 0.6 / (epoch + 1)
        }
        logger.log_metrics(metrics, step=epoch)
    
    # Log training plots
    train_losses = [1.0, 0.8, 0.6, 0.5, 0.4]
    val_losses = [1.2, 0.9, 0.7, 0.6, 0.5]
    train_cers = [0.5, 0.4, 0.3, 0.25, 0.2]
    val_cers = [0.6, 0.5, 0.35, 0.3, 0.25]
    
    logger.log_training_plots(train_losses, val_losses, train_cers, val_cers)
    
    # Set some tags
    tags = {
        "framework": "PyTorch",
        "task": "OCR",
        "dataset": "IAM"
    }
    logger.set_tags(tags)
    
    # End run
    logger.end_run()
    
    print("\n✓ MLflow logger test passed!")
    print(f"  Check mlruns/ directory for logged artifacts")

