import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.amp import autocast, GradScaler
import os
import json
import time
from sentiment_analysis import w2v_model, text_to_indices, SentimentCNN, preprocess_tweet, EarlyStopping
from scripts.load_data import load_sentiment140_data
from scripts.gpu_usage import print_gpu_memory, print_system_memory, list_tensors

"""
Model Configuration:
- Define hyperparameters for the CNN model:
  - DROPOUT: Dropout rate to prevent overfitting.
  - FILTER_SIZES: Different kernel sizes for convolution layers.
  - NUM_FILTERS: Number of filters per convolution layer.
  - EMBEDDING_DIM: Word embedding vector size (derived from w2v_model).
  - VOCAB_SIZE: Size of the vocabulary (derived from w2v_model).
  - MAX_LENGTH: Maximum token length for input tweets.

Training Configuration:
- Define parameters for model training:
  - BATCH_SIZE: Number of samples per batch during training.
  - EPOCHS: Number of full training cycles over the dataset.
  - LEARNING_RATE: Step size for optimizer updates.
  - SAMPLING_PERCENTAGE: Fraction of the dataset to be used for training.
"""

BATCH_SIZE = 1024
EPOCHS = 30
LEARNING_RATE = 0.0001
DROPOUT = 0.3
NUM_FILTERS = [128, 128, 128]
FILTER_SIZES = [3, 5, 7]
NUM_CLASSES = 1
SAMPLING_PERCENTAGE = 1
MAX_LENGTH = 15

VOCAB_SIZE = len(w2v_model.key_to_index)
EMBEDDING_DIM = w2v_model.vector_size
SAVE_PATH = '../models/CNN_model.pth'

# Function to print gradient magnitudes for monitoring
def print_gradients(model):
    grad_magnitudes = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_magnitudes[name] = param.grad.abs().mean().item()
    print("Average Gradient Magnitudes per Layer:")
    for name, grad in grad_magnitudes.items():
        print(f"  {name}: {grad:.6f}")

# Function to resume training from a saved checkpoint
def resume_training(model, optimizer, scaler, epoch=11):
    if os.path.exists(SAVE_PATH):
        print(f"Resuming training from {SAVE_PATH}...")
        checkpoint = torch.load(SAVE_PATH)

        print("Checkpoint keys:", checkpoint.keys())

        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print("Model state not found, initializing a new model.")

        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            print("Optimizer state not found, reinitializing optimizer.")

        if 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        else:
            print("Scaler state not found, reinitializing scaler.")

        epoch = checkpoint.get('epoch', 0)
        print(f"Resumed from epoch {epoch + 1}")
    else:
        print("No saved model found, starting from scratch.")

    return model, optimizer, scaler, epoch

# Main training loop
def main():
    # Load Sentiment140 dataset
    targets, tweets = load_sentiment140_data()

    # Sample a subset of the dataset
    num_samples = int(len(tweets) * SAMPLING_PERCENTAGE)
    sampled_targets = targets[:num_samples]
    sampled_tweets = tweets[:num_samples]

    # Preprocess and tokenize tweets
    tokenized_tweets = [preprocess_tweet(tweet) for tweet in sampled_tweets]
    tweets_indices = text_to_indices(tokenized_tweets, MAX_LENGTH)
    tweets_indices = np.array(tweets_indices)
    targets = np.array(sampled_targets)
    targets = targets / np.max(targets)

    # Determine the device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Clear CUDA cache to free memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Print system and GPU memory usage
    print_system_memory()
    print_gpu_memory()
    list_tensors()
    print("-" * 60)

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(tweets_indices, targets, test_size=0.2, random_state=42)

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.long).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.long).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float).to(device)

    # Reshape target tensors
    y_train_tensor = y_train_tensor.unsqueeze(1)
    y_test_tensor = y_test_tensor.unsqueeze(1)

    # Create PyTorch datasets and dataloaders
    train_data = TensorDataset(X_train_tensor, y_train_tensor)
    test_data = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

    # Initialize SentimentCNN model
    print("Creating SentimentCNN model...")
    model = SentimentCNN(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        num_classes=NUM_CLASSES,
        num_filters=NUM_FILTERS,
        filter_sizes=FILTER_SIZES,
        dropout=DROPOUT,
        embedding_model=w2v_model
    ).to(device)
    print("Model created successfully.")

    # Define optimizer, loss function, and learning rate scheduler
    optimizer = optim.Adam([{'params': model.conv1.parameters()},
                            {'params': model.conv2.parameters()},
                            {'params': model.conv3.parameters()},
                            {'params': model.fc.parameters(), 'lr': LEARNING_RATE}],
                           lr=LEARNING_RATE, weight_decay=1e-6)
    scaler = GradScaler()
    model, optimizer, scaler, epoch = resume_training(model, optimizer, scaler)

    objective_function = torch.nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
    early_stopping = EarlyStopping(patience=6, delta=0.001, save_path=SAVE_PATH)

    # Training loop
    for epoch in range(epoch, EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}...")
        model.train()
        epoch_loss = 0
        batch_losses = []

        for batch_idx, (data, target) in enumerate(train_loader):
            print("-" * 60)
            data, target = data.to(device), target.to(device)

            start_time = time.time()
            optimizer.zero_grad()

            with autocast("cuda"):
                output = model(data)
                loss = objective_function(output, target)

            scaler.scale(loss).backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            print("Gradients after backward pass:")
            print_gradients(model)

            scaler.step(optimizer)
            scaler.update()

            mean_logits = output.mean().item()
            var_logits = output.var().item()
            sigmoid_mean = torch.sigmoid(output.mean()).item()

            batch_losses.append(loss.item())
            epoch_loss += loss.item()

            end_time = time.time()
            batch_time = end_time - start_time

            print(
                f"Epoch {epoch + 1}/{EPOCHS}, Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}, Time per batch: {batch_time:.4f} seconds")
            print(
                f"Batch {batch_idx + 1}: Mean Logits: {mean_logits:.6f}, Variance: {var_logits:.6f}, Sigmoid(Mean): {sigmoid_mean:.6f}")

            print_system_memory()
            print_gpu_memory()

        torch.cuda.empty_cache()

        average_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{EPOCHS} - Average Loss: {average_epoch_loss:.4f}")

        # Validation step
        model.eval()
        val_predictions = []
        val_labels = []

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)

                predicted = (torch.sigmoid(output) > 0.5).float()

                val_predictions.append(predicted)
                val_labels.append(target)

        val_predictions = torch.cat(val_predictions, dim=0).cpu()
        val_labels = torch.cat(val_labels, dim=0).cpu()

        val_accuracy = accuracy_score(val_labels.numpy(), val_predictions.numpy())
        print(f"Epoch {epoch + 1}/{EPOCHS} - Validation Accuracy: {val_accuracy:.4f}")

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(f"Epoch {epoch + 1}/{EPOCHS} - Learning Rate: {current_lr:.6f}")

        early_stopping(val_accuracy, model, optimizer, scaler, epoch)

        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

        batch_loss_data = {'epoch': epoch, 'batch_losses': batch_losses}
        epoch_loss_data = {'epoch': epoch, 'average_epoch_loss': average_epoch_loss}
        validation_data = {'epoch': epoch, 'validation_accuracy': val_accuracy}

        # Save training data in JSON files
        with open(f'../parameters/batch_losses.json', 'a') as f:
            json.dump(batch_loss_data, f, indent=4)
            f.write("\n")

        with open(f'../parameters/epoch_losses.json', 'a') as f:
            json.dump(epoch_loss_data, f, indent=4)
            f.write("\n")

        with open('../parameters/validation_accuracy.json', 'a') as f:
            json.dump(validation_data, f, indent=4)
            f.write("\n")


if __name__ == '__main__':
    main()
