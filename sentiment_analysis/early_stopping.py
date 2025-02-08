import torch


class EarlyStopping:
    """
    Implements early stopping to halt training when validation accuracy stops improving.

    Attributes:
        patience (int): Number of epochs to wait before stopping if no improvement.
        delta (float): Minimum change in validation accuracy to qualify as improvement.
        save_path (str): Path to save the best model checkpoint.
        best_score (float or None): Best validation accuracy observed.
        counter (int): Number of consecutive epochs without improvement.
        early_stop (bool): Flag indicating whether training should be stopped.
    """

    def __init__(self, patience=5, delta=0.001, save_path=''):
        """
        Initializes the EarlyStopping object.

        Args:
            patience (int): Number of epochs to wait before stopping.
            delta (float): Minimum improvement required to reset patience.
            save_path (str): File path to save the best model.
        """
        self.patience = patience
        self.delta = delta
        self.save_path = save_path
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_accuracy, model, optimizer, scaler, epoch):
        """
        Checks validation accuracy and determines whether to stop training.

        Args:
            val_accuracy (float): The current validation accuracy.
            model (torch.nn.Module): The model being trained.
            optimizer (torch.optim.Optimizer): The optimizer used for training.
            scaler (torch.cuda.amp.GradScaler): The gradient scaler for mixed precision training.
            epoch (int): The current training epoch.
        """
        score = val_accuracy

        # If no best score has been recorded, initialize it and save the checkpoint.
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, optimizer, scaler, epoch)

        # If the score does not improve beyond delta, increase the counter.
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")

            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # If improvement is found, update best score and reset counter
            self.best_score = score
            self.save_checkpoint(model, optimizer, scaler, epoch)
            self.counter = 0

    def save_checkpoint(self, model, optimizer, scaler, epoch):
        """
        Saves the current model checkpoint.

        Args:
            model (torch.nn.Module): The model to save.
            optimizer (torch.optim.Optimizer): The optimizer state to save.
            scaler (torch.cuda.amp.GradScaler): The scaler state for mixed precision training.
            epoch (int): The epoch at which the checkpoint is saved.
        """
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
        }, self.save_path)
        print("Validation accuracy improved. Model saved.")
