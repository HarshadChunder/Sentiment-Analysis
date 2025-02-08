import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SentimentCNN(nn.Module):
    """
        A Convolutional Neural Network (CNN) model for sentiment analysis.

        The model processes tokenized text data through multiple convolutional layers
        with different filter sizes, followed by batch normalization, activation functions,
        and pooling layers. The final representation is passed through a fully connected layer
        to produce a sentiment score.

        Attributes:
            vocab_size (int): Size of the vocabulary.
            embedding_dim (int): Dimension of word embeddings.
            num_classes (int): Number of output classes.
            num_filters (list): Number of filters for each convolutional layer.
            filter_sizes (list): Kernel sizes for each convolutional layer.
            dropout (float): Dropout rate for regularization.
            embedding_model: Pretrained word embedding model.
    """
    def __init__(self, vocab_size, embedding_dim, num_classes, num_filters, filter_sizes, dropout, embedding_model=None):
        super(SentimentCNN, self).__init__()

        """
        Initializes the SentimentCNN model.

        Args:
            vocab_size (int): Size of the vocabulary.
            embedding_dim (int): Dimension of word embeddings.
            num_classes (int): Number of output classes.
            num_filters (list): Number of filters per convolutional layer.
            filter_sizes (list): Kernel sizes for convolutional layers.
            dropout (float): Dropout rate for regularization.
            embedding_model (object, optional): Pretrained embedding model.
        """

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding_model = embedding_model
        self.embeddings_cache = {}

        # Define convolutional layers with batch normalization
        self.bn1 = nn.BatchNorm2d(num_filters[0])
        self.conv1 = nn.Conv2d(1, num_filters[0], (filter_sizes[0], embedding_dim))

        self.bn2 = nn.BatchNorm2d(num_filters[1])
        self.conv2 = nn.Conv2d(1, num_filters[1], (filter_sizes[1], embedding_dim))

        self.bn3 = nn.BatchNorm2d(num_filters[2])
        self.conv3 = nn.Conv2d(1, num_filters[2], (filter_sizes[2], embedding_dim))

        # Additional convolutional and batch normalization layers for refining features
        self.bn1_2 = nn.BatchNorm2d(num_filters[0])
        self.conv1_2 = nn.Conv2d(num_filters[0], num_filters[0], (filter_sizes[0], 1))

        self.bn2_2 = nn.BatchNorm2d(num_filters[1])
        self.conv2_2 = nn.Conv2d(num_filters[1], num_filters[1], (filter_sizes[1], 1))

        self.bn3_2 = nn.BatchNorm2d(num_filters[2])
        self.conv3_2 = nn.Conv2d(num_filters[2], num_filters[2], (filter_sizes[2], 1))

        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(dropout)

        # Fully connected output layer
        self.fc = nn.Linear(num_filters[0] + num_filters[1] + num_filters[2], num_classes)

        # Initialize weights using Xavier initialization
        self.initialize_weights()

    def initialize_weights(self):
        """
        Initializes model weights using Xavier uniform initialization for layers
        with Leaky ReLU activation.
        """
        for layer in [self.conv1, self.conv2, self.conv3, self.conv1_2, self.conv2_2, self.conv3_2, self.fc]:
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                print(f"Initialized {layer} weights with Xavier uniform for Leaky ReLU.")
                gain = nn.init.calculate_gain('leaky_relu', param=math.sqrt(5))
                nn.init.xavier_uniform_(layer.weight, gain=gain)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        print("Weight stats after initialization:")
        with torch.no_grad():
            for name, param in self.named_parameters():
                if param.requires_grad and param.data.numel() > 1:
                    print(f"  {name} - Mean: {param.data.mean():.6f}, Std: {param.data.std():.6f}")

    def get_embedding(self, word_index, device):
        """
        Retrieves the embedding vector for a given word index.

        Args:
            word_index (int): Index of the word in the vocabulary.
            device (torch.device): Device to store the embedding tensor.

        Returns:
            torch.Tensor: Word embedding vector.
        """
        if word_index in self.embeddings_cache:
            return self.embeddings_cache[word_index].to(device)
        elif word_index < self.vocab_size:
            word = list(self.embedding_model.key_to_index.keys())[word_index]
            if word in self.embedding_model:
                embedding = torch.tensor(self.embedding_model[word], dtype=torch.float).to(device)
            else:
                embedding = torch.randn(self.embedding_dim, dtype=torch.float).to(device)

            # Cache the embedding for faster access
            self.embeddings_cache[word_index] = embedding
            return embedding
        else:
            return torch.randn(self.embedding_dim, dtype=torch.float).to(device)

    def forward(self, x):
        """
        Defines the forward pass of the CNN.

        Args:
            x (torch.Tensor): Input batch of tokenized text.

        Returns:
            torch.Tensor: Model output.
        """
        device = x.device

        # Convert token indices into embeddings
        batch_embeddings = torch.stack([
            torch.stack([self.get_embedding(word_idx.item(), device) for word_idx in sentence])
            for sentence in x
        ], dim=0)

        # Add a channel dimension for convolutional layers
        x = batch_embeddings.unsqueeze(1)

        # Convolution, batch normalization, activation, and pooling for filter size 1
        x1 = self.bn1(self.conv1(x))
        x1 = F.leaky_relu(x1, negative_slope=0.01)
        x1 = self.bn1_2(self.conv1_2(x1))
        x1 = F.leaky_relu(x1, negative_slope=0.01).squeeze(3)
        x1 = F.max_pool1d(x1, x1.size(2)).squeeze(2)

        # Convolution, batch normalization, activation, and pooling for filter size 2
        x2 = self.bn2(self.conv2(x))
        x2 = F.leaky_relu(x2, negative_slope=0.01)
        x2 = self.bn2_2(self.conv2_2(x2))
        x2 = F.leaky_relu(x2, negative_slope=0.01).squeeze(3)
        x2 = F.max_pool1d(x2, x2.size(2)).squeeze(2)

        # Convolution, batch normalization, activation, and pooling for filter size 3
        x3 = self.bn3(self.conv3(x))
        x3 = F.leaky_relu(x3, negative_slope=0.01)
        x3 = self.bn3_2(self.conv3_2(x3))
        x3 = F.leaky_relu(x3, negative_slope=0.01).squeeze(3)
        x3 = F.max_pool1d(x3, x3.size(2)).squeeze(2)

        # Concatenate outputs from different filter sizes
        x = torch.cat([x1, x2, x3], dim=1)

        # Apply dropout for regularization
        x = self.dropout(x)

        # Fully connected output layer
        x = self.fc(x)

        return x
