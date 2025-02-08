import json
import matplotlib.pyplot as plt

# Load data from JSON file
def load_data_from_json(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data


# Plotting function
def plot_validation_accuracy(data):
    epochs = [entry['epoch'] for entry in data]
    validation_accuracy = [entry['validation_accuracy'] for entry in data]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, validation_accuracy, marker='o', linestyle='-', color='blue', label='Validation Accuracy')

    # Add title and labels
    plt.title('Validation Accuracy Over Epochs', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Validation Accuracy', fontsize=14)
    plt.grid(visible=True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)

    # Show the plot
    plt.tight_layout()
    plt.show()


# Main function
if __name__ == '__main__':
    # Load the validation accuracy data from the JSON file
    data = load_data_from_json('../parameters/validation_accuracy.json')

    # Plot the data
    plot_validation_accuracy(data)
