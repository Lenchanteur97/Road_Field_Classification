import torch

from sklearn.metrics import roc_curve, auc

from utils import get_device

import matplotlib.pyplot as plt


def evaluate(data_loader, model, criterion=None, plot_curve=False):
    """
    Compute model loss and evaluation metrics on validation data
    If criterion is None, the loss will be None too

    :param data_loader: Data loader containing validation of test data
    :param model: Neural network to validate
    :param criterion: Loss function, if None the loss value returned is None
    :param plot_curve: Boolean, True for plotting the ROC curve (avoid during training, use for validation at end)
    :return: Accuracy, ROC-AUC Score, Loss value if criterion is not None else None
    """

    device = get_device()

    # Set model to evaluation mode
    model.eval()
    model = model.to(device)

    # Store loss
    val_loss = 0.

    # Store class probabilities to later compute metrics
    probabilities = torch.Tensor()
    true_labels = torch.Tensor()

    with torch.no_grad():
        for images, labels in data_loader:
            # Send data to the available processing unit (cuda is preferred)
            images, labels = images.to(device), labels.to(device)

            # Forward images through the network to get class probabilities
            outputs = model(images)

            # Update loss value if criterion is not None
            if criterion is not None:
                val_loss += criterion(outputs, labels).item() * images.size(0)

            # Store class probabilities and correct labels
            probabilities = torch.cat((probabilities, outputs.cpu()))
            true_labels = torch.cat((true_labels, labels.cpu()))

        # Compute average loss on validation data
        if criterion is not None:
            val_loss /= len(data_loader.dataset)
        else:
            val_loss = None

        # Compute accuracy and ROC-AUC Score
        _, predictions = torch.max(probabilities, 1)
        acc = 100 * (predictions == true_labels).sum().item() / true_labels.size(0)
        fpr, tpr, threshold = roc_curve(true_labels, probabilities[:, 1])
        auc_roc = auc(fpr, tpr)

        # Plot ROC curve (usually at testing)
        if plot_curve:
            plt.title('Receiver Operating Characteristic Curve')
            plt.plot(fpr, tpr, 'b', label='AUC = %0.3f' % auc_roc)
            plt.legend(loc='lower right')
            plt.plot([0, 1], [0, 1], 'r--')
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')
            plt.show()

    return acc, auc_roc, val_loss


def predict(data_loader, model, return_probabilities=False):
    """
    If return_probabilities is True, return probability predictions for each image contained in the data loader
    Else, perform label prediction

    :param data_loader: Data loader containing images
    :param model: Model used to classify images
    :param return_probabilities: Boolean
    :return: if return_probabilities is true return prediction probabilities, else return labels
    """

    device = get_device()

    # Set model to evaluation mode
    model.eval()
    model = model.to(device)

    # Store class probabilities
    probabilities = torch.Tensor()

    with torch.no_grad():
        for images, _ in data_loader:
            # Send data to the available processing unit (cuda is preferred)
            images = images.to(device)

            # Forward images through the network to get class prediction probabilities
            outputs = model(images)

            # Store class probabilities
            probabilities = torch.cat((probabilities, outputs.cpu()))

    if return_probabilities:
        return probabilities
    else:
        _, label_predictions = torch.max(probabilities, 1)
        return label_predictions
