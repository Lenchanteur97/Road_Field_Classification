import torch

import matplotlib.pyplot as plt

from evaluation import evaluate
from utils import get_device, plot_data


def train_model(train_data_loader, val_data_loader, model, criterion, optimizer, epochs, verbose=True):
    """
    Train a model on images contained in the data loader using the specified loss and optimizer during the given number
    of epochs

    :param train_data_loader: Data loader containing training data
    :param val_data_loader: Data loader containing validation data
    :param model: Neural network to train
    :param criterion: Loss function to minimize
    :param optimizer: Optimizer used to optimize gradients
    :param epochs: Number of iterations for training
    :param verbose: If True, print loss, accuracy and AUC ROC during training
    :return:
    """

    device = get_device()
    model = model.to(device)

    print('Training in process')
    # Lists for plots
    training_loss = []
    validation_loss = []
    training_accuracy = []
    validation_accuracy = []
    validation_roc_auc = []

    # Metrics used to select the best model to save
    best_acc, best_auc_roc = 0., 0.
    epoch_best_model = 0

    for epoch in range(epochs):
        # Set model to training mode
        model.train()

        # Store loss for future training loss plots
        tr_loss = 0.

        # Store number of correct predictions to compute accuracy
        correct_pred = 0

        for images, labels in train_data_loader:
            # Send data to the available processing unit (cuda is preferred)
            images, labels = images.to(device), labels.to(device)

            # Reset the gradients
            optimizer.zero_grad()

            # Forward images through the model
            outputs = model(images)

            # Compute loss and update model weights using the optimizer
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)  # Gradient clipping to avoid exploding gradient
            optimizer.step()

            # Record total batch loss
            tr_loss += loss.item() * images.size(0)

            # Perform label prediction using max function on model outputs
            _, predictions = torch.max(outputs, 1)
            correct_pred += (predictions == labels).sum().item()

        # Compute average training loss of the epoch
        tr_loss /= len(train_data_loader.dataset)
        training_loss.append(tr_loss)

        # Compute accuracy on training data
        accuracy = 100 * correct_pred / len(train_data_loader.dataset)
        training_accuracy.append(accuracy)

        # Compute validation loss, accuracy and AUC-ROC score (evaluate the model on validation data at the end of
        # each epoch)
        val_acc, val_auc_roc, val_loss = evaluate(val_data_loader, model, criterion)
        validation_loss.append(val_loss)
        validation_accuracy.append(val_acc)
        validation_roc_auc.append(val_auc_roc)

        # If the metrics (loss, accuracy and AUC-ROC) are better than the ones stored, update best metrics and save
        # model parameters in working directory
        if val_acc >= best_acc and val_auc_roc >= best_auc_roc:
            best_acc, best_auc_roc = val_acc, val_auc_roc
            epoch_best_model = epoch
            model.save_model()

        # Print loss and accuracy on train and validation data
        if verbose:
            print(f"Epoch {epoch + 1} | Training loss {tr_loss:.3f} | Training accuracy {accuracy:.1f}% | "
                  f"Validation loss {val_loss:.3f} | Validation accuracy {val_acc:.1f}% | Validation AUC ROC {val_auc_roc:.3f}")

    print('End of training')
    print(f'Best model selected at epoch {epoch_best_model}')

    # Plot accuracy and loss curves for both training and validation datasets
    plot_data(x=range(epochs),
              y_list=[training_loss, validation_loss],
              labels=['Training', 'Validation'],
              x_lab='Epoch',
              y_lab='Average loss value',
              title='Average loss per epoch for training and validation datasets')

    plot_data(x=range(epochs),
              y_list=[training_accuracy, validation_accuracy],
              labels=['Training', 'Validation'],
              x_lab='Epoch',
              y_lab='Accuracy (%)',
              title='Accuracy per epoch for training and validation datasets')

    # Plot roc_auc score
    plot_data(x=range(epochs),
              y_list=[validation_roc_auc],
              labels=['Validation'],
              x_lab='Epoch',
              y_lab='AUC ROC',
              title='AUC ROC Score per epoch')
