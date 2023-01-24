import torch
from torch import nn

def train_model(train_dataloader, valid_dataloader, test_dataloader, model, loss_fn, optimizer, lr_scheduler, epochs = 200):
    """
    Train the model and return the validation accuracy
    :param train_dataloader: training dataloader    
    :param valid_dataloader: validation dataloader
    :param model: model to train
    :param loss_fn: loss function
    :param optimizer: optimizer
    :param lr_scheduler: learning rate scheduler
    :param epochs: number of epochs to train
    :return: predicted labels, all training and validation accuracies, the number of epochs it ran for and the model after training
    """
    
    # initializing lists to store the accuracies, loss and setting the final_epoch to the initial number of max epochs
    train_accuracies, valid_accuracies = [], []
    stored_loss = []
    final_epoch = epochs

    # looping over the epochs
    for epoch in range(epochs):
        # iterating over the batches
        for X, y in train_dataloader:
            # Forward pass
            pred = model(X)
            # get the predicted labels from the probabilities
            y_pred = torch.log_softmax(pred, dim = 1)
            _, pred_labels = torch.max(y_pred, dim = 1) 
            # Compute loss
            loss = loss_fn(pred, y)
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            # Update the parameters
            optimizer.step()

        # Compute the training accuracy and store it
        train_accuracies.append(100 * torch.mean((pred_labels == y).float()).item())

        # getting the full validation dataset
        X, y = next(iter(valid_dataloader))
        # getting the predicted labels
        pred_labels = torch.argmax(model(X), axis = 1)

        # Compute the validation accuracy and store it
        valid_accuracies.append(100 * torch.mean((pred_labels == y).float()).item())
        print(f"Epoch {epoch+1} | Validation accuracy: {valid_accuracies[-1]:.2f}%")

        # early stopping algorithm that stops the training if the validation accuracy is within 0.1 for at least 75% of the last 200 epochs
        if (epoch > 200):
            stop = 0
            for i in range(0, 200):
                if abs(valid_accuracies[-i] - valid_accuracies[epoch]) < 0.1:
                    stop = stop + 1
            if (stop > 150):
                final_epoch = epoch

                # getting the full validation dataset
                X, y = next(iter(test_dataloader))
                # getting the predicted labels
                test_pred_labels = torch.argmax(model(X), axis = 1)

                # Compute the validation accuracy and store it
                test_accuracy = 100 * torch.mean((test_pred_labels == y).float()).item()
                print(f"Test accuracy: {test_accuracy:.2f}%")

                break
            
        if (epoch == epochs - 1):
            # getting the full validation dataset
            X, y = next(iter(test_dataloader))
            # getting the predicted labels
            test_pred_labels = torch.argmax(model(X), axis = 1)

            # Compute the validation accuracy and store it
            test_accuracy = 100 * torch.mean((test_pred_labels == y).float()).item()
            print(f"Test accuracy: {test_accuracy:.2f}%")

        # Update the learning rate
        old_lr = optimizer.param_groups[0]['lr']
        lr_scheduler.step(valid_accuracies[-1])
        new_lr = optimizer.param_groups[0]['lr']
        if old_lr != new_lr:
            print(f"Learning rate updated from {old_lr:.6f} to {new_lr:.6f}")

    return test_pred_labels, train_accuracies, valid_accuracies, test_accuracy, final_epoch, model