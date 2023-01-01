import torch
from torch import nn

def train_model(train_dataloader, test_dataloader, model, loss_fn, optimizer, lr = 0.01, epochs = 200):
    train_accuracies, test_accuracies = [], []

    for epoch in range(epochs):
        for X, y in train_dataloader:
            pred = model(X)
            y_pred = torch.log_softmax(pred, dim = 1)
            _, pred_labels = torch.max(y_pred, dim = 1) 
            #pred_labels = torch.argmax(pred, axis = 1)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_accuracies.append(100 * torch.mean((pred_labels == y).float()).item())

        X, y = next(iter(test_dataloader))
        #y_pred = torch.log_softmax(model(X), dim = 1)
        #_, pred_labels = torch.max(y_pred, dim = 1) 
        pred_labels = torch.argmax(model(X), axis = 1)
        test_accuracies.append(100 * torch.mean((pred_labels == y).float()).item())
        print(f"Epoch {epoch+1} | Test accuracy: {test_accuracies[-1]:.2f}%")

    return train_accuracies, test_accuracies