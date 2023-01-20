import torch
from torch import nn

def train_model(train_dataloader, valid_dataloader, model, loss_fn, optimizer, lr_scheduler, epochs = 200):
    train_accuracies, valid_accuracies = [], []
    stored_loss = []
    final_epoch = epochs

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

        X, y = next(iter(valid_dataloader))
        #y_pred = torch.log_softmax(model(X), dim = 1)
        #_, pred_labels = torch.max(y_pred, dim = 1) 
        pred_labels = torch.argmax(model(X), axis = 1)

        valid_accuracies.append(100 * torch.mean((pred_labels == y).float()).item())
        print(f"Epoch {epoch+1} | Validation accuracy: {valid_accuracies[-1]:.2f}%")

        if (epoch > 200):
            stop = 0
            for i in range(0, 200):
                if abs(valid_accuracies[-i] - valid_accuracies[epoch]) < 1:
                    stop = stop + 1
            if (stop > 150):
                final_epoch = epoch

                break

        # Update the learning rate
        old_lr = optimizer.param_groups[0]['lr']
        lr_scheduler.step(valid_accuracies[-1])
        new_lr = optimizer.param_groups[0]['lr']
        if old_lr != new_lr:
            print(f"Learning rate updated from {old_lr:.6f} to {new_lr:.6f}")

    return pred_labels, train_accuracies, valid_accuracies, final_epoch