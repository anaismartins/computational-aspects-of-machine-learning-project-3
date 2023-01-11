import torch
from torch import nn
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def train_model(train_dataloader, test_dataloader, model, loss_fn, optimizer, lr_scheduler, epochs = 200):
    train_accuracies, test_accuracies = [], []
    stored_loss = []

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

        if (epoch > 200):
            if (abs(test_accuracies[epoch-200] - test_accuracies[epoch]) < 0.01):
                final_epoch = epoch

                classes = ("Injection", "Blip", "Koyfish", "Low Frequency Burst", "Tomte", "Whistle", "Fast Scattering")
                cf_matrix = confusion_matrix(y, pred_labels, normalize = 'true')
                df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *10, index = [i for i in classes], columns = [i for i in classes])
                plt.figure(figsize = (12,7))
                sn.heatmap(df_cm, annot=True)
                plt.savefig('output.png')

                break

        # Update the learning rate
        old_lr = optimizer.param_groups[0]['lr']
        lr_scheduler.step(test_accuracies[-1])
        new_lr = optimizer.param_groups[0]['lr']
        if old_lr != new_lr:
            print(f"Learning rate updated from {old_lr:.6f} to {new_lr:.6f}")

    return train_accuracies, test_accuracies, final_epoch