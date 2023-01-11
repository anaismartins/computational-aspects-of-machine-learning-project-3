from sklearn.svm import OneClassSVM
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

# define outlier detection model
model = OneClassSVM(gamma='scale', nu=0.01)

data = np.load("../datasets/dataset_all_h1.npy") # uncomment this line to use the bootstrap dataset
#data = np.load("../datasets/dataset_all_h1.npy") # uncomment this line to use the original dataset
X = data[:,:-1]
y = data[:,-1]

X = torch.tensor(X, dtype=torch.float)
y = torch.tensor(y, dtype=torch.long)

# splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

# fit on majority class
X_train = X_train[y_train==0]
model.fit(X_train)

# detect outliers in the test set
y_pred = model.predict(X_test)

sum = 0
total = 0

for i in range(len(y_pred)):
    if y_pred[i] == y_test[i]:
        sum = sum + 1
        total = total + 1
    else:
        total = total + 1

accuracy = 100 * sum / total
print(accuracy)