from sklearn.svm import OneClassSVM

# define outlier detection model
model = OneClassSVM(gamma='scale', nu=0.01)