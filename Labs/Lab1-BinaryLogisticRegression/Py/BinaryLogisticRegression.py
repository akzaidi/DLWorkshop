import numpy as np
import cntk as C
import matplotlib.pyplot as plt
import sys
import os
from cntk.logging.progress_print import ProgressPrinter
from sklearn.metrics import confusion_matrix

# Select the right target device 
#C.device.try_set_default_device(cntk.device.cpu())
C.device.try_set_default_device(C.device.gpu(0))

#Load dataset
dataset = np.loadtxt('Data/cancer.csv', delimiter=',', skiprows=1, dtype=np.float32)

#Visualize dataset
colors = ['r' if l == 1. else 'b' for l in dataset[:,[2]]]
plt.scatter(dataset[:,[0]], dataset[:,[1]], c=colors)
plt.xlabel("Age")
plt.ylabel("Tumor size")
plt.show()

#Prepare training and validation features and labels
np.random.seed(0)

def preprocess_dataset(dataset):
    training = np.copy(dataset)
    np.random.shuffle(training)
    training[:,[0,1]] /= np.max(training[:,[0,1]])
    index = int(len(training) * .7) 
    return np.ascontiguousarray(training[0:index, [0,1]]), \
           np.ascontiguousarray(training[0:index, [2]]), \
           np.ascontiguousarray(training[index+1:len(training), [0,1]]), \
           np.ascontiguousarray(training[index+1:len(training), [2]])

training_features, training_labels, validation_features, validation_labels = preprocess_dataset(dataset)

#Set up a computational network
input_dim = 2
output_dim = 1 

X = C.input(shape=(input_dim))
w = C.parameter(shape=(input_dim, output_dim))
b = C.parameter(shape=(output_dim))
y = C.sigmoid(X @ w + b)

#Define a loss function
y_ = C.input((output_dim))
loss = -C.reduce_sum(y_ * C.log(y) + (1 - y_) * C.log(1 - y), axis=C.Axis.all_axes())

#Configure the SGD trainer
progress_printer = ProgressPrinter(20)
learning_rate = 0.02
lr_schedule = C.learning_rate_schedule(learning_rate, C.UnitType.sample)
learner = C.sgd(y.parameters, lr_schedule)
trainer = C.Trainer(y, (loss), [learner], [progress_printer])

#Run the trainer
num_of_sweeps = 500 
for i in range(0, num_of_sweeps):
    trainer.train_minibatch({X : training_features, y_ : training_labels})

#Plot the learned decision boundry
weights = y.parameters[0].value
bias = y.parameters[1].value
colors = ['r' if l == 1. else 'b' for l in training_labels]
plt.scatter(training_features[:,[0]], training_features[:,[1]], c=colors)
plt.plot([0, -bias[0]/weights[1][0]],
         [-bias[0]/weights[0][0], 0], c = 'g', lw =2)
plt.axis([0.2, 1.1, 0.2, 0.9])
plt.xlabel("Age")
plt.ylabel("Tumor size")
plt.show()

#Score the model on the validation dataset
result = y.eval({X: validation_features})

#Calculate confusion matrix
predicted_labels = np.round(result)
print(confusion_matrix(validation_labels, predicted_labels))