# Import the relevant components
from __future__ import print_function # Use a function definition from future version (say 3.x from 2.7 interpreter)
import numpy as np
import sys
import os
import cntk as C
import time
from cntk.logging.progress_print import ProgressPrinter

C.device.try_set_default_device(C.device.cpu())
#C.device.try_set_default_device(C.device.gpu(0))

# Ensure we always get the same amount of randomness
np.random.seed(0)

# Read a CTF formatted text (as mentioned above) using the CTF deserializer from a file
def create_reader(path, is_training, input_dim, num_label_classes):
    labelStream = C.io.StreamDef(field='labels', shape=num_label_classes, is_sparse=False)
    featureStream = C.io.StreamDef(field='features', shape=input_dim, is_sparse=False)
    deserializer = C.io.CTFDeserializer(path, C.io.StreamDefs(labels = labelStream, features = featureStream))
    return C.io.MinibatchSource(deserializer,
       randomize = is_training, max_sweeps = C.io.INFINITELY_REPEAT if is_training else 1)

# Define a computational network for multi-class logistic regression
def create_mlr_model(features, output_dim):
    input_dim = features.shape[0]
    weight_param = C.parameter(shape=(input_dim, output_dim))
    bias_param = C.parameter(shape=(output_dim))
    return C.times(features, weight_param) + bias_param

# Define the data dimensions
input_dim = 784
num_output_classes = 10

# Define features and labels
input = C.input(input_dim)
label = C.input(num_output_classes)

# Scale the input to 0-1 range by dividing each pixel by 255.
z = create_mlr_model(input/255.0, num_output_classes)

# Define loss and error functions
loss = C.cross_entropy_with_softmax(z, label)
label_error = C.classification_error(z, label)

# Instantiate the trainer object to drive the model training
learning_rate = 0.2
lr_schedule = C.learning_rate_schedule(learning_rate, C.UnitType.minibatch)
learner = C.sgd(z.parameters, lr_schedule)
progress_printer = ProgressPrinter(500)
trainer = C.Trainer(z, (loss, label_error), [learner], [progress_printer])

# Initialize the parameters for the trainer
minibatch_size = 64
num_samples_per_sweep = 60000
num_sweeps_to_train_with = 10
num_minibatches_to_train = (num_samples_per_sweep * num_sweeps_to_train_with) / minibatch_size

# Create the reader to training data set
train_file = "Data/MNIST_train.txt"
reader_train = create_reader(train_file, True, input_dim, num_output_classes)

# Map the data streams to the input and labels.
input_map = {
    label  : reader_train.streams.labels,
    input  : reader_train.streams.features
} 

start_time = time.time()
# Run the trainer on and perform model training
for i in range(0, int(num_minibatches_to_train)):
    # Read a mini batch from the training data file
    data = reader_train.next_minibatch(minibatch_size, input_map = input_map)
    trainer.train_minibatch(data)

print(time.time() - start_time)

# Read the validation data
test_file = "Data/MNIST_validate.txt"
reader_test = create_reader(test_file, False, input_dim, num_output_classes)

test_input_map = {
    label  : reader_test.streams.labels,
    input  : reader_test.streams.features,
}

# Test data for trained model
test_minibatch_size = 512
num_samples = 10000
num_minibatches_to_test = num_samples // test_minibatch_size
test_result = 0.0

for i in range(num_minibatches_to_test):
    data = reader_test.next_minibatch(test_minibatch_size, input_map = test_input_map)
    eval_error = trainer.test_minibatch(data)
    test_result = test_result + eval_error

# Average of evaluation errors of all test minibatches
print("Average validation error: {0:.2f}%".format(test_result*100 / num_minibatches_to_test))


# Read the test data set
test_file = "Data/MNIST_test.txt"
reader_test = create_reader(test_file, False, input_dim, num_output_classes)

test_input_map = {
    label  : reader_test.streams.labels,
    input  : reader_test.streams.features,
}

# Test data for trained model
test_minibatch_size = 512
num_samples = 10000
num_minibatches_to_test = num_samples // test_minibatch_size
test_result = 0.0

for i in range(num_minibatches_to_test):
    data = reader_test.next_minibatch(test_minibatch_size, input_map = test_input_map)
    eval_error = trainer.test_minibatch(data)
    test_result = test_result + eval_error

# Average of evaluation errors of all test minibatches
print("Average test error: {0:.2f}%".format(test_result*100 / num_minibatches_to_test))
