# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

#
# Fully Connected Feedforward Network for MNIST classification sample
#


import numpy as np
import sys
import os
import time

import cntk as C
from cntk.logging.progress_print import ProgressPrinter

#Configure the device
C.device.try_set_default_device(C.device.gpu(0))
#C.device.try_set_default_device(C.device.cpu())

#Define a reader based on CTF Deserializer
def create_reader(path, is_training, input_dim, label_dim):
    return C.io.MinibatchSource(C.io.CTFDeserializer(path, C.io.StreamDefs(
        features  = C.io.StreamDef(field='features', shape=input_dim, is_sparse=False),
        labels    = C.io.StreamDef(field='labels',   shape=label_dim, is_sparse=False)
    )), randomize=is_training, max_sweeps = C.io.INFINITELY_REPEAT if is_training else 1)


# Define a fully connected feedforward classification network with sigmoid neurons in the hidden layers
def create_model(features, num_hidden_layers, hidden_layers_dim, num_output_classes):
    with C.layers.default_options(init = C.layers.glorot_uniform(), activation = C.ops.relu):
        h = features
        for _ in range(num_hidden_layers):
            h = C.layers.Dense(hidden_layers_dim)(h)
        r = C.layers.Dense(num_output_classes, activation = None)(h)
        return r

# Configure a two hidden-layer FCN with softmax output and cross-entropy loss
input_dim = 784
num_hidden_layers = 2
hidden_layers_dim = 400
num_output_classes = 10

features = C.input(input_dim)
labels = C.input(num_output_classes)

z = create_model(features/255.0, num_hidden_layers, hidden_layers_dim, num_output_classes)

loss = C.cross_entropy_with_softmax(z, labels)
error = C.classification_error(z, labels)

# Configure a trainer with the SGD learner
learning_rate = 0.2
lr_schedule= C.learning_rate_schedule(learning_rate, C.UnitType.minibatch)
learner = C.sgd(z.parameters, lr_schedule)
progress_printer = ProgressPrinter(500)
trainer = C.Trainer(z, (loss, error), [learner], [progress_printer])

# Run the trainer
# Create and prime the reader with the training dataset
train_file = '../../Data/MNIST_train.txt'
reader_train = create_reader(train_file, True, input_dim, num_output_classes)

# Map the data streams to the features and labels
input_map = {
    labels : reader_train.streams.labels,
    features: reader_train.streams.features
}

# Configure sweeps and minibatches
minibatch_size = 64
num_samples_per_sweep = 50000
num_sweeps_to_train_with = 10
num_minibatches_to_train = (num_samples_per_sweep * num_sweeps_to_train_with) / minibatch_size

# Run the trainer
start_time = time.time()
for _ in range(0, int(num_minibatches_to_train)):
    data = reader_train.next_minibatch(minibatch_size, input_map = input_map)
    trainer.train_minibatch(data)
print(time.time() - start_time)

# Model validation
# Read the validation data
test_file = "../../Data/MNIST_validate.txt"
reader_test = create_reader(test_file, False, input_dim, num_output_classes)

test_input_map = {
    labels  : reader_test.streams.labels,
    features  : reader_test.streams.features,
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

# Hackathon 
# - Try relu vs sigmoid
# - Different learning rates, batch sizes, sweeps
# - SGD regularization
# Final model testing
# Read the validation data
test_file = "../../Data/MNIST_test.txt"
reader_test = create_reader(test_file, False, input_dim, num_output_classes)

test_input_map = {
    labels  : reader_test.streams.labels,
    features  : reader_test.streams.features,
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

