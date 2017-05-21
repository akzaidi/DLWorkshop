# Copyright (c) Microsoft. All rights reserved.

#
# Convolutional Neural Network for MNIST classification sample
#

import numpy as np
import sys
import os
import time
import cntk as C
from cntk.logging.progress_print import ProgressPrinter

#Configure the device
#C.device.try_set_default_device(C.device.cpu())
C.device.try_set_default_device(C.device.gpu(0))

#
# Helper functions
#

# Ensure we always get the same amount of randomness
np.random.seed(0)

# Define a reader for the CTF formatted MNIST files 
def create_reader(path, is_training, input_dim, label_dim):
    return C.io.MinibatchSource(C.io.CTFDeserializer(path, C.io.StreamDefs(
        features  = C.io.StreamDef(field='features', shape=input_dim, is_sparse=False),
        labels    = C.io.StreamDef(field='labels',   shape=label_dim, is_sparse=False)
    )), randomize=is_training, max_sweeps = C.io.INFINITELY_REPEAT if is_training else 1)

# Define a convolutional neural network - v1 
def create_cnn1_model(features, num_output_classes):
    with C.layers.default_options(init = C.layers.glorot_uniform(), activation = C.ops.relu):
        h = features
        h = C.layers.Convolution2D(filter_shape=(5,5),
                                   num_filters = 8,
                                   strides = (2,2),
                                   pad = True,
                                   name = 'first_conv')(h)
        h = C.layers.Convolution2D(filter_shape = (5,5),
                                  num_filters = 8,
                                  strides = (2, 2),
                                  pad = True,
                                  name = 'second_conv')(h)
        r = C.layers.Dense(num_output_classes, activation=None, name = 'classify')(h)
        return r

# Define a convolutional neural network - v2 
def create_cnn2_model(features, num_output_classes):
    with C.layers.default_options(init = C.layers.glorot_uniform(), activation = C.ops.relu):
        h = features
        h = C.layers.Convolution2D(filter_shape=(5,5),
                                   num_filters = 8,
                                   strides = (2,2),
                                   pad = True,
                                   name = 'first_conv')(h)
        h = C.layers.MaxPooling(filter_shape=(2,2),
                               strides=(2,2),
                               name = "first_max")(h)
        h = C.layers.Convolution2D(filter_shape = (5,5),
                                  num_filters = 8,
                                  strides = (2, 2),
                                  pad = True,
                                  name = 'second_conv')(h)
        h = C.layers.MaxPooling(filter_shape=(3,3),
                                strides=(3,3),
                                name = 'second_max')(h)
        r = C.layers.Dense(num_output_classes, activation=None, name = 'classify')(h)
        return r


# Define a trainer using a given reader and the SGD learner 
def train_model_with_SGD(model, features, labels, reader, num_samples_per_sweep, num_sweeps):
 
    # Define loss and error functions
    loss = C.cross_entropy_with_softmax(model, labels)
    error = C.classification_error(model, labels)

    # Instantiate the trainer object to drive the model training
    learning_rate = 0.2
    lr_schedule = C.learning_rate_schedule(learning_rate, C.UnitType.minibatch)
    learner = C.sgd(model.parameters, lr_schedule)
    progress_printer = ProgressPrinter(500)
    trainer = C.Trainer(model, (loss, error), [learner], [progress_printer])

   # Initialize the parameters for the trainer
    minibatch_size = 64
    num_minibatches_to_train = (num_samples_per_sweep * num_sweeps) / minibatch_size

       # Map the data streams to the input and labels.
    input_map = {
        labels  : reader.streams.labels,
        features  : reader.streams.features
    } 

    # Run the trainer on and perform model training
    start_time = time.time()
    for i in range(0, int(num_minibatches_to_train)):
        data = reader.next_minibatch(minibatch_size, input_map = input_map)
        trainer.train_minibatch(data)

    print(time.time() - start_time)

# Define the evaluater function 
def test_model(model, features, labels, reader):
    evaluator = C.Evaluator(C.classification_error(model, labels))
    input_map = {
       features : reader.streams.features,
       labels: reader.streams.labels
    }
    
    minibatch_size = 2000
    test_result = 0.0
    num_minibatches = 0
    data = reader.next_minibatch(minibatch_size, input_map = input_map)
    while bool(data):
        test_result = test_result + evaluator.test_minibatch(data)
        num_minibatches += 1
        data = reader.next_minibatch(minibatch_size, input_map = input_map)
    return None if num_minibatches == 0 else test_result*100 / num_minibatches


### v1 Model training
#
# Define MNIST data dimensions
input_dim = 784
input_dim_model = (1, 28, 28)
num_output_classes = 10

# Create inputs for features and labels
features = C.input(input_dim_model)
labels = C.input(num_output_classes)

# Create the CNN model while scaling the input to 0-1 range by dividing each pixel by 255.
z = create_cnn1_model(features/255.0, num_output_classes)

# Configure and run the trainer 
train_file = "../../Data/MNIST_train.txt"
reader = create_reader(train_file, True, input_dim, num_output_classes)
num_samples_per_sweep = 50000
num_sweeps = 10
print("Starting training of v1 model")
train_model_with_SGD(z, features, labels, reader, num_samples_per_sweep, num_sweeps)


### v1 Model evaluation
#
validation_file = "../../Data/MNIST_validate.txt"
reader = create_reader(validation_file, False, input_dim, num_output_classes)
error_rate = test_model(z, features, labels, reader)
print("Average validation error for v1 model: {0:.2f}%".format(error_rate))

### v2 Training
#
zv2 = create_cnn2_model(features/255.0, num_output_classes)
train_file = "../../Data/MNIST_train.txt"
reader = create_reader(train_file, True, input_dim, num_output_classes)
print("Starting training of v2 model")
train_model_with_SGD(zv2, features, labels, reader, num_samples_per_sweep, num_sweeps)

### v2 Model evaluation
#
validation_file = "../../Data/MNIST_validate.txt"
reader = create_reader(validation_file, False, input_dim, num_output_classes)
error_rate = test_model(zv2, features, labels, reader)
print("Average validation error for v2 model: {0:.2f}%".format(error_rate))


### Hackathon evaluation
#
# DON'T CHEAT. DON'T USE MNIST_test.txt FOR MODEL TRAINING AND VALIDATION
#
test_file = '../../Data/MNIST_test.txt'
reader = create_reader(test_file, False, input_dim, num_output_classes)
error_rate = test_model(z, features, labels, reader)
print("Average test error of the winning model: {0:.2f}%".format(error_rate))





