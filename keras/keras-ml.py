# https://stackabuse.com/image-recognition-in-python-with-tensorflow-and-keras/

# I should really be using cudas, otherwise tensorflow is running on the cpu

import numpy
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.constraints import maxnorm
from keras.utils import np_utils
from keras.datasets import cifar10

# Set random seed for purposes of reproducibility
seed = 21

# load in the data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Normalize the inputs from 0-255 to between 0 and 1 by dividing by 255
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

# One hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
class_num = y_test.shape[1]

# A Sequential model is appropriate for a plain stack of layers where each layer has exactly one
# input tensor and one output tensor. A 'tensor' is a grid of values that is not constrained to 
# be 1D (vector) or 2D (matrix) but rather has n dimensions.
model = Sequential()

# Add a convolutional layer to our model. Here we are setting it to have 32 channels/filters,
# and we are preforming the convolution with a 3x3 grid.
model.add(Conv2D(32, (3, 3), input_shape=X_train.shape[1:], padding='same'))
model.add(Activation('relu'))

# Done to prevent overfitting, Dropout(0.2) means drop a random 20% of the connections between the layers
model.add(Dropout(0.2))

# Normalize inputs heading to the next layer to ensure the network creates activations with the same distribution
model.add(BatchNormalization())

# Add another convolutional layer as before with an increased filter size to allow for more complex shapes
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))

# Add a pooling layer to make the image classifier more robust
# it is important to not have too many pooling layers, as each pooling layer discards some of the data
# Pooling makes it easier for the images to be classified as it cleans up the data (double check this), but 
# that can have the effect of making it so that there isnt enough raw data to analyze
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(BatchNormalization())


# here we add another pooling. since each image is already quite small, pooling more is going to have adverse 
# effects as mentioned above
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(BatchNormalization())
    
model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

# Once done with the convolutional layers, we need to flatten the image (look up what this means/does)
model.add(Flatten())
model.add(Dropout(0.2))

# here we are creating densly connected layers. We specify the number of neurons used in its creation
# and the number of neurons decreases, approaching the number of classes in the dataset (here 10)
model.add(Dense(256, kernel_constraint=maxnorm(3)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
    
model.add(Dense(128, kernel_constraint=maxnorm(3)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

# once we have associated the neurons with their probabilistic relationship to each image,
# we use Activation('softmax') to select the neuron with the highest probability, and associate the image with that class
model.add(Dense(class_num))
model.add(Activation('softmax'))

# Now that we have designed the model we want to use, we need to compile it. (look up what epochs are).
# 'adam' is a commonly used optimizer, as it gives great performance on most problems
epochs = 25
optimizer = 'adam'
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# print(model.summary())

# Next we will train the model by calling model.fit(). Here we are using a pre-chosen seed for reproducibility
numpy.random.seed(seed)
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=64)

# Model evaluation
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))