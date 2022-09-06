from typing import Dict, Optional, Tuple
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
import flwr as fl
import tensorflow as tf
import utils


def main() -> None:
    ##model building
    model = Sequential()
    #convolutional layer with rectified linear unit activation
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=(28, 28, 1)))
    #32 convolution filters used each of size 3x3
    #again
    model.add(Conv2D(64, (3, 3), activation='relu'))
    #64 convolution filters used each of size 3x3
    #choose the best features via pooling
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #randomly turn neurons on and off to improve convergence
    model.add(Dropout(0.25))
    #flatten since too many dimensions, we only want a classification output
    model.add(Flatten())
    #fully connected to get all relevant data
    model.add(Dense(128, activation='relu'))
    #one more dropout for convergence' sake :)
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile("adam", "binary_crossentropy", metrics=["accuracy"])

    (x_train, y_train), (x_test, y_test) = load_partition()

    history = model.fit(
        x_train,
        y_train,
        batch_size=32,
        epochs=13,
        validation_split=0.1,
    )

    loss, accuracy = model.evaluate(x_test, y_test)
    print("test_accuracy", accuracy)


def load_partition():
    (x_train, y_train), (x_test, y_test) = utils.load_data(-1)
    return (x_train.reshape(len(x_train), 28, 28, 1), y_train), (x_test.reshape(len(x_test), 28, 28, 1), y_test)

main()