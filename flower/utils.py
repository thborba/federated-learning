import dataset_without_pytorch
from dataset_without_pytorch import get_loader
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

DataClass = getattr(dataset_without_pytorch, 'PneumoniaMNIST')
CLIENTS_NUMBER = 2

def load_data(id = -1) :
    train_dataset = DataClass(split='train', download=True)
    x, y  = get_loader(dataset= train_dataset)
    test_dataset = DataClass(split='test', download=True) 
    xt, yt = get_loader(dataset= test_dataset)

    if(id == -1) : 
        return reshape_data(x, y, xt, yt)

    (x_train, y_train) = get_partial_data_according_to_client_number(x, y, id)
    (x_test, y_test)  = get_partial_data_according_to_client_number(xt, yt, id)
    
    return reshape_data(x_train, y_train, x_test, y_test)

def get_partial_data_according_to_client_number(x, y, id) :
    data_per_client = len(x) // CLIENTS_NUMBER
    return (x[id * data_per_client: (id + 1) * data_per_client], y[id * data_per_client: (id + 1) * data_per_client])

def reshape_data(x_train, y_train, x_test, y_test) : 
    return (x_train.reshape(len(x_train), 28, 28, 1), y_train), (x_test.reshape(len(x_test), 28, 28, 1), y_test)

def get_model() :
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
    return model
