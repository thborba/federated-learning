
# from tqdm import tqdm
import numpy as np

from medmnist import INFO

import dataset_without_pytorch
from dataset_without_pytorch import get_loader



def load_data(idx) :
    data_flag = 'pneumoniamnist'
    BATCH_SIZE = 4708

    info = INFO[data_flag]
    DataClass = getattr(dataset_without_pytorch, info['python_class']);

    train_dataset = DataClass(split='train', download=True) 

    train_loader = get_loader(dataset= train_dataset, batch_size=BATCH_SIZE)

    for x, y in train_loader :
       x_train = x[idx * 2354 : (idx + 1) * 2354]
       y_train = y[idx * 2354 : (idx + 1) * 2354]
       break    
    
    test_dataset = DataClass(split='test', download=True) 
    test_loader = get_loader(dataset= test_dataset, batch_size=BATCH_SIZE)

    for x, y in test_loader :
       print(x.shape)
       x_test = x[idx * 2354 : (idx + 1) * 2354]
       y_test = y[idx * 2354 : (idx + 1) * 2354]
       break

    return (x_train, y_train), (x_test, y_test) 

(x_train, y_train), (x_test, y_test)  = load_data(1)


print(x_test.shape)

