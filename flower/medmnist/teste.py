import numpy as np
from medmnist import INFO
import dataset_without_pytorch
from dataset_without_pytorch import get_loader

def load_data(idx) :
    data_flag = 'pneumoniamnist'
    info = INFO[data_flag]
    DataClass = getattr(dataset_without_pytorch, info['python_class']);

    train_dataset = DataClass(split='train', download=True)
    x, y  = get_loader(dataset= train_dataset)
    x_train = x[idx * len(x) // 2 : (idx + 1) * len(x) // 2 ]
    y_train = y[idx * len(y) // 2  : (idx + 1) * len(y) // 2 ]
 
    test_dataset = DataClass(split='test', download=True) 
    x, y = get_loader(dataset= test_dataset)
    x_test = x[idx * len(x) // 2 : (idx + 1) * len(x) // 2 ]
    y_test = y[idx * len(y) // 2  : (idx + 1) * len(y) // 2 ]

    return (x_train, y_train), (x_test, y_test) 

