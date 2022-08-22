import dataset_without_pytorch
from dataset_without_pytorch import get_loader


DataClass = getattr(dataset_without_pytorch, 'PneumoniaMNIST')

def load_data(idx) :
    train_dataset = DataClass(split='train', download=True)
    x, y  = get_loader(dataset= train_dataset)
    x_train = x[idx * len(x) // 2 : (idx + 1) * len(x) // 2 ]
    y_train = y[idx * len(y) // 2  : (idx + 1) * len(y) // 2 ]
    
    test_dataset = DataClass(split='test', download=True) 
    x, y = get_loader(dataset= test_dataset)
    x_test = x[idx * len(x) // 2 : (idx + 1) * len(x) // 2 ]
    y_test = y[idx * len(y) // 2  : (idx + 1) * len(y) // 2 ]
    return (x_train, y_train), (x_test, y_test)

def load_validation_data():
    validation_dataset = DataClass(split='val', download=True) 
    x_val, y_val = get_loader(dataset= validation_dataset)
    print(x_val.shape, y_val.shape)
    return (x_val, y_val)

