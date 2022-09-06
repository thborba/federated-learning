import dataset_without_pytorch
from dataset_without_pytorch import get_loader


DataClass = getattr(dataset_without_pytorch, 'PneumoniaMNIST')

def load_data(idx) :
    
    train_dataset = DataClass(split='train', download=True)
    x, y  = get_loader(dataset= train_dataset)

    test_dataset = DataClass(split='test', download=True) 
    xt, yt = get_loader(dataset= test_dataset)

    if(idx == -1) : 
        return (x, y), (xt, yt)

    x_train = x[idx * len(x) // 2 : (idx + 1) * len(x) // 2 ]
    y_train = y[idx * len(y) // 2  : (idx + 1) * len(y) // 2 ] 
    x_test = xt[idx * len(xt) // 2 : (idx + 1) * len(xt) // 2 ]
    y_test = yt[idx * len(yt) // 2  : (idx + 1) * len(yt) // 2 ]
    return (x_train, y_train), (x_test, y_test)

def load_test_data():
    test_dataset = DataClass(split='test', download=True) 
    x_val, y_val = get_loader(dataset= test_dataset)
    print(x_val.shape, y_val.shape)
    return (x_val, y_val)

