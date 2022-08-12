
# from tqdm import tqdm
import numpy as np

from medmnist import INFO

import dataset_without_pytorch
from dataset_without_pytorch import get_loader




data_flag = 'pneumoniamnist'
download = True

NUM_EPOCHS = 3
BATCH_SIZE = 128
lr = 0.001

info = INFO[data_flag]
task = info['task']
n_channels = info['n_channels']
n_classes = len(info['label'])

DataClass = getattr(dataset_without_pytorch, info['python_class'])

# print (DataClass)
# load the data
train_dataset = DataClass(split='train', download=download) 
# print(type(train_dataset), train_dataset)

# encapsulate data into dataloader form
train_loader = get_loader(dataset=train_dataset, batch_size=BATCH_SIZE)
# print(type(train_loader), train_loader)
# print(train_dataset)


train_dataset.montage(length=20)
print(train_loader)

# print(np.array(x).shape, y.shape)

# for x, y in train_loader:
#     print(x.shape, y.shape)
#     break