import cv2
import glob
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from data import create_imgs


def create_data(size, color, shape, data_size, set_type):  # (folder, size='big', color='blue', shape='b')
    data_map = {
        "big": {
            "colors": ['black', 'blue', 'green', 'orange', 'red'],
            "shapes": ['circle', 'ellipse', 'rectangle', 'square']
        },
        "small": {
            "colors": ['black', 'blue', 'green', 'orange', 'red'],
            "shapes": ['circle', 'ellipse', 'rectangle', 'square', 'triangle']
        }
    }

    if color is not None:
        # remove color from train set and have only that color
        # appear in the test set
        if color[0] == '-':
            data_map[size]["colors"].remove(color[1:])
        elif color[0] == '+':
            data_map[size]["colors"].clear()
            data_map[size]["colors"].append(color[1:])
    elif shape is not None:
        # remove shape from train set and have only that shape
        # appear in the test set
        if shape[0] == '-':
            data_map[size]["shapes"].remove(shape[1:])
        elif shape[0] == '+':
            data_map[size]["shapes"].clear()
            data_map[size]["shapes"].append(shape[1:])

    # generate images for data set
    if len(os.listdir('./dataset/{}'.format(set_type))) == 0:  # if data set folder is empty, create images
        create_imgs(data_map, data_size, set_type)
    img_data = []  # np.zeros((60000, 28, 28, 3))
    for i, file in enumerate(os.listdir('./dataset/{}'.format(set_type))):
        img_data.append(cv2.imread(os.path.join('./dataset/{}'.format(set_type), file)).astype(float) / 255)
        # train_data[i, :, :, :] = cv2.imread(os.path.join('./dataset/train', file)).astype(float) / 255
        # image = train_data[i, :, :, :]
    img_data = np.transpose(np.array(img_data), (0, 3, 1, 2))

    return img_data


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = torch.Tensor(data)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def iterate_through_dataloader(dataset):
    trainloader = DataLoader(dataset, batch_size=55, shuffle=True)
    for i, x in enumerate(trainloader):
        print('i: {} shape of x: {}'.format(i, x.shape))


if __name__ == '__main__':
    print('Step 1')
    data = create_data('./dataset/train/')
    print('0 row')
    x = data[0, 0, 0:5, 0:5]
    print(x.shape)  # (10, 10)
    # print(x)
    print('Step 2')
    dataset = MyDataset(data)
    y = dataset[0]
    print('0 row again')
    print(y.shape)   # (28, 28)
    # print(y[0, 0:5, 0:5])
    print('Step 3')
    iterate_through_dataloader(dataset)