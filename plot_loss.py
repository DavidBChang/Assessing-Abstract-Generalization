import numpy as np
import matplotlib.pyplot as plt


def plot(arch_name_1, arch_name_2):
    lr1 = []
    lr2 = []
    lr3 = []
    cnn_lr1 = []
    cnn_lr2 = []
    cnn_lr3 = []
    with open('results/{}/lr_1e-03/1e-03.txt'.format(arch_name_1), 'r') as f:
        lr1 = np.array([float(x) for x in f.read().split('\n')[:224]])
    with open('results/{}/lr_5e-04/5e-04.txt'.format(arch_name_1), 'r') as f:
        lr2 = np.array([float(x) for x in f.read().split('\n')[:224]])
    with open('results/{}/lr_1e-04/1e-04.txt'.format(arch_name_1), 'r') as f:
        lr3 = np.array([float(x) for x in f.read().split('\n')[:224]])

    with open('results/{}/lr_1e-03/1e-03.txt'.format(arch_name_2), 'r') as f:
        cnn_lr1 = np.array([float(x) for x in f.read().split('\n')[:224]])
    with open('results/{}/lr_5e-04/5e-04.txt'.format(arch_name_2), 'r') as f:
        cnn_lr2 = np.array([float(x) for x in f.read().split('\n')[:224]])
    with open('results/{}/lr_1e-04/1e-04.txt'.format(arch_name_2), 'r') as f:
        cnn_lr3 = np.array([float(x) for x in f.read().split('\n')[:224]])

    x = np.arange(1, 225)
    plt.plot(x, lr1, color='red', label='lr 1e-3')
    plt.plot(x, lr2, color='yellow', label='lr 5e-4')
    plt.plot(x, lr3, color='blue', label='lr 1e-4')
    plt.plot(x, cnn_lr1, color='black', linestyle='dashed', label='cnn lr 1e-3')
    plt.plot(x, cnn_lr1, color='black', linestyle='dashed', label='cnn lr 5e-4')
    plt.plot(x, cnn_lr1, color='black', linestyle='dashed', label='cnn lr 1e-4')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Relating Number of Epochs to the Loss')
    plt.show()


if __name__ == "__main__":
    arch_name_1 = 'linear'
    arch_name_2 = 'CNN'
    plot(arch_name_1, arch_name_2)
