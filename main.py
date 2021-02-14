from __future__ import print_function
import argparse
import numpy as np
import torch
import torch.utils.data
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

from create_dataset import create_data, MyDataset

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=300, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--CNN-flag', type=int, default=0)
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

train_type = 'bigblue-circle'
test_type = 'bigblue+circle'
train_data = create_data('big', 'blue', '-circle', 120, 'train/' + train_type)  # 320 distinct imgs, 38400 total
test_test_data = create_data('big', 'blue', '+circle', 120, 'test/' + test_type)   # 80 distinct imgs, 9600 total
test_train_data = create_data('big', 'blue', '-circle', 30, 'test/' + train_type)  # 320 distinct imgs, 9600 total
train_dataset = MyDataset(train_data)
test_test_dataset = MyDataset(test_test_data)
test_train_dataset = MyDataset(test_train_data)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_test_loader = DataLoader(test_test_dataset, batch_size=args.batch_size, shuffle=True)
test_train_loader = DataLoader(test_train_dataset, batch_size=args.batch_size, shuffle=True)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input, size=32):
        return input.view(input.size(0), size, 1, 1)


class VAECNN(nn.Module):
    def __init__(self, image_channels=3, h_dim=32, z_dim=20):
        super(VAECNN, self).__init__()
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=4, stride=2),  # (B, 32, 13, 13)
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=4, stride=2),  # (B, 64, 5, 5)
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=4, stride=2),  #(B, 128, 1, 1)
            nn.ReLU(),
            Flatten()
        )

        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)

        self.decoder = nn.Sequential(
            UnFlatten(),  # (B, 128, 1, 1)
            nn.ConvTranspose2d(32, 32, kernel_size=5, stride=2),  # (B, 64, 5, 5)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=5, stride=2),  # (B, 32, 13, 13)
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, kernel_size=4, stride=2),  # (B, 3, 28, 28)
            nn.Sigmoid(),
        )

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size())
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def representation(self, x):
        return self.bottleneck(self.encoder(x))[0]

    def forward(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return self.decode(z), mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        return self.decoder(z)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.z_dim = 20

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, self.z_dim)
        self.fc22 = nn.Linear(400, self.z_dim)
        self.fc3 = nn.Linear(self.z_dim, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))  # (B, 784) x (784, 400) --> (B, 400)
        return self.fc21(h1), self.fc22(h1)  # (B, 400) X (400, 20) --> (B, 20)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE().to(device)
arch_name = 'linear'
if args.CNN_flag == 1:
    model = VAECNN().to(device)
    arch_name = 'CNN'
optimizer = optim.Adam(model.parameters(), lr=args.lr)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    if (args.CNN_flag == 1):
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, *recon_x.shape[1:]), reduction='sum')
    else:
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model.forward(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))
    return train_loss / len(train_loader.dataset)


def test(epoch, test_loader, test_folder):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                # for j in range(n):
                #     print(torch.mean(data[j][0]) == 0 and torch.mean(data[j][1]) == 0
                #           and torch.mean(data[j][2]) == 0)
                # print(data[:n].shape) 8, 3, 28, 28
                comparison = torch.cat([data[:n],
                                      recon_batch.view(args.batch_size, 3, 28, 28)[:n]])
                save_image(comparison.cpu(),
                           'experiments/CNN_lr_1e-03/{}/{}/reconstruction_'.format(train_type, test_folder) + str(epoch)
                           + '.png', nrow=n)
                           # 'results/{}/lr_{:.0e}/reconstruction_'.format(arch_name, args.lr) + str(epoch) + '.png',
                           # nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return test_loss


if __name__ == "__main__":
    # losses = []
    # f = open("results/{}/lr_{:.0e}/{:.0e}.txt".format(arch_name, args.lr, args.lr), "w")
    # f.close()
    f = open("experiments/CNN_lr_1e-03/{}/train/{}_losses.txt".format(train_type, train_type), "w")
    f.close()
    f = open("experiments/CNN_lr_1e-03/{}/test_test/{}_losses.txt".format(train_type, test_type), "w")
    f.close()
    f = open("experiments/CNN_lr_1e-03/{}/test_train/{}_losses.txt".format(train_type, train_type), "w")
    f.close()

    for epoch in range(1, args.epochs + 1):
        loss = train(epoch)
        with open("experiments/CNN_lr_1e-03/{}/train/{}_losses.txt".format(train_type, train_type), 'a') as f:
            f.write('{}\n'.format(loss))
        # losses.append(loss)
        loss = test(epoch, test_test_loader, 'test_test')
        with open("experiments/CNN_lr_1e-03/{}/test_test/{}_losses.txt".format(train_type, test_type), 'a') as f:
            f.write('{}\n'.format(loss))
        loss = test(epoch, test_train_loader, 'test_train')
        with open("experiments/CNN_lr_1e-03/{}/test_train/{}_losses.txt".format(train_type, train_type), 'a') as f:
            f.write('{}\n'.format(loss))
        with torch.no_grad():
            num_samples = 64 * 3
            if args.CNN_flag == 1:
                num_samples = 64
            sample = torch.randn(num_samples, model.z_dim).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(num_samples, 3, 28, 28),
                       'experiments/CNN_lr_1e-03/{}/samples/sample_'.format(train_type) + str(epoch) + '.png')
    # plotLoss(np.array(losses))
    f.close()
