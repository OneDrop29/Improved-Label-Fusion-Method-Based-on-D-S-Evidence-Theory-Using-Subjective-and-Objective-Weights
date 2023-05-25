from torchvision.models import resnet18, vgg11, alexnet
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from torch.optim import Adam, SGD
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchmetrics.classification import MulticlassAccuracy, MulticlassRecall, MulticlassPrecision, MulticlassF1Score

import numpy as np
import torch
import torch.nn as nn

from fusion import ds_fusion_torch, contrast_algorithm_zhang_torch, contrast_algorithm_jiang_torch, contrast_algorithm_bai_torch
from utils import MeanCalculater

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
epochs = 2 ** 8
batch_size = 128
lr = 5e-3
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(model_name='resnet'):
    train_set = CIFAR10('./dataset/cifar10', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(mean=cifar10_mean, std=cifar10_std),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomCrop(size=32,
                                                  padding=int(32 * 0.125),
                                                  padding_mode='reflect'),
                        ]))

    test_set = CIFAR10('./dataset/cifar10', train=False, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize(mean=cifar10_mean, std=cifar10_std),
                           transforms.RandomHorizontalFlip(),
                           transforms.RandomCrop(size=32,
                                                 padding=int(32 * 0.125),
                                                 padding_mode='reflect'),
                       ]))

    if model_name == 'resnet':
        model = resnet18(num_classes=10)
    elif model_name == 'vgg':
        model = vgg11(num_classes=10)
    elif model_name == 'alexnet':
        model = alexnet(num_classes=10)
        model.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
    else:
        raise "no such model"

    model.to(device)
    criterion = CrossEntropyLoss()
    # optimizer = Adam(params=model.parameters(), lr=lr)
    optimizer = SGD(params=model.parameters(), lr=0.01, momentum=0.8, dampening=0.001)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
    loss_mean = MeanCalculater()

    best_acc = 0

    for epoch in tqdm(range(epochs)):
        loss_mean.clear()
        for i, data in enumerate(train_loader):
            x, y = data
            x, y = x.to(device), y.to(device)

            output = model(x)
            loss = criterion(output, y).to(device)

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
            loss_mean.add_value(loss.item())

            if i % 100 == 99:
                print(f"[{epoch + 1}, {i + 1}] loss: {loss_mean.mean_value}")

        # scheduler.step()

        if epoch % 10 == 9:
            model.eval()
            with torch.no_grad():
                acc = 0
                for i, data in enumerate(test_loader):
                    x, y = data
                    x, y = x.to(device), y.to(device)

                    output = model(x)
                    _, output_labels = torch.max(output, dim=1)
                    acc += (y == output_labels).sum()

            print(f"[{epoch + 1}] acc: {acc / len(test_set) * 100}%")
            torch.save({'model': model.state_dict(),
                        'loss': loss_mean.mean_value,
                        'optimizer': optimizer.state_dict(),
                        # 'scheduler': scheduler.state_dict(),
                        'acc': acc}, f"./models/{model_name}/{model_name}_e{epoch + 1}.pt")

            if acc > best_acc:
                best_acc = acc
                torch.save({'model': model.state_dict(),
                            'loss': loss_mean.mean_value,
                            'optimizer': optimizer.state_dict(),
                            # 'scheduler': scheduler.state_dict(),
                            'acc': best_acc}, f"./models/{model_name}/{model_name}_best.pt")

            model.train()


def calc_metrics():
    test_set = CIFAR10('./dataset/cifar10', train=False, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize(mean=cifar10_mean, std=cifar10_std),
                           transforms.RandomHorizontalFlip(),
                           transforms.RandomCrop(size=32,
                                                 padding=int(32 * 0.125),
                                                 padding_mode='reflect'),
                       ]))
    test_loader = DataLoader(test_set, batch_size=100)

    # model = resnet18(num_classes=10)
    # model_dict = torch.load("./models/resnet/resnet_best.pt")

    # model = vgg11(num_classes=10)
    # model_dict = torch.load("./models/vgg/vgg_best.pt")

    model = alexnet(num_classes=10)
    model_dict = torch.load("./models/alexnet/alexnet_best.pt")
    model.features = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 192, kernel_size=5, padding=2),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Conv2d(192, 384, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(384, 256, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2),
    )

    model.load_state_dict(model_dict['model'])
    mean_calculator = MeanCalculater()

    # metrics = MulticlassAccuracy(num_classes=10, average='micro')
    # metrics = MulticlassPrecision(num_classes=10, average='micro')
    # metrics = MulticlassRecall(num_classes=10, average='micro')
    metrics = MulticlassF1Score(num_classes=10, average='micro')

    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            output = torch.softmax(model(x), dim=-1)
            mean_calculator.add_value(metrics(output, y).item())

    print(mean_calculator.mean_value)


def get_divergence_label():
    test_set = CIFAR10('./dataset/cifar10', train=False, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize(mean=cifar10_mean, std=cifar10_std),
                           transforms.RandomHorizontalFlip(),
                           transforms.RandomCrop(size=32,
                                                 padding=int(32 * 0.125),
                                                 padding_mode='reflect'),
                       ]))
    test_loader = DataLoader(test_set, batch_size=1)

    resnet_model = resnet18(num_classes=10)
    resnet_model.load_state_dict(torch.load("./models/resnet/resnet_best.pt")['model'])

    vgg_model = vgg11(num_classes=10)
    vgg_model.load_state_dict(torch.load("./models/vgg/vgg_best.pt")['model'])

    alexnet_model = alexnet(num_classes=10)
    alexnet_model.features = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 192, kernel_size=5, padding=2),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Conv2d(192, 384, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(384, 256, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2),
    )
    alexnet_model.load_state_dict(torch.load("./models/alexnet/alexnet_best.pt")['model'])

    resnet_model.eval()
    vgg_model.eval()
    alexnet_model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            resnet_output = torch.softmax(resnet_model(x), dim=-1)
            vgg_output = torch.softmax(vgg_model(x), dim=-1)
            alexnet_output = torch.softmax(alexnet_model(x), dim=-1)

            _, resnet_label = torch.max(resnet_output, dim=1)
            _, vgg_label = torch.max(vgg_output, dim=1)
            _, alexnet_label = torch.max(alexnet_output, dim=1)

            if not (alexnet_label.item() == resnet_label.item() and alexnet_label.item() == vgg_label.item()):
                return {
                    'resnet': resnet_output,
                    'vgg': vgg_output,
                    'alexnet': alexnet_output,
                    'y': y
                }


def get_fusion_metrics():
    test_set = CIFAR10('./dataset/cifar10', train=False, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize(mean=cifar10_mean, std=cifar10_std),
                           # transforms.RandomHorizontalFlip(),
                           # transforms.RandomCrop(size=32,
                           #                       padding=int(32 * 0.125),
                           #                       padding_mode='reflect'),
                       ]))
    test_loader = DataLoader(test_set, batch_size=100)

    resnet_model = resnet18(num_classes=10)
    resnet_model.load_state_dict(torch.load("./models/resnet/resnet_best.pt")['model'])

    vgg_model = vgg11(num_classes=10)
    vgg_model.load_state_dict(torch.load("./models/vgg/vgg_best.pt")['model'])

    alexnet_model = alexnet(num_classes=10)
    alexnet_model.features = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 192, kernel_size=5, padding=2),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Conv2d(192, 384, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(384, 256, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2),
    )
    alexnet_model.load_state_dict(torch.load("./models/alexnet/alexnet_best.pt")['model'])

    metrics = MulticlassAccuracy(num_classes=10, average='micro')
    # metrics = MulticlassPrecision(num_classes=10, average='micro')
    # metrics = MulticlassRecall(num_classes=10, average='micro')
    # metrics = MulticlassF1Score(num_classes=10, average='micro')

    resnet_model.eval()
    vgg_model.eval()
    alexnet_model.eval()

    resnet_mean_calculator = MeanCalculater()
    vgg_mean_calculator = MeanCalculater()
    alexnet_mean_calculator = MeanCalculater()
    jiang_mean_calculator = MeanCalculater()
    bai_mean_calculator = MeanCalculater()
    # zhang_mean_calculator = MeanCalculater()
    cos_mean_calculator = MeanCalculater()
    js_mean_calculator = MeanCalculater()
    bd_mean_calculator = MeanCalculater()

    with torch.no_grad():
        for x, y in tqdm(test_loader):
            resnet_output = torch.softmax(resnet_model(x), dim=-1)
            vgg_output = torch.softmax(vgg_model(x), dim=-1)
            alexnet_output = torch.softmax(alexnet_model(x), dim=-1)

            jiang_output = []
            bai_output = []
            # zhang_output = []
            cos_output = []
            js_output = []
            bd_output = []
            for i in range(resnet_output.shape[0]):
                e = torch.vstack((torch.unsqueeze(resnet_output[i], dim=0),
                                  torch.unsqueeze(vgg_output[i], dim=0),
                                  torch.unsqueeze(alexnet_output[i], dim=0)))

                jiang_output.append(contrast_algorithm_jiang_torch(e))
                bai_output.append(contrast_algorithm_bai_torch(e))

                cos_output.append(ds_fusion_torch(e, distance='cos',
                                                  beta=torch.Tensor([[0.8317, 0.8726, 0.7857],
                                                                     [0.8366, 0.8735, 0.7828],
                                                                     [1, 1, 0.8]]),
                                                  W=torch.Tensor([[1, 4, 5, 9],
                                                                  [1 / 4, 1, 1, 5],
                                                                  [1 / 5, 1, 1, 5],
                                                                  [1 / 9, 1 / 5, 1 / 5, 1]])))

                js_output.append(ds_fusion_torch(e, distance='js',
                                                 beta=torch.Tensor([[0.8317, 0.8726, 0.7857],
                                                                    [0.8366, 0.8735, 0.7828],
                                                                    [1, 1, 0.8]]),
                                                 W=torch.Tensor([[1, 4, 5, 9],
                                                                 [1 / 4, 1, 1, 5],
                                                                 [1 / 5, 1, 1, 5],
                                                                 [1 / 9, 1 / 5, 1 / 5, 1]])))

                bd_output.append(ds_fusion_torch(e, distance='bd',
                                                 beta=torch.Tensor([[0.8317, 0.8726, 0.7857],
                                                                    [0.8366, 0.8735, 0.7828],
                                                                    [1, 1, 0.8]]),
                                                 W=torch.Tensor([[1, 4, 5, 9],
                                                                 [1 / 4, 1, 1, 5],
                                                                 [1 / 5, 1, 1, 5],
                                                                 [1 / 9, 1 / 5, 1 / 5, 1]])))

            resnet_mean_calculator.add_value(metrics(resnet_output, y).item())
            vgg_mean_calculator.add_value(metrics(vgg_output, y).item())
            alexnet_mean_calculator.add_value(metrics(alexnet_output, y).item())

            jiang_mean_calculator.add_value(metrics(torch.stack(jiang_output), y).item())
            bai_mean_calculator.add_value(metrics(torch.stack(bai_output), y).item())
            # zhang_mean_calculator.add_value(metrics(torch.stack(zhang_output), y).item())
            cos_mean_calculator.add_value(metrics(torch.stack(cos_output), y).item())
            js_mean_calculator.add_value(metrics(torch.stack(js_output), y).item())
            bd_mean_calculator.add_value(metrics(torch.stack(bd_output), y).item())

    data = [
        (resnet_mean_calculator.mean_value + vgg_mean_calculator.mean_value + alexnet_mean_calculator.mean_value) / 3.,
        jiang_mean_calculator.mean_value, bai_mean_calculator.mean_value, cos_mean_calculator.mean_value,
        js_mean_calculator.mean_value, bd_mean_calculator.mean_value]

    l = [i for i in range(6)]

    import matplotlib.pyplot as plt
    plt.bar(l, data, alpha=0.5, width=0.5)
    plt.xlabel("methods")
    plt.ylabel("accuracy")
    plt.ylim([0, 1])

    for i, d in zip(l, data):
        plt.text(i, d, '%.3f'%d, ha='center', va='bottom', fontsize=12)
    plt.xticks(l, ["model average", "Jiang's", "Bai's", "ours(cos)", "ours(js)", "ours\n(bhattacharyya)"])
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # train('alexnet')
    # calc_metrics()
    # out = get_divergence_label()
    # print(*out.values(), sep='\n')
    #
    # e = torch.vstack((out['resnet'], out['vgg'], out['alexnet']))
    #
    # print(ds_fusion_torch(e, distance='bd',
    #                       beta=torch.Tensor([[0.8317, 0.8726, 0.7857],
    #                                          [0.8366, 0.8735, 0.7828],
    #                                          [0.1, 1, 0.8]]),
    #                       W=torch.Tensor([[1, 4, 5, 9],
    #                                       [1 / 4, 1, 1, 5],
    #                                       [1 / 5, 1, 1, 5],
    #                                       [1 / 9, 1 / 5, 1 / 5, 1]])))
    # print(contrast_algorithm_zhang_torch(e))
    # print(contrast_algorithm_jiang(e.numpy()))

    get_fusion_metrics()
