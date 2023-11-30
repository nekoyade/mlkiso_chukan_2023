from copy import deepcopy
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# データセットの読み込み

# 読み込むデータセットは CIFAR10．
# CIFAR10 は，乗り物や動物などの小さい RGB 画像集．
# 詳しくは https://www.cs.toronto.edu/~kriz/cifar.html 参照．

training_data = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=transforms.ToTensor())

test_data = datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=transforms.ToTensor())

training_data_loader = DataLoader(training_data, batch_size=64, shuffle=True)
test_data_loader = DataLoader(test_data, batch_size=64, shuffle=True)


# 読み込んだデータセットの可視化

label_map = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck"
}

fig = plt.figure(figsize=(8, 8))

rows, cols = 5, 5
for i in range(rows * cols):
    sample_index = torch.randint(len(training_data), size=(1,)).item()
    image, label = training_data[sample_index]

    fig.add_subplot(rows, cols, i + 1)
    plt.title(label_map[label])
    plt.axis("off")
    plt.imshow(image.permute(1, 2, 0))

plt.show()


# データローダーの確認

# train_features: torch.Size([N, C, W, H])
#     N: number of batches
#     C: number of channels in images from the dataset
#     W: width of images from the dataset
#     H: height of images from the dataset

# train_labels: torch.Size([N])

train_features, train_labels = next(iter(training_data_loader))
print(f"feature batch size: {train_features.size()}")
print(f"label batch size: {train_labels.size()}")


# モデルによるデータ変換の可視化のための準備

def load_sample_images(rows, cols):
    sample_images = []
    for _ in range(rows):
        tmp_list = []
        for _ in range(cols):
            sample_index = torch.randint(len(training_data), size=(1,)).item()
            image, _ = training_data[sample_index]
            tmp_list.append(image)
        sample_images.append(tmp_list)
    return sample_images

sample_images = load_sample_images(4, 8)

def show_concatenated_images(image_list_2d, title):
    rows = len(image_list_2d)

    image_rows = []
    for i in range(rows):
        image_rows.append(torch.cat(image_list_2d[i], dim=2))
    concatenated_image = torch.cat(image_rows, dim=1)

    channels = concatenated_image.size()[0]
    concatenated_image = torch.sum(concatenated_image, dim=0) / channels

    plt.figure(figsize=(10, 10))
    plt.title(title)
    plt.imshow(concatenated_image.detach().numpy(), cmap="viridis")

def show_data_transformation(images, funcs, titles):
    print("[original images]")
    print(f"size: {images[0][0].size()}")
    show_concatenated_images(images, "original images")

    for func, title in zip(funcs, titles):
        for i in range(len(images)):
            for j in range(len(images[i])):
                images[i][j] = func(images[i][j])

        print(f"[{title}]")
        print(f"size: {images[0][0].size()}")
        show_concatenated_images(images, title)


# データ変換の可視化（プーリング層：なし）

images = deepcopy(sample_images)
funcs = [
    nn.Conv2d(3, 64, kernel_size=3, stride=1),
    nn.Conv2d(64, 128, kernel_size=3, stride=1),
]
titles = [str(f) for f in funcs]
show_data_transformation(images, funcs, titles)


# データ変換の可視化（プーリング層：MaxPool2d）

images = deepcopy(sample_images)
funcs = [
    nn.Conv2d(3, 64, kernel_size=3, stride=1),
    nn.MaxPool2d(kernel_size=2),
    nn.Conv2d(64, 128, kernel_size=3, stride=1),
    nn.MaxPool2d(kernel_size=2)
]
titles = [str(f) for f in funcs]
show_data_transformation(images, funcs, titles)


# データ変換の可視化（プーリング層：AvgPool2d）

images = deepcopy(sample_images)
funcs = [
    nn.Conv2d(3, 64, kernel_size=3, stride=1),
    nn.AvgPool2d(kernel_size=2),
    nn.Conv2d(64, 128, kernel_size=3, stride=1),
    nn.AvgPool2d(kernel_size=2)
]
titles = [str(f) for f in funcs]
show_data_transformation(images, funcs, titles)


# 訓練に使用するデバイスを確認

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using {device} device for training")


# モデルの定義（プーリング層：なし）

class CustomCNN1(nn.Module):
    def __init__(self):
        super(CustomCNN1, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, len(label_map))

    def forward(self, x):
        # Using ReLU() as an activation function
        # input batch size:           [64,   3, 32, 32]
        x = self.conv1(x)           # [64,  64, 30, 30]
        x = nn.functional.relu(x)
        x = self.conv2(x)           # [64, 128, 28, 28]
        x = nn.functional.relu(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = nn.functional.relu(x)
        x = self.fc3(x)
        return x


# モデルの定義（プーリング層：MaxPool2d）

class CustomCNN2(nn.Module):
    def __init__(self):
        super(CustomCNN2, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(128 * 6 * 6, 512)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, len(label_map))

    def forward(self, x):
        # Using ReLU() as an activation function
        # input batch size:           [64,   3, 32, 32]
        x = self.conv1(x)           # [64,  64, 30, 30]
        x = nn.functional.relu(x)
        x = self.pool1(x)           # [64,  64, 15, 15]
        x = self.conv2(x)           # [64, 128, 13, 13]
        x = nn.functional.relu(x)
        x = self.pool2(x)           # [64, 128,  6,  6]
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = nn.functional.relu(x)
        x = self.fc3(x)
        return x


# モデルの定義（プーリング層：AvgPool2d）

class CustomCNN3(nn.Module):
    def __init__(self):
        super(CustomCNN3, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1)
        self.pool1 = nn.AvgPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.pool2 = nn.AvgPool2d(kernel_size=2)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(128 * 6 * 6, 512)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, len(label_map))

    def forward(self, x):
        # Using ReLU() as an activation function
        # input batch size:           [64,   3, 32, 32]
        x = self.conv1(x)           # [64,  64, 30, 30]
        x = nn.functional.relu(x)
        x = self.pool1(x)           # [64,  64, 15, 15]
        x = self.conv2(x)           # [64, 128, 13, 13]
        x = nn.functional.relu(x)
        x = self.pool2(x)           # [64, 128,  6,  6]
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = nn.functional.relu(x)
        x = self.fc3(x)
        return x


# 学習用の関数の定義

def train_loop(model, training_data_loader, optimizer, epoch):
    train_loss_ave = 0.0
    size = len(training_data_loader.dataset)

    print(f"[epoch {epoch+1}]")

    model.train()
    start_time = time.perf_counter()
    for batch_index, (x, y) in enumerate(training_data_loader):
        # x: features
        # y: labels
        x = x.to(device)
        y = y.to(device)
        pred = model(x)  # not softmax-ed yet
        loss = nn.functional.cross_entropy(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss_ave += loss.item()

        if batch_index % 100 == 0:
            current = (batch_index + 1) * len(x)
            print(f"batch: {batch_index+1:>5d} ({current:>5d}/{size:>5d})",
                  end="  ")
            print(f"loss: {loss.item():>7f}")
    end_time = time.perf_counter()
    delta_time = end_time - start_time

    train_loss_ave /= size
    return train_loss_ave, delta_time


# テスト用の関数の定義

def test_loop(model, test_data_loader):
    correct = 0

    model.eval()
    for x, y in test_data_loader:
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        y_hat = pred.argmax(dim=1, keepdim=True)

        correct += y_hat.eq(y.view_as(y_hat)).sum().item()

    accuracy = correct / len(test_data_loader.dataset)
    print(f"test-set accuracy: {accuracy:>12f}\n")

    return accuracy


# 学習・テスト用メインループ

epochs = 5
epoch_list = [str(i + 1) for i in range(epochs)]

accuracy_dict = {1: [], 2: [], 3: []}
time_dict = {1: [], 2: [], 3: []}
time_ave_dict = {1: [], 2: [], 3: []}

def main(model_type, key):
    model = model_type().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    accuracy_dict[key] = []
    time_dict[key] = []

    for epoch in range(epochs):
        _, dt = train_loop(model, training_data_loader, optimizer, epoch)
        time_dict[key].append(dt)

        accuracy = test_loop(model, test_data_loader)

        accuracy_dict[key].append(accuracy * 100)


# 学習・テスト

main(CustomCNN1, 1)
main(CustomCNN2, 2)
main(CustomCNN3, 3)


# グラフの表示

fig = plt.figure()
ax = fig.add_subplot(111)
for i in [1, 2, 3]:
    ax.plot(epoch_list, accuracy_dict[i], label=f"CustomCNN{i}")
ax.set_ylim(0, 100)
ax.set_xlabel("epoch")
ax.set_ylabel("accuracy")
ax.legend()
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
x = np.arange(len(epoch_list))
margin = 0.2
total_width = 1.0 - margin
width = total_width / 3
for i in [1, 2, 3]:
    tmp_x = x + width*(i - 2)
    ax.bar(tmp_x, time_dict[i], width=width, label=f"CustomCNN{i}")
plt.xticks(x, epoch_list)
ax.set_xlabel("epoch")
ax.set_ylabel("total computing time [s]")
ax.legend()
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
x = np.arange(3)
tmp_list = []
for i in [1, 2, 3]:
    time_ave_dict[i] = [sum(time_dict[i]) / len(time_dict[i])]
    tmp_list.extend(time_ave_dict[i])
ax.bar(x, tmp_list, width=0.5)
plt.xticks(x, ["CustomCNN1", "CustomCNN2", "CustomCNN3"])
ax.set_xlabel("model")
ax.set_ylabel("computing time per epoch [s]")
plt.show()
