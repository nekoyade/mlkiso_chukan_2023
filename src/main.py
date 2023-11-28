import matplotlib.pyplot as plt
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


# モデルによるデータ変換の可視化

sample_images = []

rows, cols = 4, 8
for i in range(rows):
    tmp_list = []
    for j in range(cols):
        sample_index = torch.randint(len(training_data), size=(1,)).item()
        image, _ = training_data[sample_index]
        tmp_list.append(image)
    sample_images.append(tmp_list)

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

show_concatenated_images(sample_images, "sample images")

f = nn.Conv2d(3, 32, kernel_size=3, stride=1)
for i in range(rows):
    for j in range(cols):
        sample_images[i][j] = f(sample_images[i][j])

show_concatenated_images(
    sample_images, "nn.Conv2d(3, 32, kernel_size=3, stride=1)")

f = nn.MaxPool2d(kernel_size=2)
for i in range(rows):
    for j in range(cols):
        sample_images[i][j] = f(sample_images[i][j])

show_concatenated_images(
    sample_images, "nn.MaxPool2d(kernel_size=2)")

f = nn.Conv2d(32, 64, kernel_size=3, stride=1)
for i in range(rows):
    for j in range(cols):
        sample_images[i][j] = f(sample_images[i][j])

show_concatenated_images(
    sample_images, "nn.Conv2d(32, 64, kernel_size=3, stride=1)")

f = nn.MaxPool2d(kernel_size=2)
for i in range(rows):
    for j in range(cols):
        sample_images[i][j] = f(sample_images[i][j])

show_concatenated_images(
    sample_images, "nn.MaxPool2d(kernel_size=2)")


# 訓練に使用するデバイスを確認

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using {device} device for training")


# モデルの定義

class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = nn.ReLU(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = nn.ReLU(x)
        # Add more... (referred to AlexNet)
