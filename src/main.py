import matplotlib.pyplot as plt
import torch
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
