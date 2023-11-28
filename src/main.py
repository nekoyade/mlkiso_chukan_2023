import matplotlib.pyplot as plt
import torch
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
