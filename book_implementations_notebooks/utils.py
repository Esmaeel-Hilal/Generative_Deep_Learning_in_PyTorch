# Function from Generative Deep Learning (2nd Edition) by David Foster
# Source: https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition
# Licensed under MIT License


import matplotlib.pyplot as plt
import numpy as np

def display(images, n=10, size=(20, 3), cmap="gray_r", as_type="float32", save_to=None):
    """
    Displays n images from a PyTorch batch of shape [B, C, H, W]
    """
    images = images[:n].cpu().numpy()  # first n images, move to CPU
    if images.shape[1] == 1:           # remove channel dim if grayscale
        images = images[:, 0]

    plt.figure(figsize=size)
    for i in range(n):
        plt.subplot(1, n, i + 1)
        plt.imshow(images[i].astype(as_type), cmap=cmap)
        plt.axis("off")

    if save_to:
        plt.savefig(save_to)
        print(f"\nSaved to {save_to}")

    plt.show()