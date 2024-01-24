# Helper module for data visualisation

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torchvision import utils


def plot_decision_boundary2d(model, data, targets, xmargin=1, ymargin=1, showData=True):
    x_min, x_max = min(data[:, 0]) - xmargin, max(data[:, 0]) + xmargin
    y_min, y_max = min(data[:, 1]) - ymargin, max(data[:, 1]) + ymargin

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

    model_input = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float()

    Z = model(model_input.detach()).reshape(xx.shape)

    plt.contourf(xx, yy, Z.detach(), levels=50, cmap=cm.Reds)
    if showData:
        plt.scatter(data[:, 0], data[:, 1], c=targets, s=1)
    plt.show()


def view_images(images, ncol=5):
    img_grid = utils.make_grid(images, ncol)
    plt.figure(figsize=(12, 10))
    plt.imshow(img_grid.permute(1, 2, 0))
    plt.show()
