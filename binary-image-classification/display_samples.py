import matplotlib
import torch
import numpy as np
from torchvision import utils
import matplotlib.pyplot as plt

from split_dataset import get_train_val_ds

# seeding
np.random.seed(0)

train_ds, val_ds = get_train_val_ds()


def show(img, y, color=False):
    # convert tensor to np.array
    npimg = img.numpy()

    # convert to HxWxC shape
    npimg_tr = np.transpose(npimg, (1, 2, 0))

    if color == False:
        npimg_tr = npimg_tr[:, :, 0]  # just take the first channel
        plt.imshow(
            npimg_tr,
            # interpolation="nearest",
            cmap="gray",
        )
    else:
        plt.imshow(
            npimg_tr,
            # interpolation="nearest"
        )
    plt.title("label: " + str(y))
    plt.show()


total_images_to_display = 4

# train ds
rnd_inds = np.random.randint(0, len(train_ds), size=total_images_to_display)
print("image indices: ", rnd_inds)

x_grid_train = [train_ds[i][0] for i in rnd_inds]
y_grid_train = [train_ds[i][1] for i in rnd_inds]

x_grid_train = utils.make_grid(x_grid_train, nrow=total_images_to_display, padding=2)
print(x_grid_train.shape)

plt.rcParams["figure.figsize"] = (10, 5)
show(x_grid_train, y_grid_train, True)

# val ds
rnd_inds = np.random.randint(0, len(val_ds), size=total_images_to_display)
print("image indices: ", rnd_inds)

x_grid_train = [val_ds[i][0] for i in rnd_inds]
y_grid_train = [val_ds[i][1] for i in rnd_inds]

x_grid_train = utils.make_grid(x_grid_train, nrow=total_images_to_display, padding=2)
print(x_grid_train.shape)

plt.rcParams["figure.figsize"] = (10, 5)
show(x_grid_train, y_grid_train, True)
