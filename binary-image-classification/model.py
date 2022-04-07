import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from split_dataset import get_train_val_ds


# helper function to define the output shape of a conv layer
def findConv2dOutShape(H_in, W_in, conv: nn.Conv2d, pool=2):
    # get conv arguments
    kernel_size = conv.kernel_size
    stride = conv.stride
    padding = conv.padding
    dilation = conv.dilation

    # Ref: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d
    H_out = np.floor(
        (H_in + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1
    )
    W_out = np.floor(
        (W_in + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1
    )

    if pool:
        H_out /= pool
        W_out /= pool
    return int(H_out), int(W_out)


class Net(nn.Module):
    def __init__(self, params):
        super().__init__()
        C_in, H_in, W_in = params["input_shape"]
        init_f = params["initial_filters"]
        num_fc1 = params["num_fc1"]
        num_classes = params["num_classes"]
        self.dropout_rate = params["dropout_rate"]

        self.conv1 = nn.Conv2d(in_channels=C_in, out_channels=init_f, kernel_size=3)
        conv1_out_h, conv1_out_w = findConv2dOutShape(H_in, W_in, self.conv1)
        self.conv2 = nn.Conv2d(init_f, 2 * init_f, kernel_size=3)
        conv2_out_h, conv2_out_w = findConv2dOutShape(
            conv1_out_h, conv1_out_w, self.conv2
        )
        self.conv3 = nn.Conv2d(2 * init_f, 4 * init_f, kernel_size=3)
        conv3_out_h, conv3_out_w = findConv2dOutShape(
            conv2_out_h, conv2_out_w, self.conv3
        )
        self.conv4 = nn.Conv2d(4 * init_f, 8 * init_f, kernel_size=3)
        conv4_out_h, conv4_out_w = findConv2dOutShape(
            conv3_out_h, conv3_out_w, self.conv4
        )

        # compute the flatten size
        self.num_flatten = conv4_out_h * conv4_out_w * 8 * init_f
        self.fc1 = nn.Linear(self.num_flatten, num_fc1)
        self.fc2 = nn.Linear(num_fc1, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2, 2)

        x = x.view(-1, self.num_flatten)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, self.dropout_rate, training=self.training)
        x = self.fc2(x)
        out = F.log_softmax(x, dim=1)
        return out


if __name__ == "__main__":
    # # Creating dumb baselines
    # train_ds, val_ds = get_train_val_ds()

    # # get labels for validation dataset
    # y_val = [y for _, y in val_ds]

    # def accuracy(labels, out):
    #     return np.sum(out == labels) / float(len(labels))

    # # accuracy for all zero predictions
    # acc_all_zeros = accuracy(y_val, np.zeros_like(y_val))
    # print(f"{acc_all_zeros=:.2f}")

    # # accuracy for all ones predictions
    # acc_all_ones = accuracy(y_val, np.ones_like(y_val))
    # print(f"{acc_all_ones=:.2f}")

    # # accuracy for random predictions
    # acc_random = accuracy(y_val, np.random.randint(0, 2, size=len(y_val)))
    # print(f"{acc_random=:.2f}")

    # # using the helper function in an example
    # conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3)
    # h, w = findConv2dOutShape(96, 96, conv1)
    # print(f"{h=},{w=}")
    import torch
    from torchsummary import summary

    model_params = {
        "input_shape": (3, 96, 96),
        "initial_filters": 8,
        "num_fc1": 100,
        "dropout_rate": 0.25,
        "num_classes": 2,
    }

    model = Net(model_params)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(model)
    summary(model, input_size=model_params["input_shape"], device=device.type)
