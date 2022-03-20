from torchvision import transforms
from torch.utils.data import random_split

from custom_dataset import histoCancerDataset

from pathlib import Path


def get_train_val_ds():
    # Instantiate histoCancerDataset Class
    data_dir = Path(__file__).parent.resolve().joinpath("data")
    data_transformer = transforms.Compose([transforms.ToTensor()])
    histo_dataset = histoCancerDataset(
        data_dir, transform=data_transformer, data_type="train"
    )

    # Splitting the train dataset to train_ds and val_ds
    len_histo = len(histo_dataset)
    len_train = int(0.8 * len_histo)
    len_val = len_histo - len_train

    train_ds, val_ds = random_split(histo_dataset, [len_train, len_val])

    print("train_ds length:", len(train_ds))
    print("val_ds length:", len(val_ds))

    return train_ds, val_ds
