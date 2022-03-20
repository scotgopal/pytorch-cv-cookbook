from PIL import Image
import torch
from torch.utils.data import Dataset
import pandas as pd
import torchvision.transforms as transforms
import os
from pathlib import Path


def seed_all(seed_int: int = 0):
    torch.manual_seed(0)


seed_all()


class histoCancerDataset(Dataset):
    def __init__(self, data_dir, transform, data_type="train") -> None:
        super().__init__()

        # path to images
        path_to_data = os.path.join(data_dir, data_type)

        # get a list of images
        filenames = os.listdir(path_to_data)

        # get full path to images
        self.full_filenames = [os.path.join(path_to_data, f) for f in filenames]

        # labels are in a csv file named train_labels.csv
        csv_filename = data_type + "_labels.csv"
        path_to_labels = os.path.join(data_dir, csv_filename)
        labels_df = pd.read_csv(path_to_labels)

        # set dataframe index to id
        labels_df.set_index("id", inplace=True)

        # obtain labels from dataframe
        self.labels = [labels_df.loc[filename[:-4]].values[0] for filename in filenames]

        self.transform = transform

    def __len__(self):
        # return the size of the dataset
        return len(self.full_filenames)

    def __getitem__(self, idx):
        # open image, apply transforms and return with label
        image = Image.open(self.full_filenames[idx])  # PIL Image
        image = self.transform(image)
        return image, self.labels[idx]


if __name__ == "__main__":
    data_transformer = transforms.Compose([transforms.ToTensor()])
    dir_to_current_file = Path(__file__).parent.resolve()
    print(f"{dir_to_current_file=}")
    data_dir = os.path.join(dir_to_current_file, "data")

    histo_dataset = histoCancerDataset(
        data_dir=data_dir, transform=data_transformer, data_type="train"
    )
    print(len(histo_dataset))

    img, label = histo_dataset[9]
    print(img.shape, torch.min(img), torch.max(img))
