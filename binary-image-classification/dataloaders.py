from torch.utils.data import DataLoader

from split_dataset import get_train_val_ds
from transformers import train_transformer, val_transformer

train_ds, val_ds = get_train_val_ds()
train_ds.transforms = train_transformer
val_ds.transforms = val_transformer

train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=64, shuffle=False)

if __name__ == "__main__":
    # extract a batch from training data
    x, y = next(iter(train_dl))
    print(x.shape)
    print(y.shape)

    # extract a batch from validation data
    x, y = next(iter(val_dl))
    print(x.shape)
    print(y.shape)
