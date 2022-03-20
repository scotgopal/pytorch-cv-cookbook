from matplotlib import scale
from torchvision import transforms

train_transformer = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(45),
        transforms.RandomResizedCrop(96, scale=(0.08, 1), ratio=(1, 1)),
        transforms.ToTensor(),
    ]
)

val_transformer = transforms.Compose([transforms.ToTensor()])
