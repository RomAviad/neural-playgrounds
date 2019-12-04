import os
import pandas as pd
import torch

from torch.utils.data import DataLoader, Dataset
from torch.utils.model_zoo import tqdm
from mura.utils import body_part_to_one_hot
from torchvision.datasets.folder import pil_loader
from torchvision import transforms

categories = ["train", "valid"]
MURA_BASE = "resources/MURA-v1.1"


def get_study_level_data(study_type):
    """
    Returns a dict, with keys 'train' and 'valid' and respective values as study level dataframes,
    these dataframes contain three columns 'Path', 'Count', 'Label'
    Args:
        study_type (string): one of the seven study type folder names in 'train/valid/test' dataset
    """
    study_data = {}
    study_label = {"positive": 1, "negative": 0}
    for phase in categories:
        BASE_DIR = "{}/{}/{}".format(MURA_BASE, phase, study_type)
        patients = list(os.walk(BASE_DIR))[0][1]  # list of patient folder names
        study_data[phase] = pd.DataFrame(columns=["Path", "Count", "Label"])
        i = 0
        for patient in tqdm(patients):  # for each patient folder
            for study in os.listdir(BASE_DIR + patient):  # for each study in that patient folder
                label = study_label[study.split("_")[1]]  # get label 0 or 1
                path = os.path.join(BASE_DIR, patient, study) + "/"  # path to this study
                study_data[phase].loc[i] = [path, len(os.listdir(path)), label]  # add new row
                i += 1
    return study_data


def get_image_level_data(study_type):
    """
    Returns a dict, with keys 'train' and 'valid' and respective values as study level dataframes,
    these dataframes contain three columns 'Path', 'Count', 'Label'
    Args:
        study_type (string): one of the seven study type folder names in 'train/valid/test' dataset
    """
    image_data = {}
    study_label = {"positive": 1, "negative": 0}
    for phase in categories:
        BASE_DIR = "{}/{}/{}".format(MURA_BASE, phase, study_type)
        patients = list(os.walk(BASE_DIR))[0][1]  # list of patient folder names
        image_data[phase] = pd.DataFrame(columns=["Path", "Label", "Study_Type", "Study_Type_OH"])
        i = 0
        for patient in tqdm(patients):  # for each patient folder
            for study in os.listdir(os.path.join(BASE_DIR, patient)):  # for each study in that patient folder
                label = study_label[study.split('_')[1]]  # get label 0 or 1
                study_path = os.path.join(BASE_DIR, patient, study) + "/"  # path to this study
                for image in os.listdir(study_path):
                    path = os.path.join(study_path, image)
                    image_data[phase].loc[i] = [path, label, study_type, body_part_to_one_hot(study_type)]  # add new row
                    i += 1
    return image_data


class MuraDataset(Dataset):
    def __init__(self, df, transform=None):
        """
        Args:
            df (pd.DataFrame): a pandas DataFrame with image path and labels.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        study_path = self.df.iloc[idx, 0]
        count = self.df.iloc[idx, 1]
        images = []
        for filename in os.listdir(study_path):
            if filename.endswith(".png"):
                image = pil_loader(os.path.join(study_path, filename))
                images.append(self.transform(image))
        images = torch.stack(images)
        label = self.df.iloc[idx, 2]
        sample = {'images': images, 'label': label}
        return sample


class MuraImageLevelDataset(Dataset):
    def __init__(self, df, transform=None):
        """
        Args:
            df (pd.DataFrame): a pandas DataFrame with image path and labels.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path = self.df.iloc[idx, 0]
        image = self.transform(pil_loader(image_path))
        # image = torch.stack([image])
        label = self.df.iloc[idx, 1]
        image_type = self.df.iloc[idx, 2]
        return {"images": image, "label": label, "type": image_type, "type_oh": body_part_to_one_hot(image_type)}


def collate_fn(batch):
    result = {"images": [], "label": []}
    for sample in batch:
        result["images"].append(sample["images"])
        result["label"] += [sample["label"] for _ in range(len(sample["images"]))]
    result["images"] = torch.cat(result["images"])
    result["label"] = torch.Tensor(result["label"])
    return result


def get_dataloaders(data, batch_size=4, validation=False):
    imagenet_training_set_stats = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.Resize((112, 112:)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize(**imagenet_training_set_stats)
    ])
    valid_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(**imagenet_training_set_stats)
    ])

    if validation:
        train_transform = valid_transform
    image_datasets = {
        "train": MuraImageLevelDataset(data["train"], transform=train_transform),
        "valid": MuraImageLevelDataset(data["valid"], transform=valid_transform)
    }
    loaders = {key: DataLoader(image_datasets[key], batch_size=batch_size, shuffle=not validation, num_workers=batch_size) for key in
               image_datasets}
    return loaders
