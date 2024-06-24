import torch
from torchvision.datasets import ImageFolder
import numpy as np

def get_dataset_split(
        images_path: str,
        chosen_indices: list[int],
        train_split_rate: float,
        test_split_rate: float,
        seed: int,
        preprocess_fun=None
    ):
    num_classes = len(chosen_indices)

    dataset = ImageFolder(images_path, transform=preprocess_fun)

    new_class_idx_mapping = {k: v for k, v in zip(chosen_indices, range(num_classes))}
    
    dataset.classes = [dataset.classes[i] for i in chosen_indices]
    dataset.class_to_idx = {k: i for i, k in enumerate(dataset.classes)}
    
    dataset.samples = list(filter(lambda s: s[1] in chosen_indices, dataset.samples))
    dataset.samples = list(map(lambda s: (s[0], new_class_idx_mapping[s[1]]), dataset.samples))

    train_subset, test_subset = torch.utils.data.random_split(dataset, [train_split_rate , test_split_rate], torch.Generator().manual_seed(seed))

    return train_subset, test_subset


def get_chosen_classes(
        images_path: str,
        chosen_indices: list[int]
    ):
    dataset = ImageFolder(images_path)
    dataset.classes = [dataset.classes[i] for i in chosen_indices]

    return dataset.classes


def get_dl_targets(
        dataloader: torch.utils.data.DataLoader
    ):
    targets = []

    for _, y in dataloader:
        targets.extend(y.tolist())

    targets = np.array(targets)

    return targets
