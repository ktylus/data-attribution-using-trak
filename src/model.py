import torch
from torch import nn as nn
from transformers import AutoModelForImageClassification
from tqdm import tqdm
import numpy as np
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model(
	    base_path: str,
        resnet_version: str,
        num_classes: int
    ):
    model = AutoModelForImageClassification.from_pretrained(resnet_version)
    resnet_version = resnet_version.split('/')[1]
    model.classifier = nn.Sequential(
        nn.Flatten(start_dim=1, end_dim=-1),
        nn.Linear(in_features=512, out_features=num_classes)
    )
    model.num_labels = num_classes

    if not os.path.exists(base_path):
        torch.save(model.state_dict(), base_path)
    else:
        model.load_state_dict(torch.load(base_path))
        
    for param in model.classifier.parameters():
            param.requires_grad = True

    return model


def compute_val_dataset_predictions(
        model: torch.nn.Module,
        val_dl: torch.utils.data.DataLoader
    ):
    model.eval()
    preds = []

    with torch.no_grad():
        for batch in tqdm(val_dl):
            batch = [xy.to(device) for xy in batch]
            output = model(batch[0]).logits
            preds.extend(output.tolist())

    preds = torch.Tensor(preds)
    preds = torch.nn.functional.softmax(preds, dim=-1)

    return np.array(preds.tolist())


def calculate_model_loss_and_accuracy(
        model: torch.nn.Module,
        val_dl: torch.utils.data.DataLoader,
        criterion=nn.CrossEntropyLoss()
    ):
    model.eval()
    total_correct = 0
    total_outputs = 0
    val_loss = 0.0

    with torch.no_grad():
        for i, data in enumerate(tqdm(val_dl), 0):
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            val_loss += criterion(outputs.logits, labels).item()
            correct = (torch.argmax(outputs.logits, dim=-1) == labels).sum().item()

            total_correct += correct
            total_outputs += outputs.logits.shape[0]
    accuracy = total_correct / total_outputs
    val_loss = val_loss / len(val_dl)
    return val_loss, accuracy


def recall_for_class(
        model: torch.nn.Module,
        val_dl: torch.utils.data.DataLoader,
        class_id: int
    ):
    model.eval()

    n_correct = 0
    n_all = 0

    with torch.no_grad():
        for data, labels in tqdm(val_dl):
            data, labels = data.to(device), labels.to(device)
            data = data[labels == class_id]
            if len(data) == 0:
                continue
            preds = model(data).logits
            n_correct += preds.argmax(dim=-1).eq(class_id).sum().item()
            n_all += len(data)

    return n_correct / n_all
