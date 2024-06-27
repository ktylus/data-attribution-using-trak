import torch
import numpy as np
from tqdm import tqdm

from src.early_stopping import EarlyStopping
from src.class_metrics import compute_class_weights, compute_class_alignment_scores
from src.model import compute_val_dataset_predictions
from src.dataset import get_dl_targets


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(
        model: torch.nn.Module,
        train_dl: torch.utils.data.DataLoader,
        val_dl: torch.utils.data.DataLoader,
        epochs: int,
        optimizer: torch.optim.Optimizer,
        early_stopping: EarlyStopping = None,
        criterion = None,
        epochs_per_checkpoint: int = None,
        model_save_path: str = "model.pth",
    ):
    criterion = criterion or torch.nn.CrossEntropyLoss()
    model = model.to(device)

    for epoch in range(epochs):
        running_loss = 0.0

        model.train()

        train_correct = 0
        train_outputs = 0

        for i, data in enumerate(tqdm(train_dl), 0):
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs.logits, labels)

            loss.backward()
            optimizer.step()

            train_correct += (torch.argmax(outputs.logits, dim=-1) == labels).sum().item()
            train_outputs += outputs.logits.shape[0]

            running_loss += loss

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

        print(f"[Epoch {epoch + 1}] Loss: {running_loss / len(train_dl):.3f}, Train Acc: {train_correct / train_outputs:.3f}," +
              f"Valid loss: {val_loss / len(val_dl):.3f} Valid Acc: {total_correct / total_outputs:.3f}")
        
        if epochs_per_checkpoint and (epoch + 1) % epochs_per_checkpoint == 0: 
            torch.save(model.state_dict(), model_save_path.split(".")[0] + f"_cp_epoch{epoch + 1}.pth")

        if early_stopping:
            early_stopping(model, val_loss)
            if early_stopping.stop:
                print(f"Early stopping at epoch {epoch + 1}")
                model.load_state_dict(early_stopping.get_best_model_parameters())
                break


def train_without_bottom_k_alignment_scoring_data(
        base_model: torch.nn.Module,
        model: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset,
        val_dataset: torch.utils.data.Dataset,
        k: int,
        trak_scores: np.ndarray,
        batch_size: int,
        epochs: int,
        lr: float,
        criterion = None,
):
    val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    val_targets = get_dl_targets(val_dl)
    val_preds = compute_val_dataset_predictions(base_model, val_dl)
    num_classes = len(train_dataset.dataset.classes)

    class_weights = compute_class_weights(val_preds, val_targets, num_classes)
    group_alignment_scores = compute_class_alignment_scores(trak_scores, val_targets, class_weights, num_classes)
    sorted_group_alignment_scores = np.sort(group_alignment_scores)
    example_indices_to_keep = np.nonzero(~(group_alignment_scores < sorted_group_alignment_scores[k]))[0]
    
    train_data_after_trak = torch.utils.data.Subset(train_dataset, example_indices_to_keep)
    train_dl_after_trak = torch.utils.data.DataLoader(train_data_after_trak, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    early_stopping = EarlyStopping(patience=3, min_delta=0.001)
    train_model(model, train_dl_after_trak, val_dl, epochs, optimizer, early_stopping, criterion)


def train_without_top_k_alignment_scoring_data(
        base_model: torch.nn.Module,
        model: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset,
        val_dataset: torch.utils.data.Dataset,
        k: int,
        trak_scores: np.ndarray,
        batch_size: int,
        epochs: int,
        lr: float,
        criterion = None,
):
    val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    val_targets = get_dl_targets(val_dl)
    val_preds = compute_val_dataset_predictions(base_model, val_dl)
    num_classes = len(train_dataset.dataset.classes)

    class_weights = compute_class_weights(val_preds, val_targets, num_classes)
    group_alignment_scores = compute_class_alignment_scores(trak_scores, val_targets, class_weights, num_classes)
    sorted_group_alignment_scores = np.sort(group_alignment_scores)
    example_indices_to_keep = np.nonzero(~(group_alignment_scores >= sorted_group_alignment_scores[len(group_alignment_scores) - k]))[0]
    
    train_data_after_trak = torch.utils.data.Subset(train_dataset, example_indices_to_keep)
    train_dl_after_trak = torch.utils.data.DataLoader(train_data_after_trak, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    early_stopping = EarlyStopping(patience=3, min_delta=0.001)
    train_model(model, train_dl_after_trak, val_dl, epochs, optimizer, early_stopping, criterion)


def train_without_negative_alignment_scoring_data(
        base_model: torch.nn.Module,
        model: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset,
        val_dataset: torch.utils.data.Dataset,
        k: int,
        trak_scores: np.ndarray,
        batch_size: int,
        epochs: int,
        lr: float,
        criterion = None,
):
    val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    val_targets = get_dl_targets(val_dl)
    val_preds = compute_val_dataset_predictions(base_model, val_dl)
    num_classes = len(train_dataset.dataset.classes)

    class_weights = compute_class_weights(val_preds, val_targets, num_classes)
    group_alignment_scores = compute_class_alignment_scores(trak_scores, val_targets, class_weights, num_classes)
    example_indices_to_keep = np.nonzero(~(group_alignment_scores < 0))[0]
    
    train_data_after_trak = torch.utils.data.Subset(train_dataset, example_indices_to_keep)
    train_dl_after_trak = torch.utils.data.DataLoader(train_data_after_trak, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    early_stopping = EarlyStopping(patience=3, min_delta=0.001)
    train_model(model, train_dl_after_trak, val_dl, epochs, optimizer, early_stopping, criterion)