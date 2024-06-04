import torch
from tqdm import tqdm

from src.early_stopping import EarlyStopping


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(
        model: torch.nn.Module,
        train_dl: torch.utils.data.DataLoader,
        val_dl: torch.utils.data.DataLoader,
        epochs: int,
        optimizer: torch.optim.Optimizer,
        early_stopping: EarlyStopping = None,
        criterion = None
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

        if early_stopping:
            early_stopping(model, val_loss)
            if early_stopping.stop:
                print(f"Early stopping at epoch {epoch + 1}")
                model.load_state_dict(early_stopping.get_best_model_parameters())
                break