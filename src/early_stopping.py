class EarlyStopping:
    def __init__(self, patience: int, min_delta: float):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.stop = False
        self.val_loss_min = 1e10
        self.best_parameters = None

    def __call__(self, model, val_loss):
        if val_loss < self.val_loss_min - self.min_delta:
            self.best_parameters = model.state_dict()
            self.val_loss_min = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True

    def get_best_model_parameters(self):
        return self.best_parameters