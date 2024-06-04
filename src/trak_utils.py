import torch
from torch import nn as nn
from trak import modelout_functions
from tqdm import tqdm
from collections.abc import Iterable


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ResNetOutput(modelout_functions.AbstractModelOutput):
    def __init__(self, loss_temperature: float = 1.0):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.loss_temperature = loss_temperature

    @staticmethod
    def get_output(
            model: torch.nn.Module,
            weights: Iterable[torch.Tensor],
            buffers: Iterable[torch.Tensor],
            image: torch.Tensor,
            label: torch.Tensor
        ):
        for key, value in weights.items():
            weights[key] = weights[key].to(device)
        #TODO: remove this loop as you said? as it may be not needed
        output = torch.func.functional_call(model, (weights, buffers), image.unsqueeze(0))
        logits = output.logits
        bindex = torch.arange(logits.shape[0]).to(logits.device, non_blocking=False)
        logits_correct = logits[bindex, label.unsqueeze(0)]
        
        cloned_logits = logits.clone()
        
        cloned_logits[bindex, label.unsqueeze(0)] = torch.tensor(-torch.inf, device=logits.device, dtype=logits.dtype)
        
        margins = logits_correct - cloned_logits.logsumexp(dim=-1)
    
        return margins.sum()
    
    def get_out_to_loss_grad(
            self,
            model: torch.nn.Module,
            weights: Iterable[torch.Tensor],
            buffers: Iterable[torch.Tensor],
            batch: Iterable[torch.Tensor]
        ):
        for key, value in weights.items():
            weights[key] = weights[key].to(device)
        #TODO: remove this loop as you said? as it may be not needed
        images, labels = batch
        output = torch.func.functional_call(model, (weights, buffers), images)
        logits = output.logits

        ps = self.softmax(logits / self.loss_temperature)[torch.arange(logits.size(0)), labels]

        return (1 - ps).clone().detach().unsqueeze(-1)


def featurize_traker(
        traker,
        train_dl: torch.utils.data.DataLoader
    ):
    for batch in tqdm(train_dl):
        batch = [xy.cuda() for xy in batch]

        traker.featurize(batch=batch, num_samples=batch[0].shape[0])

    traker.finalize_features()


def get_traker_scores(
        traker,
        experiment_name: str,
        checkpoint,
        model_id: int,
        test_dl: torch.utils.data.DataLoader
    ):
    traker.start_scoring_checkpoint(exp_name=experiment_name, checkpoint=checkpoint, model_id=model_id, num_targets=len(test_dl.dataset))
    
    for batch in tqdm(test_dl):
        batch = [xy.cuda() for xy in batch]
        traker.score(batch=batch, num_samples=batch[0].shape[0])

    test_scores = traker.finalize_scores(exp_name=experiment_name)

    return test_scores
