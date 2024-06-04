import numpy as np
from tqdm import tqdm

# https://openreview.net/pdf?id=Agekm5fdW3 - section 2.2
def compute_class_weights(
        preds: np.ndarray,
        targets: np.ndarray,
        num_classes: int
    ):
    class_weights = []

    for i in range(num_classes):
        n_elements = np.sum(targets == i)
        # added a minus sign to the whole expression inside exp, as I believe it's incorrect in the original paper
        # before, the classes with worse scores got lower weights, but it probably should be the opposite
        class_weights.append(np.exp(-1 / n_elements * np.sum(np.log(preds[targets == i][:, i]))))
    return np.array(class_weights)

def compute_class_weights_original_formula(
        preds: np.ndarray,
        targets: np.ndarray,
        num_classes: int
    ):
    class_weights = []

    for i in range(num_classes):
        n_elements = np.sum(targets == i)
        # no minus sign before sum here
        class_weights.append(np.exp(1 / n_elements * np.sum(np.log(preds[targets == i][:, i]))))
    return np.array(class_weights)

# https://openreview.net/pdf?id=Agekm5fdW3 - section 2.2
def compute_class_alignment_scores(
        test_scores: np.ndarray,
        targets: np.ndarray,
        class_weights: np.ndarray,
        num_classes: int
    ):
    group_alignment_scores = []
    n_train_examples = test_scores.shape[0]
    n_targets = []

    for i in range(num_classes):
        n_targets.append(np.sum(targets == i))

    n_targets = np.array(n_targets)

    for i in tqdm(range(n_train_examples)):
        example_score = 0.0
        for j in range(num_classes):
            scaling_factor = class_weights[j] / n_targets[j]
            example_score += scaling_factor * np.sum(test_scores[i, targets == j])

        group_alignment_scores.append(example_score)

    return np.array(group_alignment_scores)
