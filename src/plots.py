import torch
import matplotlib.pyplot as plt
import numpy as np


def plot_single_image(
        image: torch.Tensor,
        label: str
    ):
    plt.figure(figsize=(4, 4))
    _plot(image, label)
    plt.show()


def plot_extreme_two_trak_examples_for_image(
        trak_scores: torch.Tensor,
        train_data: torch.utils.data.Subset,
        val_data: torch.utils.data.Subset,
        image_id: int,
        class_names: list[str]
    ):
    image_trak_scores = trak_scores.T[image_id]
    examples_indices_sorted_by_trak_scores = np.argsort(image_trak_scores)
    top_2_examples_indices = examples_indices_sorted_by_trak_scores[-2:]
    bottom_2_examples_indices = examples_indices_sorted_by_trak_scores[:2]
    base_image = val_data[image_id][0]

    plt.figure(figsize=(16, 12))
    plt.subplot(3, 5, 1)

    _plot(base_image, "base_image, class: " + str(class_names[val_data[image_id][1]]))

    concatenated = np.concatenate((top_2_examples_indices, bottom_2_examples_indices))

    for i, example_index in enumerate(concatenated):
        image = train_data[example_index][0]
        score = image_trak_scores[example_index]
        plt.subplot(3, 5, i + 2)
        _plot(image, "{:.3f}".format(score) + ", class: " + str(class_names[train_data[example_index][1]]))

    plt.tight_layout()
    plt.show()


def _plot(
        image: torch.Tensor,
        label: str
    ):
    plt.imshow(image.permute(1, 2, 0).cpu().numpy())
    plt.title(label)
    plt.axis("off")
