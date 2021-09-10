import torch


def predict_from_one_classifier_output(output: torch.Tensor) -> torch.Tensor:
    return torch.argmax(output, dim=1)
