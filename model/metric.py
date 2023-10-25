import torch
from monai.metrics import DiceMetric, ROCAUCMetric, CumulativeIterationMetric


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


def dice_coeff(preds, target):
    smooth = 1.
    pred = preds[3]
    num = pred.size(0)
    m1 = pred.view(num, -1) # Flatten
    m2 = target.view(num, -1) # Flatten
    intersection = (m1 * m2).sum()
    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)
    # dice = compute_dice(preds, target)


class AccMetric(CumulativeIterationMetric):
    def __init__(self) -> None:
        super().__init__()

    def _compute_tensor(self, y_pred: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        return y_pred, y
    
    def aggregate(self):
        y_pred, y = self.get_buffer()
        y_pred = torch.round(y_pred)
        return torch.sum(y_pred == y).item() / len(y)
