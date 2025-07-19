import paddle
import paddle.nn as nn

from paddleseg.cvlibs import manager
from .dice_loss import DiceLoss
from .focal_loss import MultiClassFocalLoss

@manager.LOSSES.add_component
class DiceFocalLoss(nn.Layer):
    """
    Combined loss of DiceLoss and MultiClassFocalLoss
    Args:
        num_class (int): Number of classes
        alpha (float, optional): The alpha of focal loss. Default: 1.0
        gamma (float, optional): The gamma of Focal Loss. Default: 2.0
        weight (list[float], optional): The weight for each class in DiceLoss. Default: None
        smooth (float32, optional): Laplace smoothing for DiceLoss. Default: 1.0
        ignore_index (int64, optional): Specifies a target value that is ignored. Default: 255
    """
    def __init__(self, num_class, alpha=1.0, gamma=2.0, weight=None, smooth=1.0, ignore_index=255):
        super().__init__()
        self.dice_loss = DiceLoss(
            weight=weight,
            ignore_index=ignore_index,
            smooth=smooth
        )
        self.focal_loss = MultiClassFocalLoss(
            num_class=num_class,
            alpha=alpha,
            gamma=gamma,
            ignore_index=ignore_index
        )

    def forward(self, logits, labels):
        """
        Args:
            logits (Tensor): Logit tensor, shape (N, C, H, W)
            labels (Tensor): Label tensor, shape (N, H, W)
        Returns:
            Tensor: Combined loss result
        """
        dice_loss = self.dice_loss(logits, labels)
        focal_loss = self.focal_loss(logits, labels)
        return 0.5 * dice_loss + 0.5 * focal_loss

