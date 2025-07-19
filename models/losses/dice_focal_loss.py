# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
from paddle import nn
import paddle.nn.functional as F

from paddleseg.cvlibs import manager


class DiceLoss(nn.Layer):
    """
    Dice Loss计算
    """

    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, label):
        """
        logits: [N, C, H, W]，模型预测的logits
        label: [N, H, W] 或 [N, 1, H, W]，标签
        """
        num_classes = logits.shape[1]

        if label.shape[1] != 1 and len(label.shape) != 3:
            raise ValueError("标签形状应为[N, H, W]或[N, 1, H, W]")

        if len(label.shape) == 3:
            label = paddle.unsqueeze(label, 1)

        # 将logits转换为概率值
        pred = F.softmax(logits, axis=1)

        # 计算每个类别的Dice Loss
        total_loss = 0
        for i in range(num_classes):
            if i == 0 and num_classes > 1:  # 跳过背景类别（如果存在的话）
                continue

            # 针对当前类别创建one-hot编码
            label_i = (label == i).astype('float32')
            pred_i = pred[:, i:i + 1, :, :]

            # 计算交集
            intersection = paddle.sum(label_i * pred_i)
            # 计算并集
            union = paddle.sum(label_i) + paddle.sum(pred_i)
            # Dice系数
            dice_coef = (2.0 * intersection + self.smooth) / (union + self.smooth)
            # Dice Loss
            dice_loss = 1.0 - dice_coef

            total_loss += dice_loss

        # 计算平均loss（排除背景类别）
        return total_loss / (num_classes - 1 if num_classes > 1 else 1)


class FocalLoss(nn.Layer):
    """
    Focal Loss计算
    """

    def __init__(self, alpha=0.25, gamma=2.0, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.ce_loss = nn.CrossEntropyLoss(
            weight=None,
            ignore_index=ignore_index,
            reduction='none')

    def forward(self, logits, label):
        """
        logits: [N, C, H, W]
        label: [N, H, W]
        """
        # 确保标签是二维的
        if len(label.shape) == 4:
            label = paddle.squeeze(label, axis=1)

        # 计算交叉熵损失（不进行reduction）
        ce_loss = self.ce_loss(logits, label)

        # 计算预测概率
        pred = F.softmax(logits, axis=1)

        # 创建one-hot标签
        label_one_hot = F.one_hot(label, num_classes=logits.shape[1])
        label_one_hot = paddle.transpose(label_one_hot, [0, 3, 1, 2])

        # 获取对应类别的预测概率
        pt = paddle.sum(pred * label_one_hot, axis=1)

        # 计算权重因子
        at = self.alpha * label_one_hot + (1 - self.alpha) * (1 - label_one_hot)
        at = paddle.sum(at, axis=1)

        # 计算focal loss
        focal_weight = paddle.pow(1 - pt, self.gamma)

        # 应用focal weight
        loss = ce_loss * focal_weight

        # 处理需要忽略的像素
        mask = (label != self.ignore_index).astype('float32')
        loss = loss * mask

        # 计算平均损失
        num_valid = paddle.sum(mask)
        loss = paddle.sum(loss) / (num_valid + 1e-5)

        return loss


@manager.LOSSES.add_component
class DiceFocalLoss(nn.Layer):
    """
    组合了Dice Loss和Focal Loss的损失函数，权重比例为0.5:0.5

    Args:
        dice_weight (float): Dice Loss的权重。默认为0.5。
        focal_weight (float): Focal Loss的权重。默认为0.5。
        ignore_index (int): 忽略的标签索引。默认为255。
    """

    def __init__(self, dice_weight=0.5, focal_weight=0.5, ignore_index=255):
        super(DiceFocalLoss, self).__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss(ignore_index=ignore_index)

    def forward(self, logits, label):
        """
        logits: [N, C, H, W]
        label: [N, H, W] 或 [N, 1, H, W]
        """
        dice = self.dice_loss(logits, label)
        focal = self.focal_loss(logits, label)

        return self.dice_weight * dice + self.focal_weight * focal

