import torch.nn as nn

import torch
import torch.nn as nn

class MontyHallLoss(nn.Module):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean'):
        """
        Args:
            weight (Tensor, optional): a manual rescaling weight given to the loss
            size_average (bool, optional): Deprecated (see reduction). By default, the losses are averaged over each
                loss element in the batch. Note that for some losses, there are multiple elements per sample. If the
                field size_average is set to False, the losses are instead summed for each minibatch. Ignored when
                reduce is False. Default: True
            reduce (bool, optional): Deprecated (see reduction). By default, the losses are averaged or summed over
                observations for each minibatch depending on size_average. When reduce is False, returns a loss per
                batch element instead and ignores size_average. Default: True
            reduction (string, optional): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
                'none': no reduction will be applied, 'mean': the sum of the output will be divided by the number of
                elements in the output, 'sum': the output will be summed. Note: size_average and reduce are in the
                process of being deprecated, and in the meantime, specifying either of those two args will override
                reduction. Default: 'mean'
        """
        super(MontyHallLoss, self).__init__()

        # Initialize any parameters or operations you need for your loss

    def forward(self, pred, host, target):
        """
        Args:
            input (Tensor): the model output
            target (Tensor): the ground truth

        Returns:
            Tensor: the computed loss
        """
        # Implement your custom loss computation here
        # Example: Mean Squared Error (MSE)
        null = 1/player.shape[0]
        loss = nn.functional.cross_entropy(rescaled, target)
        return loss









