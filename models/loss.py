import torch.nn.functional as F
import torch.nn as nn
from torch.nn import BCELoss
import torch
import numpy as np
from monai.losses.dice import DiceLoss
from models.registry import LOSSES

def nll_loss(output, target):
    return F.nll_loss(output, target)

def sum_tensor(inp, axes, keepdim=False):
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp

def get_tp_fp_fn_tn(net_output, gt, axes=None, mask=None, square=False):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes: can be (, ) = no summation
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
    """
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    shp_x = net_output.shape
    shp_y = gt.shape
    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x)
            if net_output.device.type == "cuda":
                y_onehot = y_onehot.cuda(net_output.device.index)
            y_onehot.scatter_(1, gt, 1)

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot
    tn = (1 - net_output) * (1 - y_onehot)

    if mask is not None:
        tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)
        tn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2
        tn = tn ** 2

    if len(axes) > 0:
        tp = sum_tensor(tp, axes, keepdim=False)
        fp = sum_tensor(fp, axes, keepdim=False)
        fn = sum_tensor(fn, axes, keepdim=False)
        tn = sum_tensor(tn, axes, keepdim=False)

    return tp, fp, fn, tn


class DeepSupervisionWrapper(nn.Module):
    def __init__(self, loss, weight_factors=None):
        """
        Wraps a loss function so that it can be applied to multiple outputs. Forward accepts an arbitrary number of
        inputs. Each input is expected to be a tuple/list. Each tuple/list must have the same length. The loss is then
        applied to each entry like this:
        l = w0 * loss(input0[0], input1[0], ...) +  w1 * loss(input0[1], input1[1], ...) + ...
        If weights are None, all w will be 1.
        """
        super(DeepSupervisionWrapper, self).__init__()
        self.weight_factors = weight_factors
        self.loss = loss

    def forward(self, *args):
        # for i in args:
        #     assert isinstance(i, (tuple, list)), "all args must be either tuple or list, got %s" % type(i)
            # we could check for equal lengths here as well but we really shouldn't overdo it with checks because
            # this code is executed a lot of times!

        if self.weight_factors is None:
            weights = [1] * len(args[0])
        else:
            weights = self.weight_factors

        # we initialize the loss like this instead of 0 to ensure it sits on the correct device, not sure if that's
        # really necessary  不需要不需要
        l = weights[0] * self.loss(args[0][0], args[1])
        for i, inputs in enumerate(args[0]):
            if i == 0:
                continue
            l += weights[i] * self.loss(inputs, args[1])
        # l = sum([self.loss(inputs, args[1]) for inputs in args[0]])
        # loss1 = self.loss(args[0][0], args[1])
        # loss2 = self.loss(args[0][1], args[1])
        # loss3 = self.loss(args[0][2], args[1])
        # loss4 = self.loss(args[0][3], args[1])
        # l = loss4 + loss3 + loss2 +loss1
        return l


class SoftDiceLoss(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.):
        """
        """
        super(SoftDiceLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)

        nominator = 2 * tp + self.smooth
        denominator = 2 * tp + fp + fn + self.smooth

        dc = nominator / (denominator + 1e-8)

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dc = dc.mean()
        return -dc


class DeepSuperDiceLoss(nn.Module):
    def __init__(self):
        super(DeepSuperDiceLoss, self).__init__()
        self.dc = DiceLoss(reduction="mean")
        self.loss = DeepSupervisionWrapper(self.dc)

    def forward(self, output, target):
        result = self.loss(output, target)
        return result


class diceloss(nn.Module):
    def __init__(self):
        super(diceloss, self).__init__()
        self.dc = DiceLoss(reduction="mean", smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)

    def forward(self, output, target):
        result = self.dc(output, target)
        return result




def _expand_binary_labels(labels, label_weights, label_channels):
    bin_labels = labels.new_full((labels.size(0), label_channels), 0)
    inds = torch.nonzero(labels >= 1).squeeze()
    if inds.numel() > 0:
        bin_labels[inds, labels[inds] - 1] = 1
    bin_label_weights = label_weights.view(-1, 1).expand(
        label_weights.size(0), label_channels)
    return bin_labels, bin_label_weights


@LOSSES.register_module
class GHMC(nn.Module):
    """GHM Classification Loss.

    Details of the theorem can be viewed in the paper
    "Gradient Harmonized Single-stage Detector".
    https://arxiv.org/abs/1811.05181

    Args:
        bins (int): Number of the unit regions for distribution calculation.
        momentum (float): The parameter for moving average.
        use_sigmoid (bool): Can only be true for BCE based loss now.
        loss_weight (float): The weight of the total GHM-C loss.
    """
    def __init__(
            self,
            bins=10,
            momentum=0,
            use_sigmoid=True,
            loss_weight=1.0):
        super(GHMC, self).__init__()
        self.bins = bins
        self.momentum = momentum
        self.edges = torch.arange(bins + 1).float().cuda() / bins
        self.edges[-1] += 1e-6
        if momentum > 0:
            self.acc_sum = torch.zeros(bins).cuda()
        self.loss_weight = loss_weight

    def forward(self, pred, target, label_weight, *args, **kwargs):
        """Calculate the GHM-C loss.

        Args:
            pred (float tensor of size [batch_num, class_num]):
                The direct prediction of classification fc layer.
            target (float tensor of size [batch_num, class_num]):
                Binary class target for each sample.
            label_weight (float tensor of size [batch_num, class_num]):
                the value is 1 if the sample is valid and 0 if ignored.
        Returns:
            The gradient harmonized loss.
        """
        # the target should be binary class label
        if pred.dim() != target.dim():
            target, label_weight = _expand_binary_labels(
                                    target, label_weight, pred.size(-1))
        target, label_weight = target.float(), label_weight.float()
        edges = self.edges
        mmt = self.momentum
        weights = torch.zeros_like(pred)

        # gradient length
        g = torch.abs(pred.detach() - target)

        valid = label_weight > 0
        tot = max(valid.float().sum().item(), 1.0)
        n = 0  # n valid bins
        for i in range(self.bins):
            inds = (g >= edges[i]) & (g < edges[i+1]) & valid
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                if mmt > 0:
                    self.acc_sum[i] = mmt * self.acc_sum[i] \
                        + (1 - mmt) * num_in_bin
                    weights[inds] = tot / self.acc_sum[i]
                else:
                    weights[inds] = tot / num_in_bin
                n += 1
        if n > 0:
            weights = weights / n

        loss = F.binary_cross_entropy(
            pred, target, weights, reduction='sum') / tot
        return loss * self.loss_weight


@LOSSES.register_module
class GHMR(nn.Module):
    """GHM Regression Loss.

    Details of the theorem can be viewed in the paper
    "Gradient Harmonized Single-stage Detector"
    https://arxiv.org/abs/1811.05181

    Args:
        mu (float): The parameter for the Authentic Smooth L1 loss.
        bins (int): Number of the unit regions for distribution calculation.
        momentum (float): The parameter for moving average.
        loss_weight (float): The weight of the total GHM-R loss.
    """
    def __init__(
            self,
            mu=0.02,
            bins=10,
            momentum=0,
            loss_weight=1.0):
        super(GHMR, self).__init__()
        self.mu = mu
        self.bins = bins
        self.edges = torch.arange(bins + 1).float().cuda() / bins
        self.edges[-1] = 1e3
        self.momentum = momentum
        if momentum > 0:
            self.acc_sum = torch.zeros(bins).cuda()
        self.loss_weight = loss_weight

    def forward(self, pred, target, label_weight, avg_factor=None):
        """Calculate the GHM-R loss.

        Args:
            pred (float tensor of size [batch_num, 4 (* class_num)]):
                The prediction of box regression layer. Channel number can be 4
                or 4 * class_num depending on whether it is class-agnostic.
            target (float tensor of size [batch_num, 4 (* class_num)]):
                The target regression values with the same size of pred.
            label_weight (float tensor of size [batch_num, 4 (* class_num)]):
                The weight of each sample, 0 if ignored.
        Returns:
            The gradient harmonized loss.
        """
        mu = self.mu
        edges = self.edges
        mmt = self.momentum

        # ASL1 loss
        diff = pred - target
        loss = torch.sqrt(diff * diff + mu * mu) - mu

        # gradient length
        g = torch.abs(diff / torch.sqrt(mu * mu + diff * diff)).detach()
        weights = torch.zeros_like(g)

        valid = label_weight > 0
        tot = max(label_weight.float().sum().item(), 1.0)
        n = 0  # n: valid bins
        for i in range(self.bins):
            inds = (g >= edges[i]) & (g < edges[i+1]) & valid
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                n += 1
                if mmt > 0:
                    self.acc_sum[i] = mmt * self.acc_sum[i] \
                        + (1 - mmt) * num_in_bin
                    weights[inds] = tot / self.acc_sum[i]
                else:
                    weights[inds] = tot / num_in_bin
        if n > 0:
            weights /= n

        loss = loss * weights
        loss = loss.sum() / tot
        return loss * self.loss_weight
    
if __name__ == "__main__":
    loss = GHMC()
    pred = torch.tensor([[0.1], [0.2], [0.8], [0.9]]).squeeze()
    target = torch.tensor([[0], [0], [0], [1]]).squeeze()
    weights = torch.tensor([[0.25], [0.25], [0.25], [.075]]).squeeze()
    print(pred.shape, target.shape, weights.shape)
    print(loss(pred.cuda(), target.cuda(), weights.cuda()))