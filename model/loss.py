import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
from monai.losses.dice import DiceLoss

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
