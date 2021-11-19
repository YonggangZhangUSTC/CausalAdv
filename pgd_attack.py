import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from ada_mt_utils import get_s_pre, SoftCrossEntropy


def adp_pgd_attack(model, images, labels, step_size, k, eps, model_g, basis, loss_fn='ce'):
    model.is_train(True)
    # 1. initialization
    soft_ce = SoftCrossEntropy().cuda()

    if k > 1:
        x = images.detach() + torch.zeros_like(images, requires_grad=False).uniform_(-eps, eps).cuda()
        x = torch.clamp(x, 0.0, 1.0)
    else:
        x = images
    ratio, adv_alpha = 0.1, 0.2
    f = model(images)[0].detach()
    s_pre = get_s_pre(f.clone().detach(), basis)
    z_s = model_g(s_pre, labels, ratio=ratio)
    p_s = F.softmax(z_s, dim=1)

    # 2. update adversarial examples
    for i in range(k):
        x.requires_grad_()
        if x.grad is not None:
            x.grad.data.fill_(0)

        with torch.enable_grad():
            f_adv, z_adv = model(x)
            if loss_fn == 'ce':
                ce_adv = nn.CrossEntropyLoss(reduce=False)(z_adv, labels).sum()
            elif loss_fn == 'cw':
                ce_adv = cw_loss(z_adv, labels).sum()
            else:
                ce_adv = None
            s_adv_pre = get_s_pre(f_adv, basis)
            z_s_adv = model_g(s_adv_pre, labels, ratio=ratio)
            kl_s_adv = soft_ce(y=p_s, z=z_s_adv)
            loss = (1 - adv_alpha) * ce_adv + adv_alpha * kl_s_adv

            grad = torch.autograd.grad(loss, [x])[0]

        x_adv = x.data + step_size * torch.sign(grad.data)
        x = Variable(torch.clamp(torch.min(torch.max(x_adv, images - eps), images + eps), 0.0, 1.0))
    model.is_train(False)
    return x


def pgd_attack(model, images, labels, step_size, k, eps, num_classes, loss_fn='ce'):
    # 1. initialization
    if k > 1:
        x = images.detach() + torch.zeros_like(images, requires_grad=False).uniform_(-eps, eps).cuda()
        x = torch.clamp(x, 0.0, 1.0)
    else:
        x = images

    # 2. update adversarial examples
    for i in range(k):
        x.requires_grad_()
        if x.grad is not None:
            x.grad.data.fill_(0)

        with torch.enable_grad():
            z_adv = model(x)
            if loss_fn == 'ce':
                loss = nn.CrossEntropyLoss(reduce=False)(z_adv, labels).sum()
            elif loss_fn == 'cw':
                loss = cw_loss(z_adv, labels, num_classes=num_classes).sum()
            else:
                loss = None
            grad = torch.autograd.grad(loss, [x])[0]

        x_adv = x.data + step_size * torch.sign(grad.data)
        x = Variable(torch.clamp(torch.min(torch.max(x_adv, images - eps), images + eps), 0.0, 1.0))
    return x


def cw_loss(output, target, confidence=50, num_classes=10):
    target = target.data
    target_onehot = torch.zeros(target.size() + (num_classes,))
    target_onehot = target_onehot.cuda()
    target_onehot.scatter_(1, target.unsqueeze(1), 1.)
    target_var = Variable(target_onehot, requires_grad=False)
    real = (target_var * output).sum(1)
    other = ((1. - target_var) * output - target_var * 10000.).max(1)[0]
    loss = - torch.clamp(real - other + confidence, min=0.)
    return loss


if __name__ == "__main__":
    print('hi')
