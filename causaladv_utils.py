import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import argparse
import os
import numpy as np
import random
import torchvision
from torchvision import transforms

gpu = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu


def get_args_cifar10():
    parser = argparse.ArgumentParser()

    # training specific args
    parser.add_argument('--dataset', type=str, default='cifar10', choices='cifar10 or cifar100')
    parser.add_argument('--download', type=bool, default=True, help="if can't find dataset, download from web")
    parser.add_argument('--image_size', type=int, default=32)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--model_name', type=str, default='adam_13-best')
    parser.add_argument('--net', type=str, default='res', choices='wrn or res')
    parser.add_argument('--gpu', type=str, default=gpu, choices='0 or 1')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    # basic parameters
    parser.add_argument('--train', type=bool, default=False)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=120)
    parser.add_argument('--base_lr', type=float, default=0.1)
    parser.add_argument('--decay_lr1', type=int, default=60)
    parser.add_argument('--decay_lr2', type=int, default=90)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=2e-4)
    parser.add_argument('--step_train_log', type=int, default=20)
    parser.add_argument('--step_test_log', type=int, default=10)
    parser.add_argument('--test_step', type=int, default=5)

    # adv parameters
    parser.add_argument('--adv_iter', type=int, default=10)
    parser.add_argument('--adv_eps', type=float, default=0.031)
    parser.add_argument('--adv_step', type=float, default=0.007)

    parser.add_argument('--adv_mode', type=str, default='adam', choices='adam or adat')
    parser.add_argument('--adv_alpha', type=float, default=1.0)
    parser.add_argument('--adv_beta', type=float, default=1.0)  # 1.0 for adam, 6.0 for adat
    parser.add_argument('--adv_ratio', type=float, default=1.0)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    return args


def pgd(model, images, labels, basis, model_g, args):
    model.eval()
    model_g.eval()
    model_g.is_update_cov(False)

    # 1. Initialization
    f, z = model(images)
    fc = f.detach().clone()
    zc = z.detach().clone()
    del f
    del z
    soft_ce = SoftCrossEntropy().to(args.device)
    ce_loss = nn.CrossEntropyLoss().to(args.device)
    kl_loss = nn.KLDivLoss(reduction='batchmean').to(args.device)

    # 2. Set input and output
    if args.adv_mode == 'adam':
        # ADA-M
        x = images.detach() + torch.zeros_like(images, requires_grad=False).uniform_(-args.adv_eps, args.adv_eps).cuda()
        x = torch.clamp(x, 0.0, 1.0)
        p_nat = labels
    else:
        # ADA-T
        x = images.detach() + 0.001 * torch.randn_like(images, requires_grad=False).cuda().detach()
        p_nat = F.softmax(zc, dim=1)
    p_s_nat = F.softmax(model_g(get_s_pre(fc.clone().detach(), basis), labels, ratio=args.adv_ratio), dim=1)

    # 3. Update adversarial examples
    for i in range(args.adv_iter):
        x.requires_grad_()
        if x.grad is not None:
            x.grad.data.fill_(0)

        with torch.enable_grad():
            f_adv, z_adv = model(x)
            loss_s_adv = soft_ce(z=model_g(get_s_pre(f_adv, basis), labels, ratio=args.adv_ratio), y=p_s_nat)

            if args.adv_mode == 'adam':
                loss_adv = ce_loss(z_adv, p_nat)
            else:
                loss_adv = kl_loss(F.log_softmax(z_adv, dim=1), p_nat)

            loss = args.adv_beta * loss_adv + args.adv_alpha * loss_s_adv
            grad = torch.autograd.grad(loss, [x])[0]

        x_adv = x.data + args.adv_step * torch.sign(grad.data)
        x_adv = torch.min(torch.max(x_adv, images - args.adv_eps), images + args.adv_eps)
        x = Variable(torch.clamp(x_adv, 0, 1.0))

    model.train()
    return x


def get_s_pre(f, basis):
    norm_f = f / (torch.sqrt(torch.sum(f ** 2, dim=1, keepdim=True)) + 1e-6)
    return torch.mm(norm_f, basis.t())


def init_anchor(weight):
    def check_vectors(v, num_reduce):
        n = v.size(0)

        # identify the bad basis
        matrix = torch.mm(v, v.t())
        ip = matrix.abs().sum(dim=1)
        index = torch.sort(ip, descending=False)[1]

        # delete the bad basis
        v = v[index[0:n - num_reduce]]
        return v

    c, d = weight.size()

    # sampling noise
    k = 4
    over_anchor = torch.randn(size=(k * d, d), requires_grad=False).float().cuda()

    # delete bad noise
    anchor = check_vectors(over_anchor, num_reduce=(k - 1) * d)

    # gram schmidt
    anchor = gram_schmidt_ort(anchor)

    return anchor


class Basis():
    def __init__(self, anchor):
        super(Basis, self).__init__()
        self.anchor = anchor
        self.estimator = EstimatorCV(feature_num=anchor.size(0), class_num=1)

    def get_basis(self, f, weight):
        # Update Sigma
        norm_f = f / (torch.sqrt(torch.sum(f ** 2, dim=1, keepdim=True)) + 1e-6)
        self.estimator.update_CV(norm_f.clone().detach(), torch.zeros(f.size(0)).long().cuda())

        # Correct weight with Sigma
        cov = self.estimator.CoVariance.clone().detach()[0]
        cor_weight = torch.mm(cov, weight.clone().detach().t())

        # Create basis with anchor and corrected weight
        basis = deco_ref(f=self.anchor.clone().detach(), ref=cor_weight.t())
        return basis


def deco_ref(f, ref):
    for i in range(ref.size(0)):
        ri = ref[i].view(1, -1)
        f = f - torch.sum(f * ri, dim=1, keepdim=True) / torch.sum(ri ** 2) * ri
        f = f / torch.sqrt(torch.sum(f ** 2, dim=1, keepdim=True))
    return f


def gram_schmidt_ort(w):
    n, d = w.size()
    o = torch.zeros_like(w, requires_grad=False).float().cuda()
    o[0] = w[0].clone().detach().view(1, -1)
    o[0] = o[0] / (o[0] ** 2).sum().sqrt()

    for i in range(1, n):
        temp = 0
        for j in range(i):
            temp = temp + torch.sum(w[i] * o[j]) * o[j]
        o[i] = w[i] - temp
        o[i] = o[i] / (o[i] ** 2).sum().sqrt()

    return o


class PredYWithS(nn.Module):
    def __init__(self, feat_dim, num_classes=10):
        super(PredYWithS, self).__init__()
        latent_dim = feat_dim // 2
        self.train_state = True
        self.update_cov = False
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.estimator = EstimatorCV(feat_dim, num_classes)

        pred = [nn.Linear(feat_dim, latent_dim), nn.BatchNorm1d(latent_dim), nn.ReLU(),
                nn.Linear(latent_dim, feat_dim), nn.ReLU()]
        fc = [nn.Linear(feat_dim, num_classes, bias=True)]
        self.pred = nn.Sequential(*pred)
        self.fc = nn.Sequential(*fc)

    def is_train(self, train):
        self.train_state = train

    def is_update_cov(self, update):
        self.update_cov = update

    def correct_z(self, z, y, cov, ratio):
        N, C, A = z.size(0), self.num_classes, self.feat_dim  # batch size, number of classes, feature dimension
        weight_m = self.fc[0].weight
        NxW_ij = weight_m.expand(N, C, A)
        NxW_kj = torch.gather(NxW_ij, 1, y.view(N, 1, 1).expand(N, C, A))
        CV_temp = cov[y]

        sigma2 = torch.bmm(torch.bmm(NxW_ij - NxW_kj, CV_temp), (NxW_ij - NxW_kj).permute(0, 2, 1))
        sigma2 = sigma2.mul(torch.eye(C).cuda().expand(N, C, C)).sum(2).view(N, C)
        aug_result = z + ratio * 0.5 * sigma2

        return aug_result

    def forward(self, s, y, ratio):
        f = self.pred(s).view(s.size(0), -1)
        z = self.fc(f)

        if self.update_cov:
            norm_f = f / (torch.sqrt(torch.sum(f ** 2, dim=1, keepdim=True)) + 1e-6)
            self.estimator.update_CV(norm_f.clone().detach(), y)

        corrected_z = self.correct_z(z, y, self.estimator.CoVariance.clone().detach(), ratio)
        return corrected_z


class EstimatorCV():
    def __init__(self, feature_num, class_num):
        super(EstimatorCV, self).__init__()
        self.class_num = class_num
        self.CoVariance = torch.zeros(class_num, feature_num, feature_num).cuda()
        self.Ave = torch.zeros(class_num, feature_num).cuda()
        self.Amount = torch.zeros(class_num).cuda()

    def update_CV(self, features, labels):
        N = features.size(0)
        C = self.class_num
        A = features.size(1)

        NxCxFeatures = features.view(
            N, 1, A
        ).expand(
            N, C, A
        )
        onehot = torch.zeros(N, C).cuda()
        onehot.scatter_(1, labels.view(-1, 1), 1)

        NxCxA_onehot = onehot.view(N, C, 1).expand(N, C, A)

        features_by_sort = NxCxFeatures.mul(NxCxA_onehot)

        Amount_CxA = NxCxA_onehot.sum(0)
        Amount_CxA[Amount_CxA == 0] = 1

        ave_CxA = features_by_sort.sum(0) / Amount_CxA

        var_temp = features_by_sort - \
                   ave_CxA.expand(N, C, A).mul(NxCxA_onehot)

        var_temp = torch.bmm(
            var_temp.permute(1, 2, 0),
            var_temp.permute(1, 0, 2)
        ).div(Amount_CxA.view(C, A, 1).expand(C, A, A))

        sum_weight_CV = onehot.sum(0).view(C, 1, 1).expand(C, A, A)

        sum_weight_AV = onehot.sum(0).view(C, 1).expand(C, A)

        weight_CV = sum_weight_CV.div(
            sum_weight_CV + self.Amount.view(C, 1, 1).expand(C, A, A)
        )
        weight_CV[weight_CV != weight_CV] = 0

        weight_AV = sum_weight_AV.div(
            sum_weight_AV + self.Amount.view(C, 1).expand(C, A)
        )
        weight_AV[weight_AV != weight_AV] = 0

        additional_CV = weight_CV.mul(1 - weight_CV).mul(
            torch.bmm(
                (self.Ave - ave_CxA).view(C, A, 1),
                (self.Ave - ave_CxA).view(C, 1, A)
            )
        )

        self.CoVariance = (self.CoVariance.mul(1 - weight_CV) + var_temp
                      .mul(weight_CV)).detach() + additional_CV.detach()

        self.Ave = (self.Ave.mul(1 - weight_AV) + ave_CxA.mul(weight_AV)).detach()

        self.Amount += onehot.sum(0)


class SoftCrossEntropy(nn.Module):
    def __init__(self, reduce=True):
        super(SoftCrossEntropy, self).__init__()
        self.reduce = reduce

    def forward(self, y, z):
        loss = - torch.sum(y * F.log_softmax(z, dim=1), dim=1)
        if self.reduce:
            return loss.mean()
        else:
            return loss


def get_cifar10(args, download=True):
    train_trans = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor()])
    test_trans = transforms.Compose([transforms.ToTensor()])

    train_set = torchvision.datasets.CIFAR10(args.data_dir, train=True, transform=train_trans, download=download)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.num_workers, pin_memory=True, drop_last=True)
    test_set = torchvision.datasets.CIFAR10(args.data_dir, train=False, transform=test_trans, download=download)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=500, shuffle=False)
    return train_loader, test_loader


def get_cifar100(args, download=True):
    train_trans = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                                      transforms.RandomRotation(15), transforms.ToTensor()])
    test_trans = transforms.Compose([transforms.ToTensor()])

    train_set = torchvision.datasets.CIFAR100(args.data_dir, train=True, transform=train_trans, download=download)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.num_workers, pin_memory=True, drop_last=True)

    test_set = torchvision.datasets.CIFAR100(args.data_dir, train=False, transform=test_trans, download=download)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=500, shuffle=False)
    return train_loader, test_loader


def get_dataset(args, download=True):
    if args.dataset == 'cifar10':
        get_data = get_cifar10
    elif args.dataset == 'cifar100':
        get_data = get_cifar100
    else:
        get_data = None
    train_loader, test_loader = get_data(args, download)

    return train_loader, test_loader


def set_deterministic(seed):
    # seed by default is None
    if seed is not None:
        print(f"Deterministic with seed = {seed}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        print("Non-deterministic")
