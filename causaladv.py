import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pgd_attack import pgd_attack, adp_pgd_attack
from wideresnet import WideResNet
from resnet import ResNet18
from causaladv_utils import pgd, set_deterministic, get_dataset, get_args_cifar10, init_anchor, PredYWithS, \
    get_s_pre, SoftCrossEntropy, Basis

set_deterministic(0)
args = get_args_cifar10()
if args.net == 'wrn':
    cnn = WideResNet
else:
    cnn = ResNet18
best_acc, best_epoch = 0, 0
train_loader, test_loader = get_dataset(args)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


def model_train():
    global best_acc, best_epoch
    file_path = os.path.join(args.output_dir, f'{args.model_name}.txt')
    with open(file_path, "w+") as file:
        file.write('Epoch\tNat-A\tAdv-L\tAdv-A\tClean\tFGSMA\tPGD20\tCWA20\n')

    # Initialize model and basis
    basis = None
    print('Initialize model')
    model = cnn(num_classes=args.num_classes).to(args.device)
    print('Initialize anchor')
    anchor = init_anchor(weight=model.fc[0].weight.data.detach())
    v_space = Basis(anchor=anchor)
    model_g = PredYWithS(feat_dim=anchor.size(1), num_classes=args.num_classes).to(args.device)
    soft_ce = SoftCrossEntropy().to(args.device)
    ce_loss = nn.CrossEntropyLoss().to(args.device)
    kl_loss = nn.KLDivLoss(reduction='batchmean').to(args.device)

    # Initialize optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.decay_lr1, args.decay_lr2], gamma=0.1)
    optimizer_g = optim.SGD(model_g.parameters(), lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler_g = optim.lr_scheduler.MultiStepLR(optimizer_g, milestones=[args.decay_lr1, args.decay_lr2], gamma=0.1)

    # The main loop
    for epoch in range(args.num_epochs):
        model.train()
        model_g.train()
        model.is_train(True)
        tr_al, tr_na, tr_aa = [], [], []
        # Gradually assign higher confidence for the learned style information
        ratio = np.clip(args.adv_ratio * epoch / args.num_epochs, 0.0, args.adv_ratio)
        alpha = np.clip(args.adv_alpha * epoch / args.num_epochs, 0.0, args.adv_alpha)

        for idx, (x, y) in enumerate(train_loader):
            # initialization
            x, y = x.to(args.device), y.to(args.device)
            f, z = model(x)
            basis = v_space.get_basis(f=f, weight=model.fc[0].weight.data)

            # Generate adversarial examples
            x_adv = pgd(model=model, images=x, labels=y, basis=basis, model_g=model_g, args=args)

            # Generate representations
            f_adv, z_adv = model(x_adv)
            s_pre = get_s_pre(f, basis)
            s_adv_pre = get_s_pre(f_adv, basis)

            # Update model g, i.e., fitting the spurious correlation between Y and S
            model_g.train()
            model_g.is_update_cov(True)
            z_s = model_g(s_pre.detach().clone(), y, ratio=ratio)
            ce_g = ce_loss(z_s, y)
            ce_g.backward()
            optimizer_g.step()
            model_g.zero_grad()

            # Update model h, i.e., aligning the adversarial distribution with the natural distribution
            # f-loss on the content information
            if args.adv_mode == 'adam':
                loss_adv = ce_loss(z_adv, y)
                loss_c = loss_adv
            else:
                loss_nat = ce_loss(z, y)
                loss_adv = kl_loss(F.log_softmax(z_adv, dim=1), F.softmax(z, dim=1))
                loss_c = loss_nat / args.adv_beta + loss_adv

            # f-loss on the style information
            model_g.eval()
            model_g.is_update_cov(False)
            z_s = model_g(s_pre, y, ratio=ratio)
            z_s_adv = model_g(s_adv_pre, y, ratio=ratio)
            loss_s = soft_ce(z=z_s_adv, y=F.softmax(z_s, dim=1))

            # total loss and backward
            loss = args.adv_beta * loss_c + alpha * loss_s
            loss.backward()
            optimizer.step()
            model.zero_grad()
            model_g.zero_grad()  # remove the gradient generated from updating h

            # Log
            acc_nat = 100. * torch.max(z.data, 1)[1].eq(y).sum().cpu().item() / y.size(0)
            acc_adv = 100. * torch.max(z_adv.data, 1)[1].eq(y).sum().cpu().item() / y.size(0)
            acc_s = 100. * torch.max(z_s.data, 1)[1].eq(y).sum().cpu().item() / y.size(0)
            acc_s_adv = 100. * torch.max(z_s_adv.data, 1)[1].eq(y).sum().cpu().item() / y.size(0)
            if idx % args.step_train_log == 0 or idx == len(train_loader) - 1:
                print('[E:{}/{},B:{}/{}], ce_adv={:.3f}, ce_g={:.3f}, ce_s_adv={:.3f}, '
                      'acc_nat={:.2f}, acc_adv={:.2f}, acc_s={:.2f}, acc_s_adv={:.2f}'.
                      format(epoch, args.num_epochs, idx, len(train_loader), loss_adv.item(), ce_g.item(),
                             loss_s.item(), acc_nat, acc_adv, acc_s, acc_s_adv))

            # Record
            tr_al.append(loss_adv.item())
            tr_na.append(acc_nat)
            tr_aa.append(acc_adv)

        scheduler.step(epoch=epoch)
        scheduler_g.step(epoch=epoch)
        if (args.decay_lr1 < epoch < args.decay_lr1 + args.test_step) or \
                (args.decay_lr2 < epoch < args.decay_lr2 + args.test_step) or \
                (epoch % args.test_step == 0) or (args.num_epochs - args.test_step < epoch):
            # Evaluate the robustness
            clean, fgsma, pgd20, cwa20 = model_eval(model=model)  # Here we do not use adaptive attacks

            # Save the best checkpoint
            if pgd20 > best_acc:
                best_acc, best_epoch = pgd20, epoch
                save_model(model=model, model_g=model_g, basis=basis, acc=best_acc, name='best')
                print('Best PGD20 accuracy {:.2f} at epoch {}'.format(best_acc, best_epoch))

            # Log
            with open(file_path, "a+") as file:
                file.write('{}    \t{:.2f}\t{:.3f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\n'.format(
                    epoch, np.mean(tr_na), np.mean(tr_al), np.mean(tr_aa), clean, fgsma, pgd20, cwa20))

    # Save the last checkpoint
    save_model(model=model, model_g=model_g, basis=basis)


def save_model(model, model_g, basis, acc=0, name=''):
    print('Saving..')
    model_path = os.path.join(args.output_dir, f'{args.dataset}-{args.model_name}-{name}.pth')
    torch.save({'net': model.state_dict(), 'acc': acc, 'g': model_g.state_dict(), 'b': basis}, model_path)
    print(f"Model saved to {model_path}")


def model_eval(model):
    clean = model_test(model=model)
    fgsma = model_robust(model=model, num_steps=1, loss_fn='ce')
    pgd20 = model_robust(model=model, num_steps=20, loss_fn='ce')
    cwa20 = model_robust(model=model, num_steps=20, loss_fn='cw')
    return clean, fgsma, pgd20, cwa20


def model_test(model):
    model.eval()
    model = model.cuda()
    model.is_train(False)
    ce_loss = nn.CrossEntropyLoss()

    correct_cnt, total_cnt = 0, 0
    for idx, (images, labels) in enumerate(test_loader):
        images, labels = images.to(args.device), labels.to(args.device)
        z = model.forward(images).detach()
        loss = ce_loss(z, labels)
        _, pred_label = torch.max(z.data, 1)
        total_cnt = total_cnt + images.data.size()[0]
        correct_cnt = correct_cnt + 100. * pred_label.eq(labels).sum().cpu().item()

        if idx % args.step_test_log == 0 or idx == len(test_loader) - 1:
            print('Batch: {} Acc = {:.2f}, Loss = {:.3f}'.format(idx + 1, correct_cnt / total_cnt, loss.item()))
    acc = correct_cnt / total_cnt
    return acc


def model_robust(model, num_steps, loss_fn, adaptive=False, model_g=None, basis=None):
    model.eval()
    model = model.cuda()
    model.is_train(False)
    ce_loss = nn.CrossEntropyLoss()
    epsilon = 0.031
    if num_steps == 1:
        step_size = epsilon
    else:
        step_size = epsilon / 10.0

    correct_cnt, total_cnt = 0, 0
    for idx, (images, labels) in enumerate(test_loader):
        images, labels = images.to(args.device), labels.to(args.device)
        if adaptive:
            images = adp_pgd_attack(model, images, labels, eps=epsilon, step_size=step_size, k=num_steps,
                                    loss_fn=loss_fn, model_g=model_g, basis=basis)
        else:
            images = pgd_attack(model, images, labels, eps=epsilon, step_size=step_size, k=num_steps, loss_fn=loss_fn,
                                num_classes=args.num_classes)
        z = model(images).detach()
        loss = ce_loss(z, labels)
        _, pred_label = torch.max(z.data, 1)
        total_cnt = total_cnt + images.data.size()[0]
        correct_cnt = correct_cnt + 100. * pred_label.eq(labels).sum().cpu().item()

        if idx % args.step_test_log == 0 or idx == len(test_loader) - 1:
            print('Batch: {} Acc = {:.2f}, Loss = {:.3f}'.format(idx + 1, correct_cnt / total_cnt, loss.item()))
    acc = correct_cnt / total_cnt
    print('Final performance step={}, acc={:.2f}'.format(num_steps, acc))
    return acc


if __name__ == "__main__":
    if args.train:
        model_train()
    else:
        # Load models and basis
        model_name = os.path.join(args.output_dir, f'{args.dataset}-{args.model_name}.pth')
        test_model = cnn(num_classes=args.num_classes).to(args.device)  # model h
        B = torch.load(model_name)['b']  # basis
        G = PredYWithS(feat_dim=B.size(1)).to(args.device)  # model g
        G.load_state_dict(torch.load(model_name)['g'])
        test_model.load_state_dict(torch.load(model_name)['net'])
        test_model.eval()

        # Robustness evaluation
        adaptive_attack = False  # Here we can use the normal attacks and adaptive attacks
        model_test(model=test_model)
        # FGSM
        model_robust(test_model, num_steps=1, loss_fn='ce', adaptive=adaptive_attack, model_g=G, basis=B)
        # PGD-20
        model_robust(test_model, num_steps=20, loss_fn='ce', adaptive=adaptive_attack, model_g=G, basis=B)
        # CW-20
        model_robust(test_model, num_steps=20, loss_fn='cw', adaptive=adaptive_attack, model_g=G, basis=B)
        # PGD-50
        model_robust(test_model, num_steps=50, loss_fn='ce', adaptive=adaptive_attack, model_g=G, basis=B)
        # CW-50
        model_robust(test_model, num_steps=50, loss_fn='cw', adaptive=adaptive_attack, model_g=G, basis=B)
        # PGD-100
        model_robust(test_model, num_steps=100, loss_fn='ce', adaptive=adaptive_attack, model_g=G, basis=B)
        # CW-100
        model_robust(test_model, num_steps=100, loss_fn='cw', adaptive=adaptive_attack, model_g=G, basis=B)
        # PGD-500
        model_robust(test_model, num_steps=500, loss_fn='ce', adaptive=adaptive_attack, model_g=G, basis=B)
        # CW-500
        model_robust(test_model, num_steps=500, loss_fn='cw', adaptive=adaptive_attack, model_g=G, basis=B)
        # PGD-1000
        model_robust(test_model, num_steps=1000, loss_fn='ce', adaptive=adaptive_attack, model_g=G, basis=B)
        # CW-1000
        model_robust(test_model, num_steps=1000, loss_fn='cw', adaptive=adaptive_attack, model_g=G, basis=B)
