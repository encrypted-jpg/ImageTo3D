import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from torch.autograd import Variable
import os
import open3d as o3d
import argparse
import json
from models import Generator, Discriminator
from tqdm import tqdm
import time
from PIL import Image
import itertools
import datetime
import random
from tensorboardX import SummaryWriter
import visdom
from imagedataset import ImageDataset
from utils import ReplayBuffer, LambdaLR, weights_init_normal


def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def print_log(fd,  message, time=True):
    if time:
        message = ' ==> '.join(
            [datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), message])
    fd.write(message + '\n')
    fd.flush()
    print(message)


def prepare_logger(log_dir="log", exp_name="exp"):
    # prepare logger directory
    make_dir(log_dir)
    make_dir(os.path.join(log_dir, exp_name))

    logger_path = os.path.join(log_dir, exp_name)
    ckpt_dir = os.path.join(log_dir, exp_name, 'checkpoints')
    epochs_dir = os.path.join(log_dir, exp_name, 'epochs')

    make_dir(logger_path)
    make_dir(ckpt_dir)
    make_dir(epochs_dir)

    logger_file = os.path.join(log_dir, exp_name, 'logger.log')
    log_fd = open(logger_file, 'a')

    print_log(log_fd, "Experiment: {}".format(exp_name), False)
    print_log(log_fd, "Logger directory: {}".format(logger_path), False)

    train_writer = SummaryWriter(os.path.join(logger_path, 'train'))
    val_writer = SummaryWriter(os.path.join(logger_path, 'val'))

    return ckpt_dir, epochs_dir, log_fd, train_writer, val_writer, logger_path


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def dataLoaders(args):
    print("[+] Loading the data...")
    folder = args.folder
    json = args.json
    batch_size = args.batch_size
    transform = [
        transforms.Resize(int(args.size*1.12), Image.BICUBIC),
        transforms.RandomCrop(args.size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]

    trainDataset = ImageDataset(
        folder, json, mode='train', transform=transform)
    testDataset = ImageDataset(
        folder, json, mode='test', transform=transform)
    valDataset = ImageDataset(folder, json, mode='val', transform=transform)

    trainLoader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True)
    testLoader = DataLoader(testDataset, batch_size=1, shuffle=False)
    valLoader = DataLoader(valDataset, batch_size=batch_size, shuffle=False)
    return trainLoader, testLoader, valLoader


def get_model():
    input_nc = 3
    output_nc = 3
    netG_A2B = Generator(input_nc, output_nc)
    netG_B2A = Generator(output_nc, input_nc)
    netD_A = Discriminator(input_nc)
    netD_B = Discriminator(output_nc)
    print(f"Generator A2B: {count_parameters(netG_A2B)}")
    print(f"Generator B2A: {count_parameters(netG_B2A)}")
    print(f"Discriminator A: {count_parameters(netD_A)}")
    print(f"Discriminator B: {count_parameters(netD_B)}")
    return netG_A2B, netG_B2A, netD_A, netD_B


def get_scheduler(optimizer_G, optimizer_D_A, optimizer_D_B, args):
    if args.scheduler == 'step':
        lr_scheduler_G = torch.optim.lr_scheduler.StepLR(
            optimizer_G, step_size=1, gamma=args.gamma)
        lr_scheduler_D_A = torch.optim.lr_scheduler.StepLR(
            optimizer_D_A, step_size=1, gamma=args.gamma)
        lr_scheduler_D_B = torch.optim.lr_scheduler.StepLR(
            optimizer_D_B, step_size=1, gamma=args.gamma)
    else:
        lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
            optimizer_G, lr_lambda=LambdaLR(args.n_epochs, args.epoch, args.decay_epoch).step)
        lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
            optimizer_D_A, lr_lambda=LambdaLR(args.n_epochs, args.epoch, args.decay_epoch).step)
        lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
            optimizer_D_B, lr_lambda=LambdaLR(args.n_epochs, args.epoch, args.decay_epoch).step)
    return lr_scheduler_G, lr_scheduler_D_A, lr_scheduler_D_B


def train(models, trainLoader, valLoader, args):
    print("[+] Training the model...")
    ckpt_dir, epochs_dir, log_fd, train_writer, val_writer, exp_path = prepare_logger(
        args.log_dir, args.exp)
    bestSavePath = os.path.join(ckpt_dir, "bestModel.pth")
    lastSavePath = os.path.join(ckpt_dir, "lastModel.pth")
    print_log(log_fd, str(args))
    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    netG_A2B, netG_B2A, netD_A, netD_B = models
    netG_A2B = netG_A2B.to(device)
    netG_B2A = netG_B2A.to(device)
    netD_A = netD_A.to(device)
    netD_B = netD_B.to(device)

    # Lossess
    criterion_GAN = torch.nn.MSELoss()  # lsgan
    # criterion_GAN = torch.nn.BCEWithLogitsLoss() #vanilla
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()

    # Optimizers & LR schedulers
    optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                                   lr=args.lr, betas=(0.5, 0.999))
    optimizer_D_A = torch.optim.Adam(
        netD_A.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_D_B = torch.optim.Adam(
        netD_B.parameters(), lr=args.lr, betas=(0.5, 0.999))

    lr_scheduler_G, lr_scheduler_D_A, lr_scheduler_D_B = get_scheduler(
        optimizer_G, optimizer_D_A, optimizer_D_B, args)

    train_step = 0
    minLoss = 1e10
    minLossEpoch = 0

    if args.resume:
        print_log(log_fd, f"Loading checkpoint from {args.modelPath}")
        checkpoint = torch.load(args.modelPath)
        netG_A2B.load_state_dict(checkpoint['netG_A2B'])
        netG_B2A.load_state_dict(checkpoint['netG_B2A'])
        netD_A.load_state_dict(checkpoint['netD_A'])
        netD_B.load_state_dict(checkpoint['netD_B'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G'])
        optimizer_D_A.load_state_dict(checkpoint['optimizer_D_A'])
        optimizer_D_B.load_state_dict(checkpoint['optimizer_D_B'])
        lr_scheduler_G.load_state_dict(checkpoint['lr_scheduler_G'])
        lr_scheduler_D_A.load_state_dict(checkpoint['lr_scheduler_D_A'])
        lr_scheduler_D_B.load_state_dict(checkpoint['lr_scheduler_D_B'])
        args.epoch = checkpoint['epoch']
        minLoss = checkpoint['loss']
        minLossEpoch = args.epoch
        print_log(
            log_fd, f"Checkpoint loaded (epoch {args.epoch}, loss {minLoss})")

    # Inputs & targets memory allocation

    Tensor = torch.cuda.FloatTensor if device.type == 'cuda' else torch.Tensor
    input_A = Tensor(args.batch_size, 3, args.size, args.size).to(device)
    input_B = Tensor(args.batch_size, 3, args.size, args.size).to(device)
    target_real = Variable(
        Tensor(args.batch_size).fill_(1.0), requires_grad=False).to(device)
    target_fake = Variable(
        Tensor(args.batch_size).fill_(0.0), requires_grad=False).to(device)

    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    G_losses = []
    D_A_losses = []
    D_B_losses = []
    to_pil = transforms.ToPILImage()

    ###### Training ######
    for epoch in range(args.epoch, args.n_epochs):
        netG_A2B.train()
        netG_B2A.train()
        netD_A.train()
        netD_B.train()
        print_log(
            log_fd, "------------------Epoch: {}------------------".format(epoch))
        train_loss = 0.0
        loader = tqdm(trainLoader)
        for i, batch in enumerate(loader):
            loader.set_description(f"Loss: {(train_loss/(i+1)):.4f}")
            # Set model input
            taxonomy_id, model_id, (A, B) = batch
            real_A = Variable(input_A.copy_(A)).to(device)
            real_B = Variable(input_B.copy_(B)).to(device)

            ###### Generators A2B and B2A ######
            optimizer_G.zero_grad()

            # # Identity loss
            # # G_A2B(B) should equal B if real B is fed
            # same_B = netG_A2B(real_B)
            # loss_identity_B = criterion_identity(
            #     same_B, real_B)*5.0  # ||Gb(b)-b||1
            # # G_B2A(A) should equal A if real A is fed
            # same_A = netG_B2A(real_A)
            # loss_identity_A = criterion_identity(
            #     same_A, real_A)*5.0  # ||Ga(a)-a||1

            # GAN loss
            fake_B = netG_A2B(real_A)
            pred_fake = netD_B(fake_B)
            loss_GAN_A2B = criterion_GAN(
                pred_fake, target_real)  # log(Db(Gb(a)))

            fake_A = netG_B2A(real_B)
            pred_fake = netD_A(fake_A)
            loss_GAN_B2A = criterion_GAN(
                pred_fake, target_real)  # log(Da(Ga(b)))

            # Cycle loss
            recovered_A = netG_B2A(fake_B)
            loss_cycle_ABA = criterion_cycle(
                recovered_A, real_A)*10.0  # ||Ga(Gb(a))-a||1

            recovered_B = netG_A2B(fake_A)
            loss_cycle_BAB = criterion_cycle(
                recovered_B, real_B)*10.0  # ||Gb(Ga(b))-b||1

            # Total loss
            loss_G = loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
            loss_G.backward()

            G_losses.append(loss_G.item())

            optimizer_G.step()
            ###################################

            ###### Discriminator A ######
            optimizer_D_A.zero_grad()

            # Real loss
            pred_real = netD_A(real_A)
            loss_D_real = criterion_GAN(pred_real, target_real)  # log(Da(a))

            # Fake loss
            fake_A = fake_A_buffer.push_and_pop(fake_A)
            pred_fake = netD_A(fake_A.detach())
            loss_D_fake = criterion_GAN(
                pred_fake, target_fake)  # log(1-Da(G(b)))

            # Total loss
            loss_D_A = (loss_D_real + loss_D_fake)*0.5
            loss_D_A.backward()

            D_A_losses.append(loss_D_A.item())

            optimizer_D_A.step()
            ###################################

            ###### Discriminator B ######
            optimizer_D_B.zero_grad()

            # Real loss
            pred_real = netD_B(real_B)
            loss_D_real = criterion_GAN(pred_real, target_real)  # log(Db(b))

            # Fake loss
            fake_B = fake_B_buffer.push_and_pop(fake_B)
            pred_fake = netD_B(fake_B.detach())
            loss_D_fake = criterion_GAN(
                pred_fake, target_fake)  # log(1-Db(G(a)))

            # Total loss
            loss_D_B = (loss_D_real + loss_D_fake)*0.5
            loss_D_B.backward()

            D_B_losses.append(loss_D_B.item())

            optimizer_D_B.step()
            ###################################
            train_loss += loss_G.item()
            train_step += 1

            train_writer.add_scalar('loss_G', loss_G.item(), train_step)
            train_writer.add_scalar('loss_D_A', loss_D_A.item(), train_step)
            train_writer.add_scalar('loss_D_B', loss_D_B.item(), train_step)

            if train_step % args.save_iter == 0:
                img_fake_A = 0.5 * (fake_A.detach().data + 1.0)
                img_fake_A = (to_pil(img_fake_A[0].data.squeeze(0).cpu()))
                img_fake_A.save(os.path.join(exp_path, "fake_A.png"))

                img_fake_B = 0.5 * (fake_B.detach().data + 1.0)
                img_fake_B = (to_pil(img_fake_B[0].data.squeeze(0).cpu()))
                img_fake_B.save(os.path.join(exp_path, "fake_B.png"))

                img_real_A = 0.5 * (real_A.detach().data + 1.0)
                img_real_A = (to_pil(img_real_A[0].data.squeeze(0).cpu()))
                img_real_A.save(os.path.join(exp_path, "real_A.png"))

                img_real_B = 0.5 * (real_B.detach().data + 1.0)
                img_real_B = (to_pil(img_real_B[0].data.squeeze(0).cpu()))
                img_real_B.save(os.path.join(exp_path, "real_B.png"))

        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

        train_loss /= len(trainLoader)
        print_log(log_fd, f"Epoch {epoch} Train Loss: {train_loss}")

        netG_A2B.eval()
        netG_B2A.eval()
        netD_A.eval()
        netD_B.eval()
        val_loss = 0.0
        with torch.no_grad():
            loader = tqdm(valLoader)
            for i, batch in enumerate(loader):
                loader.set_description(f"Loss: {(val_loss/(i+1)):.4f}")
                # Set model input
                taxonomy_id, model_id, (A, B) = batch
                real_A = Variable(input_A.copy_(A))
                real_B = Variable(input_B.copy_(B))

                ###### Generators A2B and B2A ######
                optimizer_G.zero_grad()

                # # Identity loss
                # # G_A2B(B) should equal B if real B is fed
                # same_B = netG_A2B(real_B)
                # loss_identity_B = criterion_identity(
                #     same_B, real_B)*5.0  # ||Gb(b)-b||1
                # # G_B2A(A) should equal A if real A is fed
                # same_A = netG_B2A(real_A)
                # loss_identity_A = criterion_identity(
                #     same_A, real_A)*5.0

                # GAN loss
                fake_B = netG_A2B(real_A)
                pred_fake = netD_B(fake_B)
                loss_GAN_A2B = criterion_GAN(
                    pred_fake, target_real)

                fake_A = netG_B2A(real_B)
                pred_fake = netD_A(fake_A)
                loss_GAN_B2A = criterion_GAN(
                    pred_fake, target_real)

                # Cycle loss
                recovered_A = netG_B2A(fake_B)
                loss_cycle_ABA = criterion_cycle(
                    recovered_A, real_A)*10.0

                recovered_B = netG_A2B(fake_A)
                loss_cycle_BAB = criterion_cycle(
                    recovered_B, real_B)*10.0

                # Total loss
                loss_G = loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
                val_writer.add_scalar('loss_G', loss_G.item(), i)
                val_loss += loss_G.item()

        val_loss /= len(valLoader)
        print_log(
            log_fd, f"Epoch {epoch} Val Loss: {val_loss} Learning Rate: {lr_scheduler_G.get_lr()[0]}")

        if val_loss < minLoss:
            minLoss = val_loss
            minLossEpoch = epoch
            torch.save({
                'epoch': epoch,
                'loss': val_loss,
                'netG_A2B': netG_A2B.state_dict(),
                'netG_B2A': netG_B2A.state_dict(),
                'netD_A': netD_A.state_dict(),
                'netD_B': netD_B.state_dict(),
                'optimizer_G': optimizer_G.state_dict(),
                'optimizer_D_A': optimizer_D_A.state_dict(),
                'optimizer_D_B': optimizer_D_B.state_dict(),
                'lr_scheduler_G': lr_scheduler_G.state_dict(),
                'lr_scheduler_D_A': lr_scheduler_D_A.state_dict(),
                'lr_scheduler_D_B': lr_scheduler_D_B.state_dict(),
                'loss': val_loss
            }, bestSavePath)
            print_log(log_fd, f"Epoch {epoch} Best Model Saved")

        torch.save({
            'epoch': epoch,
            'loss': val_loss,
            'netG_A2B': netG_A2B.state_dict(),
            'netG_B2A': netG_B2A.state_dict(),
            'netD_A': netD_A.state_dict(),
            'netD_B': netD_B.state_dict(),
            'optimizer_G': optimizer_G.state_dict(),
            'optimizer_D_A': optimizer_D_A.state_dict(),
            'optimizer_D_B': optimizer_D_B.state_dict(),
            'lr_scheduler_G': lr_scheduler_G.state_dict(),
            'lr_scheduler_D_A': lr_scheduler_D_A.state_dict(),
            'lr_scheduler_D_B': lr_scheduler_D_B.state_dict(),
            'loss': val_loss
        }, lastSavePath)

        print_log(log_fd, "Last Model saved (best loss {:.4f} at epoch {})" .format(
            minLoss, minLossEpoch))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default="ShapeNetRender",
                        help="Folder containing the data")
    parser.add_argument("--json", type=str, default="final.json",
                        help="JSON file containing the data")
    parser.add_argument("--log_dir", type=str, default="logs", help="Log dir")
    parser.add_argument("--exp", type=str, default="exp", help="Experiment")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--size", type=int, default=224, help="Image size")
    parser.add_argument("--epoch", type=int, default=0, help="Epoch to start")
    parser.add_argument("--scheduler", type=str,
                        default="step", help="Scheduler")
    parser.add_argument("--gamma", type=float, default=0.85, help="Gamma")
    parser.add_argument("--n_epochs", type=int, default=200,
                        help="Number of epochs")
    parser.add_argument("--decay_epoch", type=int, default=100,
                        help="Decay epoch")
    parser.add_argument("--save_iter", type=int,
                        default=10, help="Save interval")
    parser.add_argument("--lr", type=float, default=0.0002,
                        help="Learning rate")
    parser.add_argument("--gpu", type=int, default=0, help="GPU to use")
    parser.add_argument("--modelPath", type=str,
                        default="bestModel.pth", help="Path to model")
    parser.add_argument("--test", action="store_true", help="Test model")
    parser.add_argument("--testSave", action="store_true",
                        help="Save test output")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    trainLoader, testLoader, valLoader = dataLoaders(args)
    models = get_model()
    train(models, trainLoader, valLoader, args)
