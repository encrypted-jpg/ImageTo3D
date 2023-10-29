import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import numpy as np
from torch.autograd import Variable
import os
import open3d as o3d
import argparse
import json
from models.pcn import PCNEncoder, PCNDecoder
from models.base_image_encoder import BaseImageEncoder
from models.dual_image_encoder import DualImageEncoder
from models.bicyclegan import Generator
from tqdm import tqdm
import time
from PIL import Image
import cv2
import itertools
import datetime
import random
from tensorboardX import SummaryWriter
import visdom
from datasets import PCNImageDataset
from utils.utils import ReplayBuffer, LambdaLR, weights_init_normal, plot_pcd_one_view, plot_2_image_output_gt
from extensions.chamfer_dist import ChamferDistanceL1


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
    test_dir = os.path.join(log_dir, exp_name, 'test')

    make_dir(logger_path)
    make_dir(ckpt_dir)
    make_dir(epochs_dir)
    make_dir(test_dir)

    logger_file = os.path.join(log_dir, exp_name, 'logger.log')
    log_fd = open(logger_file, 'a')

    test_logger_file = os.path.join(log_dir, exp_name, 'test_logger.log')
    test_log_fd = open(test_logger_file, 'a')

    print_log(log_fd, "Experiment: {}".format(exp_name), False)
    print_log(log_fd, "Logger directory: {}".format(logger_path), False)

    train_writer = SummaryWriter(os.path.join(logger_path, 'train'))
    val_writer = SummaryWriter(os.path.join(logger_path, 'val'))

    return ckpt_dir, epochs_dir, log_fd, train_writer, val_writer, logger_path, test_dir, test_log_fd


def save_imgs(img1, img2, img3, img4, path):
    stripe_width = 10  # Adjust as needed
    image_height = 224  # Assuming all images have the same height

    # Create a new black stripe image
    black_stripe = Image.new("RGB", (stripe_width, image_height), (0, 0, 0))

    # Concatenate the images horizontally with the black stripe in between
    concatenated_image = Image.new("RGB", (4 * 224 + 3 * stripe_width, 224))
    concatenated_image.paste(img1, (0, 0))
    concatenated_image.paste(black_stripe, (224, 0))
    concatenated_image.paste(img2, (224 + stripe_width, 0))
    concatenated_image.paste(black_stripe, (2 * 224 + stripe_width, 0))
    concatenated_image.paste(img3, (2 * 224 + 2 * stripe_width, 0))
    concatenated_image.paste(black_stripe, (3 * 224 + 2 * stripe_width, 0))
    concatenated_image.paste(img4, (3 * 224 + 3 * stripe_width, 0))

    # Save the concatenated image
    concatenated_image.save(path)


def save_batch_imgs(real_A, fake_B, path):
    img_samples = None
    fake_B = [x for x in fake_B.data.cpu()]
    mids = [len(fake_B) // 4, len(fake_B) // 2, 3 * len(fake_B) // 4]
    i = 0
    img_list = []
    for img_A, f_B in zip(real_A, fake_B):
        f_B = f_B.view(1, *f_B.shape)
        img_A = img_A.view(1, *img_A.shape)
        img_sample = torch.cat((img_A, f_B), -1)
        img_sample = img_sample.view(1, *img_sample.shape)
        # Concatenate with previous samples vertically
        img_samples = img_sample if img_samples is None else torch.cat(
            (img_samples, img_sample), -2)
        i += 1
        if i in mids:
            img_list.append(img_samples)
            img_samples = None
    img_list.append(img_samples)
    img_samples = torch.cat(img_list, -1)
    save_image(img_samples[0][0], path, normalize=True)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def dataLoaders(args):
    print("[+] Loading the data...")
    folder = args.folder
    json = args.json
    batch_size = args.batch_size

    trainDataset = PCNImageDataset(
        folder, json, mode='train', b_tag=args.b_tag, img_height=args.size, img_width=args.size)
    testDataset = PCNImageDataset(
        folder, json, mode='test', b_tag=args.b_tag, img_height=args.size, img_width=args.size)
    valDataset = PCNImageDataset(
        folder, json, mode='val', b_tag=args.b_tag, img_height=args.size, img_width=args.size)

    trainLoader = DataLoader(
        trainDataset, batch_size=batch_size, shuffle=True, drop_last=True)
    testLoader = DataLoader(
        testDataset, batch_size=batch_size, shuffle=False, drop_last=True)
    valLoader = DataLoader(
        valDataset, batch_size=batch_size, shuffle=False, drop_last=True)
    return trainLoader, testLoader, valLoader


def reparameterization(mu, logvar, Tensor, args):
    std = torch.exp(logvar / 2)
    sampled_z = Variable(Tensor(np.random.normal(
        0, 1, (mu.size(0), args.latent_dim))))
    z = sampled_z * std + mu
    return z


def get_model(args):
    input_shape = (3, args.size, args.size)
    generator = Generator(args.gen_latent_dim, input_shape)
    img_encoder = DualImageEncoder(latent_dim=1024)
    encoder = PCNEncoder(latent_dim=1024)
    decoder = PCNDecoder(latent_dim=1024, num_dense=16384)
    print(
        f"Generator Parameters: {round(count_parameters(generator)/ 1e6, 4)}")
    print(
        f"Dual Image Encoder Parameters: {round(count_parameters(img_encoder)/ 1e6, 4)}")
    print(f"Encoder Parameters: {round(count_parameters(encoder)/ 1e6, 4)}")
    print(f"Decoder Parameters: {round(count_parameters(decoder)/ 1e6, 4)}")
    return generator, img_encoder, encoder, decoder


def get_scheduler(optimizer, args):
    if args.scheduler == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=1, gamma=args.gamma)
    else:
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=LambdaLR(args.n_epochs, args.epoch, args.decay_epoch).step)
    return lr_scheduler


def load_pcn_model(encoder, decoder, path):
    print(f"Loading PCN model from {path}")
    checkpoint = torch.load(path)
    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])

    for param in encoder.parameters():
        param.requires_grad = False
    for param in decoder.parameters():
        param.requires_grad = False

    print(f"PCN Model loaded")
    return encoder, decoder


def load_generator(generator, path):
    print(f"Loading Generator model from {path}")
    checkpoint = torch.load(path)
    generator.load_state_dict(checkpoint['generator'])

    for param in generator.parameters():
        param.requires_grad = False

    print(f"Generator Model loaded")
    return generator


def load_base(img_encoder, path):
    if path == "":
        return img_encoder
    print(f"Loading Base Image Encoder model from {path}")
    checkpoint = torch.load(path)
    img_encoder.first_encoder.load_state_dict(checkpoint['img_encoder'])

    print(f"Base Image Encoder Model loaded")
    return img_encoder


def train(models, trainLoader, valLoader, args):
    print("[+] Training the model...")
    ckpt_dir, epochs_dir, log_fd, train_writer, val_writer, exp_path, _, _ = prepare_logger(
        args.log_dir, args.exp)
    bestSavePath = os.path.join(ckpt_dir, "bestModel.pth")
    lastSavePath = os.path.join(ckpt_dir, "lastModel.pth")
    print_log(log_fd, str(args))
    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    generator, img_encoder, encoder, decoder = models
    generator.to(device)
    img_encoder.to(device)
    encoder.to(device)
    decoder.to(device)

    # Lossess
    chamfer = ChamferDistanceL1().to(device)
    MSE = nn.MSELoss().to(device)

    # Optimizers & LR schedulers
    optimizer = torch.optim.Adam(
        img_encoder.parameters(), lr=args.lr, betas=(0.5, 0.999))

    lr_scheduler = get_scheduler(optimizer, args)

    encoder, decoder = load_pcn_model(
        encoder, decoder, args.pcn)
    generator = load_generator(generator, args.generator)
    img_encoder = load_base(img_encoder, args.base)

    train_step = 0
    minLoss = 1e10
    minLossEpoch = 0

    if args.resume:
        print_log(log_fd, f"Loading checkpoint from {args.modelPath}")
        checkpoint = torch.load(args.modelPath)
        img_encoder.load_state_dict(checkpoint['img_encoder'])
        args.epoch = checkpoint['epoch'] + 1
        train_step = checkpoint['epoch'] * len(trainLoader)
        # minLoss = checkpoint['loss']
        # minLossEpoch = args.epoch
        lr_scheduler = get_scheduler(optimizer, args)
        print_log(
            log_fd, f"Checkpoint loaded (epoch {checkpoint['epoch']}, loss {checkpoint['loss']})")

    ###### Training ######
    for epoch in range(args.epoch, args.n_epochs):
        img_encoder.train()
        generator.eval()
        encoder.eval()
        decoder.eval()
        print_log(
            log_fd, "------------------Epoch: {}------------------".format(epoch))
        train_loss = 0.0
        loader = tqdm(trainLoader)
        for i, batch in enumerate(loader):
            loader.set_description(f"Loss: {(train_loss/(i+1)):.4f}")

            taxonomy_id, model_id, (A, B, C) = batch
            img1 = A.to(torch.float32).to(device)
            inp = B.to(torch.float32).to(device)
            gt = C.to(torch.float32).to(device)

            optimizer.zero_grad()

            sampled_z = torch.randn(
                img1.size(0), args.gen_latent_dim).to(device)
            img2 = generator(img1, sampled_z)
            img2 = torch.max(img2, torch.zeros_like(img2))
            img2 = torch.min(img2, torch.ones_like(img2))

            rep = img_encoder(img1, img2)
            base_rep = encoder(inp)

            coarse, fine = decoder(rep)

            loss1 = chamfer(coarse, gt)
            loss2 = chamfer(fine, gt)
            chamfer_loss = loss1 * args.lambda_coarse
            chamfer_loss += loss2 * (1 - args.lambda_coarse)
            mse_loss = MSE(base_rep, rep) * args.lambda_latent
            loss = chamfer_loss * args.lambda_chamfer + mse_loss
            loss.backward()
            optimizer.step()

            train_loss += loss2.item() * 1000
            train_step += 1

            train_writer.add_scalar('loss', loss.item(), train_step)

            if train_step % args.save_iter == 0:
                index = random.randint(0, inp.shape[0] - 1)
                plot_2_image_output_gt(os.path.join(exp_path, f'train_{train_step}.png'), A[index].detach().cpu().transpose(1, 0).transpose(1, 2).numpy(), img2[index].detach().cpu().transpose(1, 0).transpose(1, 2).numpy(), fine[index].detach().cpu().numpy(), gt[index].detach(
                ).cpu().numpy(), img1_title='Input Image', img2_title=f'{args.b_tag} Image', output_title='Dense Output PCD', gt_title='Ground Truth PCD', suptitle='', pcd_size=0.5, cmap='Reds', zdir='y')
                # plot_pcd_one_view(os.path.join(exp_path, f'train_{train_step}.png'),
                #                   [inp[index].detach().cpu().numpy(), coarse[index].detach().cpu().numpy(
                #                   ), fine[index].detach().cpu().numpy(), gt[index].detach().cpu().numpy()],
                #                   ['Input', 'Coarse', 'Dense', 'Ground Truth'], xlim=(-0.5, 1), ylim=(-0.5, 1), zlim=(-0.5, 1))

        # Update learning rates
        lr_scheduler.step()

        train_loss /= len(trainLoader)
        print_log(log_fd, f"Epoch {epoch} Train Loss: {train_loss}")

        ###### Validation ######
        img_encoder.eval()
        encoder.eval()
        decoder.eval()
        val_loss = 0.0
        with torch.no_grad():
            loader = tqdm(valLoader)
            for i, batch in enumerate(loader):
                loader.set_description(f"Loss: {(val_loss/(i+1)):.4f}")
                # Set model input
                taxonomy_id, model_id, (A, B, C) = batch
                img1 = A.to(torch.float32).to(device)
                inp = B.to(torch.float32).to(device)
                gt = C.to(torch.float32).to(device)

                optimizer.zero_grad()

                sampled_z = torch.randn(
                    img1.size(0), args.gen_latent_dim).to(device)
                img2 = generator(img1, sampled_z)
                img2 = torch.max(img2, torch.zeros_like(img2))
                img2 = torch.min(img2, torch.ones_like(img2))

                rep = img_encoder(img1, img2)
                base_rep = encoder(inp)

                coarse, fine = decoder(rep)

                # loss1 = chamfer(coarse, gt)
                loss2 = chamfer(fine, gt)
                chamfer_loss = loss2
                # mse_loss = MSE(base_rep, rep) * args.lambda_latent
                loss = chamfer_loss

                val_writer.add_scalar('loss', loss.item(), i)
                val_loss += loss.item() * 1000

        val_loss /= len(valLoader)
        print_log(
            log_fd, f"Epoch {epoch} Val Loss: {val_loss} Learning Rate: {lr_scheduler.get_last_lr()[0]}")

        if val_loss < minLoss:
            minLoss = val_loss
            minLossEpoch = epoch
            torch.save({
                'epoch': epoch,
                'loss': val_loss,
                'img_encoder': img_encoder.state_dict(),
                # 'optimizer': optimizer.state_dict(),
                # 'lr_scheduler': lr_scheduler.state_dict(),
            }, bestSavePath)
            print_log(log_fd, f"Epoch {epoch} Best Model Saved")

        torch.save({
            'epoch': epoch,
            'loss': val_loss,
            'img_encoder': img_encoder.state_dict(),
            # 'optimizer': optimizer.state_dict(),
            # 'lr_scheduler': lr_scheduler.state_dict(),
        }, lastSavePath)

        print_log(log_fd, "Last Model saved (best loss {:.4f} at epoch {})" .format(
            minLoss, minLossEpoch))


def test(models, testLoader, args):
    _, _, _, _, _, _, exp_path, log_fd = prepare_logger(
        args.log_dir, args.exp)
    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    generator, img_encoder, encoder, decoder = models
    generator.to(device)
    img_encoder.to(device)
    encoder.to(device)
    decoder.to(device)

    # Lossess
    chamfer = ChamferDistanceL1().to(device)

    # Optimizers & LR schedulers
    optimizer = torch.optim.Adam(
        img_encoder.parameters(), lr=args.lr, betas=(0.5, 0.999))

    encoder, decoder = load_pcn_model(
        encoder, decoder, args.pcn)
    generator = load_generator(generator, args.generator)

    # Lossess
    chamfer = ChamferDistanceL1().to(device)

    if args.modelPath:
        print_log(log_fd, f"Loading checkpoint from {args.modelPath}")
        checkpoint = torch.load(args.modelPath)
        img_encoder.load_state_dict(checkpoint['img_encoder'])
        print_log(
            log_fd, f"Checkpoint loaded (epoch {checkpoint['epoch']}, loss {checkpoint['loss']})")

    test_loss = 0.0
    img_encoder.eval()
    # encoder.eval()
    decoder.eval()
    count = 0
    with torch.no_grad():
        loader = tqdm(enumerate(testLoader), total=len(testLoader))
        for i, batch in loader:
            loader.set_description(f"Loss: {(test_loss/(i+1)):.4f}")
            # Set model input
            taxonomy_id, model_id, (A, B, C) = batch
            img1 = A.to(torch.float32).to(device)
            inp = B.to(torch.float32).to(device)
            gt = C.to(torch.float32).to(device)

            optimizer.zero_grad()

            sampled_z = torch.randn(
                img1.size(0), args.gen_latent_dim).to(device)
            img2 = generator(img1, sampled_z)
            img2 = torch.max(img2, torch.zeros_like(img2))
            img2 = torch.min(img2, torch.ones_like(img2))

            rep = img_encoder(img1, img2)
            # base_rep = encoder(inp)

            coarse, fine = decoder(rep)

            # loss1 = chamfer(coarse, gt)
            loss2 = chamfer(fine, gt)
            chamfer_loss = loss2
            # mse_loss = MSE(base_rep, rep) * args.lambda_latent
            loss = chamfer_loss
            test_loss += loss.item() * 1000

            if args.testSave:
                index = random.randint(0, inp.shape[0] - 1)
                # Save Image
                plot_2_image_output_gt(os.path.join(exp_path, f'train_{count}.png'), A[index].detach().cpu().transpose(1, 0).transpose(1, 2).numpy(), img2[index].detach().cpu().transpose(1, 0).transpose(1, 2).numpy(), fine[index].detach().cpu().numpy(), gt[index].detach(
                ).cpu().numpy(), img1_title='Input Image', img2_title=f'{args.b_tag} Image', output_title='Dense Output PCD', gt_title='Ground Truth PCD', suptitle='', pcd_size=0.5, cmap='Reds', zdir='y')
                # plot_pcd_one_view(os.path.join(exp_path, f'test_{count}.png'),
                #                   [inp[index].detach().cpu().numpy(), coarse[index].detach().cpu().numpy(
                #                   ), fine[index].detach().cpu().numpy(), gt[index].detach().cpu().numpy()],
                #                   ['Input', 'Coarse', 'Dense', 'Ground Truth'], xlim=(-0.5, 1), ylim=(-0.5, 1), zlim=(-0.5, 1))
                count += 1
    test_loss /= len(testLoader)
    print_log(log_fd, f"Test Loss: {test_loss}")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder", type=str, default="ShapeNetRender", help="Folder containing the data")
    parser.add_argument("--json", type=str, default="final.json",
                        help="JSON file containing the data")
    parser.add_argument("--b_tag", type=str, default="depth",
                        help="Tag for the B Image")
    parser.add_argument("--log_dir", type=str, default="logs", help="Log dir")
    parser.add_argument("--exp", type=str, default="exp", help="Experiment")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--size", type=int, default=256, help="Image size")
    parser.add_argument("--latent_dim", type=int,
                        default=1024, help="Latent dimension")
    parser.add_argument("--gen_latent_dim", type=int,
                        default=8, help="Generator Latent dimension")
    parser.add_argument("--epoch", type=int, default=0, help="Epoch to start")
    parser.add_argument("--scheduler", type=str,
                        default="step", help="Scheduler")
    parser.add_argument("--gamma", type=float, default=0.85, help="Gamma")
    parser.add_argument("--n_epochs", type=int, default=30,
                        help="Number of epochs")
    parser.add_argument("--decay_epoch", type=int,
                        default=10, help="Decay epoch")
    parser.add_argument("--save_iter", type=int,
                        default=1000, help="Save interval")
    parser.add_argument("--lr", type=float, default=0.0002,
                        help="Learning rate")
    parser.add_argument("--gpu", type=int, default=0, help="GPU to use")
    parser.add_argument("--modelPath", type=str, help="Path to model")
    parser.add_argument("--pcn", type=str, default='weights/pcn_base/model.pth',
                        help="Path to PCN model")
    parser.add_argument("--generator", type=str,
                        default='weights/bicycle1000/model.pth')
    parser.add_argument("--base", type=str, default="weights/base/model.pth",
                        help="Base Image Encoder model")
    parser.add_argument("--test", action="store_true", help="Test model")
    parser.add_argument("--testSave", action="store_true",
                        help="Save test output")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training")
    parser.add_argument("--lambda_coarse", type=float,
                        default=0.5, help="coarse loss weight")
    parser.add_argument("--lambda_chamfer", type=float,
                        default=2, help="chamfer loss weight")
    parser.add_argument("--lambda_pixel", type=float,
                        default=10, help="pixelwise loss weight")
    parser.add_argument("--lambda_latent", type=float,
                        default=0.5, help="latent loss weight")
    parser.add_argument("--lambda_kl", type=float,
                        default=0.01, help="kullback-leibler loss weight")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    trainLoader, testLoader, valLoader = dataLoaders(args)
    models = get_model(args)
    if args.test:
        test(models, testLoader, args)
    else:
        train(models, trainLoader, valLoader, args)
