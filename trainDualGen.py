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
from models.dual_image_generator import DualImageGenerator
from models.bicyclegan import Generator
from tqdm import tqdm
import time
from PIL import Image
import cv2
import pandas as pd
import itertools
import datetime
import random
from tensorboardX import SummaryWriter
import visdom
from datasets import PCNImageDataset
from utils.utils import LambdaLR, plot_2_image_output_gt, prepare_logger, print_log, count_parameters, plot_batch_2_image_output_gt
from extensions.chamfer_dist import ChamferDistanceL1


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
        0, 1, (mu.size(0), args.latent_dim)))).to(mu.device)
    z = sampled_z * std + mu
    return z


def get_model(args):
    input_shape = (3, args.size, args.size)
    generator = Generator(args.gen_latent_dim, input_shape)
    img_generator = DualImageGenerator(latent_dim=1024)
    encoder = PCNEncoder(latent_dim=1024)
    decoder = PCNDecoder(latent_dim=1024, num_dense=16384)
    print(
        f"Generator Parameters: {round(count_parameters(generator)/ 1e6, 4)}")
    print(
        f"Dual Image Generator Parameters: {round(count_parameters(img_generator)/ 1e6, 4)}")
    print(f"Encoder Parameters: {round(count_parameters(encoder)/ 1e6, 4)}")
    print(f"Decoder Parameters: {round(count_parameters(decoder)/ 1e6, 4)}")
    return generator, img_generator, encoder, decoder


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


def load_base(img_generator, path):
    if path == "":
        return img_generator
    # print(f"Loading Base Image Encoder model from {path}")
    # checkpoint = torch.load(path)
    # img_generator.first_encoder.load_state_dict(checkpoint['img_encoder'])

    # print(f"Base Image Encoder Model loaded")
    return img_generator


def train(models, trainLoader, valLoader, args):
    print("[+] Training the model...")
    ckpt_dir, epochs_dir, log_fd, train_writer, val_writer, exp_path, _, _ = prepare_logger(
        args.log_dir, args.exp)
    bestSavePath = os.path.join(ckpt_dir, "bestModel.pth")
    lastSavePath = os.path.join(ckpt_dir, "lastModel.pth")
    print_log(log_fd, str(args))
    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    generator, img_generator, encoder, decoder = models
    generator.to(device)
    img_generator.to(device)
    encoder.to(device)
    decoder.to(device)

    # Lossess
    chamfer = ChamferDistanceL1().to(device)
    MSE = nn.MSELoss().to(device)

    # Optimizers & LR schedulers
    optimizer = torch.optim.Adam(
        img_generator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    lr_scheduler = get_scheduler(optimizer, args)

    encoder, decoder = load_pcn_model(
        encoder, decoder, args.pcn)
    generator = load_generator(generator, args.generator)
    img_generator = load_base(img_generator, args.base)

    train_step = 0
    minLoss = 1e10
    minLossEpoch = 0

    if args.resume:
        print_log(log_fd, f"Loading checkpoint from {args.modelPath}")
        checkpoint = torch.load(args.modelPath)
        img_generator.load_state_dict(checkpoint['img_generator'])
        args.epoch = checkpoint['epoch'] + 1
        train_step = checkpoint['epoch'] * len(trainLoader)
        # minLoss = checkpoint['loss']
        # minLossEpoch = args.epoch
        lr_scheduler = get_scheduler(optimizer, args)
        print_log(
            log_fd, f"Checkpoint loaded (epoch {checkpoint['epoch']}, loss {checkpoint['loss']})")

    ###### Training ######
    for epoch in range(args.epoch, args.n_epochs):
        img_generator.train()
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

            mean, logvar = img_generator(img1, img2)
            rep = reparameterization(mean, logvar, torch.FloatTensor, args)
            base_rep = encoder(inp)

            coarse, fine = decoder(rep)

            loss1 = chamfer(coarse, gt)
            loss2 = chamfer(fine, gt)
            chamfer_loss = loss1 * args.lambda_coarse
            chamfer_loss += loss2 * (1 - args.lambda_coarse)
            chamfer_loss = chamfer_loss * args.lambda_chamfer

            mse_loss = MSE(base_rep, rep) * args.lambda_latent

            kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
            kl_loss = kl_loss * args.lambda_kl * ((train_step//2.0) + 1)

            loss = chamfer_loss + mse_loss + kl_loss
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * 1000
            train_step += 1

            train_writer.add_scalar('loss', loss.item(), train_step)
            train_writer.add_scalar(
                'chamfer_loss', chamfer_loss.item(), train_step)
            train_writer.add_scalar('mse_loss', mse_loss.item(), train_step)
            train_writer.add_scalar('kl_loss', kl_loss.item(), train_step)

            if train_step % args.save_iter == 0:
                index = random.randint(0, inp.shape[0] - 1)
                plot_batch_2_image_output_gt(os.path.join(exp_path, f'train_{train_step}.png'), A.detach().cpu().numpy(), img2.detach().cpu().numpy(), fine.detach().cpu(
                ).numpy(), gt.detach().cpu().numpy(), img1_title='Input Image', img2_title=f'{args.b_tag} Image', output_title='Dense Output PCD', gt_title='Ground Truth PCD', suptitle='', pcd_size=0.5, cmap='Reds', zdir='y')
                # plot_pcd_one_view(os.path.join(exp_path, f'train_{train_step}.png'),
                #                   [inp[index].detach().cpu().numpy(), coarse[index].detach().cpu().numpy(
                #                   ), fine[index].detach().cpu().numpy(), gt[index].detach().cpu().numpy()],
                #                   ['Input', 'Coarse', 'Dense', 'Ground Truth'], xlim=(-0.5, 1), ylim=(-0.5, 1), zlim=(-0.5, 1))

        # Update learning rates
        lr_scheduler.step()

        train_loss /= len(trainLoader)
        print_log(log_fd, f"Epoch {epoch} Train Loss: {train_loss}")

        ###### Validation ######
        img_generator.eval()
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

                mean, logvar = img_generator(img1, img2)
                rep = reparameterization(mean, logvar, torch.FloatTensor, args)
                base_rep = encoder(inp)

                coarse, fine = decoder(rep)

                loss1 = chamfer(coarse, gt)
                loss2 = chamfer(fine, gt)
                chamfer_loss = loss1 * args.lambda_coarse
                chamfer_loss += loss2 * (1 - args.lambda_coarse)
                chamfer_loss = chamfer_loss * args.lambda_chamfer

                mse_loss = MSE(base_rep, rep) * args.lambda_latent

                kl_loss = -0.5 * \
                    torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
                kl_loss = kl_loss * args.lambda_kl

                loss = chamfer_loss + mse_loss + kl_loss

                val_writer.add_scalar('loss', loss.item(), i)
                val_writer.add_scalar(
                    'chamfer_loss', chamfer_loss.item(), i)
                val_writer.add_scalar('mse_loss', mse_loss.item(), i)
                val_writer.add_scalar('kl_loss', kl_loss.item(), i)
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
                'img_generator': img_generator.state_dict(),
                # 'optimizer': optimizer.state_dict(),
                # 'lr_scheduler': lr_scheduler.state_dict(),
            }, bestSavePath)
            print_log(log_fd, f"Epoch {epoch} Best Model Saved")

        torch.save({
            'epoch': epoch,
            'loss': val_loss,
            'img_generator': img_generator.state_dict(),
            # 'optimizer': optimizer.state_dict(),
            # 'lr_scheduler': lr_scheduler.state_dict(),
        }, lastSavePath)

        print_log(log_fd, "Last Model saved (best loss {:.4f} at epoch {})" .format(
            minLoss, minLossEpoch))


def test(models, testLoader, args):
    _, _, _, _, _, _, exp_path, log_fd = prepare_logger(
        args.log_dir, args.exp)
    print_log(log_fd, str(args))
    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    generator, img_generator, encoder, decoder = models
    generator.to(device)
    img_generator.to(device)
    encoder.to(device)
    decoder.to(device)

    # Lossess
    chamfer = ChamferDistanceL1().to(device)

    # Optimizers & LR schedulers
    optimizer = torch.optim.Adam(
        img_generator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    encoder, decoder = load_pcn_model(
        encoder, decoder, args.pcn)
    generator = load_generator(generator, args.generator)

    # Lossess
    chamfer = ChamferDistanceL1().to(device)

    if args.modelPath:
        print_log(log_fd, f"Loading checkpoint from {args.modelPath}")
        checkpoint = torch.load(args.modelPath)
        img_generator.load_state_dict(checkpoint['img_generator'])
        print_log(
            log_fd, f"Checkpoint loaded (epoch {checkpoint['epoch']}, loss {checkpoint['loss']})")

    # torch.manual_seed(21)
    test_loss = 0.0
    img_generator.eval()
    # encoder.eval()
    decoder.eval()
    count = 0
    key_loss = {}
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
            # sampled_z = torch.zeros_like(sampled_z)
            img2 = generator(img1, sampled_z)
            img2 = torch.max(img2, torch.zeros_like(img2))
            img2 = torch.min(img2, torch.ones_like(img2))

            mean, logvar = img_generator(img1, img2)
            rep = reparameterization(mean, logvar, torch.FloatTensor, args)
            # base_rep = encoder(inp)

            coarse, fine = decoder(rep)

            # loss1 = chamfer(coarse, gt)
            loss2 = chamfer(fine, gt)
            chamfer_loss = loss2
            # mse_loss = MSE(base_rep, rep) * args.lambda_latent
            loss = chamfer_loss
            test_loss += loss.item() * 1000
            for i in range(img1.shape[0]):
                curr_loss = chamfer(fine[i].unsqueeze(0), gt[i].unsqueeze(0))
                if taxonomy_id[i] not in key_loss:
                    key_loss[taxonomy_id[i]] = [curr_loss.item() * 1000]
                else:
                    key_loss[taxonomy_id[i]].append((curr_loss.item() * 1000))

            if args.testSave:
                index = 0
                # Save Image
                plot_2_image_output_gt(os.path.join(exp_path, f'test_{count}.png'), A[index].detach().cpu().transpose(1, 0).transpose(1, 2).numpy(), img2[index].detach().cpu().transpose(1, 0).transpose(1, 2).numpy(), fine[index].detach().cpu().numpy(), gt[index].detach(
                ).cpu().numpy(), img1_title='Input Image', img2_title=f'{args.b_tag} Image', output_title='Dense Output PCD', gt_title='Ground Truth PCD', suptitle='', pcd_size=0.5, cmap='Reds', zdir='y')
                # plot_pcd_one_view(os.path.join(exp_path, f'test_{count}.png'),
                #                   [inp[index].detach().cpu().numpy(), coarse[index].detach().cpu().numpy(
                #                   ), fine[index].detach().cpu().numpy(), gt[index].detach().cpu().numpy()],
                #                   ['Input', 'Coarse', 'Dense', 'Ground Truth'], xlim=(-0.5, 1), ylim=(-0.5, 1), zlim=(-0.5, 1))
                count += 1
    test_loss /= len(testLoader)
    print_log(log_fd, f"Test Loss: {test_loss}")
    print_log(log_fd, "Taxonomy Losses")
    for key, value in key_loss.items():
        print_log(log_fd, f"{key}\t{sum(value)/len(value)}")
    # save dictionary as pandas dataframe
    df = pd.DataFrame.from_dict({key: round(sum(value)/len(value), 4)
                                for key, value in key_loss.items()}, orient='index')
    # sort rows based on first column
    df = df.sort_values(by=[0], ascending=False)
    df.to_csv(os.path.join(exp_path, 'test.csv'))
    for key, value in key_loss.items():
        print(round(sum(value)/len(value), 3))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder", type=str, default="../ShapeNet", help="Folder containing the data")
    parser.add_argument("--json", type=str, default="final.json",
                        help="JSON file containing the data")
    parser.add_argument("--b_tag", type=str, default="depth",
                        help="Tag for the B Image")
    parser.add_argument("--log_dir", type=str, default="logs", help="Log dir")
    parser.add_argument("--exp", type=str,
                        default="dualGen", help="Experiment")
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
    # set torch seed
    torch.manual_seed(time.time())
    trainLoader, testLoader, valLoader = dataLoaders(args)
    models = get_model(args)
    if args.test:
        test(models, testLoader, args)
    else:
        train(models, trainLoader, valLoader, args)
