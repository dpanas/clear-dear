import os, sys
import random
import argparse
import torch
import torch.utils.data

import time as tm
import numpy as np
import matplotlib.pyplot as plt

from torch import nn, optim
from torch.nn import functional as F
from torchvision.utils import save_image

import src.dear.utils as utils

from .model import *
from .sagan import *
from .config import *
from .causal import *
from .data import make_dataloader

global device
device = utils.get_device()

def main():

    global args
    args = get_config()
    args.command = 'python ' + ' '.join(sys.argv)
    args.data_dir = f'{args.data_root}/{args.dataset}'
    args.save_dir = f'{args.results_root}/{args.dataset}'
    
    global celoss
    celoss = torch.nn.BCEWithLogitsLoss()
    
    if 'pendulum' in args.dataset:
        raise Exception('Not available')
    elif args.dataset == 'MLRSNet':
        args.label_idx = [7,39,8]  
    elif args.dataset == 'xView-train_samples-test':
        args.label_idx = []
        args.label_file= None
    elif args.dataset.split('_')[0] == 'xView':
        args.label_idx = ['buildings','roads','cars']
        args.label_file = 'labels.csv'
    else:
        if args.labels == 'smile':
            args.label_idx = [31, 20, 19, 21, 23, 13]
        elif args.labels == 'age':
            args.label_idx = [39, 20, 28, 18, 13, 3]
        else:
            raise NotImplementedError("Not supported structure.")
    num_label = len(args.label_idx)

    pretr_mod = '_pretr' if args.pretrained else ''
    norm_mod = '_norm' if args.normalize else ''
    flips_mod = '_flips' if args.add_flips else ''
    save_dir = '{}/{}_{}_sup{}{}{}{}_res{}_batch{}_seed{}/'.format(
        args.save_dir, args.labels, args.prior, str(args.sup_type), pretr_mod, norm_mod, flips_mod,
        args.image_size, args.batch_size, args.seed
        )
        
    utils.make_folder(save_dir)
    if 'config.txt' in save_dir:
        print('config alredy there')
    else:
        utils.write_config_to_file(args, save_dir)


    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    train_loader, test_loader = make_dataloader(args)
    log_file_name = os.path.join(save_dir, 'log.txt')
    global log_file
    if args.start_epoch > 1:
        log_file = open(log_file_name, "at")
    else:
        log_file = open(log_file_name, "wt")

    if 'scm' in args.prior:
        A = torch.zeros((num_label, num_label))
        if args.labels == 'smile':
            A[0, 2:6] = 1
            A[1, 4] = 1
        elif args.labels  == 'bcr':  
            A[0, 1] = 1
            A[0, 2] = 1
            A[1, 2] = 1  
        elif args.labels == 'age':
            A[0, 2:6] = 1
            A[1, 2:4] = 1
        elif args.labels == 'pend':
            A[0, 2:4] = 1
            A[1, 2:4] = 1
    else:
        A = None

    print('Build models...')
    model = BGM(args.latent_dim, args.g_conv_dim, args.image_size,
                args.enc_dist, args.enc_arch, args.enc_fc_size, args.enc_noise_dim, args.dec_dist,
                args.prior, num_label, A)
    discriminator = BigJointDiscriminator(args.latent_dim, args.d_conv_dim, args.image_size,
                                          args.dis_fc_size)

    A_optimizer = None
    prior_optimizer = None
    if 'scm' in args.prior:
        enc_param = model.encoder.parameters()
        dec_param = list(model.decoder.parameters())
        prior_param = list(model.prior.parameters())
        A_optimizer = optim.Adam(prior_param[0:1], lr=args.lr_a)
        prior_optimizer = optim.Adam(prior_param[1:], lr=args.lr_p, betas=(args.beta1, args.beta2))
    else:
        enc_param = model.encoder.parameters()
        dec_param = model.decoder.parameters()
    encoder_optimizer = optim.Adam(enc_param, lr=args.lr_e, betas=(args.beta1, args.beta2))
    decoder_optimizer = optim.Adam(dec_param, lr=args.lr_g, betas=(args.beta1, args.beta2))
    D_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2))

    # Load model from checkpoint
    if args.start_epoch > 1:
        print(f'Resuming training from epoch {args.start_epoch}')
        ckpt_dir = args.ckpt_dir if args.ckpt_dir != '' else save_dir
        ckpt_ = f'{ckpt_dir}model{args.start_epoch}.sav'
        assert os.path.exists( ckpt_)
        checkpoint = torch.load( ckpt_)
        model.load_state_dict(checkpoint['model'])
        discriminator.load_state_dict(checkpoint['discriminator'])
        del checkpoint

    model = nn.DataParallel(model.to(device))
    discriminator = nn.DataParallel(discriminator.to(device))

    # Fixed noise from prior p_z for generating from G
    global fixed_noise, fixed_unif_noise, fixed_zeros
    if args.prior == 'uniform':
        fixed_noise = torch.rand(args.save_n_samples, args.latent_dim, device=device) * 2 - 1
    else:
        fixed_noise = torch.randn(args.save_n_samples, args.latent_dim, device=device)
    fixed_unif_noise = torch.rand(1, args.latent_dim, device=device) * 2 - 1
    fixed_zeros = torch.zeros(1, args.latent_dim, device=device)

    # Train
    print('Start training...')
    start = tm.time()
    for i in range(args.start_epoch, args.start_epoch + args.n_epochs):
        train(i, model, discriminator, encoder_optimizer, decoder_optimizer, D_optimizer, train_loader, args.label_idx,
                  args.print_every, save_dir, prior_optimizer, A_optimizer)
        if i % args.save_model_every == 0:
            torch.save({'model': model.module.state_dict(), 'discriminator': discriminator.module.state_dict()},
                       save_dir + 'model' + str(i) + '.sav')
    print(f'Taining took {(tm.time()-start)/60} minutes')
    torch.save(
      {'model': model.module.state_dict(), 'discriminator': discriminator.module.state_dict()},
       save_dir + 'model' + str(i) + '.sav'
       )
       

def train(epoch, model, discriminator, encoder_optimizer, decoder_optimizer, D_optimizer,
              train_loader, label_idx, print_every, save_dir,
              prior_optimizer, A_optimizer):
    model.train()
    discriminator.train()
    trav = A_optimizer is not None

    for batch_idx, (x, label, _) in enumerate(train_loader):
        start_ = tm.time()
        
        x = x.to(device)
        # supervision flag
        sup_flag = label[:, 0] != -1
        if sup_flag.sum() > 0:
            label = label[ sup_flag, :].float()
        num_labels = len(label_idx)
        label = label.to(device)

        # ================== TRAIN DISCRIMINATOR ================== #
        for _ in range(args.d_steps_per_iter):
            discriminator.zero_grad()

            # Sample z from prior p_z
            if args.prior == 'uniform':
                z = torch.rand(x.size(0), args.latent_dim, device=x.device) * 2 - 1
            else:
                z = torch.randn(x.size(0), args.latent_dim, device=x.device)

            # Get inferred latent z = E(x) and generated image x = G(z)
            if 'scm' in args.prior:
                z_fake, x_fake, z, _ = model(x, z)
            else:
                z_fake, x_fake, _ = model(x, z)

            # Compute D loss
            encoder_score = discriminator(x, z_fake.detach())
            decoder_score = discriminator(x_fake.detach(), z.detach())
            del z_fake
            del x_fake

            loss_d = F.softplus(decoder_score).mean() + F.softplus(-encoder_score).mean()
            loss_d.backward()
            D_optimizer.step()

        for _ in range(args.g_steps_per_iter):
            if args.prior == 'uniform':
                z = torch.rand(x.size(0), args.latent_dim, device=x.device) * 2 - 1
            else:
                z = torch.randn(x.size(0), args.latent_dim, device=x.device)
            if 'scm' in args.prior:
                z_fake, x_fake, z, z_fake_mean = model(x, z)
            else:
                z_fake, x_fake, z_fake_mean = model(x, z)

            # ================== TRAIN ENCODER ================== #
            model.zero_grad()
            # WITH THE GENERATIVE LOSS
            encoder_score = discriminator(x, z_fake)
            loss_encoder = encoder_score.mean()

            # WITH THE SUPERVISED LOSS
            if sup_flag.sum() > 0:
                # subset only the supervised portion!!
                label_z = z_fake_mean[sup_flag, :num_labels]
                if 'pendulum' in args.dataset:
                    if args.sup_type == 'ce':
                        # CE loss
                        sup_loss = celoss(label_z, label)
                    else:
                        # l2 loss
                        sup_loss = nn.MSELoss()(label_z, label)
                else:
                    sup_loss = celoss(label_z, label)
            else:
                sup_loss = torch.zeros([1], device=device)
            loss_encoder = loss_encoder + sup_loss * args.sup_coef

            loss_encoder.backward()
            encoder_optimizer.step()
            if 'scm' in args.prior:
                prior_optimizer.step()

            # ================== TRAIN GENERATOR ================== #
            model.zero_grad()

            decoder_score = discriminator(x_fake, z)
            # with scaling clipping for stabilization
            r_decoder = torch.exp(decoder_score.detach())
            s_decoder = r_decoder.clamp(0.5, 2)
            loss_decoder = -(s_decoder * decoder_score).mean()

            loss_decoder.backward()
            decoder_optimizer.step()
            if 'scm' in args.prior:
                model.module.prior.set_zero_grad()
                A_optimizer.step()
                prior_optimizer.step()

        # Print out losses
        if batch_idx == 0 or (batch_idx + 1) % print_every == 0:
            log = ('Train Epoch: {} ({:.0f}%)\tD loss: {:.4f}, Encoder loss: {:.4f}, Decoder loss: {:.4f}, Sup loss: {:.4f}, '
                   'E_score: {:.4f}, D score: {:.4f}'.format(
                epoch, 100. * batch_idx / len(train_loader),
                loss_d.item(), loss_encoder.item(), loss_decoder.item(), sup_loss.item(),
                encoder_score.mean().item(), decoder_score.mean().item()))
            print(f'{(tm.time() - start_)/60} | {log}')
            log_file.write(log + '\n')
            log_file.flush()

        if (epoch == 1 or epoch % args.sample_every_epoch == 0) and batch_idx == len(train_loader) - 1:
            test(epoch, batch_idx + 1, model, x[:args.save_n_recons], save_dir, trav)


def test(epoch, i, model, test_data, save_dir, trav= True):
    model.eval()
    with torch.no_grad():
        x = test_data.to(device)

        # Reconstruction
        x_recon = model(x, recon=True)
        recons = utils.draw_recon(x.cpu(), x_recon.cpu())
        del x_recon
        save_image(recons, save_dir + 'recon_' + str(epoch) + '_' + str(i) + '.png', nrow=args.nrow,
                   normalize=True, scale_each=True)

        # Generation
        sample = model(z=fixed_noise).cpu()
        save_image(sample, save_dir + 'gen_' + str(epoch) + '_' + str(i) + '.png', normalize=True, scale_each=True)

        if trav:
            # Traversal (given a fixed traversal range)
            sample = model.module.traverse(fixed_zeros).cpu()
            save_image(sample, save_dir + 'trav_' + str(epoch) + '_' + str(i) + '.png', normalize=True, scale_each=True, nrow=10)
            del sample

    model.train()

