import torch.optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils import data_processing as dp
from utils import data_analysis as da
import time
from os.path import join
import os
from shutil import copyfile
import argparse
import csv
import phantom_generation as pg
import json
from unet import model

project_dir = os.getcwd()


def write_csv(epoch, time, tloss, vloss, vloss_norm, LR, max_reserved_gpu_mem):
    headers = ["Epoch", "Time",  "train loss", "val loss", "vloss_norm", "LR", "Max. reserved GPU memory (GB)"]
    log_row = (epoch + 1,
                 "{:.2f}".format(time), "{:.2e}".format(tloss),
                 "{:.2e}".format(vloss), "{:.2e}".format(vloss_norm), LR,
                     f"{max_reserved_gpu_mem:.1f}")
    with open(f'{save_dir}/log.csv', 'a') as metrics_file:
        f_csv = csv.writer(metrics_file)
        if epoch == 0:
            f_csv.writerow(headers)
        f_csv.writerow(log_row)


def iterate_minibatch(recon_basis, gt_basis, Nop, basis, dcomp, mps, ktraj, k_proj, p_gt, k_scale, recon_scale, batch_size, shuffle=True):
    # only batch_size = 1 is supported
    assert batch_size == 1
    n = len(recon_basis)
    index = np.arange(n)
    if shuffle:
        index = np.random.permutation(index)
        recon_basis = [recon_basis[i] for i in index]
        gt_basis = [gt_basis[i] for i in index]
        Nop = [Nop[i] for i in index]
        basis = [basis[i] for i in index]
        dcomp = [dcomp[i] for i in index]
        mps = [mps[i] for i in index]
        ktraj = [ktraj[i] for i in index]
        k_proj = [k_proj[i] for i in index]
        p_gt = [p_gt[i] for i in index]
        k_scale = [k_scale[i] for i in index]
        recon_scale = [recon_scale[i] for i in index]
    for i in range(0, n, batch_size):
        yield recon_basis[i], gt_basis[i], Nop[i], basis[i], dcomp[i], mps[i], \
              ktraj[i], k_proj[i], p_gt[i], k_scale[i], recon_scale[i]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', metavar='int', nargs=1, default=['0'],
                            help='Choose which gpu to train the network')
    parser.add_argument('--dc_step_size', metavar='float', nargs=1,
                        default=['0'], help='The gradient descent step size for data consistency update')
    parser.add_argument('--p_loss', action='store_true', help='Whether to add parameter loss into the total loss function')
    parser.add_argument('--beta', metavar='float', nargs=1,
                        default=['0.1'], help='The fraction of the parameter loss in the total loss')
    parser.add_argument('--weight_decay', metavar='float', nargs=1,
                        default=['1e-5'], help='The weight on the l2 regularization on network parameters')
    parser.add_argument('--early_stop_patience', metavar='float', nargs=1,
                        default=['20'], help='The number of epochs to wait before early stop')
    parser.add_argument('--rotation_angle', action='store_true', help='Whether to optimization the rotation angle of each spiral')
    parser.add_argument('--debug', action='store_true', help='Whether to enable debug mode (only one phantom will be used for training)')
    parser.add_argument('--n_unroll', metavar='int', nargs=1, default=['3'], help='The number of algorithm unrolls in the network')
    parser.add_argument('--R', metavar='int', nargs=1, default=['6'], help='Undersampling factor')
    parser.add_argument('--full_dcomp', action='store_true', help='Whether to use density compensation function calculated from a "fully sampled" trajectory')
    parser.add_argument('--pretrain_FCNN', action='store_true', help='Whether to use pretrained full connected neural network (FCNN) for parameter estimation in joint subspace reconstrution and parameter estimation')
    parser.add_argument('--pretrain_CNN', metavar='str', nargs=1, default=None, help='Whether to use pretrained CNN')
    parser.add_argument('--share_weight', action='store_true', help='Whether to share the weights of CNNs to prevent overfitting')
    parser.add_argument('--max_n_epochs', type=int, default=500, help='The number of maximum interation for training')
    parser.add_argument('--numpoints', type=int, default=2, help='The number of neighboring point in each dimension for interpolation in NUFFT')
    parser.add_argument('--grid_size', type=float, default=1.25, help='The oversampling factor for FFT in NUFFT')
    parser.add_argument('--dataset', type=str, default='kirby21', help='Choose which dataset to train on')
    parser.add_argument('--train_mag', action='store_true', help='Whether to train CNN with magnitude images and add phase back before DC layer')
    parser.add_argument('--nufft_split_batch', action='store_true', help='Whether to apply NUFFT to coefficient images sequentially')
    parser.add_argument('--nufft_split_channel', action='store_true', help='Whether to apply NUFFT to channels sequentially')
    parser.add_argument('--ReduceLROnPlateau_patience', metavar='float', nargs=1,
                        default=['10'], help='The number of epochs to wait before reducing learning rate')
    parser.add_argument('--checkpoint', type=str, default=None, help='resume training from this checkpoint')
    parser.add_argument('--no_amp', action='store_true', help='Do not use automatic mixed precision')

    args = parser.parse_args()

    start_run_at = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    if args.checkpoint is not None:
        checkpoint = args.checkpoint
        print(f"Resume training from checkpoint {checkpoint}")
        save_dir = join(project_dir, 'model/%s' % checkpoint)
        assert os.path.isdir(save_dir)
        with open(save_dir + "/arguments.txt", 'r') as f:
            args.__dict__ = json.load(f)
        args.checkpoint = checkpoint
    else:
        save_dir = join(project_dir, 'model/%s' % start_run_at)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
        copyfile(project_dir + '/train_CNN.py', save_dir + '/train_CNN.py')
        copyfile(project_dir + '/unet/model.py', save_dir + '/model.py')
        copyfile(project_dir + '/unet/unet_parts.py', save_dir + '/unet_parts.py')
        copyfile(project_dir + '/phantom_generation.py', save_dir + '/phantom_generation.py')
        with open(save_dir + "/arguments.txt", "w") as f:
            json.dump(args.__dict__, f, indent=2)
    gpu = args.gpu[0]
    n_unroll = int(args.n_unroll[0])
    R = int(args.R[0])
    dc_step_size = float(args.dc_step_size[0])
    weight_decay = float(args.weight_decay[0])
    device = torch.device("cuda:" + gpu)
    n_basis = 5
    p_loss = args.p_loss
    rotation_angle = args.rotation_angle
    full_dcomp = args.full_dcomp
    debug = args.debug
    pretrain_FCNN = args.pretrain_FCNN
    share_weight = args.share_weight
    early_stop_patience = int(args.early_stop_patience[0])
    ReduceLROnPlateau_patience = int(args.ReduceLROnPlateau_patience[0])
    N_epochs = args.max_n_epochs
    if debug:
        matplotlib.use('TkAgg')
    if rotation_angle:
        scaling_factor = None
    else:
        scaling_factor = None
    if p_loss:
        beta = float(args.beta[0])
    if args.train_mag:
        n_ch = 1
    else:
        n_ch = 2
    cnn = model.UnrollUNet2D(in_channels=n_basis * n_ch, out_channels=n_basis * n_ch, n_unroll=n_unroll, checkpointing=True,
                             est_p=p_loss, share_weight=share_weight, rotation_angle=rotation_angle,
                             dc_step_size=dc_step_size, pretrain_FCNN=pretrain_FCNN, train_mag=args.train_mag, R=R)
    cnn.to(device)
    if pretrain_FCNN:
        optimizer = torch.optim.Adam([v for k, v in cnn.named_parameters() if k not in cnn.fnn.named_parameters()],
                                     lr=1e-4, weight_decay=1e-5)
    else:
        optimizer = torch.optim.Adam(cnn.parameters(), lr=1e-4, weight_decay=weight_decay)

    if args.checkpoint is not None:
        checkpoint = torch.load(save_dir + "/checkpoint_epoch_model_optimizer.pt", map_location='cpu')
        cnn.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        resume_epoch = checkpoint['epoch']
    else:
        resume_epoch = -1
    scaler = torch.cuda.amp.GradScaler()

    dataset = args.dataset
    if dataset[:7] == "kirby21":
        train_subject_id_debug = ["01"]
        train_subject_id = ["01", "02", "05", "07", "08"]
        val_subject_id = ["15", "16"]
        test_subject_id = ["09", '18', '28', "30", "32"]
    elif dataset == "deli_cs":
        train_subject_id_debug = ["training00"]
        train_subject_id = ["training00", "training01", "training02", "training03", "training04",
                            "training05"]
        val_subject_id = ["validation00", "validation01"]
        test_subject_id = ["testing00", "testing01", 'training06', 'training07', 'training08', 'training09']
    if debug:
        train_subject_id = train_subject_id_debug
    recon_basis, gt_basis, Nop, basis, dcomp, mps, ktraj, \
        k_proj, p_gt, k_scale, recon_scale = pg.load_dataset(dataset,
                                                             train_subject_id,
                                                             device=device,
                                                             scaling_factor=scaling_factor,
                                                             R=R,
                                                             full_dcomp=full_dcomp,
                                                             single_coil=False,
                                                             nufft_numpoints=args.numpoints,
                                                             nufft_grid_size=args.grid_size,
                                                             nufft_split_batch=args.nufft_split_batch,
                                                             nufft_split_channel=args.nufft_split_channel)
    recon_basis_val, gt_basis_val, Nop_val, basis_val, dcomp_val, mps_val, ktraj_val, \
    k_proj_val, p_gt_val, k_scale_val, recon_scale_val = pg.load_dataset(dataset, val_subject_id,
                                                                         device=device,
                                                                         scaling_factor=scaling_factor,
                                                                         R=R,
                                                                         full_dcomp=full_dcomp,
                                                                         single_coil=False,
                                                                         nufft_numpoints=args.numpoints,
                                                                         nufft_grid_size=args.grid_size,
                                                                         nufft_split_batch=args.nufft_split_batch,
                                                                         nufft_split_channel=args.nufft_split_channel)
    recon_basis_test, gt_basis_test, Nop_test, basis_test, dcomp_test, mps_test, ktraj_test, \
    k_proj_test, p_gt_test, k_scale_test, recon_scale_test = pg.load_dataset(dataset, test_subject_id,
                                                                         device=device,
                                                                         scaling_factor=scaling_factor,
                                                                         R=R,
                                                                         full_dcomp=full_dcomp,
                                                                         single_coil=False,
                                                                         nufft_numpoints=args.numpoints,
                                                                         nufft_grid_size=args.grid_size,
                                                                         nufft_split_batch=args.nufft_split_batch,
                                                                         nufft_split_channel=args.nufft_split_channel)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
                                                           patience=ReduceLROnPlateau_patience, factor=0.5)
    t_start = time.time()
    nll_mean = []
    nll_val = []
    nll_test = []
    loss_normalized_val = []

    print('Epoch \tTime \ttrain \t\tval \t\ttest \t\tLR  \t\tDC_step_size')
    nll_mean_val_min = 1e10
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(resume_epoch + 1, N_epochs):
        for x0, x, Nop_i, basis_i, dcomp_i, mps_i, ktraj_i, k_proj_i, p_gt_i, k_scale_i, recon_scale_i \
                in iterate_minibatch(recon_basis, gt_basis, Nop, basis, dcomp, mps, ktraj, k_proj, p_gt, k_scale, recon_scale, batch_size=1, shuffle=True):
            x0, x = x0.unsqueeze(0).to(device), x.unsqueeze(0).to(device)
            if dc_step_size == 1:
                if isinstance(Nop_i, list):
                    for i in range(len(Nop_i)):
                        Nop_i[i].to(device)
                else:
                    Nop_i.to(device)
                dcomp_i = dcomp_i.to(device)
                ktraj_i = ktraj_i.to(device)
                k_proj_i = k_proj_i.to(device)
            else:
                Nop_i = None
                dcomp_i = None
                ktraj_i = None
                k_proj_i = None
            mps_i = mps_i.to(device)
            basis_i = basis_i.to(device)
            p_gt_i = p_gt_i.to(device)
            k_scale_i = k_scale_i.to(device)
            recon_scale_i = recon_scale_i.to(device)
            torch.cuda.empty_cache()
            weight = 1 / torch.sum(torch.abs(dp.spatial_normalize_basis_coeff(x, basis_i)) ** 2,
                                   axis=(2, 3, 4)) ** 0.5 * (230 ** 3) ** 0.5  # (1, n_basis)
            if rotation_angle:
                x0 = x
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=not args.no_amp):
                if p_loss:
                    max_T1_T2 = torch.tensor([4000, 1500]).to(p_gt_i)
                    p_gt_i = p_gt_i / max_T1_T2.view(2, 1, 1, 1)
                    x_hat, p_hat = cnn(x0, Nop_i, basis_i, dcomp_i, mps_i, ktraj_i, k_proj_i, k_scale_i, recon_scale_i)
                    spatial_basis_loss = torch.nn.MSELoss()(torch.view_as_real(dp.spatial_normalize_basis_coeff(x, basis_i, weight=weight)),
                                             torch.view_as_real(dp.spatial_normalize_basis_coeff(x_hat, basis_i, weight=weight)))
                    if dataset == 'deli_cs':
                        p_hat = torch.clamp(p_hat, min=0, max=3)
                    else:
                        p_hat = torch.clamp(p_hat, min=0, max=1)
                    nll = (1 - beta) * spatial_basis_loss + \
                          beta * torch.mean(torch.abs(p_gt_i[p_gt_i > 0] - p_hat[p_gt_i > 0]) / torch.abs(p_gt_i[p_gt_i > 0]))
                else:
                    x_hat, basis_out = cnn(x0, Nop_i, basis_i, dcomp_i, mps_i, ktraj_i, k_proj_i, k_scale_i, recon_scale_i)
                    if dataset == 'deli_cs':
                        nll = torch.nn.MSELoss()(torch.view_as_real(dp.spatial_normalize_basis_coeff(x, basis_i, weight=weight)),
                                                 torch.view_as_real(dp.spatial_normalize_basis_coeff(x_hat * (abs(x) > 0), basis_out, weight=weight)))
                    else:
                        nll = torch.nn.MSELoss()(torch.view_as_real(dp.spatial_normalize_basis_coeff(x, basis_i, weight=weight)),
                                                 torch.view_as_real(dp.spatial_normalize_basis_coeff(x_hat, basis_out, weight=weight)))
            nll_mean.append(nll.item())
            if args.no_amp:
                nll.backward()
                optimizer.step()
            else:
                scaler.scale(nll).backward()
                scaler.step(optimizer)
                scaler.update()
            optimizer.zero_grad()
            if dc_step_size == 1:
                if isinstance(Nop_i, list):
                    for i in range(len(Nop_i)):
                        Nop_i[i].to(torch.device('cpu'))
                else:
                    Nop_i.to(torch.device('cpu'))
            del x0, x, Nop_i, basis_i, dcomp_i, mps_i, ktraj_i, k_proj_i, p_gt_i, k_scale_i, x_hat, nll
            torch.cuda.empty_cache()
        cnn.eval()
        with torch.no_grad():
            for x0, x, Nop_i, basis_i, dcomp_i, mps_i, ktraj_i, k_proj_i, p_gt_i, k_scale_i, recon_scale_i in \
                    zip(recon_basis_val, gt_basis_val, Nop_val, basis_val, dcomp_val, mps_val, ktraj_val, k_proj_val,
                        p_gt_val, k_scale_val, recon_scale_val):
                x0, x = x0.unsqueeze(0).to(device), x.unsqueeze(0).to(device)
                if dc_step_size == 1:
                    if isinstance(Nop_i, list):
                        for i in range(len(Nop_i)):
                            Nop_i[i].to(device)
                    else:
                        Nop_i.to(device)
                    dcomp_i = dcomp_i.to(device)
                    ktraj_i = ktraj_i.to(device)
                    k_proj_i = k_proj_i.to(device)
                else:
                    Nop_i = None
                    dcomp_i = None
                    ktraj_i = None
                    k_proj_i = None
                mps_i = mps_i.to(device)
                basis_i = basis_i.to(device)
                p_gt_i = p_gt_i.to(device)
                k_scale_i = k_scale_i.to(device)
                recon_scale_i = recon_scale_i.to(device)
                torch.cuda.empty_cache()
                weight = 1 / torch.sum(torch.abs(dp.spatial_normalize_basis_coeff(x, basis_i)) ** 2,
                                       axis=(2, 3, 4)) ** 0.5 * (230 ** 3) ** 0.5  # (1, n_basis)
                if rotation_angle:
                    x0 = x
                if p_loss:
                    x_hat, p_hat = cnn(x0, Nop_i, basis_i, dcomp_i, mps_i, ktraj_i, k_proj_i, k_scale_i, recon_scale_i)
                    p_gt_i = p_gt_i / max_T1_T2.view(2, 1, 1, 1)
                    spatial_basis_loss = torch.nn.MSELoss()(torch.view_as_real(dp.spatial_normalize_basis_coeff(x, basis_i, weight=weight)),
                                             torch.view_as_real(dp.spatial_normalize_basis_coeff(x_hat, basis_i, weight=weight)))
                    if dataset == 'deli_cs':
                        p_hat = torch.clamp(p_hat, min=0, max=3)
                    else:
                        p_hat = torch.clamp(p_hat, min=0, max=1)
                    nll = (1 - beta) * spatial_basis_loss + \
                          beta * torch.mean(torch.abs(p_gt_i[p_gt_i > 0] - p_hat[p_gt_i > 0]) / p_gt_i[p_gt_i > 0])
                    loss_normalized = torch.nn.MSELoss()(torch.view_as_real(dp.normalize_basis_coeff(x, basis_i)),
                                                         torch.view_as_real(dp.normalize_basis_coeff(x_hat, basis_i)))
                else:
                    x_hat, basis_out = cnn(x0, Nop_i, basis_i, dcomp_i, mps_i, ktraj_i, k_proj_i, k_scale_i, recon_scale_i)
                    if dataset == 'deli_cs':
                        nll = torch.nn.MSELoss()(torch.view_as_real(dp.spatial_normalize_basis_coeff(x, basis_i, weight=weight)),
                                                 torch.view_as_real(dp.spatial_normalize_basis_coeff(x_hat * (abs(x) > 0), basis_out, weight=weight)))
                    else:
                        nll = torch.nn.MSELoss()(torch.view_as_real(dp.spatial_normalize_basis_coeff(x, basis_i, weight=weight)),
                                                 torch.view_as_real(dp.spatial_normalize_basis_coeff(x_hat, basis_out, weight=weight)))
                    loss_normalized = torch.nn.MSELoss()(torch.view_as_real(dp.normalize_basis_coeff(x, basis_i)),
                                                         torch.view_as_real(dp.normalize_basis_coeff(x_hat, basis_out)))
                nll_val.append(nll.item())
                loss_normalized_val.append(loss_normalized.item())
                if dc_step_size == 1:
                    if isinstance(Nop_i, list):
                        for i in range(len(Nop_i)):
                            Nop_i[i].to(torch.device('cpu'))
                    else:
                        Nop_i.to(torch.device('cpu'))
                del x0, Nop_i, dcomp_i, mps_i, ktraj_i, k_proj_i, k_scale_i
                torch.cuda.empty_cache()

            for x0, x, Nop_i, basis_i, dcomp_i, mps_i, ktraj_i, k_proj_i, p_gt_i, k_scale_i, recon_scale_i in \
                    zip(recon_basis_test, gt_basis_test, Nop_test, basis_test, dcomp_test, mps_test, ktraj_test, k_proj_test,
                        p_gt_test, k_scale_test, recon_scale_test):
                x0, x = x0.unsqueeze(0).to(device), x.unsqueeze(0).to(device)
                if dc_step_size == 1:
                    if isinstance(Nop_i, list):
                        for i in range(len(Nop_i)):
                            Nop_i[i].to(device)
                    else:
                        Nop_i.to(device)
                    dcomp_i = dcomp_i.to(device)
                    ktraj_i = ktraj_i.to(device)
                    k_proj_i = k_proj_i.to(device)
                else:
                    Nop_i = None
                    dcomp_i = None
                    ktraj_i = None
                    k_proj_i = None
                mps_i = mps_i.to(device)
                basis_i = basis_i.to(device)
                p_gt_i = p_gt_i.to(device)
                k_scale_i = k_scale_i.to(device)
                recon_scale_i = recon_scale_i.to(device)
                torch.cuda.empty_cache()
                weight = 1 / torch.sum(torch.abs(dp.spatial_normalize_basis_coeff(x, basis_i)) ** 2,
                                       axis=(2, 3, 4)) ** 0.5 * (230 ** 3) ** 0.5  # (1, n_basis)
                if rotation_angle:
                    x0 = x
                if p_loss:
                    x_hat, p_hat = cnn(x0, Nop_i, basis_i, dcomp_i, mps_i, ktraj_i, k_proj_i, k_scale_i, recon_scale_i)
                    p_gt_i = p_gt_i / max_T1_T2.view(2, 1, 1, 1)
                    spatial_basis_loss = torch.nn.MSELoss()(torch.view_as_real(dp.spatial_normalize_basis_coeff(x, basis_i, weight=weight)),
                                             torch.view_as_real(dp.spatial_normalize_basis_coeff(x_hat, basis_i, weight=weight)))
                    if dataset == 'deli_cs':
                        p_hat = torch.clamp(p_hat, min=0, max=3)
                    else:
                        p_hat = torch.clamp(p_hat, min=0, max=1)
                    nll = (1 - beta) * spatial_basis_loss + \
                          beta * torch.mean(torch.abs(p_gt_i[p_gt_i > 0] - p_hat[p_gt_i > 0]) / p_gt_i[p_gt_i > 0])
                    loss_normalized = torch.nn.MSELoss()(torch.view_as_real(dp.normalize_basis_coeff(x, basis_i)),
                                                         torch.view_as_real(dp.normalize_basis_coeff(x_hat, basis_i)))
                else:
                    x_hat, basis_out = cnn(x0, Nop_i, basis_i, dcomp_i, mps_i, ktraj_i, k_proj_i, k_scale_i, recon_scale_i)
                    if dataset == 'deli_cs':
                        nll = torch.nn.MSELoss()(torch.view_as_real(dp.spatial_normalize_basis_coeff(x, basis_i, weight=weight)),
                                                 torch.view_as_real(dp.spatial_normalize_basis_coeff(x_hat * (abs(x) > 0), basis_out, weight=weight)))
                    else:
                        nll = torch.nn.MSELoss()(torch.view_as_real(dp.spatial_normalize_basis_coeff(x, basis_i, weight=weight)),
                                                 torch.view_as_real(dp.spatial_normalize_basis_coeff(x_hat, basis_out, weight=weight)))
                    loss_normalized = torch.nn.MSELoss()(torch.view_as_real(dp.normalize_basis_coeff(x, basis_i)),
                                                         torch.view_as_real(dp.normalize_basis_coeff(x_hat, basis_out)))
                nll_test.append(nll.item())
                if dc_step_size == 1:
                    if isinstance(Nop_i, list):
                        for i in range(len(Nop_i)):
                            Nop_i[i].to(torch.device('cpu'))
                    else:
                        Nop_i.to(torch.device('cpu'))
                del x0, Nop_i, dcomp_i, mps_i, ktraj_i, k_proj_i, k_scale_i
                torch.cuda.empty_cache()

        print('%.3i \t%.2f \t%.2e\t%.2e\t%.2e\t%.2e\t%.2e' % (epoch + 1,
                                                        (time.time() - t_start)/60.,
                                                        np.mean(nll_mean),
                                                        np.mean(nll_val),
                                                        np.mean(nll_test),
                                                        optimizer.param_groups[0]['lr'],
                                                        torch.sigmoid(cnn.dc_step_size * cnn.dc_step_size_sigmoid_slope)[0].cpu()),
              flush=True)
        write_csv(epoch, (time.time() - t_start) / 60., np.mean(nll_mean),
                  np.mean(nll_val), np.mean(nll_test), optimizer.param_groups[0]['lr'],
                  torch.cuda.max_memory_reserved() / 1024 ** 3)
        if np.mean(nll_val) < nll_mean_val_min:
            inpatience = 0
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': cnn.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        }, save_dir + '/checkpoint_epoch_model_optimizer.pt')
            torch.save(cnn.state_dict(), save_dir + '/cnn.pt')
            nll_mean_val_min = np.mean(nll_val)
        else:
            inpatience += 1
            if inpatience == early_stop_patience:
                break
        scheduler.step(np.mean(nll_val))
        nll_mean = []
        nll_val = []
        nll_test = []
        if epoch == 0 or not (epoch + 1) % 5:
            ### plot basis coefficient maps
            x_hat = dp.normalize_basis_coeff(x_hat, basis_i)
            x_hat = x_hat * (abs(x) > 0)
            x = dp.normalize_basis_coeff(x, basis_i)
            fig, axs = plt.subplots(2, 5, figsize=(10, 4), constrained_layout=True)
            i = 0
            for ax in axs.flat:
                if i <= 4:
                    im = ax.imshow(np.abs(x_hat.detach().cpu().numpy()[0, i, 100, :, :]), vmin=0, vmax=1,
                                   cmap='jet')
                else:
                    im = ax.imshow(np.abs(x.detach().cpu().numpy()[0, i - 5, 100, :, :]), vmin=0,
                                   vmax=1,
                                   cmap='jet')
                i += 1
            fig.colorbar(im, orientation='vertical')
            fig.savefig(save_dir + f'/epoch_{epoch + 1}.png', bbox_inches='tight')
            plt.close(fig)
            if p_loss:
                ### plot T1 and T2 maps
                p_hat = p_hat * max_T1_T2.view(2, 1, 1, 1)
                p_gt_i = p_gt_i * max_T1_T2.view(2, 1, 1, 1)
                da.PlotT1T2Map(p_gt_i.cpu().numpy()[0, 100],
                               p_gt_i.cpu().numpy()[1, 100],
                               p_hat.cpu().to(torch.float32).numpy()[0, 100],
                               p_hat.cpu().to(torch.float32).numpy()[1, 100],
                               save_dir=save_dir + f'/epoch_{epoch + 1}_T1_T2.png')
        del x_hat, basis_i, x, p_gt_i
        torch.cuda.empty_cache()
        cnn.train()
