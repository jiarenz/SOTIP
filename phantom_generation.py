import numpy as np
import torch
import os
import sigpy as sp
import sigpy.mri
from mirtorch.linear import NuSense
import torchkbnufft as tkbn
import scipy.io as sio

project_dir = os.getcwd()


def check_values(arr):
    assert np.sum(np.isnan(np.abs(arr.ravel()))) == 0, \
        ">>>>> Unexpected nan in array."
    assert np.sum(np.isinf(np.abs(arr.ravel()))) == 0, \
        ">>>>> Unexpected inf in array."


def loadDeliCSdata(trj_file: str, ksp_file: str, phi_file: str, phi_rank: int, shf_file: str,
                   akp: bool, ptt: int, device: int, R: int = 1):
    '''
    Adapted from https://github.com/SetsompopLab/MRF/blob/main/src/02_recon/load_data.py

    (trj, ksp, phi) = load_data(trj_file, ksp_file, phi_file, phi_rank):
    Load data to return NumPy arrays. This function uses the trajectory
    dimensions to remove rewinder points from the loaded k-space array.

    The trj within the 'mat' file pointed to by 'trj_file' is expected to
    be of the form trj[a, b, c, d]:
      a -> Readout dimension (data acquired during a spiral) NOT including
           rewinder points.
      b -> Non-cartesian coordinates of the bth volumetric dimension. b
           here varies from 0, 1 and 2.
      c -> Interleave dimension.
      d -> TR dimension.
    The ksp dimensions within the 'npy' file pointed to by 'ksp_file' is
    expected to be of the form ksp[e, f, g]:
      e -> Readout dimension (data acquired during a spiral) including
           rewinder points.
      f -> Coil dimension.
      g -> Combined TR and Interleave dimension (that is, c * d from trj).
           The data is expected to be ordered as ksp[:, :, 0:c] is the first
           TR, ksp[:, :, c:2*c] is the next and so on.

    The phi within the 'mat' file pointed to by 'phi_file' is expected to
    be of the form phi[d, h]
      h -> Number of subspace vectors.
    Inputs:
      trj_file (String): Path to trajectory as a MATLAB 'mat' file. Expects
                         an entry denoted 'k_3d' within the MATLAB file.
      ksp_file (String): Path to k-space file as a NumPy array.
      phi_file (String): Path to subspace basis as a MATLAB 'mat' file.
                         Expects an entry denoted 'phi' within the MATLAB
                         file.
      phi_rank (Int): Rank of subspace to use for reconstruction. If bigger
                      than h, use h.
    Returns:
      trj (Array): Non-cartesian k-space coordinates.
      ksp (Array): Acquired k-space data.
      phi (Array): Temporal subspace.
    '''
    trj = sio.loadmat(trj_file)['k_3d'].transpose((1, 0, 2, 3)).astype(np.float32)
    check_values(trj)
    assert np.abs(trj.ravel()).max() < 0.5, \
        "Trajectory must be scaled between -1/2 and 1/2."

    ksp = np.load(ksp_file, mmap_mode='r')
    ksp = ksp.astype(np.complex64)

    trj = trj[:, ptt:, ...]
    phi = sio.loadmat(phi_file)['phi'][:, :phi_rank]
    if len(phi.shape) == 1:
        phi = phi[:, None]
    phi = phi @ sp.fft(np.eye(phi.shape[-1]), axes=(0,))

    check_values(phi)
    check_values(ksp)
    return (trj, ksp, phi)


def generate_base_spiral_traj(fov=0.23, N=230, f_sampling=1, R=29, ninterleaves=1, alpha=1.3, gm=0.040, sm=150):
    traj = sigpy.mri.spiral(fov=fov, N=N, f_sampling=f_sampling, R=R, ninterleaves=ninterleaves, alpha=alpha, gm=gm, sm=sm)
    return traj


def RotateTraj(traj_coordinate: list, theta:list, rotation_order: list) -> tuple:
    kx_base, ky_base, kz_base = traj_coordinate
    out = [np.zeros_like(kx_base), np.zeros_like(ky_base), np.zeros_like(kz_base)]
    out[rotation_order[1]] = np.cos(theta[rotation_order[0]]) * traj_coordinate[rotation_order[1]] - np.sin(theta[rotation_order[0]]) * traj_coordinate[rotation_order[2]]
    temp_ = np.sin(theta[rotation_order[0]]) * traj_coordinate[rotation_order[1]] + np.cos(theta[rotation_order[0]]) * traj_coordinate[rotation_order[2]]

    out[rotation_order[2]] = np.cos(theta[rotation_order[1]]) * temp_ - np.sin(theta[rotation_order[1]]) * traj_coordinate[rotation_order[0]]
    out[rotation_order[0]] = np.sin(theta[rotation_order[1]]) * temp_ + np.cos(theta[rotation_order[1]]) * traj_coordinate[rotation_order[0]]
    return out


def TGAS(fov, N, base_traj_2d, nspokes=48, TR_ind=0, dataset="kirby21"):
    kmax_rad_per_m = 1 / fov * N / 2
    spokelength = base_traj_2d.shape[0]

    ga1 = 22.5 / 180 * np.pi
    ga2 = 23.6 / 180 * np.pi
    kx = np.zeros(shape=(spokelength, nspokes))
    ky = np.zeros(shape=(spokelength, nspokes))
    kz = np.zeros(shape=(spokelength, nspokes))

    kx_base = (base_traj_2d / kmax_rad_per_m * np.pi)[:, 0]
    ky_base = (base_traj_2d / kmax_rad_per_m * np.pi)[:, 1]
    kz_base = np.zeros_like(kx_base)

    for i in range(0, nspokes // 3):
        kx[:, i], ky[:, i], kz[:, i] = RotateTraj([kx_base, ky_base, kz_base], [ga2 * (i + TR_ind), 0, ga1 * i], [2, 0, 1])

    kx_base = np.zeros_like(kx_base)
    ky_base = (base_traj_2d / kmax_rad_per_m * np.pi)[:, 0]
    kz_base = (base_traj_2d / kmax_rad_per_m * np.pi)[:, 1]
    for i in range(nspokes // 3, nspokes // 3 * 2):
        kx[:, i], ky[:, i], kz[:, i] = RotateTraj([kx_base, ky_base, kz_base], [ga1 * (i - nspokes // 3), ga2 * (i - nspokes // 3 + TR_ind), 0], [0, 1, 2])
    kx_base = (base_traj_2d / kmax_rad_per_m * np.pi)[:, 1]
    ky_base = np.zeros_like(kx_base)
    kz_base = (base_traj_2d / kmax_rad_per_m * np.pi)[:, 0]
    for i in range(nspokes // 3 * 2, nspokes // 3 * 3):
        kx[:, i], ky[:, i], kz[:, i] = RotateTraj([kx_base, ky_base, kz_base], [0, ga1 * (i - nspokes // 3 * 2), ga2 * ((i - nspokes // 3 * 2) + TR_ind)], [1, 2, 0])

    kx = np.transpose(kx)
    ky = np.transpose(ky)
    kz = np.transpose(kz)

    ktraj = np.stack((kx.flatten(), ky.flatten(), kz.flatten()), axis=0)
    return ktraj, (kx, ky, kz)


def TGAS_w_TR(fov, N, base_traj_2d, nspokes=48, nTR=500, dataset="kirby21"):
    kx = []
    ky = []
    kz = []
    for i in range(nTR):
        _, (kx_, ky_, kz_) = TGAS(fov, N, base_traj_2d, nspokes=nspokes, TR_ind=i, dataset=dataset)
        kx.append(kx_)
        ky.append(ky_)
        kz.append(kz_)
    kx = np.stack(kx, axis=0)  # (500, 48, n_samp)
    ky = np.stack(ky, axis=0)  # (500, 48, n_samp)
    kz = np.stack(kz, axis=0)  # (500, 48, n_samp)
    ktraj = np.stack((kx.reshape(nTR, -1), ky.reshape(nTR, -1), kz.reshape(nTR, -1)), axis=1)
    return ktraj, (kx, ky, kz)


def load_data(dataset: str, subject, device, R=6, sgd=False, scaling_factor=1, full_dcomp=True, single_coil=True,
              SNR=50, nufft_numpoints=2, nufft_grid_size=1.25,
              nufft_split_batch=False, nufft_split_channel=False, deli_cs_tgas=False, opt_rot_ang=False):
    print(f"loading data of {subject}, SNR={SNR}, R={R}, sgd={str(sgd)}, scaling_factor={str(scaling_factor)}, "
          f"full_dcomp={str(full_dcomp)}, single_coil={str(single_coil)}")
    n_group = 48
    n_basis = 5
    if single_coil:
        n_c = 1
    elif dataset == 'kirby21':
        n_c = 8
    elif dataset == 'deli_cs':
        n_c = 10
    nTR = 500
    if dataset == 'deli_cs':
        im_size = (256, 256, 256)
    else:
        im_size = (230, 230, 230)

    if dataset == 'deli_cs':
        subject_id = subject[-2:]
        if R == 3:
            if deli_cs_tgas:
                trj_file = project_dir + "/reference/DeliCS/data/shared/traj_grp16_inacc2.mat"
            else:
                trj_file = project_dir + "/reference/DeliCS/data/shared/traj_grp48_inacc1.mat"
        else:
            trj_file = project_dir + "/reference/DeliCS/data/shared/traj_grp48_inacc1.mat"
        ksp_file = project_dir + f"/reference/DeliCS/data/{subject[:-2]}/case0{subject_id}/processed_ksp.npy"
        phi_file = project_dir + "/reference/DeliCS/data/shared/phi.mat"
        shf_file = project_dir + f"/reference/DeliCS/data/{subject[:-2]}/case0{subject_id}/shifts.npy"
        phi_rank = 5
        ptt = 10
        akp = False
        # trj (3, n_spiral (1678), n_group (48), n_TR)
        # ksp (n_coils, n_spiral, n_group (48), n_TR)
        # phi (n_TR, n_basis)
        ktraj, k, basis = loadDeliCSdata(trj_file, ksp_file, phi_file, phi_rank, shf_file, akp, ptt, device.index, R=R)
        if R == 3 and deli_cs_tgas:
            if not subject in ['testing02', 'testing03', 'testing04']:
                ksp_a = k[:, :, 0:32:2, 0:500:2]
                ksp_b = k[:, :, 1:32:2, 1:500:2]

                k = np.zeros(list(ksp_a.shape[:-1]) + [k.shape[-1]], dtype=k.dtype)
                k[..., 0:500:2] = ksp_a
                k[..., 1:500:2] = ksp_b
        ktraj = ktraj.transpose(3, 0, 2, 1).reshape(nTR, 3, -1)  # (n_TR, 3, n_group * n_spiral)
        k = k.transpose(3, 0, 2, 1).reshape(nTR, n_c, -1)
        ktraj = ktraj * 2 * np.pi
        ktraj = ktraj[:, ::-1, ...]
    else:
        traj = generate_base_spiral_traj()
        ktraj, _ = TGAS_w_TR(fov=0.23, N=230, base_traj_2d=traj, nspokes=n_group, nTR=nTR, dataset=dataset)  # (n_TR, 3, n_group*n_spiral)

    ### undersampling as in 10.1002/mrm.29194 ###
    if not (dataset == 'deli_cs' and R == 3 and deli_cs_tgas):
        ktraj = ktraj.reshape(nTR, 3, n_group, -1)
        ktraj_undersample = []
        for i in range(nTR):
            ktraj_undersample.append(ktraj[i:i + 1, :, int(R - i % R) % R:n_group:R, :])
        ktraj_undersample = np.concatenate(ktraj_undersample)
        ktraj = ktraj_undersample.reshape(nTR, 3, -1)
    else:
        ktraj = ktraj.copy().reshape(nTR, 3, -1)
    ###
    ktraj = ktraj.transpose(1, 0, 2)
    ktraj = torch.from_numpy(ktraj).to(torch.float32).to(device)
    dcomp = tkbn.calc_density_compensation_function(ktraj=ktraj.reshape(3, -1), im_size=im_size, numpoints=nufft_numpoints,
                                                    grid_size=tuple([int(dim * nufft_grid_size) for dim in im_size]))
    ############ Normalize dcomp ###############
    dcomp /= torch.linalg.norm(dcomp.ravel(), ord=float('inf'))
    ###
    if dataset == 'kirby21':
        k = np.load(project_dir + f'/data/Kirby21/KKI2009-{subject}/MRF_k_t_sigpy_spiral_reset_rot_every_16_grp_SNR_inf_csmp_deli_cs.npy')
    elif dataset == 'deli_cs':
        if not (dataset == 'deli_cs' and R == 3 and deli_cs_tgas):
            n_point_per_spiral = int(k.shape[-1] / 48)
            k = k.reshape(nTR, k.shape[1], n_group, n_point_per_spiral)
            k_undersample = []
            for i in range(nTR):
                k_undersample.append(k[i:i + 1, :, int(R - i % R) % R:n_group:R, :])
            k_undersample = np.concatenate(k_undersample)
            k = k_undersample.reshape(nTR, k.shape[1], -1)
            k = torch.from_numpy(k).to(torch.complex64).to(device)
        else:
            k = k.reshape(nTR, k.shape[1], -1)
            k = torch.from_numpy(k).to(torch.complex64).to(device)
    if dataset == 'kirby21':
        if not opt_rot_ang:
            n_point_per_spiral = int(k.shape[-1] / 48)
            k = k.reshape(nTR, k.shape[1], n_group, n_point_per_spiral)
            k_undersample = []
            for i in range(nTR):
                k_undersample.append(k[i:i + 1, :, int(R - i % R) % R:n_group:R, :])
            k_undersample = np.concatenate(k_undersample)
            k = k_undersample.reshape(nTR, k.shape[1], -1)
        k = torch.from_numpy(k).to(torch.complex64).to(device)
        ### Add IID Gaussian noise to each coil
        torch.random.manual_seed(666)
        k_sigma = 0.07 / SNR / 2 ** 0.5  # 0.07 is the mean signal in the MRF image-t of kirby21 01
        k += torch.randn_like(k) * k_sigma + 1j * torch.randn_like(k) * k_sigma
    ###
    ### Normalize k-t-space data
    k_scale = 1 / torch.linalg.norm(k)
    k = k * k_scale
    if dataset == 'kirby21':
        basis = np.load(project_dir + '/data/Bloch_simulation/MRF_norm_basis_TGAS_22235_entries.npy')
    elif dataset == 'deli_cs':
        basis = basis.transpose(1, 0)
    basis = basis[:n_basis].copy()
    basis = torch.from_numpy(basis).to(device)
    ########### load T1, T2 ground truth
    if dataset == 'kirby21':
        t1_gt = np.load(project_dir + f'/data/Kirby21/KKI2009-{subject}/t1.npy')
        t2_gt = np.load(project_dir + f'/data/Kirby21/KKI2009-{subject}/t2.npy')
    elif dataset == 'deli_cs':
        if subject in ['testing02', 'testing03', 'testing04']:
            t1_gt = np.load(project_dir + f"/reference/DeliCS/data/{subject[:-2]}/case000/T1_ref_6min_redo.npy")
            t2_gt = np.load(project_dir + f"/reference/DeliCS/data/{subject[:-2]}/case000/T2_ref_6min_redo.npy")
            t1_gt = np.zeros_like(t1_gt.T)
            t2_gt = np.zeros_like(t2_gt.T)
        else:
            t1_gt = np.load(project_dir + f"/reference/DeliCS/data/{subject[:-2]}/case0{subject_id}/T1_ref_6min_redo.npy")
            t2_gt = np.load(project_dir + f"/reference/DeliCS/data/{subject[:-2]}/case0{subject_id}/T2_ref_6min_redo.npy")
            t1_gt = t1_gt.T
            t2_gt = t2_gt.T
    p_gt = torch.from_numpy(np.stack((t1_gt, t2_gt))).to(torch.float32)  # (2, nx, ny, nz)
    # load brain mask
    if dataset == 'kirby21':
        mps = t1_gt > 0
        mps = mps.astype(int).reshape(1, 230, 230, 230)
    ####### Load coil sensitivity maps
    if dataset == 'kirby21':
        csmp = np.load(project_dir + f'/data/csmap_from_deli_cs.npy')
        if single_coil:
            csmp = np.ones_like(csmp)[0:1]
        mps = mps * csmp
        mps = torch.from_numpy(mps).to(torch.complex64).to(device)
    elif dataset == 'deli_cs':
        csmp = np.load(project_dir + f"/reference/DeliCS/data/{subject[:-2]}/case0{subject_id}/mps.npy")
        mps = torch.from_numpy(csmp).to(torch.complex64).to(device)
    ### Form MRI systen operator
    if nufft_split_batch and not nufft_split_channel:
        Nop = NuSense(mps[None, ...],
                      ktraj.reshape(3, -1),
                      numpoints=nufft_numpoints,
                      grid_size=nufft_grid_size,
                      sequential=True)
    elif nufft_split_batch and nufft_split_channel:
        Nop = [NuSense(mps[i].view((1, 1,) + im_size),
                      ktraj.reshape(3, -1),
                      numpoints=nufft_numpoints,
                      grid_size=nufft_grid_size,
                      sequential=True) for i in range(n_c)]
    elif not nufft_split_batch and nufft_split_channel:
        Nop = [NuSense(mps[i].view((1, 1,) + im_size).expand(n_basis, -1, -1, -1, -1),
                      ktraj.reshape(3, -1),
                      numpoints=nufft_numpoints,
                      grid_size=nufft_grid_size,
                      sequential=True) for i in range(n_c)]
    else:
        Nop = NuSense(mps.view((1, n_c,) + im_size).expand(n_basis, -1, -1, -1, -1),
                      ktraj.reshape(3, -1),
                      numpoints=nufft_numpoints,
                      grid_size=nufft_grid_size,
                      sequential=True)
    ###### P'(.)Phi' #######
    k_proj = []
    for i in range(nTR):
        k_proj.append(torch.conj(basis[:, i:i + 1]).view(n_basis, 1, 1) * k[i:i + 1, :, :])
    k_proj = torch.cat(k_proj, dim=2)
    # ############
    if dataset == 'kirby21':
        gt_basis = np.load(project_dir + f'/data/Kirby21/KKI2009-{subject}/MRF_basis_img.npy')
    elif dataset == 'deli_cs':
        if subject in ['testing02', 'testing03', 'testing04']:
            gt_basis = np.load(project_dir + f"/reference/DeliCS/data/{subject[:-2]}/case0{subject_id}/ref_2min.npy")
        else:
            gt_basis = np.load(project_dir + f"/reference/DeliCS/data/{subject[:-2]}/case0{subject_id}/ref_6min_redo.npy")
    gt_basis = torch.from_numpy(gt_basis).to(device)   # (n_basis, nx, ny, nz)
    ### reconstruct basis coefficient maps
    if nufft_split_channel and nufft_split_batch:
        recon_basis = 0
        for i in range(n_c):
            recon_basis = recon_basis + torch.cat([Nop[i].H * (dcomp * k_proj.view(n_basis, n_c, -1)[j:j+1, i:i + 1])
                                      for j in range(n_basis)])  # (n_basis, 1, nx, ny, nz)
            torch.cuda.empty_cache()
    elif nufft_split_channel and not nufft_split_batch:
        recon_basis = 0
        for i in range(n_c):
            recon_basis = recon_basis + Nop[i].H * (dcomp * k_proj.view(n_basis, n_c, -1)[:, i:i + 1])  # (n_basis, 1, nx, ny, nz)
            torch.cuda.empty_cache()
    elif not nufft_split_channel and nufft_split_batch:
        recon_basis = torch.cat(
            [(Nop.H * (dcomp * k_proj.view(n_basis, n_c, -1)[i:i + 1])) for i in
             range(n_basis)])  # (n_basis, 1, nx, ny, nz)
    else:
        recon_basis = Nop.H * (dcomp * k_proj.view(n_basis, n_c, -1))
    torch.cuda.empty_cache()
    recon_basis = recon_basis[:, 0, ...]  # (n_basis, nx, ny, nz)
    ###
    recon_basis[torch.isnan(recon_basis)] = 0
    gt_basis[torch.isnan(gt_basis)] = 0
    ##### Scale basis ########
    if scaling_factor:
        recon_basis_imag_norm = torch.sum(torch.abs(recon_basis) ** 2, axis=(1, 2, 3), keepdims=True) ** 0.5  # (n_basis, 1, 1, 1)
        recon_basis = recon_basis / recon_basis_imag_norm * (im_size[0] ** 3) ** 0.5 / scaling_factor
        basis = basis * recon_basis_imag_norm.view(n_basis, 1) / (im_size[0] ** 3) ** 0.5 * scaling_factor
        gt_basis = gt_basis / recon_basis_imag_norm * (im_size[0] ** 3) ** 0.5 / scaling_factor
    recon_basis_imag_norm = torch.linalg.norm(recon_basis)
    if dataset == 'kirby21':
        gt_basis_imag_norm = 3400
    elif dataset == 'deli_cs':
        gt_basis_imag_norm = 3400
        gt_scale = gt_basis_imag_norm / torch.linalg.norm(gt_basis)
        gt_basis *= gt_scale
        k_proj *= gt_scale
        gt_basis = gt_basis.T
    recon_scale = gt_basis_imag_norm / recon_basis_imag_norm
    recon_basis *= recon_scale
    if dataset == 'deli_cs':
        k_scale /= k_scale
        if isinstance(Nop, list):
            if nufft_split_batch:
                recon_basis_in_k = torch.cat([torch.cat(
                    [Nop[i] * recon_basis[j:j+1].unsqueeze(1) for i in
                     range(n_c)], dim=1) for j in range(n_basis)])
            else:
                recon_basis_in_k = torch.cat(
                    [Nop[i] * recon_basis.unsqueeze(1) for i in
                     range(n_c)], dim=1)
        else:
            recon_basis_in_k = torch.cat(
                [Nop * recon_basis[i:i + 1].unsqueeze(1) for i in
                 range(n_basis)])
        k_proj_norm = torch.norm(k_proj)
        recon_basis_in_k = recon_basis_in_k.view(n_basis, n_c, nTR, -1)  # (n_basis, n_c, nTR, n_group * n_ro)
        _temp = torch.sum(recon_basis_in_k * basis.view(n_basis, 1, nTR, 1), dim=0,
                          keepdim=True)  # (1, n_c, nTR, n_group * n_ro)
        torch.cuda.empty_cache()
        _temp = torch.conj(basis).view(n_basis, 1, nTR, 1) * _temp
        k_proj *= torch.norm(_temp) / k_proj_norm
        recon_scale /= gt_scale
        recon_scale /= torch.norm(_temp) / k_proj_norm
        ####
        gt_basis_mask = torch.mean(abs(gt_basis), dim=0, keepdim=True) > \
                        torch.mean(abs(gt_basis), dim=0, keepdim=True).max() / 20
        gt_basis *= gt_basis_mask
        p_gt *= gt_basis_mask.cpu()
    return recon_basis, gt_basis, Nop, basis, dcomp, mps, ktraj, k_proj, p_gt, k_scale, recon_scale


def load_dataset(dataset: str, subject_list, device, scaling_factor=1, R=6, full_dcomp=False, single_coil=True,
                 nufft_numpoints=2, nufft_grid_size=1.25, nufft_split_batch=False, nufft_split_channel=False,
                 opt_rot_ang=False):
    recon_basis = []
    gt_basis = []
    Nop = []
    basis = []
    dcomp = []
    mps = []
    ktraj = []
    k_proj = []
    p_gt = []
    k_scale = []
    recon_scale = []
    for subject in subject_list:
        if subject in ["testing02", "testing03", "testing04"]:
            deli_cs_tgas = True
        else:
            deli_cs_tgas = False
        recon_basis_temp, gt_basis_temp, Nop_temp, basis_temp, \
        dcomp_temp, mps_temp, ktraj_temp, k_proj_temp, p_gt_temp, \
            k_scale_temp, recon_scale_temp = load_data(dataset,
                                                       subject,
                                                       device=device,
                                                       R=R,
                                                       scaling_factor=scaling_factor,
                                                       full_dcomp=full_dcomp,
                                                       single_coil=single_coil,
                                                       nufft_numpoints=nufft_numpoints,
                                                       nufft_grid_size=nufft_grid_size,
                                                       nufft_split_batch=nufft_split_batch,
                                                       nufft_split_channel=nufft_split_channel,
                                                       deli_cs_tgas=deli_cs_tgas,
                                                       opt_rot_ang=opt_rot_ang)
        recon_basis.append(recon_basis_temp.cpu())  # (n_basis, nx, ny, nz)
        gt_basis.append(gt_basis_temp.cpu())  # (n_basis, nx, ny, nz)
        if isinstance(Nop_temp, list):
            for i in range(len(Nop_temp)):
                Nop_temp[i].to(torch.device("cpu"))
        else:
            Nop_temp.to(torch.device("cpu"))
        Nop.append(Nop_temp)   # (n_basis, n_c, total_n_ro) * (n_basis, 1, nx, ny, nz)
        basis.append(basis_temp.cpu())  # (n_basis, n_TR)
        dcomp.append(dcomp_temp.cpu())  # (1, 1, total_n_ro)
        mps.append(mps_temp.cpu())
        ktraj.append(ktraj_temp.cpu())
        k_proj.append(k_proj_temp.cpu())
        p_gt.append(p_gt_temp.cpu())
        k_scale.append(k_scale_temp.cpu())
        recon_scale.append(recon_scale_temp.cpu())
        del recon_basis_temp, gt_basis_temp, Nop_temp, basis_temp, \
            mps_temp, ktraj_temp, k_proj_temp, p_gt_temp, k_scale_temp, recon_scale_temp
        torch.cuda.empty_cache()
    return recon_basis, gt_basis, Nop, basis, dcomp, mps, ktraj, k_proj, p_gt, k_scale, recon_scale
