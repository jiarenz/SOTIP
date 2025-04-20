import os
from mirtorch.linear import NuSense_om
import torchkbnufft as tkbn
from unet.unet_parts import *
from torch.utils.checkpoint import checkpoint
from utils import data_processing as dp
import fcnn


def grad4one_coeff(index, Nop, recon_basis, basis, dcomp, k_proj, k_scale, recon_scale, step_size):
    # recon_basis: (1, n_basis=1, nx, ny, nz)
    n_basis = 5
    _, nTR = basis.shape
    n_c = Nop.size_out[1]
    recon_basis_in_k = torch.cat(
        [Nop(recon_basis[0, i:i + 1].unsqueeze(1)) for i in
         range(n_basis)]) * k_scale
    torch.cuda.empty_cache()
    k_proj = k_proj.view(1, n_c, nTR, -1)
    recon_basis_in_k = recon_basis_in_k.view(1, n_c, nTR, -1)  # (n_basis, n_c, nTR, n_group * n_ro)
    _temp = recon_basis_in_k * basis.view(n_basis, 1, nTR, 1)  # (1, n_c, nTR, n_group * n_ro)
    del recon_basis_in_k
    torch.cuda.empty_cache()
    k_proj = torch.conj(basis).view(n_basis, 1, nTR, 1) * _temp - k_proj
    del _temp, basis
    torch.cuda.empty_cache()
    g_DC = Nop.H(dcomp * k_proj[index].view(1, n_c, -1))  # (n_basis, 1, nx, ny, nz)
    del dcomp
    torch.cuda.empty_cache()
    recon_basis -= step_size * g_DC[0].unsqueeze(0) * recon_scale
    torch.cuda.empty_cache()


def grad4one_channel(Nop, recon_basis, basis, dcomp, k_proj, k_scale, recon_scale, step_size):
    # recon_basis: (1, n_basis=5, nx, ny, nz)
    # k_proj: ()
    n_basis, nTR = basis.shape
    n_c = Nop.size_out[1]
    if Nop.size_in[0] == n_basis:
        recon_basis_in_k = Nop(recon_basis[0].unsqueeze(1)) * k_scale
    elif Nop.size_in[0] == 1:
        recon_basis_in_k = torch.cat([Nop(recon_basis[0].unsqueeze(1)[i:i+1]) for i in range(n_basis)]) * k_scale
    # recon_basis_in_k = Nop(recon_basis[0].unsqueeze(1)) * k_scale
    torch.cuda.empty_cache()
    k_proj = k_proj.view(n_basis, n_c, nTR, -1)
    recon_basis_in_k = recon_basis_in_k.view(n_basis, n_c, nTR, -1)  # (n_basis, n_c, nTR, n_group * n_ro)
    _temp = torch.sum(recon_basis_in_k * basis.view(n_basis, 1, nTR, 1), dim=0, keepdim=True)  # (1, n_c, nTR, n_group * n_ro)
    del recon_basis_in_k
    torch.cuda.empty_cache()
    k_proj = torch.conj(basis).view(n_basis, 1, nTR, 1) * _temp - k_proj
    del _temp, basis
    torch.cuda.empty_cache()
    if Nop.size_in[0] == n_basis:
        g_DC = Nop.H(dcomp * k_proj.view(n_basis, n_c, -1))  # (n_basis, 1, nx, ny, nz)
    elif Nop.size_in[0] == 1:
        g_DC = torch.cat([Nop.H(dcomp * k_proj.view(n_basis, n_c, -1)[i:i+1]) for i in range(n_basis)])  # (n_basis, 1, nx, ny, nz)
    del dcomp
    torch.cuda.empty_cache()
    return g_DC


def DC(recon_basis, Nop, basis, dcomp, k_proj, k_scale, recon_scale, step_size, llr_scale, batch_size=500,
       checkpointing=True, scale_temp=False):
    n_basis, nTR = basis.shape
    if isinstance(Nop, list):
        n_c = len(Nop)
    else:
        n_c = Nop.size_out[1]
    if isinstance(Nop, list):
        g_DC = 0
        for i in range(n_c):
            g_DC = g_DC + checkpoint(grad4one_channel, Nop[i], recon_basis, basis, dcomp,
                       k_proj[:, i:i+1], k_scale, recon_scale, step_size, use_reentrant=False)  # recon_basis: (1, n_basis, nx, ny, nz)
            torch.cuda.empty_cache()
        recon_basis = recon_basis - step_size * g_DC[:, 0].unsqueeze(0) * recon_scale
        torch.cuda.empty_cache()
        recon_basis = recon_basis[0].unsqueeze(1)
    else:
        if Nop.size_in[0] == 1:
            recon_basis_in_k = torch.cat(
                [checkpoint(Nop, recon_basis[0, i:i + 1].unsqueeze(1), use_reentrant=False) for i in
                 range(n_basis)]) * k_scale
        elif Nop.size_in[0] == n_basis:
            recon_basis_in_k = Nop * recon_basis[0].unsqueeze(1) * k_scale
        else:
            print(f"The NUFFT operator dimension is not acceptable, the principal component dimension should be"
                  f"either 1 or n_basis")
        k_proj = k_proj.view(n_basis, n_c, nTR, -1)
        recon_basis_in_k = recon_basis_in_k.view(n_basis, n_c, nTR, -1)  # (n_basis, n_c, nTR, n_group * n_ro)
        _temp = torch.sum(recon_basis_in_k * basis.view(n_basis, 1, nTR, 1), dim=0, keepdim=True)  # (1, n_c, nTR, n_group * n_ro)
        del recon_basis_in_k
        torch.cuda.empty_cache()
        _temp = torch.conj(basis).view(n_basis, 1, nTR, 1) * _temp  # (n_basis, n_c, nTR, n_group * n_ro)
        torch.cuda.empty_cache()
        _temp = _temp - k_proj
        torch.cuda.empty_cache()
        if Nop.size_in[0] == 1:
            g_DC = torch.cat(
                [checkpoint(Nop.H, dcomp * _temp.view(n_basis, n_c, -1)[i:i + 1], use_reentrant=False) for i in
                 range(n_basis)])  # (n_basis, 1, nx, ny, nz)
        elif Nop.size_in[0] == n_basis:
            g_DC = Nop.H * (dcomp * _temp.view(n_basis, n_c, -1))  # (n_basis, 1, nx, ny, nz)
        g_DC = g_DC * recon_scale
        recon_basis = recon_basis[0].unsqueeze(1) - step_size * (nTR / batch_size) * g_DC
        del g_DC, _temp
    return recon_basis.unsqueeze(0).squeeze(2)  # (n_b=1, n_basis, nx, ny, nz)


class UnrollUNet2D(nn.Module):
    def __init__(self, in_channels, out_channels, n_unroll, checkpointing=True, est_p=False, f_maps=64,
                 share_weight=False, rotation_angle=False, dc_step_size=0, pretrain_FCNN=False, train_mag=False,
                 dataset='deli_cs', R: int=None, **kwargs):
        super(UnrollUNet2D, self).__init__()
        self.n_unroll = n_unroll
        self.unet_list = nn.ModuleList()
        self.rotation_angle = rotation_angle
        # self.dc_step_size = nn.Parameter(torch.ones(n_unroll) * dc_step_size)
        self.dc = dc_step_size > 0
        self.dc_step_size_sigmoid_slope = 1e3
        self.dc_step_size = nn.Parameter(torch.zeros(n_unroll))
        if dataset == 'deli_cs':
            self.dataset = dataset
            self.llr_scale = nn.Parameter(torch.zeros(1))
            self.llr_scale_sigmoid_slope = 1e3
        self.pretrain_FCNN = pretrain_FCNN
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.train_mag = train_mag
        if share_weight:
            unet = UNet(in_channels, out_channels)
            for i in range(n_unroll):
                self.unet_list.append(unet)
        else:
            for i in range(n_unroll):
                unet = UNet(in_channels, out_channels)
                self.unet_list.append(unet)
        self.checkpointing = checkpointing
        self.est_p = est_p
        if est_p:
            self.fcnn_batch_size = 10000
            if pretrain_FCNN:
                project_dir = os.getcwd()
                model_id = '20230112_152842'  # replace with your FCNN model ID
                fcnn_estimator = fcnn.CohenModelSubspace()
                state_dict = {k: v for k, v in torch.load(project_dir + 'model/' + model_id + '/cohen_sub.pt').items()
                              if
                              'tmp_var' not in k}
                fcnn_estimator.load_state_dict(state_dict)
                self.fcnn = fcnn_estimator
            else:
                self.fcnn = fcnn.CohenModelSubspace()
            self.im_size = (230, 230, 230)
        if rotation_angle:
            self.nufft_numpoints = 2
            self.im_size = (230, 230, 230)
            self.nufft_grid_size = 1.25
            if R == 3:
                n_group = 16
            elif R == 6:
                n_group = 8
            self.ga_kx = nn.Parameter(torch.zeros(n_group * 500))
            self.ga_ky = nn.Parameter(torch.zeros(n_group * 500))
            self.ga_kz = nn.Parameter(torch.zeros(n_group * 500))

    def forward(self, x, Nop, basis, dcomp, mps, ktraj, k_proj, k_scale, recon_scale):
        nTR = basis.shape[1]
        n_basis = x.shape[1]
        im_size = x.shape[-3:]
        n_c = mps.shape[0]
        recon_basis_norm = torch.sum(torch.abs(x) ** 2, axis=1, keepdims=True) ** 0.5  # (n_b, 1, nx, ny, nz)
        if self.rotation_angle:
            ga_kx = (torch.sigmoid(self.ga_kx * 100) - 1 / 2) * 2 * torch.pi
            ga_ky = (torch.sigmoid(self.ga_ky * 100) - 1 / 2) * 2 * torch.pi
            ga_kz = (torch.sigmoid(self.ga_kz * 100) - 1 / 2) * 2 * torch.pi
            n_spiral = ktraj.shape[2] // 8
            ktraj = ktraj.reshape(3, -1, n_spiral)
            kx = []
            ky = []
            kz = []
            for i in range(ktraj.shape[1]):
                ky_ = torch.cos(ga_kx[i]) * ktraj[1:2, i:i+1, :] - torch.sin(ga_kx[i]) * ktraj[2:3, i:i+1, :]
                kz_ = torch.sin(ga_kx[i]) * ktraj[1:2, i:i+1, :] + torch.cos(ga_kx[i]) * ktraj[2:3, i:i+1, :]

                kx_ = torch.cos(ga_ky[i]) * ktraj[0:1, i:i+1, :] + torch.sin(ga_ky[i]) * kz_
                kz_ = -torch.sin(ga_ky[i]) * ktraj[0:1, i:i+1, :] + torch.cos(ga_ky[i]) * kz_

                kx_ = torch.cos(ga_kz[i]) * kx_ - torch.sin(ga_kz[i]) * ky_
                ky_ = torch.sin(ga_kz[i]) * kx_ + torch.cos(ga_kz[i]) * ky_

                kx.append(kx_)
                ky.append(ky_)
                kz.append(kz_)

            kx = torch.cat(kx, 1)
            ky = torch.cat(ky, 1)
            kz = torch.cat(kz, 1)

            ktraj = torch.cat((kx, ky, kz), dim=0)
            del kx, ky, kz
            torch.cuda.empty_cache()
            ktraj = ktraj.reshape(3, nTR, -1)

            dcomp = tkbn.calc_density_compensation_function(ktraj=ktraj.reshape(3, -1), im_size=self.im_size,
                                                            numpoints=self.nufft_numpoints,
                                                            grid_size=tuple(
                                                                [int(dim * self.nufft_grid_size) for dim in self.im_size]))
            ############ Normalize dcomp ###############
            dcomp = dcomp / torch.linalg.norm(dcomp.ravel(), ord=float('inf'))
            dcomp = dcomp.detach()
            ###
            nufft_split_batch = False
            nufft_split_channel = True
            if nufft_split_batch:
                Nop = NuSense_om(mps.view(1, n_c, 230, 230, 230),
                                 ktraj.reshape(1, 3, -1),
                                 numpoints=self.nufft_numpoints,
                                 grid_size=self.nufft_grid_size)
            elif nufft_split_channel:
                Nop = [NuSense_om(mps[i].view((1, 1,) + im_size).expand(n_basis, -1, -1, -1, -1),
                               ktraj.reshape(1, 3, -1),
                               numpoints=self.nufft_numpoints,
                               grid_size=self.nufft_grid_size) for i in range(n_c)]
            else:
                Nop = NuSense_om(mps.view(1, n_c, 230, 230, 230).expand(n_basis, -1, -1, -1, -1),
                                 ktraj.reshape(1, 3, -1),
                                 numpoints=self.nufft_numpoints,
                                 grid_size=self.nufft_grid_size)
            del mps, ktraj
            torch.cuda.empty_cache()
            x, k_proj, k_scale, recon_scale = checkpoint(gtBasisMap2UnderBasisMap, x, Nop, basis, dcomp, use_reentrant=False)
            torch.cuda.empty_cache()
        for i in range(self.n_unroll):
            if self.train_mag:
                phase = torch.angle(x)
                x = torch.abs(x)
            if self.in_channels == 2 * n_basis or n_basis:
                x_list = []
                if self.rotation_angle:
                    n_batch = 10
                else:
                    n_batch = 10
                for j in range(0, im_size[0], n_batch):
                    if self.checkpointing:
                        x_list.append(checkpoint(self.unet_list[i], x[:, :, j:j+n_batch, :, :], use_reentrant=False))
                    else:
                        x_list.append(self.unet_list[i](x[:, :, j:j + n_batch, :, :]))
                    torch.cuda.empty_cache()
                x = torch.cat(x_list, dim=2)
                del x_list
                torch.cuda.empty_cache()
            elif self.in_channels == 2:
                x_list = []
                n_batch = 10
                for j in range(0, im_size[0], n_batch):
                    if self.checkpointing:
                        x_list.append(checkpoint(self.unet_list[i], x[:, :, j:j+n_batch, :, :], use_reentrant=False))
                    else:
                        x_list.append(self.unet_list[i](x[:, :, j:j + n_batch, :, :]))
                x = torch.cat(x_list, dim=2)
            x = x * recon_basis_norm
            x = x / (recon_basis_norm + 1e-20)
            if self.train_mag:
                x = x * torch.exp(1j * phase)
            if self.dc:
                dc_step_size = torch.sigmoid(self.dc_step_size[i] * self.dc_step_size_sigmoid_slope)
                if self.dataset == "deli_cs":
                    llr_scale = torch.sigmoid(self.llr_scale * self.llr_scale_sigmoid_slope)
                    x = checkpoint(DC, x, Nop, basis, dcomp, k_proj, k_scale, recon_scale, dc_step_size,
                                   llr_scale, use_reentrant=False)   # (n_b, n_basis, nx, ny, nz)
                else:
                    llr_scale = None
                    x = checkpoint(DC, x, Nop, basis, dcomp, k_proj, k_scale, recon_scale, dc_step_size,
                                   llr_scale, use_reentrant=False)   # (n_b, n_basis, nx, ny, nz)
        del Nop, dcomp, k_proj
        torch.cuda.empty_cache()
        if self.est_p:
            ### parameter reconstruction
            if self.pretrain_FCNN:
                x_flatten = dp.normalize_basis_coeff(x, basis)
                x_flatten = torch.view_as_real(x_flatten.reshape(5, -1).permute(1, 0)).reshape(-1, n_basis * 2)  # (nx * ny * nz, n_basis * 2)
                p_hat = []
                for i in range(0, x_flatten.shape[0], self.fcnn_batch_size):
                    if self.checkpointing:
                        p_hat.append(checkpoint(self.fcnn, x_flatten[i: i + self.fcnn_batch_size], use_reentrant=False))
                    else:
                        p_hat.append(self.fcnn(x_flatten[i: i + self.fcnn_batch_size]))
                del x_flatten
                p_hat = torch.cat(p_hat).permute(1, 0).reshape((2,) + im_size)
                max_T1_T2 = torch.tensor([3817, 1442]).to(p_hat)
                p_hat = p_hat / max_T1_T2.view(2, 1, 1, 1)
                return x, p_hat
            else:
                x_flatten = x
                x_flatten = torch.view_as_real(x_flatten.reshape(n_basis, -1).permute(1, 0)).reshape(-1, n_basis * 2)   # (nx * ny * nz, n_basis * 2)
                p_hat = []
                for i in range(0, x_flatten.shape[0], self.fcnn_batch_size):
                    if self.checkpointing:
                        p_hat.append(checkpoint(self.fcnn, x_flatten[i: i + self.fcnn_batch_size], use_reentrant=False))
                    else:
                        p_hat.append(self.fcnn(x_flatten[i: i + self.fcnn_batch_size]))
                del x_flatten
                p_hat = torch.cat(p_hat).permute(1, 0).reshape((2,) + im_size)
                return x, p_hat
        else:
            return x, basis


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, norm=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        n_f = 128
        self.inc = (DoubleConv(n_channels, n_f, batch_norm=False))
        self.down1 = (Down(n_f, n_f * 2))
        self.down2 = (Down(n_f * 2, n_f * 2 ** 2))
        self.down3 = (Down(n_f * 2 ** 2, n_f * 2 ** 3))
        factor = 2 if bilinear else 1
        self.up2 = (Up(n_f * 2 ** 3, n_f * 2 ** 2 // factor, bilinear))
        self.up3 = (Up(n_f * 2 ** 2, n_f * 2 // factor, bilinear))
        self.up4 = (Up(n_f * 2, n_f, bilinear))
        self.outc = (OutConv(n_f, n_classes))
        self.norm = norm

    def forward(self, x):
        n_batch, n_basis, nx, ny, nz = x.shape
        complex_input = x.is_complex()
        if complex_input:
            if self.n_channels == 2:
                x0 = torch.view_as_real(x).permute(0, 5, 1, 2, 3, 4).reshape(2, n_basis * nx, ny, nz)
            else:
                x0 = torch.view_as_real(x).permute(0, 5, 1, 2, 3, 4).reshape(2 * n_basis, nx, ny, nz)
        else:
            x0 = x.reshape(n_basis, nx, ny, nz)
        if self.norm:
            if self.n_channels == 2:
                mean = torch.mean(x0, dim=(2, 3), keepdim=True)  # (2, n_basis * nx, 1, 1)
                std = torch.std(x0, dim=(2, 3), keepdim=True)  # (2, n_basis * nx, 1, 1)
                x0 = (x0 - mean) / (std + 1e-10)  # (2, n_basis * nx, ny, nz)
            else:
                mean = torch.mean(x0, dim=(2, 3), keepdim=True)  # (2 * n_basis, nx, 1, 1)
                std = torch.std(x0, dim=(2, 3), keepdim=True)
                x0 = (x0 - mean) / (std + 1e-10)
        x0 = x0.permute(1, 0, 2, 3)
        x1 = self.inc(x0)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up2(x4, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)  # (230, 10, 230, 230)
        if self.norm:
            if self.n_channels == 2:
                x = x.permute(1, 0, 2, 3)  # (2, n_basis * nx, ny, nz)
                x = x * std + mean
            else:
                x = x.permute(1, 0, 2, 3)  # (2 * n_basis, nx, ny, nz)
                x = x * std + mean
                x = x.permute(1, 0, 2, 3)  # (nx, 2 * n_basis, ny, nz)
        if complex_input:
            if self.n_channels == 2:
                x = torch.view_as_complex(
                    x.view(2, n_basis, nx, ny, nz).permute(1, 2, 3, 4, 0).contiguous()).unsqueeze(0)  # (n_b, n_basis, nx, ny, nz)
            else:
                x = torch.view_as_complex(
                    x.view(nx, 2, n_basis, ny, nz).permute(2, 0, 3, 4, 1).contiguous()).unsqueeze(0)  # (n_b, n_basis, nx, ny, nz)
        else:
            x = x.permute(1, 0, 2, 3).unsqueeze(0)
        return x


def UndersampleBasisImage4one_channel(Nop, recon_basis, basis, dcomp):
    # recon_basis: (1, n_basis=5, nx, ny, nz)
    n_basis, nTR = basis.shape
    n_c = Nop.size_out[1]
    recon_basis_in_k = Nop(recon_basis[0].unsqueeze(1))
    torch.cuda.empty_cache()
    recon_basis_in_k = recon_basis_in_k.view(n_basis, n_c, nTR, -1)  # (n_basis, n_c, nTR, n_group * n_ro)
    _temp = torch.sum(recon_basis_in_k * basis.view(n_basis, 1, nTR, 1), dim=0, keepdim=True)  # (1, n_c, nTR, n_group * n_ro)
    del recon_basis_in_k
    torch.cuda.empty_cache()
    _temp = torch.conj(basis).view(n_basis, 1, nTR, 1) * _temp  # (n_basis, n_c, nTR, n_group * n_ro)
    del basis
    torch.cuda.empty_cache()
    g_DC = Nop.H(dcomp * _temp.view(n_basis, n_c, -1))  # (n_basis, 1, nx, ny, nz)
    del dcomp
    torch.cuda.empty_cache()
    return _temp, g_DC


def gtBasisMap2UnderBasisMap(gt_basis, Nop, basis, dcomp, SNR=50):
    n_basis, nTR = basis.shape
    torch.random.manual_seed(666)
    # 0.07 is the mean signal in the MRF image-t of kirby21 01
    # We want the noise on the basis maps projected to the time domain to be what we want (sigma/SNR).
    sigma = 0.07 / SNR / 2 ** 0.5 * (nTR / n_basis) ** 0.5
    gt_basis = gt_basis + torch.randn_like(gt_basis) * sigma + 1j * torch.randn_like(gt_basis) * sigma
    torch.cuda.empty_cache()
    if isinstance(Nop, list):
        n_c = 8
        recon_basis = 0
        temp = []
        for i in range(n_c):
            _temp, _recon_basis = checkpoint(UndersampleBasisImage4one_channel, Nop[i], gt_basis, basis,
                                     dcomp, use_reentrant=False)  # recon_basis: (1, n_basis, nx, ny, nz)
            recon_basis = recon_basis + _recon_basis
            temp.append(_temp)
            del _recon_basis
            torch.cuda.empty_cache()
        temp = torch.cat(temp, dim=1)
        torch.cuda.empty_cache()
        recon_basis_imag_norm = torch.linalg.norm(recon_basis)
        gt_basis_imag_norm = 3400
        recon_scale = gt_basis_imag_norm / recon_basis_imag_norm
        recon_basis = recon_basis * recon_scale
        k_scale = torch.ones_like(recon_scale)
        return recon_basis.unsqueeze(0).squeeze(2), temp, k_scale.detach(), recon_scale.detach()
    elif Nop.size_in[0] == 1:
        recon_basis_in_k = torch.cat(
            [checkpoint(Nop, gt_basis[0, i:i + 1].unsqueeze(1), use_reentrant=False) for i in
             range(n_basis)])
    elif Nop.size_in[0] == n_basis:
        recon_basis_in_k = Nop * gt_basis[0].unsqueeze(1)
    del gt_basis
    torch.cuda.empty_cache()
    n_c = Nop.size_out[1]
    recon_basis_in_k = recon_basis_in_k.view(n_basis, n_c, nTR, -1)    # (n_basis, n_c, nTR, n_ro)
    _temp = torch.sum(recon_basis_in_k * basis.view(n_basis, 1, nTR, 1), dim=0, keepdim=True)  # (1, n_c, nTR, n_ro)
    del recon_basis_in_k
    torch.cuda.empty_cache()
    k_scale = 1 / torch.linalg.norm(_temp)
    _temp = _temp * k_scale
    _temp = torch.conj(basis).view(n_basis, 1, nTR, 1) * _temp  # (n_basis, n_c, nTR, n_ro)
    del basis
    torch.cuda.empty_cache()
    _temp = _temp.view(n_basis, n_c, -1)
    if Nop.size_in[0] == 1:
        recon_basis = torch.cat(
            [checkpoint(Nop.H, dcomp * _temp.view(n_basis, n_c, -1)[i:i + 1], use_reentrant=False) for i in
             range(n_basis)])  # (n_basis, 1, nx, ny, nz)
    elif Nop.size_in[0] == n_basis:
        recon_basis = Nop.H * (dcomp * _temp)  # (n_basis, 1, nx, ny, nz)
    del dcomp, Nop
    torch.cuda.empty_cache()
    recon_basis_imag_norm = torch.linalg.norm(recon_basis)
    gt_basis_imag_norm = 3400
    recon_scale = gt_basis_imag_norm / recon_basis_imag_norm
    recon_basis = recon_basis * recon_scale
    return recon_basis.unsqueeze(0).squeeze(2), _temp, k_scale.detach(), recon_scale.detach()  # (n_b=1, n_basis, nx, ny, nz)
