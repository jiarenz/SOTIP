import torch
from mirtorch.linear.linearmaps import LinearMap


class BasisProj(LinearMap):
    """
    Linear operator projecting time domain signal to its bases, returns corresponding coefficients.
    Attributes:
        dim: assign the dimension to apply operation
        basis: tensor of shape (n_basis, nt)
    """

    def __init__(self,
                 size_in,
                 basis):
        self.im_size = size_in[1:]
        self.nt = basis.shape[1]
        self.n_basis = basis.shape[0]
        self.size_out = (self.nt, 1) + self.im_size
        super(BasisProj, self).__init__(size_in, self.size_out)
        self.A = torch.transpose(basis, 0, 1)  # (nt, n_basis)

    def _apply(self, coeff):
        """
        coeff: tensor of shape (n_basis, nx, ny, nz)
        """
        n_basis = coeff.shape[0]
        coeff = coeff.view(n_basis, -1)
        y = torch.mm(self.A, coeff)  # (nt, nx*ny*nz)
        y = y.view(self.size_out)
        return y

    def _apply_adjoint(self, image):
        """
        image: tensor of shape (nt, 1, nx, ny, nz)
        """
        image = image.view(self.nt, -1)  # (nt, nx*ny*nz)
        y = torch.mm(self.A.H, image)  # (n_basis, nx*ny*nz)
        y = y.view((self.n_basis,) + self.im_size)
        return y


def spatial_normalize_basis_coeff(basis_coeff, basis, weight=None):
    """
    weight: (n_batch, n_basis)
    """
    n_basis = basis.shape[0]
    basis_norm = torch.norm(basis, dim=1).view(1, n_basis, 1, 1, 1)
    basis_coeff = basis_coeff * basis_norm
    if weight is not None:
        basis_coeff = basis_coeff * weight.view(1, n_basis, 1, 1, 1)
    return basis_coeff


def normalize_basis_coeff(basis_coeff, basis):
    n_basis = basis.shape[0]
    basis_norm = torch.norm(basis, dim=1).view(1, n_basis, 1, 1, 1)
    basis_coeff_norm = basis_coeff * basis_norm
    basis_coeff_norm = basis_coeff_norm + 1e-10
    recon_basis_norm = torch.sum(torch.abs(basis_coeff_norm) ** 2, axis=1, keepdims=True) ** 0.5
    basis_coeff_norm = basis_coeff_norm / recon_basis_norm
    basis_coeff_norm[basis_coeff == 0] = 0
    return basis_coeff_norm