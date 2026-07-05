"""
Multigrid utilities for AMG operations

Includes:
    - Coarse grid operator computation
    - Relaxation (smoothing) matrices
    - Two-grid convergence analysis
    - Loss functions for P-value prediction
"""

import torch
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
from scipy.sparse.linalg import inv as sparse_inv, eigsh
from typing import Tuple, Optional


def compute_coarse_grid_operator(A: np.ndarray, P: np.ndarray, R: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute coarse grid operator A_c = R * A * P

    Parameters:
        A: fine grid operator [n, n]
        P: prolongation matrix [n, n_c]
        R: restriction matrix [n_c, n] (default: R = P^T)

    Returns:
        A_c: coarse grid operator [n_c, n_c]
    """
    if R is None:
        R = P.T

    A_c = R @ A @ P
    return A_c


def jacobi_relaxation_matrix(A: np.ndarray, omega: float = 0.8) -> np.ndarray:
    """
    Compute Jacobi relaxation matrix S = I - omega * D^{-1} * A

    Parameters:
        A: system matrix [n, n]
        omega: damping parameter

    Returns:
        S: Jacobi iteration matrix [n, n]
    """
    n = A.shape[0]
    I = np.eye(n)

    # Extract diagonal
    D = np.diag(np.diag(A))
    D_inv = np.linalg.inv(D)

    S = I - omega * D_inv @ A
    return S


def gauss_seidel_relaxation_matrix(A: np.ndarray) -> np.ndarray:
    """
    Compute Gauss-Seidel relaxation matrix S = -(D+L)^{-1} * U

    Parameters:
        A: system matrix [n, n]

    Returns:
        S: Gauss-Seidel iteration matrix [n, n]
    """
    # Split A into L, D, U
    L = np.tril(A, k=-1)  # Lower triangular
    D = np.diag(np.diag(A))  # Diagonal
    U = np.triu(A, k=1)  # Upper triangular

    # Compute -(D+L)^{-1} * U
    DL = D + L
    DL_inv = np.linalg.inv(DL)
    S = -DL_inv @ U

    return S


def two_grid_error_matrix(A: np.ndarray, P: np.ndarray, S: np.ndarray,
                          R: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute two-grid error propagation matrix M = S * C * S

    where C = I - P * A_c^{-1} * R * A is the coarse correction operator

    Parameters:
        A: fine grid operator [n, n]
        P: prolongation matrix [n, n_c]
        S: smoother iteration matrix [n, n]
        R: restriction matrix [n_c, n] (default: R = P^T)

    Returns:
        M: two-grid error matrix [n, n]
    """
    n = A.shape[0]
    I = np.eye(n)

    if R is None:
        R = P.T

    # Compute coarse grid operator
    A_c = compute_coarse_grid_operator(A, P, R)

    # Invert coarse grid operator
    A_c_inv = np.linalg.inv(A_c)

    # Compute coarse correction operator C = I - P * A_c^{-1} * R * A
    C = I - P @ A_c_inv @ R @ A

    # Two-grid error matrix: M = S * C * S
    M = S @ C @ S

    return M


def convergence_factor_frobenius(M: np.ndarray) -> float:
    """
    Compute Frobenius norm of error matrix ||M||_F

    This is a differentiable approximation to the spectral radius

    Parameters:
        M: error propagation matrix [n, n]

    Returns:
        Frobenius norm
    """
    return np.linalg.norm(M, ord='fro')


def convergence_factor_spectral(M: np.ndarray) -> float:
    """
    Compute spectral radius rho(M) = max|lambda_i(M)|

    This is the actual convergence factor but is not differentiable

    Parameters:
        M: error propagation matrix [n, n]

    Returns:
        Spectral radius
    """
    eigenvalues = np.linalg.eigvals(M)
    return np.max(np.abs(eigenvalues))


class TwoGridLoss(torch.nn.Module):
    """
    Differentiable two-grid convergence loss for training P-value predictor

    Computes loss based on the two-grid error matrix:
        M = S * C * S
    where C = I - P * A_c^{-1} * R * A

    This implementation uses PyTorch operations to maintain differentiability.

    Loss can be:
        - Frobenius norm: ||M||_F (differentiable, for training)
        - Spectral radius: rho(M) (for evaluation, not differentiable)
    """

    def __init__(self, loss_type: str = 'frobenius', epsilon: float = 1e-4):
        """
        Initialize loss

        Parameters:
            loss_type: 'frobenius' or 'spectral'
            epsilon: Regularization for numerical stability (default: 1e-4)
        """
        super().__init__()
        self.loss_type = loss_type
        self.epsilon = epsilon

    def two_grid_error_matrix_torch(self, A, P, S, R=None):
        """
        Compute two-grid error matrix using PyTorch operations (differentiable)

        Args:
            A: system matrix [n, n] (torch.Tensor or numpy)
            P: prolongation matrix [n, n_c] (torch.Tensor with gradients!)
            S: smoother matrix [n, n] (torch.Tensor or numpy)
            R: restriction matrix [n_c, n] (default: P^T)

        Returns:
            M: two-grid error matrix [n, n] (torch.Tensor)
        """
        # Determine device from P (which should be a tensor with gradients)
        if isinstance(P, torch.Tensor):
            device = P.device
            dtype = P.dtype
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            dtype = torch.float32

        # Convert to torch tensors if needed (but keep P's gradients!)
        if not isinstance(A, torch.Tensor):
            A = torch.from_numpy(A).to(dtype).to(device)
        if not isinstance(P, torch.Tensor):
            P = torch.from_numpy(P).to(dtype).to(device)
        if not isinstance(S, torch.Tensor):
            S = torch.from_numpy(S).to(dtype).to(device)

        n = A.shape[0]
        I = torch.eye(n, device=device, dtype=dtype)

        # Restriction matrix (default: P^T)
        if R is None:
            R = P.t()
        elif not isinstance(R, torch.Tensor):
            R = torch.from_numpy(R).to(dtype).to(device)

        # Compute coarse grid operator: A_c = R * A * P
        A_c = R @ A @ P  # Shape: [n_c, n_c]

        # Add regularization to diagonal for numerical stability
        # Increased epsilon to handle ill-conditioned matrices during training
        n_c = A_c.shape[0]
        A_c_reg = A_c + self.epsilon * torch.eye(n_c, device=device, dtype=dtype)

        # Invert coarse grid operator with robust error handling
        try:
            # First try direct inversion
            A_c_inv = torch.linalg.inv(A_c_reg)
        except torch._C._LinAlgError:
            try:
                # If that fails, try with more regularization
                A_c_reg2 = A_c + (self.epsilon * 10) * torch.eye(n_c, device=device, dtype=dtype)
                A_c_inv = torch.linalg.inv(A_c_reg2)
            except torch._C._LinAlgError:
                try:
                    # Last resort: pseudo-inverse (slower but more stable)
                    # Add extra regularization before pseudo-inverse
                    A_c_reg3 = A_c + (self.epsilon * 100) * torch.eye(n_c, device=device, dtype=dtype)
                    A_c_inv = torch.linalg.pinv(A_c_reg3)
                except torch._C._LinAlgError:
                    # If all fails, use identity (no coarse correction)
                    # This prevents training from crashing
                    A_c_inv = torch.eye(n_c, device=device, dtype=dtype)
                    print("Warning: Using identity for A_c_inv due to numerical issues")

        # Compute coarse correction: C = I - P * A_c^{-1} * R * A
        C = I - P @ A_c_inv @ R @ A  # Shape: [n, n]

        # Two-grid error matrix: M = S * C * S
        M = S @ C @ S  # Shape: [n, n]

        return M

    def forward(self, A_list, P_list, S_list):
        """
        Compute loss for a batch

        Parameters:
            A_list: list of system matrices (numpy or torch)
            P_list: list of predicted prolongation matrices (numpy or torch)
            S_list: list of smoother matrices (numpy or torch)

        Returns:
            Average loss over batch (torch.Tensor, differentiable)
        """
        # FIX: Use list to collect losses, then stack and mean
        # This preserves the computation graph properly
        losses = []

        for A, P, S in zip(A_list, P_list, S_list):
            # Compute two-grid error matrix (all PyTorch operations)
            M = self.two_grid_error_matrix_torch(A, P, S)

            # Compute loss
            if self.loss_type == 'frobenius':
                # Frobenius norm: sqrt(sum of squared elements)
                # ||M||_F = sqrt(trace(M^T * M))
                loss = torch.norm(M, p='fro')  # Differentiable!
            else:  # spectral
                # Spectral radius (not differentiable, use for evaluation only)
                eigenvalues = torch.linalg.eigvals(M)
                loss = torch.max(torch.abs(eigenvalues)).real

            losses.append(loss)

        # Stack losses and compute mean (preserves gradients!)
        if len(losses) == 0:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            return torch.tensor(0.0, device=device, requires_grad=True)

        avg_loss = torch.stack(losses).mean()

        return avg_loss


def compute_two_grid_convergence(A: np.ndarray, P: np.ndarray, S: np.ndarray,
                                 num_iterations: int = 10) -> Tuple[float, np.ndarray]:
    """
    Compute two-grid convergence factor by power iteration

    Parameters:
        A: system matrix
        P: prolongation matrix
        S: smoother matrix
        num_iterations: number of power iterations

    Returns:
        Tuple of (convergence_factor, error_history)
    """
    n = A.shape[0]

    # Random initial error
    e = np.random.randn(n)
    e = e / np.linalg.norm(e)

    # Compute two-grid error matrix
    M = two_grid_error_matrix(A, P, S)

    # Power iteration
    error_norms = []
    for _ in range(num_iterations):
        e = M @ e
        error_norm = np.linalg.norm(e)
        error_norms.append(error_norm)

    # Estimate convergence factor from error reduction
    error_norms = np.array(error_norms)
    convergence_factor = np.exp(np.mean(np.log(error_norms[1:] / error_norms[:-1])))

    return convergence_factor, error_norms


def sparse_two_grid_error_matrix(A_sparse, P_sparse, S_sparse,
                                 R_sparse=None) -> csr_matrix:
    """
    Compute two-grid error matrix for sparse matrices

    Parameters:
        A_sparse: sparse system matrix
        P_sparse: sparse prolongation matrix
        S_sparse: sparse smoother matrix
        R_sparse: sparse restriction matrix (default: P^T)

    Returns:
        M: sparse two-grid error matrix
    """
    from scipy.sparse import eye, csr_matrix
    from scipy.sparse.linalg import spsolve

    n = A_sparse.shape[0]
    I = eye(n, format='csr')

    if R_sparse is None:
        R_sparse = P_sparse.T

    # Compute coarse grid operator A_c = R * A * P
    A_c = R_sparse @ A_sparse @ P_sparse

    # For sparse inversion, we compute P * (A_c \ (R * A))
    # which is more efficient than computing A_c^{-1} explicitly
    RA = R_sparse @ A_sparse

    # Solve A_c * X = RA for X
    # X will have shape [n_c, n]
    X = np.zeros((A_c.shape[0], n))
    for i in range(n):
        X[:, i] = spsolve(A_c, RA[:, i].toarray().flatten())

    # Coarse correction: C = I - P * X
    C = I - P_sparse @ csr_matrix(X)

    # Two-grid error: M = S * C * S
    M = S_sparse @ C @ S_sparse

    return M



'''
class TwoGridLoss(torch.nn.Module):
    """
    Differentiable two-grid convergence loss for training P-value predictor.

    Improvements / fixes over original:
    - Enforce that P (predicted prolongation) is a torch.Tensor with requires_grad=True.
    - Use torch.linalg.solve instead of explicit inverse when possible (more stable).
    - Robust fallback strategies with broader exception handling.
    - Correct tensor initializations for batch_loss (device/dtype safe).
    - Use eigvalsh for symmetric matrices when computing spectral radius.
    - Normalize Frobenius penalty optionally by matrix size (keeps scale stable).
    """

    def __init__(self, loss_type: str = 'frobenius', epsilon: float = 1e-6,
                 normalize_frob: bool = True):
        """
        Params:
            loss_type: 'frobenius' or 'spectral' (spectral only for evaluation; not differentiable)
            epsilon: small ridge added to coarse operator for numerical stability
            normalize_frob: if True, normalize ||M||_F by sqrt(n_entries)
        """
        super().__init__()
        if loss_type not in ('frobenius', 'spectral'):
            raise ValueError("loss_type must be 'frobenius' or 'spectral'")
        self.loss_type = loss_type
        self.epsilon = float(epsilon)
        self.normalize_frob = bool(normalize_frob)

    def two_grid_error_matrix_torch(self, A, P, S, R=None):
        """
        Compute two-grid error matrix M = S * C * S in a differentiable way.

        Requirements:
            - P must be a torch.Tensor and must have requires_grad=True (predicted P).
            - A, S, R may be numpy arrays or torch tensors; they will be converted to match P's device/dtype.
        """
        if not isinstance(P, torch.Tensor):
            raise RuntimeError("Predicted P must be a torch.Tensor. If you passed numpy, convert to torch.Tensor first.")
        
        if torch.is_grad_enabled() and (not P.requires_grad):
            raise RuntimeError("Predicted P must have requires_grad=True when gradients are enabled (training). "
                            "In evaluation (torch.no_grad()) P can be a tensor without gradients.")

        device = P.device
        dtype = P.dtype

        # Convert A, S, R to torch tensors on same device/dtype
        if not isinstance(A, torch.Tensor):
            A = torch.from_numpy(np.asarray(A)).to(dtype).to(device)
        else:
            A = A.to(dtype).to(device)

        if not isinstance(S, torch.Tensor):
            S = torch.from_numpy(np.asarray(S)).to(dtype).to(device)
        else:
            S = S.to(dtype).to(device)

        if R is None:
            R = P.t()
        else:
            if not isinstance(R, torch.Tensor):
                R = torch.from_numpy(np.asarray(R)).to(dtype).to(device)
            else:
                R = R.to(dtype).to(device)

        n = A.shape[0]
        I = torch.eye(n, device=device, dtype=dtype)

        # Compute coarse operator A_c = R * A * P
        A_c = R @ A @ P  # shape [n_c, n_c]

        # Regularize diagonal for stability
        n_c = A_c.shape[0]
        A_c_reg = A_c + (self.epsilon * torch.eye(n_c, device=device, dtype=dtype))

        # Instead of computing explicit inverse, solve A_c_reg @ X = R @ A  -> X = solve(A_c_reg, R@A)
        # This yields P @ X = P @ A_c_reg^{-1} @ R @ A without explicit inverse
        try:
            RA = R @ A  # shape [n_c, n]
            # torch.linalg.solve supports batched RHS; RA shape is OK
            X = torch.linalg.solve(A_c_reg, RA)
            C = I - P @ X  # [n, n]
        except Exception as e:
            # Fallback strategies: increase regularization, then try pseudo-inverse
            warnings.warn(f"Numerical issue solving for A_c inverse: {e}. "
                          "Applying stronger regularization / pseudo-inverse fallback.")
            try:
                A_c_reg2 = A_c + (self.epsilon * 10.0) * torch.eye(n_c, device=device, dtype=dtype)
                X = torch.linalg.solve(A_c_reg2, RA)
                C = I - P @ X
            except Exception:
                # Last resort: pseudo-inverse (preserves differentiability through pinv)
                try:
                    A_c_pinv = torch.linalg.pinv(A_c_reg2)
                    C = I - P @ (A_c_pinv @ RA)
                except Exception:
                    # If everything fails, return the identity correction (no coarse correction)
                    warnings.warn("All coarse inversion attempts failed. Using C = I (no coarse correction).")
                    C = I

        # Two-grid error matrix
        M = S @ C @ S
        return M

    def forward(self, A_list, P_list, S_list):
        """
        A_list, P_list, S_list: iterables of same length describing a batch.
        Returns mean loss over batch.
        """
        if len(A_list) == 0:
            # Return zero tensor on default device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            return torch.tensor(0., device=device)

        losses = []
        for A, P, S in zip(A_list, P_list, S_list):
            # Validate P is tensor with grad
            if not isinstance(P, torch.Tensor):
                raise RuntimeError("Predicted P must be a torch.Tensor. If you passed numpy, convert to torch.Tensor first.")

            if torch.is_grad_enabled() and (not P.requires_grad):
                raise RuntimeError("Predicted P must have requires_grad=True when gradients are enabled (training). "
                                "In evaluation (torch.no_grad()) P can be a tensor without gradients.")

            device = P.device
            dtype = P.dtype

            # Ensure A and S placed on same device/dtype
            if not isinstance(A, torch.Tensor):
                A_t = torch.from_numpy(np.asarray(A)).to(dtype).to(device)
            else:
                A_t = A.to(dtype).to(device)
            if not isinstance(S, torch.Tensor):
                S_t = torch.from_numpy(np.asarray(S)).to(dtype).to(device)
            else:
                S_t = S.to(dtype).to(device)

            # Compute two-grid error matrix M
            M = self.two_grid_error_matrix_torch(A_t, P, S_t)

            if self.loss_type == 'frobenius':
                frob = torch.norm(M, p='fro')
                if self.normalize_frob:
                    n_entries = torch.tensor(M.numel(), dtype=frob.dtype, device=frob.device)
                    frob = frob / torch.sqrt(n_entries)
                loss_val = frob
            else:  # spectral
                # Prefer symmetric eig solver if M is symmetric (numerically)
                try:
                    if torch.allclose(M, M.transpose(-2, -1), atol=1e-6):
                        eigs = torch.linalg.eigvalsh(M)
                        loss_val = torch.max(torch.abs(eigs))
                    else:
                        eigs = torch.linalg.eigvals(M)
                        # eigvals may be complex; take abs of complex numbers then max
                        loss_val = torch.max(torch.abs(eigs))
                        # ensure real scalar
                        loss_val = loss_val.real if hasattr(loss_val, 'real') else loss_val
                except Exception as e:
                    warnings.warn(f"Eigen decomposition failed: {e}. Falling back to Frobenius norm.")
                    # Fallback: use Frobenius
                    loss_val = torch.norm(M, p='fro')
            losses.append(loss_val)

        # Stack and mean in a way that preserves device/dtype
        loss_tensor = torch.stack(losses).mean()
        return loss_tensor
'''