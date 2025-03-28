import numpy as np
from memory_profiler import profile

# Set random seed for reproducibility
np.random.seed(0)


# Generate random data for A and B
def generate_data(n, m, p):
    A = np.random.rand(n, m)
    B = np.random.rand(m, n, p)
    return A, B


@profile
def test_einsum(A, B):
    pij = np.einsum("j,jil->il", A[:, 1], B)[:, 1]
    print(pij.shape)
    denom = np.einsum("j,jil->i", A[:, 1], B)
    diag_values = pij
    return pij, denom, diag_values


@profile
def test_matrix_multiplication(A, B):
    pij = A[:, 1] @ B.transpose((1, 0, 2))
    denom = pij.sum(1)
    diag_values = pij[:, 1]
    return pij, denom, diag_values


if __name__ == "__main__":
    n = 10000
    m = 10000
    p = 50

    A, B = generate_data(n, m, p)

    print("Testing einsum approach...")
    test_einsum(A, B)

    print("\nTesting matrix multiplication approach...")
    test_matrix_multiplication(A, B)
