import numpy as np


def lu_factorize(a: np.ndarray, tol: float = 1e-12):
    """LU-разложение с частичным выбором главного элемента по столбцу."""
    a = np.array(a, dtype=float, copy=True)
    n = a.shape[0]

    if a.ndim != 2 or a.shape[1] != n:
        raise ValueError("Матрица должна быть квадратной")

    l = np.eye(n, dtype=float)
    u = a.copy()
    p = np.arange(n)
    sign = 1

    for col in range(n - 1):
        pivot = col + int(np.argmax(np.abs(u[col:, col])))
        if abs(u[pivot, col]) < tol:
            raise ValueError("Матрица вырождена или близка к вырожденной")

        if pivot != col:
            u[[col, pivot], :] = u[[pivot, col], :]
            p[[col, pivot]] = p[[pivot, col]]
            if col > 0:
                l[[col, pivot], :col] = l[[pivot, col], :col]
            sign *= -1

        pivot_val = u[col, col]
        for row in range(col + 1, n):
            mult = u[row, col] / pivot_val
            l[row, col] = mult
            u[row, col:] -= mult * u[col, col:]

    if abs(u[-1, -1]) < tol:
        raise ValueError("Матрица вырождена или близка к вырожденной")

    return l, u, p, sign


def forward_substitution(l: np.ndarray, b: np.ndarray):
    n = l.shape[0]
    y = np.zeros(n, dtype=float)
    for i in range(n):
        y[i] = b[i] - np.dot(l[i, :i], y[:i])
    return y


def backward_substitution(u: np.ndarray, y: np.ndarray):
    n = u.shape[0]
    x = np.zeros(n, dtype=float)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(u[i, i + 1 :], x[i + 1 :])) / u[i, i]
    return x


def solve_via_lu(l: np.ndarray, u: np.ndarray, p: np.ndarray, b: np.ndarray):
    b_permuted = np.asarray(b, dtype=float)[p]
    y = forward_substitution(l, b_permuted)
    return backward_substitution(u, y)


def inverse_from_lu(l: np.ndarray, u: np.ndarray, p: np.ndarray):
    n = l.shape[0]
    inv = np.empty((n, n), dtype=float)
    for j in range(n):
        e = np.zeros(n, dtype=float)
        e[j] = 1.0
        inv[:, j] = solve_via_lu(l, u, p, e)
    return inv


def main():
    a = np.array(
        [
            [7.0, -5.0, 6.0, 7.0],
            [8.0, -1.0, -9.0, 1.0],
            [-3.0, 8.0, 8.0, 8.0],
            [2.0, -3.0, 6.0, -4.0],
        ],
        dtype=float,
    )
    b = np.array([120.0, 31.0, 6.0, 25.0], dtype=float)

    l, u, p, sign = lu_factorize(a)
    x = solve_via_lu(l, u, p, b)
    determinant = sign * float(np.prod(np.diag(u)))
    a_inv = inverse_from_lu(l, u, p)

    print("Матрица L (нижняя треугольная):")
    print(l)
    print("\nМатрица U (верхняя треугольная):")
    print(u)
    print(f"\nВектор перестановок: {p}")
    print("-" * 30)
    print(f"Решение системы x: {x}")
    print(f"Определитель det(A): {determinant:.1f}")
    print("\nОбратная матрица A^-1:")
    print(np.round(a_inv, 4))


if __name__ == "__main__":
    main()
