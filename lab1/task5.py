import numpy as np


def householder_qr(a: np.ndarray):
    """QR-разложение отражениями Хаусхолдера."""
    r = np.array(a, dtype=float, copy=True)
    n = r.shape[0]

    if r.ndim != 2 or r.shape[1] != n:
        raise ValueError("Матрица должна быть квадратной")

    q = np.eye(n, dtype=float)

    for k in range(n - 1):
        x = r[k:, k]
        norm_x = np.linalg.norm(x)
        if norm_x == 0:
            continue

        sign = 1.0 if x[0] >= 0 else -1.0
        v = x.copy()
        v[0] += sign * norm_x
        v_norm = np.linalg.norm(v)
        if v_norm == 0:
            continue
        v /= v_norm

        r[k:, k:] -= 2.0 * np.outer(v, v @ r[k:, k:])
        q[:, k:] -= 2.0 * np.outer(q[:, k:] @ v, v)

    return q, r


def lower_triangle_norm(a: np.ndarray):
    n = a.shape[0]
    total = 0.0
    for i in range(1, n):
        for j in range(i):
            total += a[i, j] ** 2
    return float(np.sqrt(total))


def extract_eigenvalues_from_quasitriangular(t: np.ndarray, eps: float):
    n = t.shape[0]
    eigenvalues = []
    i = 0

    while i < n:
        if i == n - 1 or abs(t[i + 1, i]) < eps:
            eigenvalues.append(t[i, i])
            i += 1
            continue

        block = t[i : i + 2, i : i + 2]
        trace = block[0, 0] + block[1, 1]
        det = block[0, 0] * block[1, 1] - block[0, 1] * block[1, 0]
        disc = trace * trace - 4.0 * det

        if disc >= 0:
            root = np.sqrt(disc)
            eigenvalues.extend([(trace + root) / 2.0, (trace - root) / 2.0])
        else:
            root = np.sqrt(-disc) * 1j
            eigenvalues.extend([(trace + root) / 2.0, (trace - root) / 2.0])

        i += 2

    return eigenvalues


def qr_algorithm(a: np.ndarray, eps: float = 1e-6, max_iter: int = 1000):
    t = np.array(a, dtype=float, copy=True)

    for _ in range(max_iter):
        if lower_triangle_norm(t) < eps:
            break

        q, r = householder_qr(t)
        t = r @ q

    return extract_eigenvalues_from_quasitriangular(t, eps)


def main():
    a = np.array(
        [
            [-1, 8, 5],
            [8, -4, 4],
            [2, 9, -2],
        ],
        dtype=float,
    )

    eigenvals = qr_algorithm(a, eps=1e-6)

    print("Собственные значения, найденные QR-алгоритмом:")
    for i, val in enumerate(eigenvals, start=1):
        print(f"λ{i} = {val:.6f}")

    np_eigenvals = np.linalg.eigvals(a)
    print("\nСобственные значения от numpy.linalg.eigvals:")
    for i, val in enumerate(np_eigenvals, start=1):
        print(f"λ{i} = {val:.6f}")


if __name__ == "__main__":
    main()
