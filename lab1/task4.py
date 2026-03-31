import numpy as np
import matplotlib.pyplot as plt


def off_diagonal_frobenius(a: np.ndarray) -> float:
    diag_only = np.diag(np.diag(a))
    return float(np.linalg.norm(a - diag_only, ord="fro"))


def jacobi_eigen_solver(a: np.ndarray, eps: float = 1e-9, max_iter: int = 100):
    """Метод вращений Якоби для симметричной матрицы."""
    a = np.array(a, dtype=float, copy=True)
    n = a.shape[0]

    if a.ndim != 2 or a.shape[1] != n:
        raise ValueError("Матрица должна быть квадратной")
    if not np.allclose(a, a.T):
        raise ValueError("Метод вращений Якоби применим к симметричным матрицам")

    vectors = np.eye(n, dtype=float)
    history = [off_diagonal_frobenius(a)]

    for _ in range(max_iter):
        # Ищем самый крупный внедиагональный элемент
        p, q = 0, 1
        best = abs(a[p, q])
        for i in range(n - 1):
            for j in range(i + 1, n):
                cand = abs(a[i, j])
                if cand > best:
                    best = cand
                    p, q = i, j

        if best < eps:
            break

        if a[p, p] == a[q, q]:
            phi = np.pi / 4
        else:
            phi = 0.5 * np.arctan2(2 * a[p, q], a[p, p] - a[q, q])

        c = np.cos(phi)
        s = np.sin(phi)

        rotation = np.eye(n)
        rotation[p, p] = c
        rotation[q, q] = c
        rotation[p, q] = -s
        rotation[q, p] = s

        a = rotation.T @ a @ rotation
        vectors = vectors @ rotation
        history.append(off_diagonal_frobenius(a))

        if history[-1] < eps:
            break

    eigenvalues = np.diag(a).copy()
    return eigenvalues, vectors, history


def check_eigenpairs(a: np.ndarray, eigenvalues: np.ndarray, eigenvectors: np.ndarray):
    checks = []
    n = eigenvalues.size

    for i in range(n):
        lam = eigenvalues[i]
        vec = eigenvectors[:, i]

        left = a @ vec
        right = lam * vec
        abs_err = np.linalg.norm(left - right)

        denom = np.linalg.norm(right)
        rel_err = abs_err / denom if denom > 1e-15 else abs_err

        checks.append((lam, abs_err, rel_err))

        print(f"\nСобственная пара #{i + 1}:")
        print(f"  λ = {lam:.6f}")
        print("  v =", np.array2string(vec, precision=6, suppress_small=True))
        print("  A·v =", np.array2string(left, precision=6, suppress_small=True))
        print("  λ·v =", np.array2string(right, precision=6, suppress_small=True))

    return checks


def main():
    a_mat = np.array([[2, -9, 4], [-9, 0, 9], [4, 9, 6]], dtype=float)

    eigenvalues, eigenvectors, err_history = jacobi_eigen_solver(a_mat, eps=1e-9, max_iter=100)

    print(f"Собственные значения:\n{eigenvalues}")
    print(f"\nСобственные векторы (столбцы):\n{eigenvectors}")
    print(f"\nИтоговая погрешность: {err_history[-1]:.2e}")

    print(f"{'Итерация':<10} | {'Погрешность':<15}")
    print("-" * 30)
    for i, err in enumerate(err_history):
        print(f"{i:<10} | {err:.2e}")

    check_eigenpairs(a_mat, eigenvalues, eigenvectors)

    plt.figure(figsize=(8, 5))
    plt.plot(range(len(err_history)), err_history, "o-r", label="Текущая ошибка")
    plt.axhline(y=1e-9, color="gray", linestyle="--", label="Порог eps")
    plt.yscale("log")
    plt.xlabel("Номер итерации (поворота)")
    plt.ylabel("Погрешность (log scale)")
    plt.title("Сходимость метода вращений Якоби")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
