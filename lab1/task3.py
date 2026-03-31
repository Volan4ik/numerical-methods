import numpy as np


def build_iteration_form(a: np.ndarray, b: np.ndarray):
    """Переход к x = Bx + c для методов Якоби/Зейделя."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("Матрица A должна быть квадратной")
    if b.size != a.shape[0]:
        raise ValueError("Размерность b не согласована с A")

    diag = np.diag(a)
    if np.any(diag == 0):
        raise ValueError("На диагонали A есть нулевой элемент")

    n = a.shape[0]
    b_mat = np.zeros((n, n), dtype=float)
    c_vec = b / diag

    for i in range(n):
        for j in range(n):
            if i != j:
                b_mat[i, j] = -a[i, j] / diag[i]

    return b_mat, c_vec


def jacobi_method(a: np.ndarray, b: np.ndarray, eps: float = 1e-6, max_iter: int = 1000):
    b_mat, c_vec = build_iteration_form(a, b)
    norm_b = np.linalg.norm(b_mat, ord=np.inf)

    x_prev = c_vec.copy()
    history = []

    for it in range(1, max_iter + 1):
        x_next = c_vec + b_mat @ x_prev
        delta = np.linalg.norm(x_next - x_prev, ord=np.inf)

        if norm_b < 1:
            error_est = norm_b * delta / (1 - norm_b)
        else:
            error_est = delta

        history.append(error_est)
        if error_est < eps:
            return x_next, it, history

        x_prev = x_next

    return x_prev, max_iter, history


def gauss_seidel_method(a: np.ndarray, b: np.ndarray, eps: float = 1e-6, max_iter: int = 1000):
    b_mat, c_vec = build_iteration_form(a, b)
    n = c_vec.size

    x_prev = c_vec.copy()
    history = []

    for it in range(1, max_iter + 1):
        x_next = x_prev.copy()

        for i in range(n):
            left = np.dot(b_mat[i, :i], x_next[:i])
            right = np.dot(b_mat[i, i + 1 :], x_prev[i + 1 :])
            x_next[i] = c_vec[i] + left + right

        error_est = np.linalg.norm(x_next - x_prev, ord=np.inf)
        history.append(error_est)

        if error_est < eps:
            return x_next, it, history

        x_prev = x_next

    return x_prev, max_iter, history


def main():
    a = np.array(
        [
            [-18, 9, -1, -8],
            [6, 22, 9, 0],
            [-4, 2, -16, 9],
            [1, 6, -1, -14],
        ],
        dtype=float,
    )
    b = np.array([-60, -109, -103, -33], dtype=float)

    x_exact = np.linalg.solve(a, b)
    print("Точное решение (np.linalg.solve):", x_exact)

    eps = 1e-6
    x_jacobi, jacobi_iters, _ = jacobi_method(a, b, eps=eps)
    print("\nМетод простых итераций (Якоби):")
    print("Приближенное решение:", x_jacobi)
    print("Число итераций:", jacobi_iters)
    print("Невязка (max|x - x_exact|):", np.max(np.abs(x_jacobi - x_exact)))

    x_seidel, seidel_iters, _ = gauss_seidel_method(a, b, eps=eps)
    print("\nМетод Зейделя:")
    print("Приближенное решение:", x_seidel)
    print("Число итераций:", seidel_iters)
    print("Невязка (max|x - x_exact|):", np.max(np.abs(x_seidel - x_exact)))

    print("\nАнализ зависимости числа итераций от точности:")
    for eps_test in (1e-3, 1e-6, 1e-9):
        _, jacobi_n, _ = jacobi_method(a, b, eps=eps_test, max_iter=10000)
        _, seidel_n, _ = gauss_seidel_method(a, b, eps=eps_test, max_iter=10000)
        print(f"eps = {eps_test:.0e}: Якоби = {jacobi_n}, Зейдель = {seidel_n}")


if __name__ == "__main__":
    main()
