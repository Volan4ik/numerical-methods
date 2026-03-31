import numpy as np


def thomas_algorithm(lower, diag, upper, rhs, tol: float = 1e-14):
    """Решение трехдиагональной СЛАУ методом прогонки."""
    a = np.asarray(lower, dtype=float)
    b = np.asarray(diag, dtype=float)
    c = np.asarray(upper, dtype=float)
    d = np.asarray(rhs, dtype=float)

    n = d.size
    if not (a.size == b.size == c.size == n):
        raise ValueError("Все диагонали и правая часть должны иметь одинаковую длину")

    cp = np.zeros(n, dtype=float)
    dp = np.zeros(n, dtype=float)

    if abs(b[0]) < tol:
        raise ValueError("Нулевой ведущий элемент в первой строке")

    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]

    for i in range(1, n):
        denom = b[i] - a[i] * cp[i - 1]
        if abs(denom) < tol:
            raise ValueError(f"Нулевой знаменатель на шаге {i}")

        cp[i] = c[i] / denom if i < n - 1 else 0.0
        dp[i] = (d[i] - a[i] * dp[i - 1]) / denom

    x = np.zeros(n, dtype=float)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]

    return x


def main():
    a = [0, 2, -7, 2, -7]
    b = [-14, 7, -18, -13, -7]
    c = [6, 0, -9, 2, 0]
    d = [82, -51, -46, 111, 35]

    solution = thomas_algorithm(a, b, c, d)

    print("Решение системы:")
    for i, value in enumerate(solution, start=1):
        print(f"x{i} = {value:.6f}")


if __name__ == "__main__":
    main()
