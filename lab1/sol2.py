from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple


Vector = List[float]
Matrix = List[List[float]]


def validate_tridiagonal(lower: Sequence[float], main: Sequence[float], upper: Sequence[float], rhs: Sequence[float]) -> None:
    n = len(main)
    if n == 0:
        raise ValueError("Главная диагональ не должна быть пустой.")
    if len(lower) != n - 1 or len(upper) != n - 1 or len(rhs) != n:
        raise ValueError("Размеры диагоналей или вектора правых частей не согласованы.")


def thomas_algorithm(
    lower: Sequence[float],
    main: Sequence[float],
    upper: Sequence[float],
    rhs: Sequence[float],
) -> Tuple[Vector, Vector, Vector]:
    validate_tridiagonal(lower, main, upper, rhs)

    n = len(main)
    alpha = [0.0] * n
    beta = [0.0] * n

    if abs(main[0]) < 1e-12:
        raise ValueError("Нулевой элемент на главной диагонали в первой строке.")

    alpha[0] = -upper[0] / main[0] if n > 1 else 0.0
    beta[0] = rhs[0] / main[0]

    for i in range(1, n):
        denominator = main[i] + lower[i - 1] * alpha[i - 1]
        if abs(denominator) < 1e-12:
            raise ValueError(f"Нулевой знаменатель на шаге {i}.")

        if i < n - 1:
            alpha[i] = -upper[i] / denominator
        beta[i] = (rhs[i] - lower[i - 1] * beta[i - 1]) / denominator

    solution = [0.0] * n
    solution[-1] = beta[-1]
    for i in range(n - 2, -1, -1):
        solution[i] = alpha[i] * solution[i + 1] + beta[i]

    return solution, alpha, beta


def build_matrix(lower: Sequence[float], main: Sequence[float], upper: Sequence[float]) -> Matrix:
    n = len(main)
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        matrix[i][i] = float(main[i])
        if i > 0:
            matrix[i][i - 1] = float(lower[i - 1])
        if i < n - 1:
            matrix[i][i + 1] = float(upper[i])
    return matrix


def multiply_matrix_vector(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> Vector:
    return [sum(row[j] * vector[j] for j in range(len(vector))) for row in matrix]


def format_number(value: float) -> str:
    if abs(value) < 1e-12:
        value = 0.0
    return f"{value:10.6f}"


def format_vector(vector: Iterable[float]) -> str:
    return "[" + ", ".join(f"{value:.6f}" for value in vector) + "]"


def format_matrix(matrix: Sequence[Sequence[float]]) -> str:
    return "\n".join(" ".join(format_number(value) for value in row) for row in matrix)


def main() -> None:
    # Коэффициенты из текущего варианта на изображении.
    lower_diagonal = [2.0, -7.0, 2.0, -7.0]
    main_diagonal = [-14.0, 7.0, -18.0, -13.0, -7.0]
    upper_diagonal = [6.0, 0.0, -9.0, 2.0]
    rhs = [82.0, -51.0, -46.0, 111.0, 35.0]

    matrix = build_matrix(lower_diagonal, main_diagonal, upper_diagonal)
    solution, alpha, beta = thomas_algorithm(lower_diagonal, main_diagonal, upper_diagonal, rhs)
    check = multiply_matrix_vector(matrix, solution)

    print("Трехдиагональная матрица A:")
    print(format_matrix(matrix))

    print("\nВектор правых частей b:")
    print(format_vector(rhs))

    print("\nКоэффициенты прямой прогонки alpha:")
    print(format_vector(alpha))

    print("\nКоэффициенты прямой прогонки beta:")
    print(format_vector(beta))

    print("\nРешение системы Ax = b:")
    print(format_vector(solution))

    print("\nПроверка A * x:")
    print(format_vector(check))


if __name__ == "__main__":
    main()
