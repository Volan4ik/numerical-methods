from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple


Vector = List[float]
Matrix = List[List[float]]


def clone_matrix(matrix: Sequence[Sequence[float]]) -> Matrix:
    return [list(map(float, row)) for row in matrix]


def validate_system(matrix: Sequence[Sequence[float]], rhs: Sequence[float]) -> None:
    n = len(matrix)
    if n == 0 or len(rhs) != n:
        raise ValueError("Матрица и вектор правых частей должны быть непустыми и согласованными по размеру.")
    if any(len(row) != n for row in matrix):
        raise ValueError("Матрица должна быть квадратной.")
    if any(abs(matrix[i][i]) < 1e-12 for i in range(n)):
        raise ValueError("На главной диагонали не должно быть нулевых элементов.")


def build_iteration_form(matrix: Sequence[Sequence[float]], rhs: Sequence[float]) -> Tuple[Matrix, Vector]:
    validate_system(matrix, rhs)

    n = len(matrix)
    alpha = [[0.0] * n for _ in range(n)]
    beta = [0.0] * n

    for i in range(n):
        diagonal = matrix[i][i]
        beta[i] = rhs[i] / diagonal
        for j in range(n):
            if i != j:
                alpha[i][j] = -matrix[i][j] / diagonal

    return alpha, beta


def multiply_matrix_vector(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> Vector:
    return [sum(row[j] * vector[j] for j in range(len(vector))) for row in matrix]


def subtract_vectors(left: Sequence[float], right: Sequence[float]) -> Vector:
    return [left[i] - right[i] for i in range(len(left))]


def vector_norm_inf(vector: Sequence[float]) -> float:
    return max(abs(value) for value in vector)


def matrix_norm_inf(matrix: Sequence[Sequence[float]]) -> float:
    return max(sum(abs(value) for value in row) for row in matrix)


def simple_iterations(
    matrix: Sequence[Sequence[float]],
    rhs: Sequence[float],
    eps: float,
    max_iterations: int = 10000,
) -> Tuple[Vector, int, float]:
    alpha, beta = build_iteration_form(matrix, rhs)
    current = [0.0] * len(matrix)

    for iteration in range(1, max_iterations + 1):
        next_vector = [
            beta[i] + sum(alpha[i][j] * current[j] for j in range(len(matrix)))
            for i in range(len(matrix))
        ]
        step = vector_norm_inf(subtract_vectors(next_vector, current))
        residual = vector_norm_inf(subtract_vectors(multiply_matrix_vector(matrix, next_vector), rhs))
        if step < eps and residual < eps:
            return next_vector, iteration, residual
        current = next_vector

    raise ValueError("Метод простых итераций не достиг заданной точности.")


def seidel_method(
    matrix: Sequence[Sequence[float]],
    rhs: Sequence[float],
    eps: float,
    max_iterations: int = 10000,
) -> Tuple[Vector, int, float]:
    alpha, beta = build_iteration_form(matrix, rhs)
    current = [0.0] * len(matrix)

    for iteration in range(1, max_iterations + 1):
        previous = current[:]
        for i in range(len(matrix)):
            left_sum = sum(alpha[i][j] * current[j] for j in range(i))
            right_sum = sum(alpha[i][j] * previous[j] for j in range(i + 1, len(matrix)))
            current[i] = beta[i] + left_sum + right_sum

        step = vector_norm_inf(subtract_vectors(current, previous))
        residual = vector_norm_inf(subtract_vectors(multiply_matrix_vector(matrix, current), rhs))
        if step < eps and residual < eps:
            return current, iteration, residual

    raise ValueError("Метод Зейделя не достиг заданной точности.")


def format_number(value: float) -> str:
    if abs(value) < 1e-12:
        value = 0.0
    return f"{value:10.6f}"


def format_matrix(matrix: Sequence[Sequence[float]]) -> str:
    return "\n".join(" ".join(format_number(value) for value in row) for row in matrix)


def format_vector(vector: Iterable[float]) -> str:
    return "[" + ", ".join(f"{value:.6f}" for value in vector) + "]"


def main() -> None:
    matrix_a = [
        [-18.0, 9.0, -1.0, -8.0],
        [6.0, 22.0, 9.0, 0.0],
        [-4.0, 2.0, -16.0, 9.0],
        [1.0, 6.0, -1.0, -14.0],
    ]
    vector_b = [-60.0, -109.0, -103.0, -33.0]
    epsilon = 1e-6

    alpha, beta = build_iteration_form(matrix_a, vector_b)
    jacobi_solution, jacobi_iterations, jacobi_residual = simple_iterations(matrix_a, vector_b, epsilon)
    seidel_solution, seidel_iterations, seidel_residual = seidel_method(matrix_a, vector_b, epsilon)

    print("Матрица A:")
    print(format_matrix(matrix_a))

    print("\nВектор b:")
    print(format_vector(vector_b))

    print("\nТочность epsilon:")
    print(f"{epsilon:.1e}")

    print("\nМатрица alpha для итерационной формы x = beta + alpha * x:")
    print(format_matrix(alpha))

    print("\nВектор beta:")
    print(format_vector(beta))

    print("\nМетод простых итераций:")
    print(f"Решение: {format_vector(jacobi_solution)}")
    print(f"Число итераций: {jacobi_iterations}")
    print(f"Невязка ||Ax - b||_inf: {jacobi_residual:.6e}")

    print("\nМетод Зейделя:")
    print(f"Решение: {format_vector(seidel_solution)}")
    print(f"Число итераций: {seidel_iterations}")
    print(f"Невязка ||Ax - b||_inf: {seidel_residual:.6e}")

    print("\nАнализ:")
    print(f"||alpha||_inf = {matrix_norm_inf(alpha):.6f}")
    if seidel_iterations < jacobi_iterations:
        print("Метод Зейделя потребовал меньше итераций, чем метод простых итераций.")
    elif seidel_iterations > jacobi_iterations:
        print("Метод простых итераций потребовал меньше итераций, чем метод Зейделя.")
    else:
        print("Оба метода достигли заданной точности за одинаковое число итераций.")


if __name__ == "__main__":
    main()
