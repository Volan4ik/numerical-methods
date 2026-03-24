from __future__ import annotations

import cmath
import math
from typing import Iterable, List, Sequence, Tuple


Matrix = List[List[float]]


def clone_matrix(matrix: Sequence[Sequence[float]]) -> Matrix:
    return [list(map(float, row)) for row in matrix]


def identity_matrix(size: int) -> Matrix:
    return [[1.0 if i == j else 0.0 for j in range(size)] for i in range(size)]


def validate_square(matrix: Sequence[Sequence[float]]) -> None:
    size = len(matrix)
    if size == 0 or any(len(row) != size for row in matrix):
        raise ValueError("Матрица должна быть квадратной и непустой.")


def transpose(matrix: Sequence[Sequence[float]]) -> Matrix:
    return [list(row) for row in zip(*matrix)]


def multiply_matrices(left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]) -> Matrix:
    rows = len(left)
    cols = len(right[0])
    inner = len(right)
    return [
        [sum(left[i][k] * right[k][j] for k in range(inner)) for j in range(cols)]
        for i in range(rows)
    ]


def subtract_matrices(left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]) -> Matrix:
    return [
        [left[i][j] - right[i][j] for j in range(len(left[0]))]
        for i in range(len(left))
    ]


def matrix_norm_inf(matrix: Sequence[Sequence[float]]) -> float:
    return max(sum(abs(value) for value in row) for row in matrix)


def lower_triangle_norm(matrix: Sequence[Sequence[float]]) -> float:
    total = 0.0
    for i in range(1, len(matrix)):
        for j in range(min(i, len(matrix[0]))):
            total += matrix[i][j] * matrix[i][j]
    return math.sqrt(total)


def matrix_minus_shift(matrix: Sequence[Sequence[float]], shift: float) -> Matrix:
    result = clone_matrix(matrix)
    for i in range(len(result)):
        result[i][i] -= shift
    return result


def add_shift(matrix: Sequence[Sequence[float]], shift: float) -> Matrix:
    result = clone_matrix(matrix)
    for i in range(len(result)):
        result[i][i] += shift
    return result


def householder_qr(matrix: Sequence[Sequence[float]]) -> Tuple[Matrix, Matrix]:
    rows = len(matrix)
    cols = len(matrix[0])
    r_matrix = clone_matrix(matrix)
    q_matrix = identity_matrix(rows)

    for k in range(min(rows - 1, cols)):
        x = [r_matrix[i][k] for i in range(k, rows)]
        norm_x = math.sqrt(sum(value * value for value in x))
        if norm_x < 1e-12:
            continue

        sign = 1.0 if x[0] >= 0 else -1.0
        v = x[:]
        v[0] += sign * norm_x
        norm_v = math.sqrt(sum(value * value for value in v))
        if norm_v < 1e-12:
            continue
        v = [value / norm_v for value in v]

        for j in range(k, cols):
            projection = sum(v[i - k] * r_matrix[i][j] for i in range(k, rows))
            for i in range(k, rows):
                r_matrix[i][j] -= 2.0 * v[i - k] * projection

        for i in range(rows):
            projection = sum(q_matrix[i][j] * v[j - k] for j in range(k, rows))
            for j in range(k, rows):
                q_matrix[i][j] -= 2.0 * projection * v[j - k]

    return q_matrix, r_matrix


def qr_algorithm(
    matrix: Sequence[Sequence[float]],
    eps: float,
    max_iterations: int = 1000,
) -> Tuple[Matrix, List[float], int]:
    validate_square(matrix)
    current = clone_matrix(matrix)
    history = [lower_triangle_norm(current)]

    for iteration in range(1, max_iterations + 1):
        shift = current[-1][-1]
        q_matrix, r_matrix = householder_qr(matrix_minus_shift(current, shift))
        current = add_shift(multiply_matrices(r_matrix, q_matrix), shift)

        for i in range(1, len(current)):
            threshold = eps * (abs(current[i - 1][i - 1]) + abs(current[i][i]) + 1.0)
            if abs(current[i][i - 1]) < threshold:
                current[i][i - 1] = 0.0

        history.append(lower_triangle_norm(current))
        if history[-1] < eps:
            return current, history, iteration

    raise ValueError("QR-алгоритм не достиг заданной точности.")


def eigenvalues_from_quasi_triangular(matrix: Sequence[Sequence[float]], eps: float) -> List[complex]:
    values: List[complex] = []
    i = len(matrix) - 1

    while i >= 0:
        if i == 0 or abs(matrix[i][i - 1]) < eps:
            values.append(complex(matrix[i][i], 0.0))
            i -= 1
            continue

        a = matrix[i - 1][i - 1]
        b = matrix[i - 1][i]
        c = matrix[i][i - 1]
        d = matrix[i][i]

        trace = a + d
        determinant = a * d - b * c
        discriminant = cmath.sqrt(trace * trace - 4.0 * determinant)
        values.append((trace + discriminant) / 2.0)
        values.append((trace - discriminant) / 2.0)
        i -= 2

    return list(reversed(values))


def format_number(value: float) -> str:
    if abs(value) < 1e-12:
        value = 0.0
    return f"{value:10.6f}"


def format_complex(value: complex) -> str:
    if abs(value.imag) < 1e-10:
        return f"{value.real:.6f}"
    sign = "+" if value.imag >= 0 else "-"
    return f"{value.real:.6f} {sign} {abs(value.imag):.6f}i"


def format_matrix(matrix: Sequence[Sequence[float]]) -> str:
    return "\n".join(" ".join(format_number(value) for value in row) for row in matrix)


def main() -> None:
    matrix_a = [
        [-1.0, 8.0, 5.0],
        [8.0, -4.0, 4.0],
        [2.0, 9.0, -2.0],
    ]
    epsilon = 1e-6

    q_matrix, r_matrix = householder_qr(matrix_a)
    reconstructed = multiply_matrices(q_matrix, r_matrix)
    qr_residual = matrix_norm_inf(subtract_matrices(reconstructed, matrix_a))
    orthogonality_residual = matrix_norm_inf(
        subtract_matrices(multiply_matrices(transpose(q_matrix), q_matrix), identity_matrix(len(matrix_a)))
    )

    quasi_triangular, history, iterations = qr_algorithm(matrix_a, epsilon)
    eigenvalues = eigenvalues_from_quasi_triangular(quasi_triangular, epsilon)
    eigenvalues.sort(key=lambda value: (value.real, value.imag), reverse=True)

    print("Матрица A:")
    print(format_matrix(matrix_a))

    print("\nТочность epsilon:")
    print(f"{epsilon:.1e}")

    print("\nQR-разложение матрицы A:")
    print("Матрица Q:")
    print(format_matrix(q_matrix))
    print("\nМатрица R:")
    print(format_matrix(r_matrix))
    print(f"\n||A - Q*R||_inf = {qr_residual:.6e}")
    print(f"||Q^T*Q - I||_inf = {orthogonality_residual:.6e}")

    print("\nМатрица после QR-итераций:")
    print(format_matrix(quasi_triangular))

    print("\nСобственные значения:")
    for index, value in enumerate(eigenvalues, start=1):
        print(f"{index}. lambda = {format_complex(value)}")

    print("\nИстория нормы нижнетреугольной части:")
    for iteration, error in enumerate(history):
        print(f"Итерация {iteration:2d}: {error:.6e}")

    print("\nАнализ:")
    print(f"Всего QR-итераций: {iterations}")
    print("По мере итераций матрица приближается к квазитреугольному виду,")
    print("а собственные значения считываются с диагонали или из 2x2 блоков.")


if __name__ == "__main__":
    main()
