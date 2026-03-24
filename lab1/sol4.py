from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple


Vector = List[float]
Matrix = List[List[float]]


def clone_matrix(matrix: Sequence[Sequence[float]]) -> Matrix:
    return [list(map(float, row)) for row in matrix]


def identity_matrix(size: int) -> Matrix:
    return [[1.0 if i == j else 0.0 for j in range(size)] for i in range(size)]


def validate_symmetric(matrix: Sequence[Sequence[float]]) -> None:
    n = len(matrix)
    if n == 0 or any(len(row) != n for row in matrix):
        raise ValueError("Матрица должна быть квадратной и непустой.")

    for i in range(n):
        for j in range(i + 1, n):
            if abs(matrix[i][j] - matrix[j][i]) > 1e-12:
                raise ValueError("Матрица должна быть симметричной.")


def off_diagonal_norm(matrix: Sequence[Sequence[float]]) -> float:
    total = 0.0
    for i in range(len(matrix)):
        for j in range(i + 1, len(matrix)):
            total += 2.0 * matrix[i][j] * matrix[i][j]
    return math.sqrt(total)


def max_off_diagonal_index(matrix: Sequence[Sequence[float]]) -> Tuple[int, int]:
    n = len(matrix)
    row = 0
    col = 1
    max_value = abs(matrix[row][col])

    for i in range(n):
        for j in range(i + 1, n):
            current = abs(matrix[i][j])
            if current > max_value:
                max_value = current
                row, col = i, j

    return row, col


def jacobi_rotation_method(
    matrix: Sequence[Sequence[float]],
    eps: float,
    max_iterations: int = 100,
) -> Tuple[Vector, Matrix, List[float], int]:
    validate_symmetric(matrix)

    n = len(matrix)
    current = clone_matrix(matrix)
    eigenvectors = identity_matrix(n)
    history = [off_diagonal_norm(current)]

    for iteration in range(1, max_iterations + 1):
        p, q = max_off_diagonal_index(current)
        if abs(current[p][q]) < eps:
            break

        if abs(current[p][p] - current[q][q]) < 1e-12:
            angle = math.copysign(math.pi / 4, current[p][q])
        else:
            angle = 0.5 * math.atan2(2.0 * current[p][q], current[p][p] - current[q][q])

        cosine = math.cos(angle)
        sine = math.sin(angle)

        app = current[p][p]
        aqq = current[q][q]
        apq = current[p][q]

        current[p][p] = cosine * cosine * app + 2.0 * sine * cosine * apq + sine * sine * aqq
        current[q][q] = sine * sine * app - 2.0 * sine * cosine * apq + cosine * cosine * aqq
        current[p][q] = 0.0
        current[q][p] = 0.0

        for k in range(n):
            if k == p or k == q:
                continue

            akp = current[k][p]
            akq = current[k][q]
            current[k][p] = cosine * akp + sine * akq
            current[p][k] = current[k][p]
            current[k][q] = -sine * akp + cosine * akq
            current[q][k] = current[k][q]

        for k in range(n):
            vkp = eigenvectors[k][p]
            vkq = eigenvectors[k][q]
            eigenvectors[k][p] = cosine * vkp + sine * vkq
            eigenvectors[k][q] = -sine * vkp + cosine * vkq

        history.append(off_diagonal_norm(current))
        if history[-1] < eps:
            return [current[i][i] for i in range(n)], eigenvectors, history, iteration

    iterations_done = len(history) - 1
    return [current[i][i] for i in range(n)], eigenvectors, history, iterations_done


def extract_sorted_eigenpairs(eigenvalues: Sequence[float], eigenvectors: Sequence[Sequence[float]]) -> List[Tuple[float, Vector]]:
    pairs = []
    for column in range(len(eigenvalues)):
        vector = [eigenvectors[row][column] for row in range(len(eigenvectors))]
        pairs.append((eigenvalues[column], vector))

    return sorted(pairs, key=lambda pair: pair[0], reverse=True)


def multiply_matrix_vector(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> Vector:
    return [sum(row[j] * vector[j] for j in range(len(vector))) for row in matrix]


def residual_norm(matrix: Sequence[Sequence[float]], eigenvalue: float, eigenvector: Sequence[float]) -> float:
    left = multiply_matrix_vector(matrix, eigenvector)
    right = [eigenvalue * value for value in eigenvector]
    return max(abs(left[i] - right[i]) for i in range(len(left)))


def format_number(value: float) -> str:
    if abs(value) < 1e-12:
        value = 0.0
    return f"{value:10.6f}"


def format_vector(vector: Iterable[float]) -> str:
    return "[" + ", ".join(f"{value:.6f}" for value in vector) + "]"


def format_matrix(matrix: Sequence[Sequence[float]]) -> str:
    return "\n".join(" ".join(format_number(value) for value in row) for row in matrix)


def main() -> None:
    matrix_a = [
        [2.0, -9.0, 4.0],
        [-9.0, 0.0, 9.0],
        [4.0, 9.0, 6.0],
    ]
    epsilon = 1e-6

    eigenvalues, eigenvectors, history, iterations = jacobi_rotation_method(matrix_a, epsilon)
    eigenpairs = extract_sorted_eigenpairs(eigenvalues, eigenvectors)

    print("Матрица A:")
    print(format_matrix(matrix_a))

    print("\nТочность epsilon:")
    print(f"{epsilon:.1e}")

    print("\nСобственные значения и собственные векторы:")
    for index, (eigenvalue, eigenvector) in enumerate(eigenpairs, start=1):
        print(f"{index}. lambda = {eigenvalue:.6f}")
        print(f"   v = {format_vector(eigenvector)}")
        print(f"   ||A*v - lambda*v||_inf = {residual_norm(matrix_a, eigenvalue, eigenvector):.6e}")

    print("\nИстория нормы внедиагональных элементов:")
    for iteration, error in enumerate(history):
        print(f"Итерация {iteration:2d}: {error:.6e}")

    print("\nАнализ:")
    print(f"Всего итераций: {iterations}")
    print("С ростом числа итераций норма внедиагональной части убывает,")
    print("поэтому приближение к диагональной форме и точность собственных значений улучшаются.")


if __name__ == "__main__":
    main()
