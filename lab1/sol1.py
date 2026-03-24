from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple


Matrix = List[List[float]]
Vector = List[float]


def clone_matrix(matrix: Sequence[Sequence[float]]) -> Matrix:
    return [list(map(float, row)) for row in matrix]


def lu_decomposition(matrix: Sequence[Sequence[float]]) -> Tuple[Matrix, List[int], int]:
    n = len(matrix)
    if n == 0 or any(len(row) != n for row in matrix):
        raise ValueError("Матрица должна быть квадратной и непустой.")

    lu = clone_matrix(matrix)
    permutation = list(range(n))
    swap_count = 0

    for k in range(n):
        pivot_row = max(range(k, n), key=lambda i: abs(lu[i][k]))
        pivot_value = lu[pivot_row][k]
        if abs(pivot_value) < 1e-12:
            raise ValueError("Матрица вырождена, LU-разложение невозможно.")

        if pivot_row != k:
            lu[k], lu[pivot_row] = lu[pivot_row], lu[k]
            permutation[k], permutation[pivot_row] = permutation[pivot_row], permutation[k]
            swap_count += 1

        for i in range(k + 1, n):
            lu[i][k] /= lu[k][k]
            for j in range(k + 1, n):
                lu[i][j] -= lu[i][k] * lu[k][j]

    return lu, permutation, swap_count


def split_lu(lu: Sequence[Sequence[float]]) -> Tuple[Matrix, Matrix]:
    n = len(lu)
    lower = [[0.0] * n for _ in range(n)]
    upper = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i > j:
                lower[i][j] = lu[i][j]
            elif i == j:
                lower[i][j] = 1.0
                upper[i][j] = lu[i][j]
            else:
                upper[i][j] = lu[i][j]

    return lower, upper


def permutation_matrix(permutation: Sequence[int]) -> Matrix:
    n = len(permutation)
    matrix = [[0.0] * n for _ in range(n)]
    for i, source_row in enumerate(permutation):
        matrix[i][source_row] = 1.0
    return matrix


def permute_vector(vector: Sequence[float], permutation: Sequence[int]) -> Vector:
    return [float(vector[index]) for index in permutation]


def forward_substitution(lower: Sequence[Sequence[float]], vector: Sequence[float]) -> Vector:
    n = len(lower)
    result = [0.0] * n
    for i in range(n):
        subtotal = sum(lower[i][j] * result[j] for j in range(i))
        result[i] = vector[i] - subtotal
    return result


def backward_substitution(upper: Sequence[Sequence[float]], vector: Sequence[float]) -> Vector:
    n = len(upper)
    result = [0.0] * n
    for i in range(n - 1, -1, -1):
        subtotal = sum(upper[i][j] * result[j] for j in range(i + 1, n))
        pivot = upper[i][i]
        if abs(pivot) < 1e-12:
            raise ValueError("Матрица вырождена, обратный ход невозможен.")
        result[i] = (vector[i] - subtotal) / pivot
    return result


def solve_with_lu(lu: Sequence[Sequence[float]], permutation: Sequence[int], vector: Sequence[float]) -> Vector:
    lower, upper = split_lu(lu)
    permuted_vector = permute_vector(vector, permutation)
    intermediate = forward_substitution(lower, permuted_vector)
    return backward_substitution(upper, intermediate)


def determinant_from_lu(lu: Sequence[Sequence[float]], swap_count: int) -> float:
    determinant = -1.0 if swap_count % 2 else 1.0
    for i in range(len(lu)):
        determinant *= lu[i][i]
    return determinant


def inverse_matrix(lu: Sequence[Sequence[float]], permutation: Sequence[int]) -> Matrix:
    n = len(lu)
    inverse = []
    for column in range(n):
        basis_vector = [0.0] * n
        basis_vector[column] = 1.0
        inverse_column = solve_with_lu(lu, permutation, basis_vector)
        inverse.append(inverse_column)

    return [[inverse[j][i] for j in range(n)] for i in range(n)]


def multiply_matrix_vector(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> Vector:
    return [sum(row[j] * vector[j] for j in range(len(vector))) for row in matrix]


def format_number(value: float) -> str:
    if abs(value) < 1e-12:
        value = 0.0
    return f"{value:10.6f}"


def format_matrix(matrix: Sequence[Sequence[float]]) -> str:
    return "\n".join(" ".join(format_number(value) for value in row) for row in matrix)


def format_vector(vector: Iterable[float]) -> str:
    return "[" + ", ".join(f"{value:.6f}" for value in vector) + "]"


def main() -> None:
    # Коэффициенты считаны по приложенному варианту.
    matrix_a = [
        [7.0, -5.0, 6.0, 7.0],
        [8.0, -1.0, -9.0, 1.0],
        [-3.0, 8.0, 8.0, 8.0],
        [2.0, -3.0, 6.0, -4.0],
    ]
    vector_b = [120.0, 31.0, 6.0, 25.0]

    lu, permutation, swap_count = lu_decomposition(matrix_a)
    lower, upper = split_lu(lu)
    solution = solve_with_lu(lu, permutation, vector_b)
    determinant = determinant_from_lu(lu, swap_count)
    inverse = inverse_matrix(lu, permutation)
    check = multiply_matrix_vector(matrix_a, solution)

    print("Матрица A:")
    print(format_matrix(matrix_a))
    print("\nВектор b:")
    print(format_vector(vector_b))

    print("\nМатрица перестановок P:")
    print(format_matrix(permutation_matrix(permutation)))

    print("\nМатрица L:")
    print(format_matrix(lower))

    print("\nМатрица U:")
    print(format_matrix(upper))

    print("\nРешение системы Ax = b:")
    print(format_vector(solution))

    print("\nПроверка A * x:")
    print(format_vector(check))

    print("\nОпределитель det(A):")
    print(f"{determinant:.6f}")

    print("\nОбратная матрица A^(-1):")
    print(format_matrix(inverse))


if __name__ == "__main__":
    main()
