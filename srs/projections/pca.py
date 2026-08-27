"""Principal component projection of samples in CLR space."""

from dataclasses import dataclass
from math import log10, sqrt

# Minimum sample coverage required for an element to enter the projection.
DEFAULT_MIN_COVERAGE = 0.9

# Stops an unusual covariance matrix from looping indefinitely.
_MAX_SWEEPS = 100


@dataclass
class PCAModel:
    """A fitted projection: which elements, their centre, and the axes."""

    symbols: list
    means: list
    components: list
    explained_variance_ratio: list

    def project(self, values):
        """Project a sample, or return ``None`` when coverage is too low."""
        vector = _clr_vector(values, self.symbols, self.means)
        if vector is None:
            return None

        centred = [value - mean for value, mean in zip(vector, self.means)]
        return [
            sum(value * weight for value, weight in zip(centred, component))
            for component in self.components
        ]


def _clr_vector(values, symbols, fallback):
    """Build a CLR vector and fill missing elements with fitted means."""
    present = [values.get(symbol) for symbol in symbols]
    measured = [value for value in present if value is not None and value > 0]

    # Fewer than half the elements measured means the position would be driven
    # by the filler rather than by the sample.
    if len(measured) < max(2, len(symbols) // 2):
        return None

    logs = [log10(value) for value in measured]
    centre = sum(logs) / len(logs)

    return [
        (log10(value) - centre) if (value is not None and value > 0) else fallback[index]
        for index, value in enumerate(present)
    ]


def _canonical_component(vector):
    """Give an eigenvector a stable sign without changing its direction."""
    if not vector:
        return vector

    largest = max(range(len(vector)), key=lambda index: abs(vector[index]))
    if vector[largest] < 0:
        return [-value for value in vector]
    return vector


def _jacobi_eigen(matrix):
    """Return eigenpairs for a symmetric matrix using Jacobi rotations."""
    size = len(matrix)
    a = [row[:] for row in matrix]
    vectors = [
        [1.0 if row == column else 0.0 for column in range(size)]
        for row in range(size)
    ]

    for _ in range(_MAX_SWEEPS):
        off_diagonal = sqrt(sum(
            a[p][q] ** 2
            for p in range(size)
            for q in range(p + 1, size)
        ))
        if off_diagonal < 1e-9:
            break

        for p in range(size):
            for q in range(p + 1, size):
                if abs(a[p][q]) < 1e-12:
                    continue

                theta = (a[q][q] - a[p][p]) / (2 * a[p][q])
                sign = 1.0 if theta >= 0 else -1.0
                t = sign / (abs(theta) + sqrt(theta * theta + 1))
                cos = 1 / sqrt(t * t + 1)
                sin = t * cos

                for k in range(size):
                    akp, akq = a[k][p], a[k][q]
                    a[k][p] = cos * akp - sin * akq
                    a[k][q] = sin * akp + cos * akq
                for k in range(size):
                    apk, aqk = a[p][k], a[q][k]
                    a[p][k] = cos * apk - sin * aqk
                    a[q][k] = sin * apk + cos * aqk
                for k in range(size):
                    vkp, vkq = vectors[k][p], vectors[k][q]
                    vectors[k][p] = cos * vkp - sin * vkq
                    vectors[k][q] = sin * vkp + cos * vkq

    eigenvalues = [a[i][i] for i in range(size)]
    pairs = [
        (
            eigenvalues[i],
            _canonical_component([vectors[row][i] for row in range(size)]),
        )
        for i in range(size)
    ]
    pairs.sort(key=lambda pair: -pair[0])
    return pairs


def fit_pca(rows, n_components=2, min_coverage=DEFAULT_MIN_COVERAGE):
    """Fit PCA, returning ``None`` when the input is too small or sparse."""
    rows = [row for row in rows if row]
    if len(rows) < n_components + 1:
        return None

    counts = {}
    for row in rows:
        for symbol, value in row.items():
            if value is not None and value > 0:
                counts[symbol] = counts.get(symbol, 0) + 1

    threshold = len(rows) * min_coverage
    symbols = sorted(
        symbol
        for symbol, count in counts.items()
        if count >= threshold
    )
    if len(symbols) < n_components + 1:
        return None

    # Build vectors once, then use them for the mean and covariance passes.
    vectors = []
    zero = [0.0] * len(symbols)
    for row in rows:
        vector = _clr_vector(row, symbols, zero)
        if vector is not None:
            vectors.append(vector)

    if len(vectors) < n_components + 1:
        return None

    size = len(symbols)
    means = [
        sum(vector[index] for vector in vectors) / len(vectors)
        for index in range(size)
    ]

    covariance = [[0.0] * size for _ in range(size)]
    for vector in vectors:
        centred = [value - mean for value, mean in zip(vector, means)]
        for a in range(size):
            left = centred[a]
            if not left:
                continue
            row_a = covariance[a]
            for b in range(a, size):
                row_a[b] += left * centred[b]

    divisor = max(1, len(vectors) - 1)
    for a in range(size):
        for b in range(a, size):
            covariance[a][b] /= divisor
            covariance[b][a] = covariance[a][b]

    pairs = _jacobi_eigen(covariance)
    total = sum(max(0.0, value) for value, _ in pairs) or 1.0

    return PCAModel(
        symbols=symbols,
        means=means,
        components=[vector for _, vector in pairs[:n_components]],
        explained_variance_ratio=[
            max(0.0, value) / total
            for value, _ in pairs[:n_components]
        ],
    )


def project_points(model, entries):
    """Project ``(id, kind, values)`` entries, skipping unplaceable samples."""
    if model is None:
        return []

    points = []
    for identifier, kind, values in entries:
        position = model.project(values)
        if position is None:
            continue

        points.append({
            "id": identifier,
            "x": position[0],
            "y": position[1],
            "kind": kind,
        })

    return points
