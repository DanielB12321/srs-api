"""Offline t-SNE projection of the reference library."""

from math import exp, log
import random

DEFAULT_PERPLEXITY = 30.0
DEFAULT_ITERATIONS = 500
DEFAULT_LEARNING_RATE = 200.0

# Early exaggeration separates clusters before the normal optimisation stage.
_EARLY_EXAGGERATION = 4.0
_EARLY_ITERATIONS = 100


def _squared_distances(vectors):
    n = len(vectors)
    distances = [[0.0] * n for _ in range(n)]

    for i in range(n):
        vi = vectors[i]
        for j in range(i + 1, n):
            vj = vectors[j]
            total = 0.0
            for a, b in zip(vi, vj):
                diff = a - b
                total += diff * diff
            distances[i][j] = total
            distances[j][i] = total

    return distances


def _binary_search_sigma(row, target_entropy, tolerance=1e-5, steps=50):
    """Find the bandwidth that gives this point the requested perplexity."""
    low, high = 1e-20, 1e20
    beta = 1.0

    for _ in range(steps):
        weights = [exp(-value * beta) for value in row]
        total = sum(weights) or 1e-12
        entropy = log(total) + beta * sum(
            value * weight for value, weight in zip(row, weights)
        ) / total

        if abs(entropy - target_entropy) < tolerance:
            break

        if entropy > target_entropy:
            low = beta
            beta = beta * 2 if high == 1e20 else (beta + high) / 2
        else:
            high = beta
            beta = beta / 2 if low == 1e-20 else (beta + low) / 2

    weights = [exp(-value * beta) for value in row]
    total = sum(weights) or 1e-12
    return [weight / total for weight in weights]


def _joint_probabilities(vectors, perplexity):
    n = len(vectors)
    distances = _squared_distances(vectors)
    target_entropy = log(perplexity)
    conditional = []

    for i in range(n):
        row = [distances[i][j] for j in range(n) if j != i]
        probabilities = _binary_search_sigma(row, target_entropy)

        full = [0.0] * n
        position = 0
        for j in range(n):
            if j == i:
                continue
            full[j] = probabilities[position]
            position += 1
        conditional.append(full)

    # Make the affinity between each pair symmetric.
    joint = [[0.0] * n for _ in range(n)]
    scale = 2 * n
    for i in range(n):
        for j in range(n):
            joint[i][j] = (conditional[i][j] + conditional[j][i]) / scale

    return joint


def fit_tsne(
    vectors,
    perplexity=DEFAULT_PERPLEXITY,
    iterations=DEFAULT_ITERATIONS,
    learning_rate=DEFAULT_LEARNING_RATE,
    seed=7,
    progress=None,
):
    """Return seeded two-dimensional positions in input order."""
    n = len(vectors)
    if n < 3:
        return [[0.0, 0.0] for _ in range(n)]

    perplexity = min(perplexity, max(2.0, (n - 1) / 3))
    joint = _joint_probabilities(vectors, perplexity)

    rng = random.Random(seed)
    embedding = [[rng.gauss(0, 1e-2), rng.gauss(0, 1e-2)] for _ in range(n)]
    velocity = [[0.0, 0.0] for _ in range(n)]
    gains = [[1.0, 1.0] for _ in range(n)]

    for iteration in range(iterations):
        exaggeration = (
            _EARLY_EXAGGERATION if iteration < _EARLY_ITERATIONS else 1.0
        )
        momentum = 0.5 if iteration < 20 else 0.8

        # Student-t affinities in the embedding, and the normaliser for them.
        numerators = [[0.0] * n for _ in range(n)]
        total = 0.0
        for i in range(n):
            yi = embedding[i]
            for j in range(i + 1, n):
                yj = embedding[j]
                dx = yi[0] - yj[0]
                dy = yi[1] - yj[1]
                value = 1.0 / (1.0 + dx * dx + dy * dy)
                numerators[i][j] = value
                numerators[j][i] = value
                total += 2 * value
        total = total or 1e-12

        gradient = [[0.0, 0.0] for _ in range(n)]
        for i in range(n):
            yi = embedding[i]
            gxi = gyi = 0.0
            for j in range(n):
                if i == j:
                    continue
                value = numerators[i][j]
                force = (joint[i][j] * exaggeration - value / total) * value
                gxi += force * (yi[0] - embedding[j][0])
                gyi += force * (yi[1] - embedding[j][1])
            gradient[i][0] = 4 * gxi
            gradient[i][1] = 4 * gyi

        for i in range(n):
            for axis in range(2):
                # Adjust each step size when the gradient direction changes.
                if (gradient[i][axis] > 0) != (velocity[i][axis] > 0):
                    gains[i][axis] += 0.2
                else:
                    gains[i][axis] = max(0.01, gains[i][axis] * 0.8)

                velocity[i][axis] = (
                    momentum * velocity[i][axis]
                    - learning_rate * gains[i][axis] * gradient[i][axis]
                )
                embedding[i][axis] += velocity[i][axis]

        centre_x = sum(point[0] for point in embedding) / n
        centre_y = sum(point[1] for point in embedding) / n
        for point in embedding:
            point[0] -= centre_x
            point[1] -= centre_y

        if progress is not None and (iteration + 1) % 25 == 0:
            progress(iteration + 1, iterations)

    return embedding
