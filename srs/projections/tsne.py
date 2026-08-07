"""
t-SNE in pure Python, for offline exploration of the reference library.

This is deliberately not on the request path. It costs minutes rather than the
fraction of a second PCA costs, and more importantly it has no out-of-sample
extension: placing one newly uploaded sample would mean recomputing the entire
embedding. Use it to look at how the library's own deposit classes separate,
and use PCA when a user's sample has to appear on the plot.

The implementation is the standard one: match a probability distribution over
neighbours in the high-dimensional space with one in two dimensions, minimising
the Kullback-Leibler divergence between them by gradient descent.
"""

from math import exp, log, sqrt
import random

DEFAULT_PERPLEXITY = 30.0
DEFAULT_ITERATIONS = 500
DEFAULT_LEARNING_RATE = 200.0

# The first iterations run with the affinities exaggerated, which pushes
# clusters apart early and leaves room between them for the rest of the run.
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

    # Symmetrise, so the affinity between two points does not depend on which
    # of them is doing the looking.
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
    """
    Embed high-dimensional vectors into two dimensions.

    Returns a list of [x, y] pairs in the order given. Seeded so a rerun over
    an unchanged library reproduces the same picture.
    """
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
                # Gains rise where the gradient keeps its sign and fall where
                # it oscillates, which is the usual t-SNE step-size heuristic.
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
