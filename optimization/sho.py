"""Sea-horse optimization (SHO).

The optimizer is deliberately independent of Ultralytics so that its update
rules can be tested with inexpensive mathematical objectives before running
YOLO training.
"""
from __future__ import annotations

from collections.abc import Callable
from math import gamma

import numpy as np


def initialization(pop: int, dim: int, lb, ub, rng: np.random.Generator | None = None) -> np.ndarray:
    """Sample an initial population uniformly inside the search bounds."""
    if pop < 2 or dim < 1:
        raise ValueError("pop must be at least 2 and dim must be positive")
    lower, upper = _bounds(lb, ub, dim)
    return (rng or np.random.default_rng()).uniform(lower, upper, size=(pop, dim))


def levy(dim: int | tuple[int, ...], beta: float = 1.5, rng: np.random.Generator | None = None) -> np.ndarray:
    """Return a Levy-flight step using Mantegna's method."""
    shape = (dim,) if isinstance(dim, int) else tuple(dim)
    if not shape or any(size < 1 for size in shape) or not 0 < beta <= 2:
        raise ValueError("dim must be positive and beta must be in (0, 2]")
    generator = rng or np.random.default_rng()
    sigma = (
        gamma(1 + beta)
        * np.sin(np.pi * beta / 2)
        / (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
    ) ** (1 / beta)
    numerator = generator.normal(0, sigma, shape)
    denominator = np.maximum(np.abs(generator.normal(0, 1, shape)), np.finfo(float).eps) ** (1 / beta)
    return numerator / denominator


def sho(
    pop: int,
    max_iter: int,
    lb,
    ub,
    dim: int,
    fobj: Callable[[np.ndarray], float],
    seed: int | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Minimize ``fobj`` with the sea-horse optimization algorithm.

    Returns:
        Best objective value, best position, and the best value per iteration.
    """
    if max_iter < 1 or pop % 2:
        raise ValueError("max_iter must be positive and pop must be even")
    lower, upper = _bounds(lb, ub, dim)
    rng = np.random.default_rng(seed)
    population = initialization(pop, dim, lower, upper, rng)
    fitness = np.asarray([float(fobj(candidate)) for candidate in population])
    if not np.all(np.isfinite(fitness)):
        raise ValueError("fobj must return finite values")

    best_index = int(np.argmin(fitness))
    best_position = population[best_index].copy()
    best_fitness = float(fitness[best_index])
    convergence = np.empty(max_iter, dtype=float)

    for iteration in range(1, max_iter + 1):
        beta = rng.normal(size=(pop, dim))
        elite = np.tile(best_position, (pop, 1))
        r1 = rng.normal(size=pop)
        step_length = levy((pop, dim), rng=rng)
        moved = np.empty_like(population)
        for index, position in enumerate(population):
            if r1[index] > 0:
                for coordinate in range(dim):
                    theta = rng.random() * 2 * np.pi
                    row = 0.05 * np.exp(theta * 0.05)
                    x = row * np.cos(theta)
                    y = row * np.sin(theta)
                    z = row * theta
                    moved[index, coordinate] = position[coordinate] + step_length[index, coordinate] * (
                        (elite[index, coordinate] - position[coordinate]) * x * y * z + elite[index, coordinate]
                    )
            else:
                for coordinate in range(dim):
                    moved[index, coordinate] = position[coordinate] + rng.random() * 0.05 * beta[index, coordinate] * (
                        position[coordinate] - beta[index, coordinate] * elite[index, coordinate]
                    )
        moved = np.clip(moved, lower, upper)

        alpha = (1 - iteration / max_iter) ** (2 * iteration / max_iter)
        r2 = rng.random(pop)
        hunted = np.empty_like(moved)
        for index in range(pop):
            for coordinate in range(dim):
                if r2[index] >= 0.1:
                    hunted[index, coordinate] = alpha * (elite[index, coordinate] - rng.random() * moved[index, coordinate]) + (
                        1 - alpha
                    ) * elite[index, coordinate]
                else:
                    hunted[index, coordinate] = (1 - alpha) * (
                        moved[index, coordinate] - rng.random() * elite[index, coordinate]
                    ) + alpha * moved[index, coordinate]
        hunted = np.clip(hunted, lower, upper)
        hunted_fitness = np.asarray([float(fobj(candidate)) for candidate in hunted])

        order = np.argsort(hunted_fitness)
        fathers = hunted[order[: pop // 2]]
        mothers = hunted[order[pop // 2 :]]
        children = np.empty((pop // 2, dim))
        for index in range(pop // 2):
            mixing = rng.random()
            children[index] = mixing * fathers[index] + (1 - mixing) * mothers[index]
        children = np.clip(children, lower, upper)
        children_fitness = np.asarray([float(fobj(candidate)) for candidate in children])

        candidates = np.concatenate((hunted, children))
        candidate_fitness = np.concatenate((hunted_fitness, children_fitness))
        selected = np.argsort(candidate_fitness)[:pop]
        population = candidates[selected]
        fitness = candidate_fitness[selected]
        if fitness[0] < best_fitness:
            best_fitness = float(fitness[0])
            best_position = population[0].copy()
        convergence[iteration - 1] = best_fitness

    return best_fitness, best_position, convergence


def _bounds(lb, ub, dim: int) -> tuple[np.ndarray, np.ndarray]:
    """Normalize scalar or vector bounds and validate them."""
    lower = np.broadcast_to(np.asarray(lb, dtype=float), (dim,)).copy()
    upper = np.broadcast_to(np.asarray(ub, dtype=float), (dim,)).copy()
    if lower.shape != (dim,) or upper.shape != (dim,) or np.any(~np.isfinite(lower)) or np.any(~np.isfinite(upper)):
        raise ValueError("bounds must be finite scalars or vectors with length dim")
    if np.any(lower >= upper):
        raise ValueError("each lower bound must be smaller than its upper bound")
    return lower, upper