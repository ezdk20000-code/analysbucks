"""Работа с коэффициентами: вычет маржи букмекера.

Почему степенной метод (power), а не пропорциональный: букмекеры закладывают
больше маржи в коэффициенты на аутсайдеров, чем на фаворитов (favorite–longshot bias).
Пропорциональный метод это игнорирует и недооценивает фаворитов. На истории
2010–2026 (АПЛ + Ла Лига) степенной метод откалиброван заметно лучше.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq


def fair_power(odds) -> np.ndarray:
    """Справедливые вероятности: p_i = (1/o_i)^k, где k подобран так, чтобы сумма = 1.

    odds — массив формы (n, k) или (k,). Строки с NaN возвращаются как NaN.
    """
    O = np.atleast_2d(np.asarray(odds, dtype=float))
    out = np.full_like(O, np.nan)
    for i, row in enumerate(O):
        if np.isnan(row).any() or (row <= 1).any():
            continue
        inv = 1 / row
        if abs(inv.sum() - 1) < 1e-9:
            out[i] = inv
            continue
        try:
            k = brentq(lambda k: (inv ** k).sum() - 1, 0.2, 5)
            out[i] = inv ** k
        except ValueError:
            out[i] = inv / inv.sum()
    return out if np.ndim(odds) > 1 else out[0]


def fair_prop(odds) -> np.ndarray:
    O = np.atleast_2d(np.asarray(odds, dtype=float))
    inv = 1 / O
    out = inv / inv.sum(axis=1, keepdims=True)
    return out if np.ndim(odds) > 1 else out[0]


def margin(odds) -> float:
    return float((1 / np.asarray(odds, float)).sum() - 1)
