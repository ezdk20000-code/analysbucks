"""Мани-менеджмент: размер ставки и ограничения риска.

Критерий Келли максимизирует ожидаемый логарифм банка, то есть даёт самый быстрый
«геометрический» рост на дистанции, НО только если вероятности оценены верно.
Полный Келли при ошибке оценки даёт огромные просадки, поэтому используется доля
(обычно 1/4). Ниже — набор предохранителей, которые срабатывают последовательно.
"""
from __future__ import annotations

from dataclasses import dataclass


def kelly(p: float, odds: float) -> float:
    """Доля банка по Келли для одной ставки: f* = (p*k - 1) / (k - 1). Не меньше 0."""
    if odds <= 1 or p <= 0:
        return 0.0
    return max((p * odds - 1) / (odds - 1), 0.0)


@dataclass
class StakePlan:
    stake: float
    fraction: float
    reason: str


class Staker:
    """Считает ставку с учётом стратегии и всех лимитов.

    cfg — раздел `staking` из config.yaml.
    """

    def __init__(self, cfg: dict, start_bank: float):
        self.c = cfg
        self.start_bank = start_bank
        self.peak = start_bank

    def update_peak(self, bank: float) -> None:
        self.peak = max(self.peak, bank)

    def risk_multiplier(self, bank: float) -> tuple[float, str]:
        if bank < self.start_bank * self.c["stop_loss_pct"]:
            return 0.0, "стоп-лосс: банк ниже порога, ставки остановлены"
        if bank < self.peak * (1 - self.c["drawdown_brake"]):
            return 0.5, "просадка от пика — ставки уменьшены вдвое"
        return 1.0, ""

    def size(self, p: float, odds: float, bank: float, legs: int = 1) -> StakePlan:
        c = self.c
        mult, why = self.risk_multiplier(bank)
        if mult == 0:
            return StakePlan(0.0, 0.0, why)
        strat = c["strategy"]
        if strat == "kelly":
            f = c["kelly_fraction"] * kelly(p, odds)
        elif strat == "percent":
            f = c["percent"] if p * odds > 1 else 0.0
        elif strat == "flat":
            f = (c["flat_unit"] * self.start_bank / bank) if p * odds > 1 and bank > 0 else 0.0
        else:
            raise ValueError(f"Неизвестная стратегия {strat}")
        cap = c["max_stake_pct"]
        if legs > 1:
            # экспресс: дисперсия выше — отдельный (меньший) лимит
            cap = c.get("max_express_pct", cap / 2)
        f = min(f, cap) * mult
        stake = round(bank * f, 2)
        if stake < c.get("min_stake", 0):
            return StakePlan(0.0, 0.0, "ставка меньше минимальной у букмекера")
        return StakePlan(stake, f, why or strat)

    def apply_exposure_cap(self, stakes: list[float], bank: float) -> list[float]:
        """Если суммарно на день больше лимита — пропорционально урезаем все ставки."""
        total = sum(stakes)
        cap = bank * self.c["max_daily_exposure_pct"]
        if total <= cap or total == 0:
            return stakes
        k = cap / total
        return [round(s * k, 2) for s in stakes]
