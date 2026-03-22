"""
Astar Island — Simulation Engine v2
====================================
Recalibrated against 14 rounds of GT data.

GT transition patterns (14-round average):
  Empty  → [0.844, 0.110, 0.008, 0.010, 0.029]
  Settle → [0.440, 0.320, 0.004, 0.026, 0.210]
  Port   → [0.469, 0.101, 0.185, 0.022, 0.221]
  Forest → [0.076, 0.140, 0.009, 0.013, 0.761]
  Mountain → [0, 0, 0, 0, 0, 1.0]
"""
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class Settlement:
    row: int
    col: int
    population: float = 100.0
    food: float = 50.0
    wealth: float = 0.0
    defense: float = 10.0
    tech: float = 1.0
    is_port: bool = False
    longships: int = 0
    faction: int = 0
    age: int = 0


@dataclass
class SimParams:
    """Tunable parameters. Calibrated against GT data."""
    # Growth
    food_per_plains: float = 3.0
    food_per_forest: float = 2.5
    food_per_ocean: float = 1.0
    growth_rate: float = 0.08
    max_pop: float = 500.0
    port_pop_threshold: float = 80.0
    longship_rate: float = 0.3
    expand_pop: float = 120.0
    expand_prob: float = 0.25
    expand_radius: int = 4
    forest_clear_prob: float = 0.08

    # Conflict
    raid_range: float = 3.5
    raid_prob: float = 0.25
    desperate_mult: float = 2.5
    raid_damage: float = 0.25
    loot_frac: float = 0.4
    conquer_pop: float = 20.0

    # Trade
    trade_range: float = 5.0
    trade_wealth: float = 5.0
    trade_food: float = 3.0
    tech_rate: float = 0.1

    # Winter
    winter_mean: float = 1.0
    winter_std: float = 0.5
    food_consume: float = 0.4
    starve_loss: float = 0.5
    collapse_threshold: float = 8.0

    # Environment
    ruin_nature_rate: float = 0.20
    ruin_settle_rate: float = 0.12
    ruin_to_forest: float = 0.55
    forest_spread_prob: float = 0.005

    n_years: int = 50


NEIGHBORS_4 = [(-1, 0), (0, -1), (0, 1), (1, 0)]
NEIGHBORS_8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


class Simulation:
    def __init__(self, initial_grid: np.ndarray, params: SimParams, rng: np.random.Generator):
        self.grid = initial_grid.copy()
        self.H, self.W = self.grid.shape
        self.p = params
        self.rng = rng
        self.settlements: Dict[Tuple[int, int], Settlement] = {}
        self.faction_counter = 0
        self._init()

    def _init(self):
        for r in range(self.H):
            for c in range(self.W):
                v = self.grid[r, c]
                if v in (1, 2):
                    s = Settlement(
                        row=r, col=c,
                        population=100 + self.rng.uniform(-30, 30),
                        food=60 + self.rng.uniform(-20, 20),
                        wealth=self.rng.uniform(5, 30),
                        defense=10 + self.rng.uniform(-3, 3),
                        tech=1.0 + self.rng.uniform(0, 0.5),
                        is_port=(v == 2),
                        longships=2 if v == 2 else 0,
                        faction=self.faction_counter,
                    )
                    self.settlements[(r, c)] = s
                    self.faction_counter += 1

    def _is_ocean(self, r, c):
        return 0 <= r < self.H and 0 <= c < self.W and self.grid[r, c] == 10

    def _is_coastal(self, r, c):
        for dr, dc in NEIGHBORS_4:
            if self._is_ocean(r + dr, c + dc):
                return True
        return False

    def _land_neighbors(self, r, c, radius=1):
        cells = []
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.H and 0 <= nc < self.W:
                    if self.grid[nr, nc] != 10:
                        cells.append((nr, nc))
        return cells

    def _dist(self, r1, c1, r2, c2):
        return np.sqrt((r1 - r2) ** 2 + (c1 - c2) ** 2)

    def run(self) -> np.ndarray:
        for year in range(self.p.n_years):
            self._growth()
            self._conflict()
            self._trade()
            self._winter()
            self._environment()
        return self.grid.copy()

    def _remove_dead(self):
        dead = [k for k, s in self.settlements.items() if s.population <= 0]
        for k in dead:
            self.grid[k[0], k[1]] = 3
            del self.settlements[k]

    def _growth(self):
        p = self.p
        self._remove_dead()
        to_expand = []

        for (r, c), s in list(self.settlements.items()):
            food_gain = 0.0
            for nr, nc in self._land_neighbors(r, c):
                t = self.grid[nr, nc]
                if t in (0, 11):
                    food_gain += p.food_per_plains
                elif t == 4:
                    food_gain += p.food_per_forest
            for dr, dc in NEIGHBORS_4:
                if self._is_ocean(r + dr, c + dc):
                    food_gain += p.food_per_ocean
            if s.is_port:
                food_gain += p.food_per_ocean * 3

            s.food += food_gain
            s.age += 1

            surplus = s.food - s.population * p.food_consume
            if surplus > 0:
                s.population = min(s.population + p.growth_rate * surplus, p.max_pop)

            if not s.is_port and s.population >= p.port_pop_threshold and self._is_coastal(r, c):
                s.is_port = True
                self.grid[r, c] = 2

            if s.is_port:
                s.longships = min(s.longships + p.longship_rate, 10)

            if s.population >= p.expand_pop:
                to_expand.append(s)

        for s in to_expand:
            if self.rng.random() > p.expand_prob:
                continue
            candidates = []
            for nr, nc in self._land_neighbors(s.row, s.col, radius=p.expand_radius):
                t = self.grid[nr, nc]
                if t in (0, 11, 3, 4) and (nr, nc) not in self.settlements:
                    d = self._dist(s.row, s.col, nr, nc)
                    candidates.append((nr, nc, d))
            if not candidates:
                continue
            candidates.sort(key=lambda x: x[2])
            top_n = max(1, len(candidates) // 3)
            idx = self.rng.integers(top_n)
            nr, nc, _ = candidates[idx]
            new_pop = s.population * 0.25
            s.population -= new_pop
            is_port = self._is_coastal(nr, nc) and new_pop >= p.port_pop_threshold * 0.3
            new_s = Settlement(
                row=nr, col=nc, population=new_pop,
                food=s.food * 0.2, wealth=s.wealth * 0.1,
                defense=s.defense * 0.5, tech=s.tech,
                is_port=is_port, longships=0, faction=s.faction,
            )
            s.food *= 0.8
            self.settlements[(nr, nc)] = new_s
            self.grid[nr, nc] = 2 if is_port else 1

        # Forest clearing by settlements
        for (r, c), s in list(self.settlements.items()):
            if s.population < 50:
                continue
            for nr, nc in self._land_neighbors(r, c, radius=2):
                if self.grid[nr, nc] == 4 and (nr, nc) not in self.settlements:
                    if self.rng.random() < p.forest_clear_prob * (s.population / 200):
                        self.grid[nr, nc] = 11

    def _conflict(self):
        p = self.p
        items = list(self.settlements.items())
        if len(items) < 2:
            return
        self.rng.shuffle(items)

        for (r, c), attacker in items:
            if attacker.population <= 0:
                continue
            food_ratio = attacker.food / max(attacker.population * p.food_consume, 1)
            desperate = food_ratio < 0.5
            prob = p.raid_prob * (p.desperate_mult if desperate else 1.0)
            if self.rng.random() > prob:
                continue
            eff_range = p.raid_range + (np.sqrt(attacker.longships) * 1.5 if attacker.longships > 0 else 0)
            targets = []
            for (tr, tc), defender in self.settlements.items():
                if (tr, tc) == (r, c) or defender.population <= 0 or defender.faction == attacker.faction:
                    continue
                d = self._dist(r, c, tr, tc)
                if d <= eff_range:
                    targets.append(((tr, tc), defender, d))
            if not targets:
                continue
            targets.sort(key=lambda x: x[2])
            (tr, tc), defender, _ = targets[0]
            atk = attacker.population * (1 + attacker.tech * 0.1)
            dfn = defender.population * defender.defense * 0.1 * (1 + defender.tech * 0.1)
            if atk > dfn * 0.5:
                dmg = min(p.raid_damage * (atk / max(dfn, 1)), 0.6)
                defender.population *= (1 - dmg)
                loot = defender.wealth * p.loot_frac * dmg
                attacker.wealth += loot
                defender.wealth -= loot
                attacker.food += defender.food * p.loot_frac * dmg * 0.5
                defender.food *= (1 - p.loot_frac * dmg * 0.5)
                attacker.population *= (1 - dmg * 0.15)
                if defender.population < p.conquer_pop:
                    defender.faction = attacker.faction

    def _trade(self):
        p = self.p
        ports = [(k, s) for k, s in self.settlements.items() if s.is_port and s.population > 0]
        for i, ((r1, c1), s1) in enumerate(ports):
            for (r2, c2), s2 in ports[i + 1:]:
                if s1.faction != s2.faction and self.rng.random() > 0.3:
                    continue
                d = self._dist(r1, c1, r2, c2)
                if d <= p.trade_range:
                    s1.wealth += p.trade_wealth
                    s2.wealth += p.trade_wealth
                    s1.food += p.trade_food
                    s2.food += p.trade_food
                    avg = (s1.tech + s2.tech) / 2
                    s1.tech += (avg - s1.tech) * p.tech_rate
                    s2.tech += (avg - s2.tech) * p.tech_rate

    def _winter(self):
        p = self.p
        severity = max(0.1, self.rng.normal(p.winter_mean, p.winter_std))
        for (r, c), s in list(self.settlements.items()):
            if s.population <= 0:
                continue
            consumption = s.population * p.food_consume * severity
            s.food -= consumption
            if s.food < 0:
                s.population *= (1 - p.starve_loss * severity)
                s.food = 0
                if s.population < p.collapse_threshold:
                    self.grid[r, c] = 3
                    del self.settlements[(r, c)]

    def _environment(self):
        p = self.p
        ruin_cells = [(r, c) for r in range(self.H) for c in range(self.W)
                      if self.grid[r, c] == 3 and (r, c) not in self.settlements]

        for r, c in ruin_cells:
            reclaimed = False
            for nr, nc in self._land_neighbors(r, c, radius=2):
                if (nr, nc) in self.settlements:
                    s = self.settlements[(nr, nc)]
                    if s.population > self.p.expand_pop * 0.4 and self.rng.random() < p.ruin_settle_rate:
                        is_port = self._is_coastal(r, c) and s.is_port
                        new_pop = s.population * 0.1
                        s.population -= new_pop
                        new_s = Settlement(
                            row=r, col=c, population=new_pop,
                            food=s.food * 0.05, wealth=0,
                            defense=s.defense * 0.3, tech=s.tech,
                            is_port=is_port, longships=0, faction=s.faction,
                        )
                        s.food *= 0.95
                        self.settlements[(r, c)] = new_s
                        self.grid[r, c] = 2 if is_port else 1
                        reclaimed = True
                        break
            if reclaimed:
                continue
            if self.rng.random() < p.ruin_nature_rate:
                if self.rng.random() < p.ruin_to_forest:
                    self.grid[r, c] = 4
                else:
                    self.grid[r, c] = 11

        # Forest spread
        if p.forest_spread_prob > 0:
            for r in range(self.H):
                for c in range(self.W):
                    if self.grid[r, c] in (0, 11) and (r, c) not in self.settlements:
                        for dr, dc in NEIGHBORS_4:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < self.H and 0 <= nc < self.W and self.grid[nr, nc] == 4:
                                if self.rng.random() < p.forest_spread_prob:
                                    self.grid[r, c] = 4
                                break


# ─── Public API ─────────────────────────────────────────

GRID_TO_CLASS = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 10: 0, 11: 0}


def simulate_once(initial_grid: np.ndarray, params: SimParams, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sim = Simulation(initial_grid, params, rng)
    final = sim.run()
    cls = np.zeros((sim.H, sim.W), dtype=int)
    for gv, c in GRID_TO_CLASS.items():
        cls[final == gv] = c
    return cls


def simulate_distribution(initial_grid: np.ndarray, params: SimParams,
                          n_sims: int = 200, base_seed: int = 42) -> np.ndarray:
    H, W = initial_grid.shape
    counts = np.zeros((H, W, 6), dtype=np.float64)
    for i in range(n_sims):
        cls = simulate_once(initial_grid, params, seed=base_seed + i)
        for c in range(6):
            counts[:, :, c] += (cls == c)
    probs = counts / n_sims
    probs = np.clip(probs, 1e-6, None)
    probs /= probs.sum(axis=-1, keepdims=True)
    return probs
