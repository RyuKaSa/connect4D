"""
Glicko-2 rating system.

Reference: Mark Glickman, "Example of the Glicko-2 system" (2013).
http://www.glicko.net/glicko/glicko2.pdf

Usage:
    player = Rating()                    # 1500 / 350 / 0.06
    player = glicko2_update(player, opponents, outcomes)

All internal math uses Glicko-2 scale (mu/phi/sigma).
Public-facing Rating stores Glicko-1 scale (rating/rd) for readability.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# ── Constants ──────────────────────────────────────────────
MU_SCALE = 173.7178        # Glicko-1 ↔ Glicko-2 conversion factor
BASE_RATING = 1500.0
BASE_RD = 350.0
BASE_VOL = 0.06
TAU = 0.5                  # system volatility constraint (Glickman recommends 0.3–1.2)
EPSILON = 1e-6             # convergence threshold for volatility iteration


# ── Rating object ──────────────────────────────────────────

@dataclass
class Rating:
    """A player's Glicko-2 rating (stored in Glicko-1 scale for readability)."""
    rating: float = BASE_RATING
    rd: float = BASE_RD
    vol: float = BASE_VOL
    games: int = 0

    # Glicko-2 internal scale
    @property
    def mu(self) -> float:
        return (self.rating - BASE_RATING) / MU_SCALE

    @property
    def phi(self) -> float:
        return self.rd / MU_SCALE

    def to_dict(self) -> dict:
        return {
            "rating": round(self.rating, 2),
            "rd": round(self.rd, 2),
            "vol": round(self.vol, 6),
            "games": self.games,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Rating:
        return cls(
            rating=d["rating"],
            rd=d["rd"],
            vol=d["vol"],
            games=d.get("games", 0),
        )


# ── Core update ────────────────────────────────────────────

def glicko2_update(
    player: Rating,
    opponents: list[Rating],
    outcomes: list[float],       # 1.0 = win, 0.5 = draw, 0.0 = loss
) -> Rating:
    """Compute a new Rating for *player* after a rating period.

    *opponents* and *outcomes* are parallel lists: one entry per game played.
    If no games were played, RD increases (uncertainty grows).
    """
    mu = player.mu
    phi = player.phi
    sigma = player.vol

    if not opponents:
        # No games: RD increases via volatility
        phi_star = math.sqrt(phi ** 2 + sigma ** 2)
        return Rating(
            rating=player.rating,
            rd=phi_star * MU_SCALE,
            vol=sigma,
            games=player.games,
        )

    # Step 3: Compute variance (v) and delta
    g_vals = [_g(opp.phi) for opp in opponents]
    e_vals = [_E(mu, opp.mu, g_j) for opp, g_j in zip(opponents, g_vals)]

    v_inv = sum(g_j ** 2 * e_j * (1 - e_j) for g_j, e_j in zip(g_vals, e_vals))
    v = 1.0 / v_inv

    delta = v * sum(
        g_j * (s_j - e_j)
        for g_j, e_j, s_j in zip(g_vals, e_vals, outcomes)
    )

    # Step 4: Compute new volatility (sigma')
    sigma_new = _new_volatility(sigma, phi, v, delta)

    # Step 5: Pre-rating RD
    phi_star = math.sqrt(phi ** 2 + sigma_new ** 2)

    # Step 6: New rating and RD
    phi_new = 1.0 / math.sqrt(1.0 / phi_star ** 2 + 1.0 / v)
    mu_new = mu + phi_new ** 2 * sum(
        g_j * (s_j - e_j)
        for g_j, e_j, s_j in zip(g_vals, e_vals, outcomes)
    )

    return Rating(
        rating=BASE_RATING + mu_new * MU_SCALE,
        rd=phi_new * MU_SCALE,
        vol=sigma_new,
        games=player.games + len(opponents),
    )


# ── Glicko-2 helper functions ─────────────────────────────

def _g(phi: float) -> float:
    """Reduce impact of opponents with high RD."""
    return 1.0 / math.sqrt(1 + 3 * phi ** 2 / math.pi ** 2)


def _E(mu: float, mu_j: float, g_j: float) -> float:
    """Expected outcome against opponent j."""
    return 1.0 / (1 + math.exp(-g_j * (mu - mu_j)))


def _new_volatility(sigma: float, phi: float, v: float, delta: float) -> float:
    """Illinois algorithm (Step 4 of Glickman's paper)."""
    a = math.log(sigma ** 2)
    tau2 = TAU ** 2

    def f(x: float) -> float:
        ex = math.exp(x)
        d2 = delta ** 2
        p2 = phi ** 2
        num1 = ex * (d2 - p2 - v - ex)
        den1 = 2 * (p2 + v + ex) ** 2
        return num1 / den1 - (x - a) / tau2

    # Bracket: A = a, find B
    A = a
    if delta ** 2 > phi ** 2 + v:
        B = math.log(delta ** 2 - phi ** 2 - v)
    else:
        k = 1
        while f(a - k * TAU) < 0:
            k += 1
        B = a - k * TAU

    # Bisection (Illinois variant)
    fA = f(A)
    fB = f(B)

    while abs(B - A) > EPSILON:
        C = A + (A - B) * fA / (fB - fA)
        fC = f(C)
        if fC * fB <= 0:
            A = B
            fA = fB
        else:
            fA /= 2
        B = C
        fB = fC

    return math.exp(A / 2)