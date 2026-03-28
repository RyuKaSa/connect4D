"""
Connect Four 3D — Headless arena.

Usage:
    python arena.py                       # default matchups
    python arena.py random random 200     # custom: agent1 agent2 n_games
"""

from __future__ import annotations

import sys
import time

from engine import Game
from agents import RandomAgent, MCTSAgent


def run_match(
    agent0,
    agent1,
    n_games: int = 100,
    verbose: bool = True,
) -> tuple[list[int], int]:
    """Play *n_games* between two agents. Returns ([w0, w1], draws)."""
    agents = [agent0, agent1]
    wins = [0, 0]
    draws = 0
    t0 = time.perf_counter()

    for i in range(n_games):
        game = Game()
        while not game.is_over():
            move = agents[game.turn].choose_move(game)
            game.place(move)

        if game.result == 0:
            wins[0] += 1
        elif game.result == 1:
            wins[1] += 1
        else:
            draws += 1

        if verbose:
            elapsed = time.perf_counter() - t0
            print(
                f"  game {i + 1:>{len(str(n_games))}}/{n_games}  "
                f"[{wins[0]}-{wins[1]}-{draws}]  "
                f"{elapsed:.1f}s",
                end="\r",
            )

    elapsed = time.perf_counter() - t0

    if verbose:
        print()
        print(f"\n{'─' * 44}")
        print(f"  {agent0.name}  vs  {agent1.name}  ({n_games} games, {elapsed:.1f}s)")
        print(f"{'─' * 44}")
        print(f"  {agent0.name:>12}: {wins[0]:>4} wins  ({100 * wins[0] / n_games:5.1f}%)")
        print(f"  {agent1.name:>12}: {wins[1]:>4} wins  ({100 * wins[1] / n_games:5.1f}%)")
        print(f"  {'Draws':>12}: {draws:>4}       ({100 * draws / n_games:5.1f}%)")
        print(f"{'─' * 44}\n")

    return wins, draws


# ── Agent lookup by name ──────────────────────────────────

_AGENTS = {
    "random": RandomAgent,
    "mcts-1k": lambda: MCTSAgent(1000),
    "mcts-2k": lambda: MCTSAgent(2000),
    "mcts-5k": lambda: MCTSAgent(5000),
}


def _resolve(name: str):
    key = name.lower()
    if key in _AGENTS:
        factory = _AGENTS[key]
        return factory() if callable(factory) else factory
    raise SystemExit(
        f"Unknown agent '{name}'. Options: {', '.join(_AGENTS.keys())}"
    )


# ── CLI ────────────────────────────────────────────────────

if __name__ == "__main__":
    args = sys.argv[1:]

    if args:
        if len(args) < 2:
            print("Usage: python arena.py <agent1> <agent2> [n_games]")
            print(f"Agents: {', '.join(_AGENTS.keys())}")
            sys.exit(1)
        a0 = _resolve(args[0])
        a1 = _resolve(args[1])
        n = int(args[2]) if len(args) > 2 else 50
        run_match(a0, a1, n_games=n)
    else:
        # Default matchups
        print("=== Random vs Random ===")
        run_match(RandomAgent(), RandomAgent(), n_games=100)

        print("=== MCTS-1k vs Random ===")
        run_match(MCTSAgent(1000), RandomAgent(), n_games=20)