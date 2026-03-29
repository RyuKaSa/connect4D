"""
Connect Four 3D — Headless arena with Glicko-2 ratings.

Usage:
    python arena.py <agent1> <agent2> [n_games]     # head-to-head match
    python arena.py tournament [n_games_per_pair]    # round-robin + ratings
    python arena.py ratings                          # show current leaderboard

All matchups alternate colors (half the games each side) to cancel
first-player advantage. Ratings persist in ratings.json.
"""

from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

from engine import Game
from agents import RandomAgent, MCTSAgent
from glicko2 import Rating, glicko2_update

RATINGS_FILE = Path(__file__).parent / "ratings.json"


# ── Match runner ───────────────────────────────────────────

def run_match(
    agent0,
    agent1,
    n_games: int = 100,
    verbose: bool = True,
) -> tuple[list[int], int]:
    """Play *n_games* between two agents with color alternation.

    Odd-indexed games swap who plays first, so each agent gets
    roughly equal time as P1 and P2. Returns ([w_agent0, w_agent1], draws).
    """
    wins = [0, 0]
    draws = 0
    t0 = time.perf_counter()

    for i in range(n_games):
        # Alternate who goes first
        if i % 2 == 0:
            players = [agent0, agent1]
            mapping = {0: 0, 1: 1}      # game player → agent index
        else:
            players = [agent1, agent0]
            mapping = {0: 1, 1: 0}

        game = Game()
        while not game.is_over():
            move = players[game.turn].choose_move(game)
            game.place(move)

        if game.result == "draw":
            draws += 1
        else:
            winner_agent = mapping[game.result]
            wins[winner_agent] += 1

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
        w0, w1 = wins
        total = n_games
        print(f"\n{'─' * 50}")
        print(f"  {agent0.name}  vs  {agent1.name}  ({total} games, {elapsed:.1f}s)")
        print(f"  (color-alternated: {total // 2} games each side)")
        print(f"{'─' * 50}")
        print(f"  {agent0.name:>12}: {w0:>4} wins  ({100 * w0 / total:5.1f}%)")
        print(f"  {agent1.name:>12}: {w1:>4} wins  ({100 * w1 / total:5.1f}%)")
        print(f"  {'Draws':>12}: {draws:>4}       ({100 * draws / total:5.1f}%)")
        print(f"{'─' * 50}\n")

    return wins, draws


# ── Ratings I/O ────────────────────────────────────────────

def load_ratings() -> dict[str, Rating]:
    if RATINGS_FILE.exists():
        data = json.loads(RATINGS_FILE.read_text())
        return {name: Rating.from_dict(d) for name, d in data.items()}
    return {}


def save_ratings(ratings: dict[str, Rating]) -> None:
    data = {name: r.to_dict() for name, r in ratings.items()}
    RATINGS_FILE.write_text(json.dumps(data, indent=2) + "\n")


def print_leaderboard(ratings: dict[str, Rating]) -> None:
    if not ratings:
        print("No ratings yet. Run a tournament first.")
        return

    ranked = sorted(ratings.items(), key=lambda kv: kv[1].rating, reverse=True)
    print(f"\n{'─' * 52}")
    print(f"  {'Rank':>4}  {'Agent':<14} {'Rating':>7} {'RD':>6} {'Games':>6}")
    print(f"{'─' * 52}")
    for i, (name, r) in enumerate(ranked, 1):
        print(f"  {i:>4}  {name:<14} {r.rating:>7.1f} {r.rd:>6.1f} {r.games:>6}")
    print(f"{'─' * 52}\n")


# ── Tournament ─────────────────────────────────────────────

def run_tournament(agent_map: dict[str, object], n_per_pair: int = 20) -> None:
    """Round-robin tournament: every pair plays n_per_pair games (color-alternated).
    Updates Glicko-2 ratings and saves them."""

    names = list(agent_map.keys())
    ratings = load_ratings()

    # Ensure all agents have a rating entry
    for name in names:
        if name not in ratings:
            ratings[name] = Rating()

    # Collect per-agent results for this rating period
    # results[name] = list of (opponent_rating_snapshot, outcome)
    results: dict[str, list[tuple[Rating, float]]] = {n: [] for n in names}

    pairs = list(itertools.combinations(names, 2))
    total_pairs = len(pairs)

    print(f"\n{'═' * 52}")
    print(f"  TOURNAMENT: {len(names)} agents, {total_pairs} pairs, "
          f"{n_per_pair} games/pair")
    print(f"{'═' * 52}\n")

    for pair_idx, (n0, n1) in enumerate(pairs, 1):
        print(f"  [{pair_idx}/{total_pairs}]  {n0}  vs  {n1}")
        a0 = agent_map[n0]
        a1 = agent_map[n1]
        wins, draws = run_match(a0, a1, n_games=n_per_pair, verbose=True)

        # Record outcomes for Glicko-2
        # Agent 0's perspective
        r0_snap = Rating(rating=ratings[n1].rating, rd=ratings[n1].rd, vol=ratings[n1].vol)
        r1_snap = Rating(rating=ratings[n0].rating, rd=ratings[n0].rd, vol=ratings[n0].vol)

        for _ in range(wins[0]):
            results[n0].append((r0_snap, 1.0))
            results[n1].append((r1_snap, 0.0))
        for _ in range(wins[1]):
            results[n0].append((r0_snap, 0.0))
            results[n1].append((r1_snap, 1.0))
        for _ in range(draws):
            results[n0].append((r0_snap, 0.5))
            results[n1].append((r1_snap, 0.5))

    # Glicko-2 batch update (one rating period = entire tournament)
    print(f"\n  Updating Glicko-2 ratings...")
    for name in names:
        opps = [r for r, _ in results[name]]
        outs = [o for _, o in results[name]]
        ratings[name] = glicko2_update(ratings[name], opps, outs)

    save_ratings(ratings)
    print(f"  Saved to {RATINGS_FILE}")
    print_leaderboard(ratings)


# ── Agent registry ─────────────────────────────────────────

AGENT_REGISTRY: dict[str, callable] = {
    "random":   RandomAgent,
    "mcts-1k":  lambda: MCTSAgent(1000),
    "mcts-2k":  lambda: MCTSAgent(2000),
    "mcts-5k":  lambda: MCTSAgent(5000),
}


def _resolve(name: str):
    key = name.lower()
    if key in AGENT_REGISTRY:
        factory = AGENT_REGISTRY[key]
        return factory() if callable(factory) else factory
    raise SystemExit(
        f"Unknown agent '{name}'. Options: {', '.join(AGENT_REGISTRY.keys())}"
    )


# ── CLI ────────────────────────────────────────────────────

def _usage():
    agents = ", ".join(AGENT_REGISTRY.keys())
    print("Usage:")
    print(f"  python arena.py <agent1> <agent2> [n_games]              # head-to-head")
    print(f"  python arena.py tournament [-n N] [agent1 agent2 ...]    # round-robin + Glicko-2")
    print(f"  python arena.py ratings                                  # show leaderboard")
    print(f"\nAgents: {agents}")
    print(f"\nExamples:")
    print(f"  python arena.py mcts-1k random 50")
    print(f"  python arena.py tournament -n 20 random mcts-1k mcts-2k")
    print(f"  python arena.py tournament                               # all agents, 20 games/pair")


if __name__ == "__main__":
    args = sys.argv[1:]

    if not args:
        _usage()
        sys.exit(0)

    cmd = args[0].lower()

    if cmd == "ratings":
        print_leaderboard(load_ratings())

    elif cmd == "tournament":
        rest = args[1:]
        n = 20
        # Parse optional -n flag
        if len(rest) >= 2 and rest[0] == "-n":
            n = int(rest[1])
            rest = rest[2:]
        # Remaining args are agent names; if none, use all
        if rest:
            names = [r.lower() for r in rest]
            for name in names:
                if name not in AGENT_REGISTRY:
                    raise SystemExit(
                        f"Unknown agent '{name}'. Options: {', '.join(AGENT_REGISTRY.keys())}"
                    )
        else:
            names = list(AGENT_REGISTRY.keys())
        agent_map = {name: _resolve(name) for name in names}
        run_tournament(agent_map, n_per_pair=n)

    elif cmd in AGENT_REGISTRY or len(args) >= 2:
        if len(args) < 2:
            _usage()
            sys.exit(1)
        a0 = _resolve(args[0])
        a1 = _resolve(args[1])
        n = int(args[2]) if len(args) > 2 else 50
        run_match(a0, a1, n_games=n)

    else:
        _usage()
        sys.exit(1)