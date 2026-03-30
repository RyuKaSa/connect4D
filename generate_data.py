"""
Connect Four 3D — Generate training data from MCTS self-play.

Runs MCTS-vs-MCTS games and records (state, move, result) triples.
Output: a .npz file with arrays ready for PyTorch training.

Usage:
    python generate_data.py                         # defaults: 500 games, MCTS-5k
    python generate_data.py --games 2000 --iters 5000 --output data.npz

The state encoding is canonical (current player's pieces first) so the
network learns to play from either side.  The result is stored from the
perspective of the player who made the move (+1 win, -1 loss, 0 draw).
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from engine import Game
from agents import MCTSAgent, board_to_features


def generate_games(
    n_games: int = 500,
    mcts_iters: int = 5000,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Play *n_games* of MCTS self-play, collecting training data.

    Returns:
        states:  (N, 128) float32 — board features at each move
        moves:   (N,)     int64   — peg chosen by MCTS
        results: (N,)     float32 — game outcome from mover's perspective
    """
    agent = MCTSAgent(iterations=mcts_iters)

    all_states: list[list[float]] = []
    all_moves: list[int] = []
    all_players: list[int] = []  # who moved (needed to compute results)
    game_boundaries: list[int] = []  # index where each game starts

    t0 = time.perf_counter()

    for g_idx in range(n_games):
        game = Game()
        game_start = len(all_states)
        game_boundaries.append(game_start)

        while not game.is_over():
            # Snapshot state BEFORE the move (from current player's perspective)
            features = board_to_features(game.board)
            player = game.turn
            move = agent.choose_move(game)

            all_states.append(features)
            all_moves.append(move)
            all_players.append(player)

            game.place(move)

        # Assign results for this game
        result = game.result  # 0, 1, or "draw"

        if verbose:
            elapsed = time.perf_counter() - t0
            n_samples = len(all_states)
            print(
                f"  game {g_idx + 1:>{len(str(n_games))}}/{n_games}  "
                f"samples={n_samples}  {elapsed:.1f}s",
                end="\r",
            )

    # Convert results: +1 if the mover won, -1 if lost, 0 if draw
    all_results = []
    game_boundaries.append(len(all_states))  # sentinel
    for g_idx in range(n_games):
        start = game_boundaries[g_idx]
        end = game_boundaries[g_idx + 1]
        # Recover the game result — we need to replay to find it
        # Actually we stored all_players, so we can look at the final game
        # Let's just replay quickly
        game = Game()
        for i in range(start, end):
            game.place(all_moves[i])
        result = game.result

        for i in range(start, end):
            player = all_players[i]
            if result == "draw":
                all_results.append(0.0)
            elif result == player:
                all_results.append(1.0)
            else:
                all_results.append(-1.0)

    if verbose:
        elapsed = time.perf_counter() - t0
        print(f"\n  Done: {len(all_states)} samples from {n_games} games in {elapsed:.1f}s\n")

    states = np.array(all_states, dtype=np.float32)
    moves = np.array(all_moves, dtype=np.int64)
    results = np.array(all_results, dtype=np.float32)

    return states, moves, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate MCTS self-play data")
    parser.add_argument("--games", type=int, default=500, help="Number of games")
    parser.add_argument("--iters", type=int, default=5000, help="MCTS iterations per move")
    parser.add_argument("--output", type=str, default="training_data.npz", help="Output file")
    args = parser.parse_args()

    print(f"\n  Generating {args.games} games with MCTS-{args.iters}...")
    print(f"  Output: {args.output}\n")

    states, moves, results = generate_games(
        n_games=args.games,
        mcts_iters=args.iters,
    )

    np.savez_compressed(args.output, states=states, moves=moves, results=results)
    print(f"  Saved {len(states)} samples to {args.output}")
    print(f"  States shape: {states.shape}")
    print(f"  Moves shape:  {moves.shape}")
    print(f"  Results distribution: wins={np.sum(results == 1)}, "
          f"losses={np.sum(results == -1)}, draws={np.sum(results == 0)}")