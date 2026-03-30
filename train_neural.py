"""
Connect Four 3D — Neural agent training pipeline.

Three stages:
  1. Behavioral cloning  — learn to imitate MCTS from recorded data
  2. DAgger               — play with learned policy, relabel with MCTS expert
  3. RL fine-tuning       — play against MCTS, improve via REINFORCE

Usage:
    # Stage 1: train on pre-generated data
    python train_neural.py bc --data training_data.npz --epochs 50

    # Stage 2: DAgger (generates new data using the learned policy + MCTS expert)
    python train_neural.py dagger --model neural_model.pt --rounds 3 --games 200

    # Stage 3: RL fine-tune against MCTS
    python train_neural.py rl --model neural_model.pt --games 500 --iters 5000

    # Quick eval: play the neural agent against MCTS in the arena
    python arena.py neural mcts-5k 50
"""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from engine import Game
from agents import (
    ConnectFourNet,
    MCTSAgent,
    NeuralAgent,
    RandomAgent,
    board_to_features,
)


# ── Training utilities ─────────────────────────────────────

def load_data(path: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load .npz data and return (states, moves, results) tensors."""
    data = np.load(path)
    states = torch.tensor(data["states"], dtype=torch.float32)
    moves = torch.tensor(data["moves"], dtype=torch.long)
    results = torch.tensor(data["results"], dtype=torch.float32)
    return states, moves, results


# ── D4 symmetry augmentation (8x data for free) ──────────
#
# The 4×4 board has dihedral symmetry D4: 4 rotations × 2 reflections.
# Pieces stack along Z, so Z is invariant. We permute (x,y) in the XY plane.
# Bit index = x + 4*y + 16*z,  Peg id = x + 4*y.

def _build_d4_tables():
    """Precompute permutation tables for the 8 D4 symmetries."""
    transforms = [
        lambda x, y: (x, y),           # identity
        lambda x, y: (3 - y, x),       # rot90
        lambda x, y: (3 - x, 3 - y),   # rot180
        lambda x, y: (y, 3 - x),       # rot270
        lambda x, y: (3 - x, y),       # flip X
        lambda x, y: (x, 3 - y),       # flip Y
        lambda x, y: (y, x),           # flip diagonal
        lambda x, y: (3 - y, 3 - x),   # flip anti-diagonal
    ]

    bit_perms = []   # each is a list of 64 ints: new_index[old_index]
    peg_perms = []   # each is a list of 16 ints: new_peg[old_peg]

    for T in transforms:
        bp = [0] * 64
        pp = [0] * 16
        for x in range(4):
            for y in range(4):
                nx, ny = T(x, y)
                old_peg = x + 4 * y
                new_peg = nx + 4 * ny
                pp[old_peg] = new_peg
                for z in range(4):
                    old_bit = x + 4 * y + 16 * z
                    new_bit = nx + 4 * ny + 16 * z
                    bp[old_bit] = new_bit
        bit_perms.append(bp)
        peg_perms.append(pp)

    return bit_perms, peg_perms

_D4_BIT_PERMS, _D4_PEG_PERMS = _build_d4_tables()


def augment_d4(
    states: np.ndarray,   # (N, 128)
    moves: np.ndarray,    # (N,)
    results: np.ndarray,  # (N,)
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply all 8 D4 symmetries, returning 8N samples.

    Skips the identity to avoid duplicating originals, then stacks.
    """
    n = len(states)
    all_states = [states]      # start with originals
    all_moves = [moves]
    all_results = [results]

    for sym_idx in range(1, 8):   # skip identity (index 0)
        bp = _D4_BIT_PERMS[sym_idx]
        pp = _D4_PEG_PERMS[sym_idx]

        # Permute features: first 64 bits and second 64 bits use same permutation
        new_states = np.empty_like(states)
        for old, new in enumerate(bp):
            new_states[:, new] = states[:, old]            # channel 0 (own pieces)
            new_states[:, 64 + new] = states[:, 64 + old]  # channel 1 (opp pieces)

        # Permute peg IDs
        peg_map = np.array(pp, dtype=np.int64)
        new_moves = peg_map[moves]

        all_states.append(new_states)
        all_moves.append(new_moves)
        all_results.append(results)   # results don't change under symmetry

    return (
        np.concatenate(all_states),
        np.concatenate(all_moves),
        np.concatenate(all_results),
    )


def train_epoch(
    model: ConnectFourNet,
    optimizer: torch.optim.Optimizer,
    states: torch.Tensor,
    moves: torch.Tensor,
    batch_size: int = 256,
) -> float:
    """One epoch of supervised policy training.  Returns mean loss."""
    model.train()
    n = len(states)
    perm = torch.randperm(n)
    total_loss = 0.0
    n_batches = 0

    for i in range(0, n, batch_size):
        idx = perm[i : i + batch_size]
        x = states[idx]
        y = moves[idx]

        logits = model(x)
        loss = F.cross_entropy(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def evaluate_accuracy(
    model: ConnectFourNet,
    states: torch.Tensor,
    moves: torch.Tensor,
) -> float:
    """Top-1 accuracy on held-out data."""
    model.eval()
    with torch.no_grad():
        logits = model(states)
        preds = logits.argmax(dim=1)
    return (preds == moves).float().mean().item()


# ── Stage 1: Behavioral Cloning ───────────────────────────

def train_bc(
    data_path: str,
    model_path: str = "neural_model.pt",
    epochs: int = 50,
    lr: float = 1e-3,
    val_split: float = 0.1,
):
    """Train a policy network to imitate MCTS moves."""
    print(f"\n{'═' * 50}")
    print(f"  BEHAVIORAL CLONING")
    print(f"{'═' * 50}\n")

    data = np.load(data_path)
    raw_states = data["states"]
    raw_moves = data["moves"]
    raw_results = data["results"]
    n = len(raw_states)
    print(f"  Raw data: {n} samples from {data_path}")

    # Train/val split BEFORE augmentation (val stays un-augmented for honest eval)
    n_val = int(n * val_split)
    perm = np.random.permutation(n)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    val_states = torch.tensor(raw_states[val_idx], dtype=torch.float32)
    val_moves = torch.tensor(raw_moves[val_idx], dtype=torch.long)

    # Augment training set with D4 symmetries (8x)
    aug_states, aug_moves, aug_results = augment_d4(
        raw_states[train_idx], raw_moves[train_idx], raw_results[train_idx]
    )
    train_states = torch.tensor(aug_states, dtype=torch.float32)
    train_moves = torch.tensor(aug_moves, dtype=torch.long)

    print(f"  Train: {len(train_states)} (augmented 8x from {len(train_idx)})")
    print(f"  Val: {len(val_states)} (un-augmented)")

    model = ConnectFourNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    best_val_acc = 0.0
    t0 = time.perf_counter()

    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, optimizer, train_states, train_moves)
        val_acc = evaluate_accuracy(model, val_states, val_moves)
        train_acc = evaluate_accuracy(model, train_states, train_moves)
        elapsed = time.perf_counter() - t0

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_path)

        print(
            f"  epoch {epoch:>3}/{epochs}  "
            f"loss={loss:.4f}  "
            f"train_acc={train_acc:.3f}  "
            f"val_acc={val_acc:.3f}  "
            f"best={best_val_acc:.3f}  "
            f"{elapsed:.1f}s"
            f"{'  *' if val_acc >= best_val_acc else ''}"
        )

    elapsed = time.perf_counter() - t0
    print(f"\n  Best val accuracy: {best_val_acc:.3f}")
    print(f"  Model saved to {model_path}")
    print(f"  Time: {elapsed:.1f}s\n")


# ── Stage 2: DAgger ───────────────────────────────────────

def run_dagger(
    model_path: str = "neural_model.pt",
    data_path: str = "training_data.npz",
    rounds: int = 3,
    games_per_round: int = 200,
    mcts_iters: int = 5000,
    epochs_per_round: int = 20,
    lr: float = 5e-4,
):
    """DAgger: play with learned policy, relabel states with MCTS expert,
    aggregate into training set, retrain."""
    print(f"\n{'═' * 50}")
    print(f"  DAGGER ({rounds} rounds, {games_per_round} games each)")
    print(f"{'═' * 50}\n")

    # Load existing data
    existing = np.load(data_path)
    all_states = list(existing["states"])
    all_moves = list(existing["moves"])
    all_results = list(existing["results"])
    print(f"  Starting with {len(all_states)} existing samples")

    expert = MCTSAgent(iterations=mcts_iters)

    for round_idx in range(1, rounds + 1):
        print(f"\n  ── DAgger round {round_idx}/{rounds} ──")

        # Load current model as the rollout policy
        learner = NeuralAgent(model_path)
        new_states = []
        new_moves = []
        new_results = []

        t0 = time.perf_counter()

        for g_idx in range(games_per_round):
            game = Game()
            episode_states = []
            episode_expert_moves = []
            episode_players = []

            while not game.is_over():
                features = board_to_features(game.board)
                player = game.turn

                # Learner picks the move to play (defines the visited state distribution)
                learner_move = learner.choose_move(game)
                # Expert labels the state (what MCTS would play from here)
                expert_move = expert.choose_move(game)

                episode_states.append(features)
                episode_expert_moves.append(expert_move)
                episode_players.append(player)

                # Execute the LEARNER's move (DAgger key insight)
                game.place(learner_move)

            # Compute results for this episode
            result = game.result
            for i, player in enumerate(episode_players):
                if result == "draw":
                    new_results.append(0.0)
                elif result == player:
                    new_results.append(1.0)
                else:
                    new_results.append(-1.0)

            new_states.extend(episode_states)
            new_moves.extend(episode_expert_moves)

            elapsed = time.perf_counter() - t0
            print(
                f"    game {g_idx + 1:>{len(str(games_per_round))}}/{games_per_round}  "
                f"new_samples={len(new_states)}  "
                f"moves={len(episode_states)}  "
                f"result={'W' if result == 0 else 'L' if result == 1 else 'D'}  "
                f"{elapsed:.1f}s",
                end="\r",
            )

        # Aggregate
        print()  # newline after \r progress
        all_states.extend(new_states)
        all_moves.extend(new_moves)
        all_results.extend(new_results)

        print(f"  Aggregated: {len(all_states)} total samples")

        # Save aggregated data
        np.savez_compressed(
            data_path,
            states=np.array(all_states, dtype=np.float32),
            moves=np.array(all_moves, dtype=np.int64),
            results=np.array(all_results, dtype=np.float32),
        )

        # Retrain on full aggregated dataset (augmented with D4)
        raw_s = np.array(all_states, dtype=np.float32)
        raw_m = np.array(all_moves, dtype=np.int64)
        raw_r = np.array(all_results, dtype=np.float32)
        aug_s, aug_m, aug_r = augment_d4(raw_s, raw_m, raw_r)
        states_t = torch.from_numpy(aug_s)
        moves_t = torch.from_numpy(aug_m)
        print(f"  Training on {len(states_t)} samples (augmented 8x from {len(raw_s)})")

        model = ConnectFourNet()
        model.load_state_dict(
            torch.load(model_path, map_location="cpu", weights_only=True)
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

        for epoch in range(1, epochs_per_round + 1):
            loss = train_epoch(model, optimizer, states_t, moves_t)
            acc = evaluate_accuracy(model, states_t, moves_t)
            print(
                f"    epoch {epoch:>2}/{epochs_per_round}  "
                f"loss={loss:.4f}  acc={acc:.3f}"
            )

        torch.save(model.state_dict(), model_path)
        print(f"  Model saved to {model_path}")

    print(f"\n  DAgger complete. Final dataset: {len(all_states)} samples\n")


# ── Stage 3: RL Fine-tuning (REINFORCE with curriculum) ───

# Curriculum: start easy, ramp up when win rate is high enough
RL_CURRICULUM = [
    (50,   0.55),   # beat MCTS-50 at 55%+ → advance
    (200,  0.50),   # beat MCTS-200 at 50%+ → advance
    (500,  0.45),   # beat MCTS-500 at 45%+ → advance
    (1000, 0.40),   # beat MCTS-1000 at 40%+ → advance
    (2000, 0.35),   # beat MCTS-2000 at 35%+ → advance
    (5000, 0.30),   # final boss
]

def train_rl(
    model_path: str = "neural_model.pt",
    n_games: int = 500,
    opponent_iters: int = 5000,  # ignored if curriculum, kept for CLI compat
    lr: float = 1e-4,
    gamma: float = 1.0,
    eval_every: int = 25,
    window: int = 50,           # rolling window for win rate
):
    """REINFORCE with curriculum: start against weak MCTS, ramp up as win rate
    improves. Saves best model checkpoint and rolls back on collapse."""
    print(f"\n{'═' * 56}")
    print(f"  RL FINE-TUNING (REINFORCE, curriculum)")
    print(f"{'═' * 56}\n")

    # Back up the pre-RL model so we can recover
    backup_path = model_path.replace(".pt", "_pre_rl.pt")
    shutil.copy2(model_path, backup_path)
    print(f"  Backed up pre-RL model to {backup_path}")

    model = ConnectFourNet()
    model.load_state_dict(
        torch.load(model_path, map_location="cpu", weights_only=True)
    )
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Curriculum state
    level = 0
    opp_iters, advance_wr = RL_CURRICULUM[level]
    opponent = MCTSAgent(iterations=opp_iters)
    print(f"  Starting at level {level}: MCTS-{opp_iters} (advance at {advance_wr*100:.0f}%)\n")

    wins, losses, draws = 0, 0, 0
    recent_results: list[float] = []   # 1.0 = win, 0.0 = loss, 0.5 = draw
    best_wr = 0.0
    t0 = time.perf_counter()

    for g_idx in range(n_games):
        game = Game()
        neural_color = g_idx % 2

        log_probs = []

        while not game.is_over():
            if game.turn == neural_color:
                features = board_to_features(game.board)
                x = torch.tensor([features], dtype=torch.float32)
                logits = model(x).squeeze(0)

                legal = game.legal_pegs()
                mask = torch.full((16,), float("-inf"))
                for p in legal:
                    mask[p] = 0.0
                masked_logits = logits + mask

                probs = F.softmax(masked_logits, dim=0)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
                log_probs.append(dist.log_prob(action))

                game.place(action.item())
            else:
                move = opponent.choose_move(game)
                game.place(move)

        # Compute reward
        if game.result == "draw":
            reward = 0.0
            draws += 1
            recent_results.append(0.5)
        elif game.result == neural_color:
            reward = 1.0
            wins += 1
            recent_results.append(1.0)
        else:
            reward = -1.0
            losses += 1
            recent_results.append(0.0)

        # Rolling window
        if len(recent_results) > window:
            recent_results.pop(0)

        # REINFORCE with baseline (mean reward over window)
        if log_probs:
            baseline = sum(recent_results) / len(recent_results) if recent_results else 0.0
            # Shift reward: advantage = reward - baseline
            advantage = reward - baseline
            loss = -advantage * torch.stack(log_probs).sum()
            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping to prevent catastrophic updates
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Progress
        elapsed = time.perf_counter() - t0
        total = wins + losses + draws
        wr = wins / total * 100 if total > 0 else 0
        rolling_wr = sum(recent_results) / len(recent_results) if recent_results else 0
        print(
            f"  game {g_idx + 1:>{len(str(n_games))}}/{n_games}  "
            f"lvl={level} MCTS-{opp_iters}  "
            f"W/L/D={wins}/{losses}/{draws}  "
            f"roll={rolling_wr*100:.0f}%  "
            f"{'W' if reward > 0 else 'L' if reward < 0 else 'D'}  "
            f"{elapsed:.1f}s",
            end="\r",
        )

        # Summary at checkpoints
        if (g_idx + 1) % eval_every == 0:
            print(
                f"  game {g_idx + 1:>{len(str(n_games))}}/{n_games}  "
                f"lvl={level} MCTS-{opp_iters}  "
                f"W/L/D={wins}/{losses}/{draws}  "
                f"overall={wr:.1f}%  "
                f"rolling={rolling_wr*100:.0f}%  "
                f"{elapsed:.1f}s"
            )

            # Save best model
            if rolling_wr > best_wr and len(recent_results) >= eval_every:
                best_wr = rolling_wr
                torch.save(model.state_dict(), model_path)
                print(f"    → saved best model (rolling {best_wr*100:.0f}%)")

        # Curriculum advancement
        if (len(recent_results) >= window
                and rolling_wr >= advance_wr
                and level < len(RL_CURRICULUM) - 1):
            level += 1
            opp_iters, advance_wr = RL_CURRICULUM[level]
            opponent = MCTSAgent(iterations=opp_iters)
            recent_results.clear()
            print(f"\n  ★ Advanced to level {level}: MCTS-{opp_iters} "
                  f"(advance at {advance_wr*100:.0f}%)")

    print()
    # Final save only if we improved
    total = wins + losses + draws
    wr = wins / total * 100 if total > 0 else 0
    elapsed = time.perf_counter() - t0

    if best_wr > 0:
        print(f"\n  Best rolling win rate: {best_wr*100:.1f}%")
        print(f"  Best model saved to {model_path}")
    else:
        # Restore backup — RL didn't help
        shutil.copy2(backup_path, model_path)
        print(f"\n  No improvement during RL — restored pre-RL model from {backup_path}")

    print(f"  Final: W/L/D={wins}/{losses}/{draws}  overall={wr:.1f}%")
    print(f"  Time: {elapsed:.1f}s\n")


# ── CLI ────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train neural Connect Four agent")
    sub = parser.add_subparsers(dest="stage", required=True)

    # Behavioral cloning
    bc_parser = sub.add_parser("bc", help="Behavioral cloning from MCTS data")
    bc_parser.add_argument("--data", default="training_data.npz")
    bc_parser.add_argument("--model", default="neural_model.pt")
    bc_parser.add_argument("--epochs", type=int, default=50)
    bc_parser.add_argument("--lr", type=float, default=1e-3)

    # DAgger
    dag_parser = sub.add_parser("dagger", help="DAgger: iterative relabeling")
    dag_parser.add_argument("--model", default="neural_model.pt")
    dag_parser.add_argument("--data", default="training_data.npz")
    dag_parser.add_argument("--rounds", type=int, default=3)
    dag_parser.add_argument("--games", type=int, default=200)
    dag_parser.add_argument("--iters", type=int, default=5000, help="MCTS iters for expert")
    dag_parser.add_argument("--epochs", type=int, default=20)
    dag_parser.add_argument("--lr", type=float, default=5e-4)

    # RL
    rl_parser = sub.add_parser("rl", help="REINFORCE with curriculum learning")
    rl_parser.add_argument("--model", default="neural_model.pt")
    rl_parser.add_argument("--games", type=int, default=500)
    rl_parser.add_argument("--lr", type=float, default=1e-4)
    rl_parser.add_argument("--eval-every", type=int, default=25)
    rl_parser.add_argument("--window", type=int, default=50, help="Rolling window for win rate")

    args = parser.parse_args()

    if args.stage == "bc":
        train_bc(
            data_path=args.data,
            model_path=args.model,
            epochs=args.epochs,
            lr=args.lr,
        )
    elif args.stage == "dagger":
        run_dagger(
            model_path=args.model,
            data_path=args.data,
            rounds=args.rounds,
            games_per_round=args.games,
            mcts_iters=args.iters,
            epochs_per_round=args.epochs,
            lr=args.lr,
        )
    elif args.stage == "rl":
        train_rl(
            model_path=args.model,
            n_games=args.games,
            lr=args.lr,
            eval_every=args.eval_every,
            window=args.window,
        )