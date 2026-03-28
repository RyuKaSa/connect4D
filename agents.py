"""
Connect Four 3D — Agent interface and implementations.

Agent protocol: any object with `.name` (str) and `.choose_move(game) -> int`.
"""

from __future__ import annotations

import math
import random
from typing import Protocol, runtime_checkable

from engine import Game


# ── Agent protocol ─────────────────────────────────────────

@runtime_checkable
class Agent(Protocol):
    name: str

    def choose_move(self, game: Game) -> int:
        """Return the peg id (0..15) to play on."""
        ...


# ── Human (placeholder — moves come from the UI) ──────────

class HumanAgent:
    name = "Human"

    def choose_move(self, game: Game) -> int:
        raise NotImplementedError("HumanAgent is controlled via the UI")


# ── Random ─────────────────────────────────────────────────

class RandomAgent:
    name = "Random"

    def choose_move(self, game: Game) -> int:
        return random.choice(game.legal_pegs())


# ── MCTS (UCT with random rollouts) ───────────────────────

class MCTSAgent:
    """Monte-Carlo Tree Search using Upper Confidence bounds for Trees.

    Pure random rollouts — no heuristic, no neural net.
    Terminal detection via bitmask is very fast, so even modest
    iteration counts give strong play on 4×4×4.
    """

    def __init__(self, iterations: int = 2000, c: float = 1.414) -> None:
        self.iterations = iterations
        self.c = c
        self.name = f"MCTS-{iterations}"

    def choose_move(self, game: Game) -> int:
        legal = game.legal_pegs()
        if len(legal) == 1:
            return legal[0]

        root = _Node(
            move=-1,
            player=1 - game.turn,   # dummy: "opponent moved to reach this state"
            legal_moves=legal,
        )

        for _ in range(self.iterations):
            node = root
            sim = game.copy()

            # ── Selection ──
            while not node.untried and node.children:
                node = self._best_child(node)
                sim.place(node.move)

            # ── Expansion ──
            if node.untried and not sim.is_over():
                move = random.choice(node.untried)
                node.untried.remove(move)
                player = sim.turn
                sim.place(move)
                child = _Node(
                    move=move,
                    player=player,
                    legal_moves=sim.legal_pegs() if not sim.is_over() else [],
                    parent=node,
                )
                node.children.append(child)
                node = child

            # ── Rollout ──
            while not sim.is_over():
                sim.place(random.choice(sim.legal_pegs()))

            # ── Backpropagate ──
            result = sim.result  # 0, 1, or "draw"
            while node is not None:
                node.n += 1
                if result == node.player:
                    node.w += 1.0
                elif result == "draw":
                    node.w += 0.5
                node = node.parent

        # Pick the most-visited child (robust selection).
        if not root.children:
            return random.choice(legal)
        return max(root.children, key=lambda c: c.n).move

    def _best_child(self, node: _Node) -> _Node:
        c = self.c
        log_n = math.log(node.n)
        return max(
            node.children,
            key=lambda ch: ch.w / ch.n + c * math.sqrt(log_n / ch.n),
        )


class _Node:
    """A node in the MCTS tree."""

    __slots__ = ("move", "player", "parent", "children", "w", "n", "untried")

    def __init__(
        self,
        move: int,
        player: int,
        legal_moves: list[int],
        parent: _Node | None = None,
    ) -> None:
        self.move = move
        self.player = player      # player who made *this* move
        self.parent = parent
        self.children: list[_Node] = []
        self.w = 0.0              # wins from self.player's perspective
        self.n = 0                # visit count
        self.untried = list(legal_moves)


# ── Registry (used by the UI to cycle through agents) ─────

AGENT_FACTORIES: list[tuple[str, callable]] = [
    ("Human",     lambda: HumanAgent()),
    ("Random",    lambda: RandomAgent()),
    ("MCTS-1k",   lambda: MCTSAgent(1000)),
    ("MCTS-5k",   lambda: MCTSAgent(5000)),
]