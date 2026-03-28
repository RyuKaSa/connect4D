"""
Connect Four 3D — Pure engine (no rendering, no dependencies).

Board: two uint64 bitboards (one per player) on a 4×4×4 grid.
Bit index = x + 4*y + 16*z   (x, y, z each in 0..3)
Peg id    = x + 4*y           (column on the board, 0..15)

Game: wraps Board with move history, undo, and result tracking.
"""

from __future__ import annotations

GRID = 4
NUM_PEGS = GRID * GRID  # 16
NUM_CELLS = GRID ** 3   # 64
FULL_BOARD = (1 << NUM_CELLS) - 1


# ── Win masks (76 lines) ──────────────────────────────────

def _build_win_masks() -> tuple[int, ...]:
    masks: list[int] = []

    def bit(x: int, y: int, z: int) -> int:
        return 1 << (x + 4 * y + 16 * z)

    def line(*coords: tuple[int, int, int]) -> int:
        m = 0
        for x, y, z in coords:
            m |= bit(x, y, z)
        return m

    # Axis-aligned rows (3 axes × 16 = 48)
    for y in range(4):
        for z in range(4):
            masks.append(line(*[(x, y, z) for x in range(4)]))
    for x in range(4):
        for z in range(4):
            masks.append(line(*[(x, y, z) for y in range(4)]))
    for x in range(4):
        for y in range(4):
            masks.append(line(*[(x, y, z) for z in range(4)]))

    # Face diagonals (3 planes × 4 slices × 2 = 24)
    for z in range(4):
        masks.append(line(*[(i, i, z) for i in range(4)]))
        masks.append(line(*[(i, 3 - i, z) for i in range(4)]))
    for y in range(4):
        masks.append(line(*[(i, y, i) for i in range(4)]))
        masks.append(line(*[(i, y, 3 - i) for i in range(4)]))
    for x in range(4):
        masks.append(line(*[(x, i, i) for i in range(4)]))
        masks.append(line(*[(x, i, 3 - i) for i in range(4)]))

    # Space diagonals (4)
    masks.append(line(*[(i, i, i) for i in range(4)]))
    masks.append(line(*[(i, i, 3 - i) for i in range(4)]))
    masks.append(line(*[(i, 3 - i, i) for i in range(4)]))
    masks.append(line(*[(i, 3 - i, 3 - i) for i in range(4)]))

    return tuple(masks)


WIN_MASKS = _build_win_masks()
assert len(WIN_MASKS) == 76


# ── Board ──────────────────────────────────────────────────

class Board:
    """Immutable-style board state (mutated in place for speed, but copyable)."""

    __slots__ = ("bbs", "turn")

    def __init__(self) -> None:
        self.bbs: list[int] = [0, 0]  # bitboard per player
        self.turn: int = 0             # 0 or 1

    # -- Queries --

    def combined(self) -> int:
        return self.bbs[0] | self.bbs[1]

    def stack_height(self, peg: int) -> int:
        x = peg % GRID
        y = peg // GRID
        combined = self.combined()
        h = 0
        for z in range(GRID):
            if combined & (1 << (x + 4 * y + 16 * z)):
                h += 1
            else:
                break  # stacking means no gaps
        return h

    def legal_pegs(self) -> list[int]:
        return [p for p in range(NUM_PEGS) if self.stack_height(p) < GRID]

    def is_legal(self, peg: int) -> bool:
        return 0 <= peg < NUM_PEGS and self.stack_height(peg) < GRID

    def check_win(self, player: int) -> bool:
        b = self.bbs[player]
        return any((b & m) == m for m in WIN_MASKS)

    def is_draw(self) -> bool:
        return self.combined() == FULL_BOARD

    def is_terminal(self) -> bool:
        return self.check_win(0) or self.check_win(1) or self.is_draw()

    # -- Mutations --

    def place(self, peg: int) -> int:
        """Place current player's piece on *peg*. Returns the z level used.
        Raises ValueError if the move is illegal."""
        z = self.stack_height(peg)
        if z >= GRID:
            raise ValueError(f"Peg {peg} is full")
        bit = 1 << (peg % GRID + 4 * (peg // GRID) + 16 * z)
        self.bbs[self.turn] |= bit
        self.turn ^= 1
        return z

    def remove(self, peg: int, z: int, player: int) -> None:
        """Undo a placement (used by Game.undo)."""
        bit = 1 << (peg % GRID + 4 * (peg // GRID) + 16 * z)
        self.bbs[player] &= ~bit
        self.turn = player  # restore turn to the player who made the move

    def copy(self) -> Board:
        b = Board.__new__(Board)
        b.bbs = self.bbs[:]
        b.turn = self.turn
        return b


# ── Game ───────────────────────────────────────────────────

class Game:
    """Board + history + result. The object the renderer and agents interact with."""

    __slots__ = ("board", "history", "result")

    def __init__(self) -> None:
        self.board = Board()
        self.history: list[tuple[int, int, int]] = []  # (peg, z, player)
        self.result: int | str | None = None            # None | 0 | 1 | "draw"

    # -- Forwarding queries --

    @property
    def turn(self) -> int:
        return self.board.turn

    def legal_pegs(self) -> list[int]:
        return self.board.legal_pegs()

    def is_over(self) -> bool:
        return self.result is not None

    # -- Actions --

    def place(self, peg: int) -> bool:
        """Attempt to place a piece. Returns True on success."""
        if self.result is not None:
            return False
        if not self.board.is_legal(peg):
            return False

        player = self.board.turn
        z = self.board.place(peg)
        self.history.append((peg, z, player))

        if self.board.check_win(player):
            self.result = player
        elif self.board.is_draw():
            self.result = "draw"
        return True

    def undo(self) -> tuple[int, int, int] | None:
        """Undo the last move. Returns (peg, z, player) or None."""
        if not self.history:
            return None
        peg, z, player = self.history.pop()
        self.board.remove(peg, z, player)
        self.result = None  # reopen game
        return peg, z, player

    def reset(self) -> None:
        self.board = Board()
        self.history.clear()
        self.result = None

    def copy(self) -> Game:
        g = Game.__new__(Game)
        g.board = self.board.copy()
        g.history = self.history[:]
        g.result = self.result
        return g