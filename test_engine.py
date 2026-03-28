"""Smoke tests for engine.py — run with: python test_engine.py"""
from engine import Game, Board, WIN_MASKS, GRID, NUM_PEGS

def test_win_mask_count():
    assert len(WIN_MASKS) == 76

def test_basic_place_and_turn():
    g = Game()
    assert g.turn == 0
    assert g.place(0)
    assert g.turn == 1
    assert g.place(0)
    assert g.turn == 0
    # Peg 0 now has 2 pieces stacked
    assert g.board.stack_height(0) == 2

def test_full_peg_rejected():
    g = Game()
    for _ in range(GRID):
        assert g.place(0)
    # 5th placement on same peg should fail
    assert not g.place(0)

def test_vertical_win():
    """Player 0 stacks 4 on peg 0 (alternating with player 1 on peg 1)."""
    g = Game()
    for _ in range(4):
        g.place(0)  # P0
        if not g.is_over():
            g.place(1)  # P1
    # P0 placed on peg 0 at z=0,1,2,3 — vertical win (Z-pillar)
    assert g.result == 0

def test_undo():
    g = Game()
    g.place(5)
    g.place(3)
    assert len(g.history) == 2
    peg, z, player = g.undo()
    assert peg == 3 and z == 0 and player == 1
    assert g.turn == 1
    assert g.board.stack_height(3) == 0

def test_undo_reopens_game():
    g = Game()
    # Force vertical win for P0 on peg 0
    for _ in range(4):
        g.place(0)
        if not g.is_over():
            g.place(1)
    assert g.is_over()
    g.undo()  # undo P1's last move (or the winning move)
    # After undoing the last move before the win, game might reopen
    # Actually, let's undo until the winning move is removed
    while g.is_over():
        g.undo()
    assert not g.is_over()

def test_legal_pegs():
    g = Game()
    assert len(g.legal_pegs()) == NUM_PEGS
    # Fill peg 7 completely
    for _ in range(GRID):
        g.place(7)
        if not g.is_over():
            g.place(8)  # alternate
    if not g.is_over():
        legal = g.legal_pegs()
        assert 7 not in legal
        assert 8 in legal  # partially filled

def test_copy_independence():
    g = Game()
    g.place(0)
    g2 = g.copy()
    g2.place(1)
    assert g.board.stack_height(1) == 0  # original unaffected
    assert g2.board.stack_height(1) == 1

def test_draw():
    """Fill the entire board without anyone winning (contrived)."""
    # This is hard to guarantee no win, so just test the draw detection mechanism
    b = Board()
    b.bbs[0] = 0  # won't be a real game, just check the flag
    b.bbs[1] = 0
    # Manually set all bits
    b.bbs[0] = (1 << 64) - 1
    assert b.is_draw()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ✓ {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")