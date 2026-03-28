"""
Connect Four 3D — Visual prototype with win detection.
76 precomputed win masks on a 4×4×4 bitboard.
Click a peg to stack a piece (alternating white/black).
Z = undo, R = reset, Right-drag = orbit, Scroll = zoom.
"""

from ursina import *

app = Ursina(title="Connect Four 3D", borderless=False)
window.color = Color(0.2, 0.2, 0.2, 1)


# ── Win masks (76 lines on 4×4×4) ─────────────────────────
# Bit index = x + 4*y + 16*z   (x,y,z each 0..3)

def _build_win_masks():
    masks = []

    def bit(x, y, z):
        return 1 << (x + 4 * y + 16 * z)

    def mask_from(coords):
        m = 0
        for x, y, z in coords:
            m |= bit(x, y, z)
        return m

    # ── Straight lines along each axis ──
    # X-rows: fix y,z, vary x  (16 lines)
    for y in range(4):
        for z in range(4):
            masks.append(mask_from([(x, y, z) for x in range(4)]))

    # Y-rows: fix x,z, vary y  (16 lines)
    for x in range(4):
        for z in range(4):
            masks.append(mask_from([(x, y, z) for y in range(4)]))

    # Z-pillars: fix x,y, vary z  (16 lines)
    for x in range(4):
        for y in range(4):
            masks.append(mask_from([(x, y, z) for z in range(4)]))

    # ── Face diagonals (2 per face × 4 slices × 3 planes = 24) ──
    # XY plane diags: fix z, vary x and y together  (8 lines)
    for z in range(4):
        masks.append(mask_from([(i, i, z) for i in range(4)]))
        masks.append(mask_from([(i, 3 - i, z) for i in range(4)]))

    # XZ plane diags: fix y, vary x and z together  (8 lines)
    for y in range(4):
        masks.append(mask_from([(i, y, i) for i in range(4)]))
        masks.append(mask_from([(i, y, 3 - i) for i in range(4)]))

    # YZ plane diags: fix x, vary y and z together  (8 lines)
    for x in range(4):
        masks.append(mask_from([(x, i, i) for i in range(4)]))
        masks.append(mask_from([(x, i, 3 - i) for i in range(4)]))

    # ── Space diagonals (4 lines) ──
    masks.append(mask_from([(i, i, i) for i in range(4)]))
    masks.append(mask_from([(i, i, 3 - i) for i in range(4)]))
    masks.append(mask_from([(i, 3 - i, i) for i in range(4)]))
    masks.append(mask_from([(i, 3 - i, 3 - i) for i in range(4)]))

    return tuple(masks)

WIN_MASKS = _build_win_masks()
assert len(WIN_MASKS) == 76, f"Expected 76 win masks, got {len(WIN_MASKS)}"


# ── Constants ──────────────────────────────────────────────
GRID = 4
SPACING = 1.2
PEG_WIDTH = 0.08
PEG_HEIGHT = 1.7
PIECE_RADIUS = 0.4
PIECE_HEIGHT = 0.4
SLAB_THICKNESS = 0.4

COLORS = {
    0: color.white,
    1: Color(0.12, 0.12, 0.12, 1),
}
PEG_COLOR = Color(0.55, 0.51, 0.47, 1)
SLAB_COLOR = Color(0.12, 0.4, 0.18, 1)


# ── Game state ─────────────────────────────────────────────
boards = [0, 0]          # bitboard per player (uint64)
stacks = {i: [] for i in range(GRID * GRID)}  # visual entities
history = []              # list of (peg_id, z, player)
turn = [0]
game_over = [False]
winner = [None]           # None, 0, 1, or "draw"


# ── Engine helpers ─────────────────────────────────────────
def bit_index(peg_id, z):
    x = peg_id % GRID
    y = peg_id // GRID
    return x + 4 * y + 16 * z

def stack_height(peg_id):
    """Count filled slots on a peg using both bitboards."""
    x = peg_id % GRID
    y = peg_id // GRID
    h = 0
    combined = boards[0] | boards[1]
    for z in range(4):
        if combined & (1 << bit_index(peg_id, z)):
            h += 1
    return h

def check_win(player):
    b = boards[player]
    return any((b & mask) == mask for mask in WIN_MASKS)

def is_draw():
    return (boards[0] | boards[1]) == (1 << 64) - 1


# ── Visual helpers ─────────────────────────────────────────
def peg_pos(peg_id):
    col = peg_id % GRID
    row = peg_id // GRID
    offset = (GRID - 1) / 2 * SPACING
    return (col * SPACING - offset, row * SPACING - offset)

def piece_y(level):
    return SLAB_THICKNESS / 2 + PIECE_HEIGHT / 2 + level * (PIECE_HEIGHT + 0.02)


# ── Actions ────────────────────────────────────────────────
def place_piece(pid):
    if game_over[0]:
        return False
    z = stack_height(pid)
    if z >= GRID:
        return False

    player = turn[0]
    boards[player] |= (1 << bit_index(pid, z))

    px, pz = peg_pos(pid)
    piece = Entity(
        parent=scene,
        model="sphere",
        color=COLORS[player],
        scale=(PIECE_RADIUS * 2, PIECE_HEIGHT, PIECE_RADIUS * 2),
        position=(px, piece_y(z), pz),
    )
    stacks[pid].append(piece)
    history.append((pid, z, player))

    # Check win / draw
    if check_win(player):
        game_over[0] = True
        winner[0] = player
        show_game_over(f"Player {player + 1} ({'White' if player == 0 else 'Black'}) wins!")
    elif is_draw():
        game_over[0] = True
        winner[0] = "draw"
        show_game_over("Draw!")
    else:
        turn[0] = 1 - player
        info_text.text = f"turn: P{turn[0]+1} ({'White' if turn[0]==0 else 'Black'})"

    return True

def undo():
    if not history:
        return
    # If game was over, reopen it
    if game_over[0]:
        game_over[0] = False
        winner[0] = None
        hide_game_over()

    pid, z, player = history.pop()
    boards[player] &= ~(1 << bit_index(pid, z))
    piece = stacks[pid].pop()
    destroy(piece)
    turn[0] = player  # it was this player's move, so it's their turn again
    info_text.text = f"undo → turn: P{turn[0]+1} ({'White' if turn[0]==0 else 'Black'})"

def reset():
    boards[0] = 0
    boards[1] = 0
    for pid in range(GRID * GRID):
        for piece in stacks[pid]:
            destroy(piece)
        stacks[pid] = []
    history.clear()
    turn[0] = 0
    game_over[0] = False
    winner[0] = None
    hide_game_over()
    info_text.text = "reset | turn: P1 (White)"


# ── Game over overlay ─────────────────────────────────────
game_over_bg = Entity(
    parent=camera.ui,
    model="quad",
    color=Color(0, 0, 0, 0.6),
    scale=(2, 2),
    z=-1,
    visible=False,
)
game_over_text = Text(
    text="",
    parent=camera.ui,
    scale=3,
    origin=(0, 0),
    z=-2,
    visible=False,
)
game_over_sub = Text(
    text="Z = undo last move  |  R = new game",
    parent=camera.ui,
    scale=1.5,
    origin=(0, 0),
    y=-0.08,
    z=-2,
    visible=False,
)

def show_game_over(msg):
    game_over_text.text = msg
    game_over_bg.visible = True
    game_over_text.visible = True
    game_over_sub.visible = True
    info_text.text = msg

def hide_game_over():
    game_over_bg.visible = False
    game_over_text.visible = False
    game_over_sub.visible = False


# ── Scene ──────────────────────────────────────────────────

# Base slab
Entity(
    parent=scene,
    model="cube",
    color=SLAB_COLOR,
    scale=(GRID * SPACING + 0.8, SLAB_THICKNESS, GRID * SPACING + 0.8),
    position=(0, 0, 0),
)

# Pegs
for pid in range(GRID * GRID):
    px, pz = peg_pos(pid)
    Entity(
        parent=scene,
        model="cube",
        color=PEG_COLOR,
        scale=(PEG_WIDTH, PEG_HEIGHT, PEG_WIDTH),
        position=(px, SLAB_THICKNESS / 2 + PEG_HEIGHT / 2, pz),
    )

# Click targets (first row visible for hitbox debugging)
hit_targets = []
for pid in range(GRID * GRID):
    px, pz = peg_pos(pid)
    row = pid // GRID
    debug_visible = (row == 0)
    t = Entity(
        parent=scene,
        model="cube",
        color=Color(1, 0, 0, 0.15) if debug_visible else color.clear,
        scale=(SPACING * 0.2, PEG_HEIGHT, SPACING * 0.2),
        position=(px, SLAB_THICKNESS / 2 + PEG_HEIGHT / 2, pz),
        collider="box",
    )
    t.peg_id = pid
    hit_targets.append(t)


# ── Orbit Camera ──────────────────────────────────────────
cam_pivot = Entity()
cam_pivot.rotation = (40, 0, 0)
cam_distance = [12]
camera.parent = cam_pivot
camera.position = (0, 0, -cam_distance[0])
camera.rotation = (0, 0, 0)
camera.fov = 50
orbit_dragging = [False]


# ── HUD ────────────────────────────────────────────────────
debug_text = Text(
    text="hover: —",
    position=(-0.85, 0.47),
    scale=1.2,
    background=True,
)
info_text = Text(
    text="turn: P1 (White)",
    position=(-0.85, 0.42),
    scale=1,
    background=True,
)

# ── Hover highlight ────────────────────────────────────────
highlight = Entity(
    parent=scene,
    model="cube",
    color=color.yellow,
    scale=(SPACING * 0.85, 0.05, SPACING * 0.85),
    position=(0, -100, 0),
)


# ── Update ─────────────────────────────────────────────────
def update():
    if orbit_dragging[0]:
        cam_pivot.rotation_y += mouse.velocity[0] * 200
        cam_pivot.rotation_x -= mouse.velocity[1] * 200
        cam_pivot.rotation_x = clamp(cam_pivot.rotation_x, 5, 85)

    hit = mouse.hovered_entity
    if hit and hasattr(hit, "peg_id") and not game_over[0]:
        pid = hit.peg_id
        px, pz = peg_pos(pid)
        level = stack_height(pid)
        if level >= GRID:
            highlight.visible = False
            debug_text.text = f"hover: peg {pid}  FULL"
        else:
            y = piece_y(level)
            highlight.position = (px, y, pz)
            highlight.visible = True
            debug_text.text = (
                f"hover: peg {pid}  col={pid % GRID} row={pid // GRID}  "
                f"stack={level}/{GRID}"
            )
    else:
        highlight.visible = False
        if not game_over[0]:
            debug_text.text = f"hover: {hit.name if hit else '—'}"


def input(key):
    if key == "right mouse down":
        orbit_dragging[0] = True
    elif key == "right mouse up":
        orbit_dragging[0] = False
    elif key == "scroll up":
        cam_distance[0] = max(4, cam_distance[0] - 1)
        camera.position = (0, 0, -cam_distance[0])
    elif key == "scroll down":
        cam_distance[0] = min(25, cam_distance[0] + 1)
        camera.position = (0, 0, -cam_distance[0])
    elif key == "left mouse down":
        hit = mouse.hovered_entity
        if hit and hasattr(hit, "peg_id"):
            if not place_piece(hit.peg_id):
                if not game_over[0]:
                    debug_text.text = f"peg {hit.peg_id} FULL"
    elif key == "z":
        undo()
    elif key == "r":
        reset()


print("\n=== Connect Four 3D ===")
print("  Left-click   = place piece")
print("  Right-drag   = orbit camera")
print("  Scroll       = zoom")
print("  Z            = undo")
print("  R            = reset")
print(f"  Win masks    = {len(WIN_MASKS)} lines\n")

app.run()