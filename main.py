"""
Connect Four 3D — Ursina renderer.
All game logic lives in engine.py; this file handles visuals + input.

  Left-click   = place piece
  Right-drag   = orbit camera
  Scroll       = zoom
  Z            = undo
  R            = reset
"""

from ursina import *
from engine import Game, GRID

app = Ursina(title="Connect Four 3D", borderless=False)
window.color = Color(0.2, 0.2, 0.2, 1)

# ── Constants ──────────────────────────────────────────────
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
game = Game()
piece_entities: dict[int, list[Entity]] = {i: [] for i in range(GRID * GRID)}

# ── Visual helpers ─────────────────────────────────────────
def peg_pos(peg_id):
    col = peg_id % GRID
    row = peg_id // GRID
    offset = (GRID - 1) / 2 * SPACING
    return (col * SPACING - offset, row * SPACING - offset)

def piece_y(level):
    return SLAB_THICKNESS / 2 + PIECE_HEIGHT / 2 + level * (PIECE_HEIGHT + 0.02)

def player_label(p):
    return f"P{p+1} ({'White' if p == 0 else 'Black'})"

# ── Actions ────────────────────────────────────────────────
def do_place(pid):
    if game.is_over():
        return False
    player = game.turn
    z = game.board.stack_height(pid)
    if not game.place(pid):
        return False

    # Spawn visual piece
    px, pz = peg_pos(pid)
    piece = Entity(
        parent=scene,
        model="sphere",
        color=COLORS[player],
        scale=(PIECE_RADIUS * 2, PIECE_HEIGHT, PIECE_RADIUS * 2),
        position=(px, piece_y(z), pz),
    )
    piece_entities[pid].append(piece)

    # Update HUD
    if game.result is not None:
        if game.result == "draw":
            show_game_over("Draw!")
        else:
            show_game_over(f"{player_label(game.result)} wins!")
    else:
        info_text.text = f"turn: {player_label(game.turn)}"
    return True

def do_undo():
    result = game.undo()
    if result is None:
        return
    pid, z, player = result
    hide_game_over()
    piece = piece_entities[pid].pop()
    destroy(piece)
    info_text.text = f"undo → turn: {player_label(game.turn)}"

def do_reset():
    game.reset()
    for pid in range(GRID * GRID):
        for piece in piece_entities[pid]:
            destroy(piece)
        piece_entities[pid] = []
    hide_game_over()
    info_text.text = f"reset | turn: {player_label(game.turn)}"

# ── Game over overlay ─────────────────────────────────────
game_over_bg = Entity(
    parent=camera.ui, model="quad", color=Color(0, 0, 0, 0.6),
    scale=(2, 2), z=-1, visible=False,
)
game_over_text = Text(
    text="", parent=camera.ui, scale=3, origin=(0, 0), z=-2, visible=False,
)
game_over_sub = Text(
    text="Z = undo last move  |  R = new game",
    parent=camera.ui, scale=1.5, origin=(0, 0), y=-0.08, z=-2, visible=False,
)

def show_game_over(msg):
    game_over_text.text = msg
    game_over_bg.visible = game_over_text.visible = game_over_sub.visible = True
    info_text.text = msg

def hide_game_over():
    game_over_bg.visible = game_over_text.visible = game_over_sub.visible = False

# ── Scene ──────────────────────────────────────────────────
# Base slab
Entity(
    parent=scene, model="cube", color=SLAB_COLOR,
    scale=(GRID * SPACING + 0.8, SLAB_THICKNESS, GRID * SPACING + 0.8),
    position=(0, 0, 0),
)

# Pegs + click targets
hit_targets = []
for pid in range(GRID * GRID):
    px, pz = peg_pos(pid)
    Entity(
        parent=scene, model="cube", color=PEG_COLOR,
        scale=(PEG_WIDTH, PEG_HEIGHT, PEG_WIDTH),
        position=(px, SLAB_THICKNESS / 2 + PEG_HEIGHT / 2, pz),
    )
    t = Entity(
        parent=scene, model="cube", color=color.clear,
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
    text="hover: —", position=(-0.85, 0.47), scale=1.2, background=True,
)
info_text = Text(
    text="turn: P1 (White)", position=(-0.85, 0.42), scale=1, background=True,
)

# ── Hover highlight ────────────────────────────────────────
highlight = Entity(
    parent=scene, model="cube", color=color.yellow,
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
    if hit and hasattr(hit, "peg_id") and not game.is_over():
        pid = hit.peg_id
        px, pz = peg_pos(pid)
        level = game.board.stack_height(pid)
        if level >= GRID:
            highlight.visible = False
            debug_text.text = f"hover: peg {pid}  FULL"
        else:
            highlight.position = (px, piece_y(level), pz)
            highlight.visible = True
            debug_text.text = (
                f"hover: peg {pid}  col={pid % GRID} row={pid // GRID}  "
                f"stack={level}/{GRID}"
            )
    else:
        highlight.visible = False
        if not game.is_over():
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
            if not do_place(hit.peg_id):
                if not game.is_over():
                    debug_text.text = f"peg {hit.peg_id} FULL"
    elif key == "z":
        do_undo()
    elif key == "r":
        do_reset()

print("\n=== Connect Four 3D ===")
print("  Left-click   = place piece")
print("  Right-drag   = orbit camera")
print("  Scroll       = zoom")
print("  Z            = undo")
print("  R            = reset\n")

app.run()