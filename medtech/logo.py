"""Animated MedTech logo (dots fill in, then a shine sweeps; press any key to skip)."""
import base64, random, shutil, time, zlib
from rich.console import Console
from rich.live import Live
from rich.text import Text

LOGO = (
    "eNrtmGuOwyAMhP/vafD9L9dVN2Ht8TjBxCmt1EptXg0ffjAYRNZ8fkTags8ngWUZWJ5vXHTUJPj3nY0/64CZGOvUfF69zuL92397"
    "AOrBgtZuLHV+C9iY9c92EXevXAQrszS6daS+3czZNFh50kJbd/ZfIAT7ddXVpkET5+5NK8LNHILmR1ytkwmNNiDo06HdAbjfA0d3"
    "HumRD/GRSSFYt8AGkrZb2Jias9gbocNsImmfgcWxzcdggRZYs/qRz4F8VkNGGduofOj7gjo3Au5ElWVGLUAnHZd3ecBiAbUwwrX7"
    "1HvFuSOfXE4LnGKav2jPomjls9qpkZMse2aOwetjrjbz7abI0JqKY7P9cz7JuxpnJjMHRPnOQpJRLnE11oF7yVQNHhqeJOI1gMst"
    "3/BZiM8EBAXTDjCaeIMzciyZLtmwN0HCt7EOnMzHgX+dbOz4xrJQ8oWAUEPoQHITSkWViSnrM4c/r6m5bCXLaiw2Fi6DWSmQqirn"
    "wKD31Kh68KGwvGqZmsieQnAdNbVaLMWmFm2F1JSAlGJTi7ZS7gx4mcWFMV62l/kFvw+YrP14Bd3IhsU0GFeNZIfgFnAoIeyqGBxO"
    "/a44KHV1UGKr5dltYBJJ2JC6EyzNbVbikXfhUoyD1QxuiJl/V43j+PLs8VcyV4MfXHZWRQ=="
)
BLUE, MID, BRIGHT, DIM = "#007fa4", "#3fc1e8", "#e8fbff", "#2b6f87"

def _bitmap():
    rows = zlib.decompress(base64.b64decode("".join(LOGO))).decode().split("\n")
    return [[ch == "1" for ch in r] for r in rows]

def _fit(bm):
    """Shrink the logo so it always fits the terminal."""
    sz = shutil.get_terminal_size((100, 30))
    h, w = len(bm), len(bm[0])
    s = min(1, (sz.lines - 11) / h, (sz.columns - 2) / w)
    H, W = max(1, int(h * s)), max(1, int(w * s))
    out = []
    for Y in range(H):
        y0 = int(Y / s); y1 = max(y0 + 1, int((Y + 1) / s))
        row = []
        for X in range(W):
            x0 = int(X / s); x1 = max(x0 + 1, int((X + 1) / s))
            blk = [bm[y][x] for y in range(y0, min(y1, h)) for x in range(x0, min(x1, w))]
            row.append(bool(blk) and sum(blk) * 2 >= len(blk))
        out.append(row)
    return out

def _key():
    try:
        import msvcrt
        if msvcrt.kbhit():
            msvcrt.getch(); return True
    except ImportError:
        pass
    return False

def draw_logo():
    """Logo appears little by little (dots, then filled), then a shine sweeps over it.
    Press any key to skip."""
    g = _fit(_bitmap())
    H, W = len(g), len(g[0])
    top = H  # find the bottom bar, fill the tiny letter holes, write "S M U" in it
    while top > 0 and sum(g[top - 1]) >= 0.6 * W: top -= 1
    label = {}
    if H - top >= 3 and W >= 9:
        for y in range(top, H): g[y] = [True] * W
        ly = (top + H - 1) // 2
        label = {(ly, (W - 5) // 2 + i): ch for i, ch in enumerate("S M U")}
    cells = [(y, x) for y in range(H) for x in range(W) if g[y][x]]
    random.shuffle(cells)
    cells.sort(key=lambda p: -p[0] + random.random() * H * 0.3)  # trunk/bottom first
    st = [[0] * W for _ in range(H)]  # 0 empty, 1 dot, 2 filled

    def render(shine=None, hint=""):
        t = Text()
        for y in range(H):
            for x in range(W):
                v = st[y][x]
                if v == 0: t.append(" ")
                elif v == 1: t.append(".", style=DIM)
                else:
                    d = abs(x + 2 * y - shine) if shine is not None else 99
                    if (y, x) in label: t.append(label[(y, x)], style="bold " + BRIGHT)
                    else: t.append("$", style=BRIGHT if d < 3 else MID if d < 7 else BLUE)
            t.append("\n")
        t.append(hint, style=DIM)
        return t

    hint = "(press any key to skip)"
    n, step = len(cells), max(1, len(cells) // 90)
    lag = max(8, n // 25)
    with Live(render(hint=hint), console=Console(), refresh_per_second=60) as live:
        skipped = False
        for i in range(0, n, step):
            for y, x in cells[i:i + step]: st[y][x] = 1
            for y, x in cells[max(0, i - lag):max(0, i + step - lag)]: st[y][x] = 2
            live.update(render(hint=hint)); time.sleep(0.02)
            if _key(): skipped = True; break
        for y, x in cells: st[y][x] = 2
        if not skipped:
            for _ in range(2):
                for pos in range(-8, W + 2 * H + 8, 3):
                    live.update(render(pos, hint)); time.sleep(0.012)
                    if _key(): skipped = True; break
                if skipped: break
        live.update(render())
