"""Nocturnal Valley pocket owl box, v3. Builds owl_base.stl and owl_lid.stl (mm).
Z up, owl front toward -Y. Organic form from owl_field (signed distance), details with Manifold."""
import os, sys, math, time
import numpy as np
import trimesh
from manifold3d import Manifold, Mesh, CrossSection, FillRule
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties
from fontTools.ttLib import TTFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import owl_field as of
from sdf import field_to_manifold, field_levels_to_manifolds, to_trimesh

T0 = time.time()
def log(*a):
    print(f"[{time.time() - T0:6.1f}s]", *a, flush=True)

# ---------------- global dimensions (final mm) ----------------
S = 0.72                 # design -> pocket scale (about 63 mm tall)
RES = 0.25               # sculpt grid
SPLIT = 47.0 * S         # lid / base joint height
LIP = 4.0                # friction lip height
WALL = 2.6
LID_IN = 1.2             # lid recess surface (offset inside the skin)
LIP_OUT = 1.4            # lip outer surface  -> 0.2 mm radial clearance
FLOOR = 2.2
FONT = os.path.join(HERE, "fonts", "Poppins-Bold.ttf")   # bundled, SIL Open Font License
CACHE = os.path.join(HERE, "cache"); os.makedirs(CACHE, exist_ok=True)
OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)

BIG = 300
def box(x0, x1, y0, y1, z0, z1):
    return Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])
above = lambda z: box(-BIG, BIG, -BIG, BIG, z, BIG)
below = lambda z: box(-BIG, BIG, -BIG, BIG, -BIG, z)
between = lambda a, b: box(-BIG, BIG, -BIG, BIG, a, b)

FM = lambda X, Y, Z: S * of.F_main_dist(X / S, Y / S, Z / S)
FO = lambda X, Y, Z: S * of.F_outer(X / S, Y / S, Z / S)
BOUNDS = [(-27.0, 27.0), (-25.0, 24.0), (-1.0, 66.0)]

# ---------------- sculpt ----------------
log("sculpting outer form")
outer = field_to_manifold(FO, BOUNDS, RES, smooth=6, cache=os.path.join(CACHE, f"outer3_{RES}.npz")) ^ above(0)
LEVELS = [1.0, 0.7, 0.55, 0.0, -0.8, -0.98, -1.08, -LID_IN, -LIP_OUT, -1.6, -WALL]
log("offset surfaces", LEVELS)
M = field_levels_to_manifolds(FM, BOUNDS, RES, LEVELS, smooth=6, cache_prefix=os.path.join(CACHE, f"maind_{RES}"))
log("sculpt done")


def tmesh(m):
    t = to_trimesh(m)
    return t, t.vertex_normals


# ---------------- relief wrapping ----------------
def ray_surface(mesh_m, origin_fn, dir_fn):
    """Returns f(u, v) -> (points, unit normals) by ray casting onto a manifold's surface."""
    tm, vn = tmesh(mesh_m)
    def f(u, v, batch=40000):
        P = np.zeros((len(u), 3)); N = np.zeros((len(u), 3))
        for s in range(0, len(u), batch):
            uu, vv = u[s:s + batch], v[s:s + batch]
            o, d = origin_fn(uu, vv), dir_fn(uu, vv)
            loc, ri, ti = tm.ray.intersects_location(o, d, multiple_hits=False)
            assert len(ri) == len(uu), f"{len(uu) - len(ri)} rays missed"
            P[s + ri] = loc
            bary = trimesh.triangles.points_to_barycentric(tm.triangles[ti], loc)
            N[s + ri] = np.einsum("ij,ijk->ik", bary, vn[tm.faces[ti]])
        N /= np.linalg.norm(N, axis=1, keepdims=True)
        return P, N
    return f


def wrap(flat, surface, refine=0.35, flip=False):
    """Warp a flat solid in (u, v, h) onto a surface: point = P(u, v) + h * N(u, v)."""
    flat = flat.refine_to_length(refine)
    def w(p):
        P, N = surface(p[:, 0], p[:, 1])
        return P + (-1 if flip else 1) * p[:, 2:3] * N
    return flat.warp_batch(w)


def ramp_tile(cs, v_top, v_bot, h_top, h_bot, depth=1.0):
    """Extrude a 2D tile to height h(v), linear from h_top at v_top to h_bot at v_bot."""
    hmax = max(h_top, h_bot)
    t = cs.extrude(hmax + depth).translate([0, 0, -depth])
    k = (h_bot - h_top) / (v_top - v_bot)
    L = math.hypot(k, 1)
    return t.trim_by_plane([0, -k / L, -1 / L], -(h_top + k * v_top) / L)


FRONT = ray_surface(M[0.0], lambda u, v: np.stack([u, np.full_like(u, -100.0), v], 1),
                    lambda u, v: np.tile([0.0, 1.0, 0.0], (len(u), 1)))

# ---------------- chest bib: overlapping fish-scale feathers ----------------
log("chest feathers")
BIB_C, BIB_H = (0.0, 19.0), (12.8, 9.6)          # centre (x, z), half sizes
RV, DU, DV, PROUD = 3.1, 5.7, 3.2, 0.8
tiles = []
v = BIB_C[1] + BIB_H[1] - RV
row = 0
while v > BIB_C[1] - BIB_H[1] - RV:
    off = 0 if row % 2 == 0 else DU / 2
    for u in np.arange(-BIB_H[0] - DU, BIB_H[0] + DU, DU) + off:
        tiles.append(ramp_tile(CrossSection.circle(RV, 40).translate([u, v]), v + RV, v - RV, 0.0, PROUD))
    v -= DV; row += 1
bib_clip = CrossSection.circle(1, 96).scale(list(BIB_H)).translate(list(BIB_C)).extrude(4).translate([0, 0, -2])
bib = wrap(Manifold.batch_boolean(tiles, 0) ^ bib_clip, FRONT)

# ---------------- wing coverts: smaller scales on the upper wing ----------------
log("wing feathers")
wing_parts = []
for sx in (-1, 1):
    side = ray_surface(outer, lambda u, v, sx=sx: np.stack([np.full_like(u, sx * 100.0), sx * u, v], 1),
                       lambda u, v, sx=sx: np.tile([-sx * 1.0, 0.0, 0.0], (len(u), 1)))
    wc, wh = (3.2, 19.8), (8.6, 8.4)
    tl = []
    v = wc[1] + wh[1] - 2.6; row = 0
    while v > wc[1] - wh[1] - 2.6:
        off = 0 if row % 2 == 0 else 1.9
        for u in np.arange(wc[0] - wh[0] - 4, wc[0] + wh[0] + 4, 3.8) + off:
            shape = CrossSection.circle(1, 32).scale([2.1, 2.9]).translate([u, v])
            tl.append(ramp_tile(shape, v + 2.9, v - 2.9, 0.0, 0.6))
        v -= 2.7; row += 1
    clip = CrossSection.circle(1, 96).scale(list(wh)).translate(list(wc)).extrude(4).translate([0, 0, -2])
    flat = Manifold.batch_boolean(tl, 0) ^ clip
    if sx < 0:
        flat = flat.mirror([1, 0, 0])            # left side: u = -y keeps the warp right-side-out
    wing_parts.append(wrap(flat, side))
wings_relief = Manifold.batch_boolean(wing_parts, 0)

# ---------------- feather ruff around the neck (hides the lid seam) ----------------
log("neck ruff")
R0 = 17.0
N_RUFF = 17


def radial_prism(cs, r_in=3.0, r_out=40.0):
    """2D shape in (u = R0 * theta, v = z) -> solid wedge spanning radii r_in..r_out."""
    flat = cs.extrude(r_out - r_in).translate([0, 0, r_in]).refine_to_length(0.6)
    def w(p):
        th = p[:, 0] / R0
        return np.stack([p[:, 2] * np.cos(th), p[:, 2] * np.sin(th), p[:, 1]], 1)
    return flat.warp_batch(w)


def teardrop(u, v_top, v_tip, half_w):
    return CrossSection.batch_hull([CrossSection.circle(half_w, 32).translate([u, v_top]),
                                    CrossSection.circle(0.35, 12).translate([u, v_tip])])


step = 2 * math.pi * R0 / N_RUFF
RADIAL = ray_surface(M[0.0], lambda u, v: np.stack([np.zeros_like(u), np.zeros_like(u), v], 1),
                     lambda u, v: np.stack([np.cos(u / R0), np.sin(u / R0), np.zeros_like(u)], 1))
ruff_tiles = []
for i in range(N_RUFF):
    u = -math.pi * R0 / 2 + i * step
    for du, vc, r, h0, h1 in [(0.0, SPLIT - 2.4, 3.2, 0.75, 0.95), (step / 2, SPLIT - 4.3, 3.0, 0.2, 0.6)]:
        cs = CrossSection.circle(r, 40).translate([u + du, vc])
        t = ramp_tile(cs, vc + r, vc - r, h0, h1)
        ruff_tiles.append(wrap(t, RADIAL, refine=0.35))
ruff_band = between(SPLIT - 1.3, SPLIT) ^ (M[0.7] - M[-0.8])
ruff = (Manifold.batch_boolean(ruff_tiles, 0) + ruff_band) ^ below(SPLIT)

# ---------------- back lettering ----------------
log("back lettering")
font = FontProperties(fname=FONT)


def text_cs(s, size, center=True):
    tp = TextPath((0, 0), s, size=size, prop=font)
    polys = [np.asarray(p, float) for p in tp.to_polygons(closed_only=True) if len(p) >= 3]
    cs = CrossSection(polys, FillRule.EvenOdd)
    if center:
        x0, y0, x1, y1 = cs.bounds()
        cs = cs.translate([-(x0 + x1) / 2, -y0])
    return cs


LINES = [("NOCTURNAL", 4.3, 24.6), ("VALLEY", 6.0, 18.9), ("MUSIC & ARTS", 3.75, 14.8),
         ("FESTIVAL", 3.75, 10.9), ("SEPT 24-26 2026", 3.2, 7.2)]
prisms = []
for s, size, z in LINES:
    cs = text_cs(s, size)
    b = cs.bounds()
    log(f"   {s!r}: {b[2] - b[0]:.1f} x {b[3] - b[1]:.1f} mm")
    p = cs.mirror([1, 0]).extrude(40).rotate([-90, 0, 0]).mirror([0, 0, 1]).translate([0, 0, z])
    prisms.append(p)
back_text = Manifold.batch_boolean(prisms, 0) ^ (M[0.7] - M[-0.8]) ^ box(-BIG, BIG, 3, BIG, 0, BIG)

# ---------------- eyes: pupils, catchlights, iris ring ----------------
log("eye details")
eye_cut, eye_add = [], []
eR = of._eye_R * S


def cyl_y(r, x, z, y0=-40, y1=0, n=64):
    return Manifold.cylinder(y1 - y0, r, r, n).rotate([-90, 0, 0]).translate([x, y0, z])


for e in of.EYES:
    c = [v * S for v in e["c"]]
    ball = Manifold.sphere(eR, 128).translate(c)
    ex, ez = e["x"] * S, e["z"] * S
    shell = lambda depth: Manifold.sphere(eR + 2, 128).translate(c) - Manifold.sphere(eR - depth, 128).translate(c)
    y1 = e["apex_y"] * S + 2.5
    eye_cut.append(cyl_y(2.9, ex, ez, y1=y1) ^ shell(0.75))                            # pupil
    eye_cut.append((cyl_y(4.75, ex, ez, y1=y1) - cyl_y(4.3, ex, ez, y1=y1)) ^ shell(0.35))  # iris ring
    sx = 1 if ex > 0 else -1
    eye_add.append(cyl_y(0.85, ex + sx * 1.0, ez + 1.1) ^ ball)                        # catchlight

# ---------------- assemble the outside ----------------
log("assembling outside")
skin = Manifold.batch_boolean([outer, bib, wings_relief, ruff, back_text] + eye_add, 0) ^ above(0)
skin = skin - Manifold.batch_boolean(eye_cut, 0)

# ---------------- interior: cavity, lip, snap bead, key ----------------
log("container geometry")
cav = M[-WALL] ^ above(FLOOR)
Z_BEAD = SPLIT + LIP - 1.5
lip = (M[-LIP_OUT] - M[-WALL]) ^ between(SPLIT - 1, SPLIT + LIP)
bead = ((M[-1.08] - M[-1.6]) ^ between(Z_BEAD - 0.35, Z_BEAD + 0.35)) + \
       ((M[-LID_IN] - M[-1.6]) ^ between(Z_BEAD - 0.7, Z_BEAD + 0.7))
key_rib = (M[-0.98] - M[-1.6]) ^ box(-1.2, 1.2, 0, BIG, SPLIT, SPLIT + LIP - 0.6)

lid_recess = M[-LID_IN] ^ between(SPLIT - 1, SPLIT + LIP + 0.3)
bead_groove = M[-0.98] ^ between(Z_BEAD - 0.8, Z_BEAD + 0.8)
key_groove = M[-0.8] ^ box(-1.5, 1.5, 0, BIG, SPLIT - 1, SPLIT + LIP + 0.3)


def inner_top():
    lo, hi = SPLIT, 70.0
    for _ in range(50):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if FM(np.float64(0), np.float64(0), np.float64(mid)) < -WALL else (lo, mid)
    return lo


APEX = inner_top() - 1.2                    # 45-degree ceiling: lid prints upright with no supports
cone = Manifold.cylinder(80, 80, 0, 128).translate([0, 0, APEX - 80])
log(f"   lid ceiling apex z={APEX:.2f}")

# ---------------- hidden interior text & vortex ----------------
log("interior text")
ft = TTFont(FONT); UPM = ft["head"].unitsPerEm; CMAP = ft.getBestCmap(); HMTX = ft["hmtx"]


def advance(ch, size):
    if ch == "*":
        return 0.9 * size
    return HMTX[CMAP[ord(ch)]][0] / UPM * size


def sparkle(size):
    """Four-point sparkle used as a separator."""
    r1, r2 = 0.46 * size, 0.2 * size
    pts = []
    for i in range(8):
        a = math.pi / 2 + i * math.pi / 4
        r = r1 if i % 2 == 0 else r2
        pts.append((r * math.cos(a), r * math.sin(a) + 0.35 * size))
    return CrossSection([np.array(pts)], FillRule.Positive)


def glyph(ch, size):
    """Glyph centred on its advance, baseline at v = 0."""
    if ch == "*":
        return sparkle(size)
    if ch == " ":
        return None
    cs = text_cs(ch, size, center=False)
    return cs.translate([-advance(ch, size) / 2, 0])


# floor: spiral of the dedication, letters' tops pointing outward, running clockwise
FLOOR_TXT = "HAPPY TEN YEARS TO DOWN RIGHT ENTERTAINMENT *"
FS = 3.0                                        # em size -> 2.1 mm capitals, 0.5 mm strokes
floor_poly = (cav ^ between(FLOOR, FLOOR + 0.05)).slice(FLOOR + 0.02)
fp = np.vstack(floor_poly.to_polygons())
r_floor = float(np.min(np.hypot(fp[:, 0], fp[:, 1])))
PITCH = 2.85
r_base0 = r_floor - 0.9 - 0.705 * FS            # baseline radius of the first (outer) turn
log(f"   floor inscribed radius {r_floor:.2f}, spiral starts at r={r_base0:.2f}")
theta = math.radians(100)
th0 = theta
floor_glyphs = []
for ch in FLOOR_TXT:
    adv = advance(ch, FS)
    rb = r_base0 - PITCH * (th0 - theta) / (2 * math.pi)
    theta -= adv / 2 / rb
    rb = r_base0 - PITCH * (th0 - theta) / (2 * math.pi)
    g = glyph(ch, FS)
    if g is not None:
        g = g.rotate(math.degrees(theta) - 90).translate([rb * math.cos(theta), rb * math.sin(theta)])
        floor_glyphs.append(g)
    theta -= adv / 2 / rb
r_end = r_base0 - PITCH * (th0 - theta) / (2 * math.pi)
log(f"   floor spiral: {math.degrees(th0 - theta) / 360:.2f} turns, ends at r={r_end:.2f}")
# hypnotic centre: a raised Archimedean spiral ridge
cpts = []
for t in np.linspace(0, 2.6 * 2 * math.pi, 400):
    r = 0.6 + (r_end - 1.1) * t / (2.6 * 2 * math.pi)
    cpts.append((r * math.cos(-t + theta), r * math.sin(-t + theta)))
cpts = np.array(cpts)
ridge = CrossSection.batch_boolean(
    [CrossSection.batch_hull([CrossSection.circle(0.3, 10).translate(list(cpts[i])),
                              CrossSection.circle(0.3, 10).translate(list(cpts[i + 1]))]) for i in range(len(cpts) - 1)], 0)
floor_art = CrossSection.batch_boolean(floor_glyphs + [ridge], 0)
floor_text = floor_art.extrude(1.1).translate([0, 0, FLOOR - 0.5]) ^ (M[-1.6] ^ above(FLOOR - 1))

# wall: as much of the lineup as fits, on a helix covering the whole inside wall, readable from inside
from lineup import LINEUP
TOP_BILL = ["DAILY BREAD", "EMANCIPATOR", "G JONES", "STS9", "THE DISCO BISCUITS", "JADE CICADA", "DETOX UNIT",
            "MICKMAN", "AB-SOUL", "SKREAM", "TROYBOI", "OTT.", "BLAAP", "IVY LAB", "MINNESOTA"]
ORDER = TOP_BILL + [n for n in LINEUP if n not in TOP_BILL]
WS = 2.9                                         # em -> 2.05 mm capitals, 0.5 mm strokes
CAP = 0.705 * WS
W_PITCH = 2.75
Z0 = SPLIT + LIP - 0.45 - CAP                    # baseline of the top turn (on the lip's inner face)
Z_BOT = 3.8                                      # baseline of the lowest turn
TURNS = (Z0 - Z_BOT) / W_PITCH
TH0 = math.pi / 2                                # start at the back, read clockwise (left to right from inside)


def wall_radius(th, z, level=-WALL):
    lo, hi = np.zeros_like(th), np.full_like(th, 30.0)
    for _ in range(40):
        mid = (lo + hi) / 2
        inside = FM(mid * np.cos(th), mid * np.sin(th), z) < level
        lo, hi = np.where(inside, mid, lo), np.where(inside, hi, mid)
    return lo


ths = np.linspace(TH0, TH0 - 2 * math.pi * TURNS, int(TURNS * 1800))
zb = Z0 - W_PITCH * (TH0 - ths) / (2 * math.pi)
rr = wall_radius(ths, zb + CAP / 2)
arc = np.concatenate([[0], np.cumsum(np.hypot(rr[1:] * np.diff(-ths), np.diff(rr)))])   # true arc length
U_MAX = arc[-1]
theta_of_u = lambda u: np.interp(u, arc, ths)
base_of_u = lambda u: np.interp(u, arc, zb)


def adv(ch, size):
    return 0.95 * size if ch == "*" else advance(ch, size)


words = ["ASTRAL VALLEY ART PARK", "NOCTURNAL VALLEY"]
used = sum(adv(c, WS) for c in "*".join(words)) + adv("*", WS)
included = []
for n in ORDER:
    w = sum(adv(c, WS) for c in n) + adv("*", WS)
    if used + w > U_MAX - 1.0:
        continue                                 # skip names that don't fit; a shorter one later may
    used += w; included.append(n)
WALL_TXT = "*".join(words + included) + "*"
left_out = [n for n in ORDER if n not in included]
log(f"   wall helix: {TURNS:.1f} turns, {U_MAX:.0f} mm of path, {len(included)} of {len(ORDER)} artists fit")
log(f"   left out: {', '.join(left_out)}")
wall_glyphs = []
u = 0.0
for ch in WALL_TXT:
    a = adv(ch, WS)
    uc = u + a / 2
    g = sparkle(WS * 0.8) if ch == "*" else glyph(ch, WS)
    if g is not None:
        slope = (base_of_u(uc + 0.5) - base_of_u(uc - 0.5)) / 1.0
        g = g.rotate(math.degrees(math.atan(slope))).translate([uc, float(base_of_u(uc))])
        wall_glyphs.append(g)
    u += a
wall_flat = CrossSection.batch_boolean(wall_glyphs, 0).extrude(1.4).translate([0, 0, -0.8])
WALLSURF = ray_surface(M[-WALL],
                       lambda u, v: np.stack([np.zeros_like(u), np.zeros_like(u), v], 1),
                       lambda u, v: np.stack([np.cos(theta_of_u(u)), np.sin(theta_of_u(u)), np.zeros_like(u)], 1))
# the cavity surface normal points into the wall, so relief toward the viewer is -N
wall_text = wrap(wall_flat, WALLSURF, refine=0.3, flip=True)
with open(os.path.join(OUT, "wall_lineup.txt"), "w") as fh:
    fh.write("On the wall: " + ", ".join(included) + "\n\nLeft out: " + ", ".join(left_out) + "\n")

# lid ceiling: vortex of swirling grooves cut into the 45-degree cone
log("lid vortex")


def r_meet(ang):
    lo, hi = 0.0, 30.0
    for _ in range(50):
        mid = (lo + hi) / 2
        x, y, z = mid * math.cos(ang), mid * math.sin(ang), APEX - mid
        lo, hi = (mid, hi) if FM(np.float64(x), np.float64(y), np.float64(z)) < -WALL else (lo, mid)
    return lo


R_VORTEX = min(r_meet(a) for a in np.linspace(0, 2 * math.pi, 24, endpoint=False)) - 0.8
log(f"   vortex radius {R_VORTEX:.2f}")
arms = []
N_ARMS = 9
for k in range(N_ARMS):
    pts = []
    for t in np.linspace(0, 1, 90):
        r = 0.9 + (R_VORTEX - 0.9) * t
        a = 2 * math.pi * k / N_ARMS + 2.4 * math.log(r / 0.9) * 0.9
        pts.append((r * math.cos(a), r * math.sin(a), 0.25 + 0.35 * t))
    segs = [CrossSection.batch_hull([CrossSection.circle(pts[i][2], 10).translate(list(pts[i][:2])),
                                     CrossSection.circle(pts[i + 1][2], 10).translate(list(pts[i + 1][:2]))])
            for i in range(len(pts) - 1)]
    arms.append(CrossSection.batch_boolean(segs, 0))
vortex_2d = CrossSection.batch_boolean(arms + [CrossSection.circle(0.9, 32)], 0)
vflat = vortex_2d.extrude(1.4).translate([0, 0, -0.6]).refine_to_length(0.4)
vortex = vflat.warp_batch(lambda p: np.stack([p[:, 0], p[:, 1], APEX - np.hypot(p[:, 0], p[:, 1]) + p[:, 2]], 1))

# ---------------- split into base and lid ----------------
log("splitting")
base = (skin ^ below(SPLIT)) + lip + bead + key_rib
base = base - (cav ^ below(SPLIT + LIP + 1))
base = base + floor_text + (wall_text ^ M[-0.8])

lid = skin ^ above(SPLIT)
lid = lid - lid_recess - bead_groove - key_groove - ((cav ^ cone) ^ above(SPLIT)) - vortex

def keep_main(m):
    parts = sorted(m.decompose(), key=lambda p: -p.volume())
    dropped = sum(p.volume() for p in parts[1:])
    if len(parts) > 1:
        log(f"   dropped {len(parts) - 1} boolean slivers ({dropped:.4f} mm3)")
    return parts[0]


base, lid = keep_main(base), keep_main(lid)
def simplify_safely(m, tolerances):
    """Shrink the triangle count (deviation in mm) but only keep a result that is still one valid,
    watertight solid. Falls back to the full-resolution mesh."""
    t = to_trimesh(m)
    t.merge_vertices()                     # round-trip at STL precision first: simplifies far more reliably
    m = Manifold(Mesh(vert_properties=np.asarray(t.vertices, np.float32), tri_verts=np.asarray(t.faces, np.uint32)))
    for tol in tolerances:
        s = m.simplify(tol)
        if s.status().name == "NoError" and len(s.decompose()) == 1 and to_trimesh(s).is_watertight:
            log(f"   simplified within {tol} mm: {m.num_tri()} -> {s.num_tri()} triangles")
            return s
    log("   kept at full resolution")
    return m


assembled = base + lid                     # full-resolution preview model, made before simplifying
base = simplify_safely(base, (0.008, 0.004))
lid = simplify_safely(lid, (0.004, 0.002))
for name, m in (("owl_base", base), ("owl_lid", lid)):
    t = to_trimesh(m)
    if name == "owl_lid":
        t.apply_translation([0, 0, -SPLIT])
    t.export(os.path.join(OUT, f"{name}.stl"))
    bb = t.bounds
    log(f"{name}: status {m.status().name}, genus {m.genus()}, watertight {t.is_watertight}, "
        f"bodies {len(m.decompose())}, size {np.round(bb[1] - bb[0], 1)}, vol {t.volume / 1000:.1f} cm3, tris {len(t.faces)}")

to_trimesh(assembled).export(os.path.join(OUT, "owl_assembled.stl"))
cv = (cav ^ below(SPLIT + LIP)).volume() + ((cav ^ cone) ^ above(SPLIT + LIP)).volume()
log(f"storage volume ~{cv / 1000:.1f} ml")
np.savez(os.path.join(OUT, "meta.npz"), split=SPLIT, lip=LIP, floor=FLOOR, apex=APEX)
