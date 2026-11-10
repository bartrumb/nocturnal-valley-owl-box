"""The v3 owl as a signed-distance field. Z up, front toward -Y, bed at z = 0."""
import numpy as np
from sdf import smin, smax, ellipsoid, sphere, round_cone, chain, rot_xyz, circle2, vesica2

# ---------- main form (this alone defines the container walls) ----------
BODY_C, BODY_R = (0.0, 0.0, 21.0), (30.0, 27.0, 31.0)
HEAD_C, HEAD_R = (0.0, 0.0, 59.0), (27.5, 25.0, 22.0)
NECK_K = 20.0


def F_main(X, Y, Z):
    return smin(ellipsoid(X, Y, Z, BODY_C, BODY_R), ellipsoid(X, Y, Z, HEAD_C, HEAD_R), NECK_K)


def F_main_dist(X, Y, Z, h=0.02):
    """F_main divided by its gradient length: a near-true distance, so offset surfaces
    (wall, lip, clearances) stay the intended thickness even in the smooth neck blend."""
    f = F_main(X, Y, Z)
    gx = F_main(X + h, Y, Z) - F_main(X - h, Y, Z)
    gy = F_main(X, Y + h, Z) - F_main(X, Y - h, Z)
    gz = F_main(X, Y, Z + h) - F_main(X, Y, Z - h)
    g = np.sqrt(gx * gx + gy * gy + gz * gz) / (2 * h)
    return f / np.maximum(g, 0.2)


def front_y(x, z, level=0.0):
    """y of the main form's front surface (F_main = level) at (x, z), by bisection."""
    lo, hi = -45.0, 0.0
    for _ in range(60):
        mid = (lo + hi) / 2
        v = F_main(np.float64(x), np.float64(mid), np.float64(z)) - level
        lo, hi = (lo, mid) if v < 0 else (mid, hi)
    return (lo + hi) / 2


# ---------- face ----------
EYE_X, EYE_Z = 11.6, 60.5
PAD = 2.0                                  # facial disc height
DISC_R, DISC_C = 12.6, (11.2, 60.8)        # heart = two discs + chin point
CHIN = (0.0, 47.6)
EYE_BASE_R, EYE_H = 8.6, 3.2               # eyeball cap: base radius, height above the disc
_eye_R = (EYE_BASE_R ** 2 + EYE_H ** 2) / (2 * EYE_H)
EYES = []
for sx in (-1, 1):
    yf = front_y(sx * EYE_X, EYE_Z, level=PAD)          # disc surface
    apex = yf - EYE_H
    EYES.append(dict(x=sx * EYE_X, z=EYE_Z, apex_y=apex, c=(sx * EYE_X, apex + _eye_R, EYE_Z)))


def heart2(x, z):
    d = smin(circle2(x, z, (-DISC_C[0], DISC_C[1]), DISC_R), circle2(x, z, (DISC_C[0], DISC_C[1]), DISC_R), 2.0)
    # chin point: triangle from the lower disc edges down to CHIN
    v0 = np.array([-14.0, 57.0]); v1 = np.array([14.0, 57.0]); v2 = np.array(CHIN)
    def half(p, a, b):
        n = np.array([b[1] - a[1], -(b[0] - a[0])]); n /= np.linalg.norm(n)
        return (x - a[0]) * n[0] + (z - a[1]) * n[1]
    tri = np.maximum(np.maximum(half(None, v0, v2), half(None, v2, v1)), -(z - 57.0) - 20)
    tri = np.maximum(tri, z - 57.5)
    return smin(d, tri, 2.5)


def face(X, Y, Z, d):
    pad = smax(heart2(X, Z), F_main(X, Y, Z) - PAD, 0.9)
    pad = smax(pad, Y + 6.0, 1.0)                              # front only
    d = smin(d, pad, 0.7)
    for e in EYES:
        d = smin(d, sphere(X, Y, Z, e["c"], _eye_R), 0.6)
        yb = e["apex_y"] + EYE_H                                   # disc surface at the eye
        q = np.sqrt((X - e["x"]) ** 2 + (Z - e["z"]) ** 2) - (EYE_BASE_R + 0.5)
        d = smin(d, np.sqrt(q * q + (Y - yb + 0.2) ** 2) - 1.15, 0.5)
    # beak: short hooked cone between the eyes
    yb = front_y(0.0, 56.5, level=PAD)
    d = smin(d, chain(X, Y, Z, [(0, yb + 2.5, 58.5), (0, yb - 1.8, 55.2), (0, yb - 2.6, 51.6)], [3.3, 2.3, 0.7]), 0.9)
    return d


# ---------- ear tufts: short blunt feather clumps (pocket-safe) ----------
def tufts(X, Y, Z, d):
    for sx in (-1, 1):
        base = np.array([sx * 15.5, -2.5, 72.0])
        for tip, r1, r2 in [((sx * 24.0, -2.5, 84.5), 5.8, 2.2), ((sx * 19.5, -3.5, 85.5), 5.0, 2.0)]:
            s = 0.62                                            # flatten front-to-back
            Yf = (Y - base[1]) / s + base[1]
            blade = round_cone(X, Yf, Z, base, (tip[0], (tip[1] - base[1]) / s + base[1], tip[2]), r1, r2) * s
            d = smin(d, blade, 3.2)
    return d


# ---------- wings with primary feather tips ----------
def wings(X, Y, Z, d):
    for sx in (-1, 1):
        R = rot_xyz(-10, sx * 9, 0)
        w = ellipsoid(X, Y, Z, (sx * 24.5, 4.0, 25.0), (6.4, 14.5, 15.5), rot=R)
        d = smin(d, w, 3.4)
    return d


# ---------- feet: short chunky toes resting on the bed (pocket-safe) ----------
def feet(X, Y, Z, d):
    for xf in (-9.0, 9.0):
        for dx in (-3.6, 0.0, 3.6):
            a = (xf + dx * 0.4, -15.5, 3.4)
            b = (xf + dx * 1.15, -24.0, 2.3)
            c = (xf + dx * 1.25, -26.2, 1.4)              # blunt claw
            d = smin(d, chain(X, Y, Z, [a, b, c], [2.9, 2.3, 1.2], k=0.5), 1.4)
    return d


def F_outer(X, Y, Z):
    d = F_main(X, Y, Z)
    d = face(X, Y, Z, d)
    d = tufts(X, Y, Z, d)
    d = wings(X, Y, Z, d)
    d = feet(X, Y, Z, d)
    return d                                                   # flat bottom is cut after meshing
