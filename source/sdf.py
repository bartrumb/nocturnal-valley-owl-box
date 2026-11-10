"""Signed-distance sculpting helpers: primitives, smooth booleans, and meshing to a Manifold.
Units: mm. Z up, owl front toward -Y. Negative = inside."""
import numpy as np
import trimesh
from manifold3d import Manifold, Mesh
from skimage.measure import marching_cubes


# ---------- smooth booleans ----------
def smin(a, b, k):
    if k <= 0:
        return np.minimum(a, b)
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b * (1 - h) + a * h - k * h * (1 - h)


def smax(a, b, k):
    return -smin(-a, -b, k)


# ---------- primitives (vectorized over X, Y, Z arrays) ----------
def ellipsoid(X, Y, Z, c, r, rot=None):
    x, y, z = X - c[0], Y - c[1], Z - c[2]
    if rot is not None:                       # rot: 3x3 world->local
        x, y, z = (rot[0, 0] * x + rot[0, 1] * y + rot[0, 2] * z,
                   rot[1, 0] * x + rot[1, 1] * y + rot[1, 2] * z,
                   rot[2, 0] * x + rot[2, 1] * y + rot[2, 2] * z)
    px, py, pz = x / r[0], y / r[1], z / r[2]
    k0 = np.sqrt(px * px + py * py + pz * pz)
    k1 = np.sqrt((px / r[0]) ** 2 + (py / r[1]) ** 2 + (pz / r[2]) ** 2)
    return k0 * (k0 - 1.0) / np.maximum(k1, 1e-9)


def sphere(X, Y, Z, c, r):
    return np.sqrt((X - c[0]) ** 2 + (Y - c[1]) ** 2 + (Z - c[2]) ** 2) - r


def round_cone(X, Y, Z, a, b, r1, r2):
    """Inigo Quilez's exact round cone between points a (radius r1) and b (radius r2)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    ba = b - a
    l2 = float(ba @ ba); rr = r1 - r2; a2 = l2 - rr * rr; il2 = 1.0 / l2
    pax, pay, paz = X - a[0], Y - a[1], Z - a[2]
    y = pax * ba[0] + pay * ba[1] + paz * ba[2]
    z = y - l2
    qx, qy, qz = pax * l2 - ba[0] * y, pay * l2 - ba[1] * y, paz * l2 - ba[2] * y
    x2 = qx * qx + qy * qy + qz * qz
    y2 = y * y * l2
    z2 = z * z * l2
    k = np.sign(rr) * rr * rr * x2
    d1 = np.sqrt(x2 + z2) * il2 - r2
    d2 = np.sqrt(x2 + y2) * il2 - r1
    d3 = (np.sqrt(np.maximum(x2 * a2 * il2, 0)) + y * rr) * il2 - r1
    return np.where(np.sign(z) * a2 * z2 > k, d1, np.where(np.sign(y) * a2 * y2 < k, d2, d3))


def chain(X, Y, Z, pts, radii, k=0.0):
    """Tapered tube through a list of points (smooth-unioned round cones)."""
    d = None
    for i in range(len(pts) - 1):
        s = round_cone(X, Y, Z, pts[i], pts[i + 1], radii[i], radii[i + 1])
        d = s if d is None else smin(d, s, k)
    return d


def rot_xyz(ax, ay, az):
    """world->local rotation for Euler angles in degrees (applied x, then y, then z)."""
    ax, ay, az = np.radians([ax, ay, az])
    Rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
    Ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
    Rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
    return (Rz @ Ry @ Rx).T


# ---------- 2D helpers (in a plane, used for extruded shapes) ----------
def circle2(u, v, c, r):
    return np.sqrt((u - c[0]) ** 2 + (v - c[1]) ** 2) - r


def vesica2(u, v, length, width):
    """Pointed leaf/lens shape with its tips on the u axis (IQ's sdVesica)."""
    b = length / 2
    r = (b * b / (width / 2) + width / 2) / 2
    d = r - width / 2
    px, py = np.abs(v), np.abs(u)
    tip = np.sqrt(px ** 2 + (py - b) ** 2)
    body = np.sqrt((px + d) ** 2 + py ** 2) - r
    return np.where((py - b) * d > px * b, tip, body)


# ---------- meshing ----------
def grid_axes(bounds, res):
    return [np.arange(lo, hi + res * 0.5, res, dtype=np.float32) for lo, hi in bounds]


def field_to_manifold(field_fn, bounds, res, level=0.0, chunk=40, smooth=8, cache=None):
    """Evaluate field on a grid (in z-chunks), marching-cube it, Taubin-smooth, return Manifold."""
    import os
    if cache and os.path.exists(cache):
        z = np.load(cache)
        verts, faces = z["v"], z["f"]
    else:
        xs, ys, zs = grid_axes(bounds, res)
        vol = np.empty((len(xs), len(ys), len(zs)), np.float32)
        for k0 in range(0, len(zs), chunk):
            X, Y, Z = np.meshgrid(xs, ys, zs[k0:k0 + chunk], indexing="ij")
            vol[:, :, k0:k0 + chunk] = field_fn(X, Y, Z)
        # guarantee a closed surface: everything on the grid boundary is outside
        for sl in [(0,), (-1,)]:
            vol[sl[0], :, :] = np.maximum(vol[sl[0], :, :], level + res)
            vol[:, sl[0], :] = np.maximum(vol[:, sl[0], :], level + res)
            vol[:, :, sl[0]] = np.maximum(vol[:, :, sl[0]], level + res)
        verts, faces, _, _ = marching_cubes(vol, level, spacing=(res, res, res), allow_degenerate=False)
        verts = verts + np.array([xs[0], ys[0], zs[0]], np.float32)
        del vol
        tm = trimesh.Trimesh(verts, faces, process=True)
        if smooth:
            trimesh.smoothing.filter_taubin(tm, lamb=0.5, nu=-0.53, iterations=smooth)
        if tm.volume < 0:
            tm.invert()
        verts, faces = np.asarray(tm.vertices, np.float32), np.asarray(tm.faces, np.uint32)
        if cache:
            np.savez(cache, v=verts, f=faces)
    m = Manifold(Mesh(vert_properties=np.ascontiguousarray(verts, np.float32),
                      tri_verts=np.ascontiguousarray(faces, np.uint32)))
    assert m.status().name == "NoError", m.status()
    return m


def field_levels_to_manifolds(field_fn, bounds, res, levels, chunk=40, smooth=6, cache_prefix=None):
    """One field evaluation, several iso-levels (offset surfaces). Returns {level: Manifold}."""
    import os
    out, todo = {}, []
    for lv in levels:
        fn = f"{cache_prefix}_{lv:+.2f}.npz" if cache_prefix else None
        if fn and os.path.exists(fn):
            out[lv] = field_to_manifold(None, None, None, cache=fn)
        else:
            todo.append((lv, fn))
    if todo:
        xs, ys, zs = grid_axes(bounds, res)
        vol = np.empty((len(xs), len(ys), len(zs)), np.float32)
        for k0 in range(0, len(zs), chunk):
            X, Y, Z = np.meshgrid(xs, ys, zs[k0:k0 + chunk], indexing="ij")
            vol[:, :, k0:k0 + chunk] = field_fn(X, Y, Z)
        for lv, fn in todo:
            v = vol.copy()
            for ax in range(3):
                idx = [slice(None)] * 3
                for e in (0, -1):
                    idx[ax] = e
                    v[tuple(idx)] = np.maximum(v[tuple(idx)], lv + res)
            verts, faces, _, _ = marching_cubes(v, lv, spacing=(res, res, res), allow_degenerate=False)
            del v
            verts = verts + np.array([xs[0], ys[0], zs[0]], np.float32)
            tm = trimesh.Trimesh(verts, faces, process=True)
            if smooth:
                trimesh.smoothing.filter_taubin(tm, lamb=0.5, nu=-0.53, iterations=smooth)
            if tm.volume < 0:
                tm.invert()
            V, F = np.asarray(tm.vertices, np.float32), np.asarray(tm.faces, np.uint32)
            if fn:
                np.savez(fn, v=V, f=F)
            out[lv] = Manifold(Mesh(vert_properties=np.ascontiguousarray(V), tri_verts=np.ascontiguousarray(F)))
            assert out[lv].status().name == "NoError", (lv, out[lv].status())
    return out


def to_trimesh(m):
    me = m.to_mesh()
    return trimesh.Trimesh(np.asarray(me.vert_properties)[:, :3], np.asarray(me.tri_verts), process=False)
