#!/usr/bin/env python3
"""
capsid_render.py - render an icosahedral phage capsid from major capsid protein (MCP) PDB file(s).

Deps: numpy, scipy, matplotlib   (no PyMOL/ChimeraX needed)

Input: one or more PDB files. Anything goes; the mode is picked automatically (--mode auto):
  biomt        a file with REMARK 350 BIOMT operators (e.g. an RCSB assembly file, >=20 ops)
               -> exact expansion straight from the file.
  icosahedral  the chains are an asymmetric unit already in a standard icosahedral (222)
               orientation (VIPERdb-style). The 60 icosahedral operators are applied; both
               222 settings are tried and the one with sensible interfaces is used.
  lattice      anything else (a bare MCP monomer, AlphaFold model, a few chains in arbitrary
               orientation). Builds a Caspar-Klug T-lattice MODEL: 12 pentamers + 10(T-1)
               hexamers, MCP copies placed on all 60T positions. This is a schematic of what
               the capsid looks like, not a refined structure. Needs --T (default 7).
               Template use: first chain -> pentamer subunits, remaining chains (if any)
               -> hexamer subunits (cycled around each ring); with one chain it is used for all.

Optional extra structural proteins (every one is optional; give any combination):
  --portal   PDB   portal protein (replaces the pentamer at one vertex, which becomes the tail axis)
  --tail     PDB   tail tube / sheath protein (helical stack of rings; --tail-length/--tail-rise/--tail-twist)
  --tapemeasure PDB  tape measure protein (runs along the tail axis, visible with --cut 0)
  --tailfiber PDB  tail fibers (--fiber-n copies around the tail tip, --fiber-angle)
  --decoration PDB capsid decoration protein (--deco-site quasi3|3fold|2fold|5fold|hexon|penton)
  Each takes FILE.pdb or FILE.pdb:A,B to pick chains. A single chain is replicated into a ring
  (--portal-n 12, --tail-n 6, ...); a multi-chain file is treated as an already assembled oligomer.
  Parts are placed schematically from the capsid's 5-fold axis - a model, not a docked structure.

Output: <out>.png and <out>.pdf (multi-view figure with scale bar; PDF fonts are Type 42,
        so text stays editable in Illustrator/Inkscape). --raw also writes per-view RGBA PNGs.

Examples:
  python capsid_render.py mcp.pdb --T 7
  python capsid_render.py 1ohg_assembly.pdb -o lambda --views 5 --cut 0
  python capsid_render.py gp5_A.pdb gp5_B.pdb --T 7l --color capsomer --diameter 650
"""
import argparse
import math
import os
import sys
from collections import OrderedDict

import numpy as np

PHI = (1 + 5 ** 0.5) / 2
PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2",
           "#EECA3B", "#B279A2", "#FF9DA6", "#9D755D", "#BAB0AC"]
NUCLEIC = {"DA", "DC", "DG", "DT", "DU", "A", "C", "G", "U", "I"}


def log(*a):
    print(*a, file=sys.stderr)


def hex2rgb(h):
    return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], np.float32) / 255.0


def unit(v):
    return v / np.linalg.norm(v)


def rotz(t):
    c, s_ = math.cos(t), math.sin(t)
    return np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1.]])


def rotx(t):
    c, s_ = math.cos(t), math.sin(t)
    return np.array([[1., 0, 0], [0, c, -s_], [0, s_, c]])


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


# --------------------------------------------------------------------------- PDB input
def parse_pdb(path):
    """-> (OrderedDict chain -> (n,3) CA coords, OrderedDict biomolecule -> [groups])."""
    res, biomols = OrderedDict(), OrderedDict()
    mol = grp = None
    with open(path) as fh:
        for line in fh:
            rec = line[:6]
            if rec == "ENDMDL":
                break
            if rec in ("ATOM  ", "HETATM"):
                if rec == "HETATM" and line[17:20] != "MSE":
                    continue
                if line[16] not in " A1" or line[17:20].strip() in NUCLEIC:
                    continue
                try:
                    xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
                except ValueError:
                    continue
                r = res.setdefault((line[21], line[22:27]), {"ca": None, "xyz": []})
                r["xyz"].append(xyz)
                if line[12:16].strip() == "CA":
                    r["ca"] = xyz
            elif rec == "REMARK" and line[7:10] == "350":
                t = line[10:].strip()
                if t.startswith("BIOMOLECULE:"):
                    mol = t.split(":")[1].strip()
                    biomols[mol] = []
                    grp = None
                elif t.startswith("APPLY THE FOLLOWING TO CHAINS:") and mol is not None:
                    grp = {"chains": [c.strip() for c in t.split(":", 1)[1].split(",") if c.strip()],
                           "ops": OrderedDict()}
                    biomols[mol].append(grp)
                elif t.startswith("AND CHAINS:") and grp is not None:
                    grp["chains"] += [c.strip() for c in t.split(":", 1)[1].split(",") if c.strip()]
                elif t.startswith("BIOMT") and grp is not None:
                    tok = t.split()
                    grp["ops"].setdefault(int(tok[1]), np.zeros((3, 4)))[int(tok[0][5:]) - 1] = \
                        [float(v) for v in tok[2:6]]
    chains = OrderedDict()
    for (ch, _), r in res.items():
        chains.setdefault(ch, []).append(r["ca"] if r["ca"] is not None else tuple(np.mean(r["xyz"], 0)))
    chains = OrderedDict((c, np.array(v, float)) for c, v in chains.items() if len(v) >= 20)
    return chains, biomols


def load_inputs(paths, chain_sel, bio_sel):
    """-> asu: list of (label, xyz); biomt: list of (asu_index, op_id, xyz) or None."""
    asu, biomt = [], []
    for fi, p in enumerate(paths):
        chains, biomols = parse_pdb(p)
        if chain_sel:
            chains = OrderedDict((c, v) for c, v in chains.items() if c in chain_sel)
        if not chains:
            sys.exit(f"No usable protein chains found in {p}" + (f" for --chain {chain_sel}" if chain_sel else ""))
        base = os.path.splitext(os.path.basename(p))[0]
        first = len(asu)
        for c, v in chains.items():
            asu.append((f"{base}:{c}" if len(paths) > 1 else c, v))
        index = {c: first + i for i, c in enumerate(chains)}
        # pick biomolecule with a full complement of operators
        pick = None
        for mid, groups in biomols.items():
            nops = max((len(g["ops"]) for g in groups), default=0)
            if (bio_sel is None and nops >= 20) or (bio_sel is not None and mid == str(bio_sel)):
                pick = groups
                break
        if pick:
            for g in pick:
                for n, M in g["ops"].items():
                    for c in (g["chains"] or list(chains)):
                        if c in chains:
                            biomt.append((index[c], (fi, n), chains[c] @ M[:, :3].T + M[:, 3], M[:, :3]))
    return asu, (biomt or None)


# ------------------------------------------------------------------- icosahedral symmetry
def rot_axis(axis, ang):
    K = skew(unit(np.asarray(axis, float)))
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K


def icosahedral_group(setting=1):
    """60 proper rotations, identity first. 2-folds on x,y,z; setting 1: 5-folds at (0,+-1,+-phi)
    cyclic, setting 2: (0,+-phi,+-1) cyclic (the two standard 222 conventions)."""
    ax5 = (0, 1, PHI) if setting == 1 else (0, PHI, 1)
    gens = [rot_axis(ax5, 2 * math.pi / 5), np.diag([-1., -1., 1.]), np.diag([1., -1., -1.]),
            np.array([[0, 0, 1.], [1, 0, 0], [0, 1, 0]])]
    ops, seen, queue = [np.eye(3)], {tuple(np.round(np.eye(3), 4).ravel())}, [np.eye(3)]
    while queue:
        a = queue.pop()
        for g in gens:
            m = g @ a
            key = tuple(np.round(m, 4).ravel())
            if key not in seen:
                seen.add(key)
                ops.append(m)
                queue.append(m)
    assert len(ops) == 60, len(ops)
    return ops


def axes_from_ops(rots):
    """find representative 5-,3-,2-fold axes from a list of 3x3 rotations."""
    found = {}
    for R in rots:
        c = np.clip((np.trace(R) - 1) / 2, -1, 1)
        ang = math.degrees(math.acos(c))
        for name, a0 in (("5", 72), ("3", 120), ("2", 180)):
            if name not in found and abs(ang - a0) < 1.0:
                w, v = np.linalg.eig(R)
                found[name] = unit(np.real(v[:, np.argmin(np.abs(w - 1))]))
    return found


def fit_icosahedral(ca):
    """try both 222 settings; score = inter-copy contacts minus clashes (per ASU residue)."""
    from scipy.spatial import cKDTree
    best = None
    for setting in (1, 2):
        ops = icosahedral_group(setting)
        d, _ = cKDTree(np.concatenate([ca @ R.T for R in ops[1:]])).query(ca)
        contact, clash = float(np.mean(d < 10.0)), float(np.mean(d < 3.0))
        score = contact - 5 * clash
        if best is None or score > best[0]:
            best = (score, setting, contact, clash)
    return best


# --------------------------------------------------------------------------- T-lattice
def parse_T(s, hk):
    if hk:
        return tuple(hk)
    s = str(s).lower()
    hand = "d"
    if s[-1] in "dl":
        hand, s = s[-1], s[:-1]
    T = int(s)
    for h in range(0, 60):
        for k in range(0, h + 1):
            if h * h + h * k + k * k == T:
                return (h, k) if (hand == "d" or k in (0, h)) else (k, h)
    ok = sorted({h * h + h * k + k * k for h in range(1, 12) for k in range(0, h + 1)})
    sys.exit(f"T={T} is not a valid Caspar-Klug number. Valid: {ok[:20]} ...")


def icosahedron(setting=1):
    base = []
    for s1 in (1, -1):
        for s2 in (1, -1):
            if setting == 1:
                base += [(0, s1, s2 * PHI), (s1, s2 * PHI, 0), (s2 * PHI, 0, s1)]
            else:
                base += [(0, s1 * PHI, s2), (s2, 0, s1 * PHI), (s1 * PHI, s2, 0)]
    V = np.array(base, float)
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    D = np.linalg.norm(V[:, None] - V[None], axis=2)
    E = D[D > 1e-6].min()
    faces = []
    for i in range(12):
        for j in range(i + 1, 12):
            for k in range(j + 1, 12):
                if abs(D[i, j] - E) < 1e-6 and abs(D[j, k] - E) < 1e-6 and abs(D[i, k] - E) < 1e-6:
                    a, b, c = V[i], V[j], V[k]
                    faces.append((i, j, k) if np.cross(b - a, c - a) @ (a + b + c) > 0 else (i, k, j))
    assert len(faces) == 20
    return V, faces, E


def lattice_capsomers(h, k):
    """capsomer centres (unit icosahedron, vertex radius 1), face-normal sums, pentamer flags,
    plus the centres of all 20T lattice triangles (true + quasi 3-fold sites)."""
    V, faces, E = icosahedron(1)
    a1, a2 = np.array([1., 0.]), np.array([.5, math.sqrt(3) / 2])
    P1, P2 = h * a1 + k * a2, -k * a1 + (h + k) * a2
    Minv = np.linalg.inv(np.column_stack([P1, P2]))
    bary, n = [], h + k + 1
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            b = Minv @ (i * a1 + j * a2)
            if b[0] > -1e-9 and b[1] > -1e-9 and 1 - b[0] - b[1] > -1e-9:
                bary.append((1 - b[0] - b[1], b[0], b[1]))
    pts, nsum, key2i = [], [], {}
    tri_p, tri_n, tri_v = [], [], []

    def to3d(cart, A, B, C):
        b = Minv @ cart
        return (1 - b.sum()) * A + b[0] * B + b[1] * C, min(1 - b.sum(), b[0], b[1])

    for f in faces:
        A, B, C = V[list(f)]
        fn = unit(np.cross(B - A, C - A))
        for b0, b1, b2 in bary:
            p = b0 * A + b1 * B + b2 * C
            key = tuple(np.round(p, 5))
            if key not in key2i:
                key2i[key] = len(pts)
                pts.append(p)
                nsum.append(np.zeros(3))
            nsum[key2i[key]] += fn
        for i in range(-n, n + 1):
            for j in range(-n, n + 1):
                for corners in (((i, j), (i + 1, j), (i, j + 1)), ((i + 1, j), (i, j + 1), (i + 1, j + 1))):
                    cl = np.mean(corners, axis=0)
                    p, mb = to3d(cl[0] * a1 + cl[1] * a2, A, B, C)
                    if mb > 1e-9:
                        c0 = corners[0]
                        tri_p.append(p)
                        tri_n.append(fn)
                        tri_v.append(to3d(c0[0] * a1 + c0[1] * a2, A, B, C)[0])
    P, N = np.array(pts), np.array(nsum)
    N /= np.linalg.norm(N, axis=1, keepdims=True)
    T = h * h + h * k + k * k
    assert len(P) == 10 * T + 2, (len(P), 10 * T + 2)
    assert len(tri_p) == 20 * T, (len(tri_p), 20 * T)
    return dict(P=P, N=N, pent=np.linalg.norm(P, axis=1) > 1 - 1e-6, E=E,
                tri_p=np.array(tri_p), tri_n=np.array(tri_n), tri_v=np.array(tri_v))


def principal_frame(X, radial="minor", flip=False):
    """centre X and express it in a right-handed frame [in-plane1, in-plane2, radial(outward)]."""
    Xc = X - X.mean(0)
    _, v = np.linalg.eigh(np.cov(Xc.T))
    ax = [v[:, 2], v[:, 1], v[:, 0]]  # major, mid, minor
    ax = [a if np.sum((Xc @ a) ** 3) >= 0 else -a for a in ax]
    i1, i2, i3 = {"minor": (0, 1, 2), "mid": (0, 2, 1), "major": (1, 2, 0)}[radial]
    e1, e2 = ax[i1], ax[i2]
    if np.cross(e1, e2) @ ax[i3] < 0:
        e2 = -e2
    F = np.column_stack([e1, e2, np.cross(e1, e2)])
    if flip:
        F = F @ np.diag([1., -1., -1.])
    return Xc @ F


class Assembly:
    pass


def build_lattice(templates, args):
    from scipy.spatial import cKDTree, ConvexHull
    h, k = parse_T(args.T, args.hk)
    T = h * h + h * k + k * k
    L = lattice_capsomers(h, k)
    P, Nf, pent, E = L["P"], L["N"], L["pent"], L["E"]
    s = args.roundness
    Rs = math.sqrt(5 * math.sqrt(3) * E * E / (4 * math.pi))  # equal-area sphere radius
    U = P / np.linalg.norm(P, axis=1, keepdims=True)
    Pb = (1 - s) * P + s * Rs * U
    Nb = (1 - s) * Nf + s * U
    Nb /= np.linalg.norm(Nb, axis=1, keepdims=True)
    vfac = (1 - s) + s * Rs

    Y = [principal_frame(X, args.radial, args.flip) for X in templates]
    a_sub = ConvexHull(Y[0][:, :2]).volume * args.packing  # 2-D hull area of the footprint
    if args.diameter:
        S = (args.diameter / 2 - max(y[:, 2].max() for y in Y)) / vfac
    else:
        S = math.sqrt(60 * T * a_sub / (5 * math.sqrt(3))) / E
    d_unit = E / math.sqrt(T)

    _, nb = cKDTree(Pb).query(Pb, k=7)
    psi = math.radians(args.twist)
    hex_t = list(range(1, len(Y))) or [0]
    out, cidx, sidx, cap, tans = [], [], [], [], []
    sub = 0
    for ci in range(len(Pb)):
        n_c = Nb[ci]
        nsub = 5 if pent[ci] else 6
        t1 = unit(np.cross(n_c, np.eye(3)[np.argmin(np.abs(n_c))]))
        t2 = np.cross(n_c, t1)
        dv = Pb[nb[ci, 1:nsub + 1]] - Pb[ci]
        alpha = np.arctan2(dv @ t2, dv @ t1)
        phi0 = np.angle(np.mean(np.exp(1j * nsub * alpha))) / nsub
        rho = d_unit / 3 * (0.85 if pent[ci] else 1.0)
        tans.append(math.cos(phi0) * t1 + math.sin(phi0) * t2)
        for j in range(nsub):
            th = phi0 + 2 * math.pi * (j + args.ring_offset) / nsub
            u = math.cos(th) * t1 + math.sin(th) * t2
            m = math.cos(psi) * u + math.sin(psi) * np.cross(n_c, u)
            W = np.column_stack([m, np.cross(n_c, m), n_c])
            t = 0 if pent[ci] else hex_t[j % len(hex_t)]
            out.append(Y[t] @ W.T + (Pb[ci] + rho * u) * S)
            cidx.append(np.full(len(Y[t]), t))
            sidx.append(np.full(len(Y[t]), sub))
            cap.append(np.full(len(Y[t]), int(pent[ci])))
            sub += 1
    A = Assembly()
    A.coords, A.chain_idx = np.concatenate(out), np.concatenate(cidx)
    A.sub_idx, A.cap = np.concatenate(sidx), np.concatenate(cap)
    A.rots = icosahedral_group(1)
    A.axes = axes_from_ops(A.rots)
    tans = np.array(tans)
    A.sites = {"hexon": (Pb[~pent] * S, Nb[~pent], tans[~pent]), "penton": (Pb[pent] * S, Nb[pent], tans[pent])}
    tp = L["tri_p"]
    tu = tp / np.linalg.norm(tp, axis=1, keepdims=True)
    Tb = (1 - s) * tp + s * Rs * tu
    Tn = (1 - s) * L["tri_n"] + s * tu
    Tn /= np.linalg.norm(Tn, axis=1, keepdims=True)
    tt = L["tri_v"] - tp
    tt -= np.sum(tt * Tn, axis=1, keepdims=True) * Tn
    tt /= np.linalg.norm(tt, axis=1, keepdims=True)
    A.sites["quasi3"] = (Tb * S, Tn, tt)
    A.info = f"T={T} ({h},{k}) lattice model, {sub} subunits"
    return A


# ---------------------------------------------------------------------------- assembly
def assemble(args, asu, biomt):
    mode = args.mode
    labels = [a[0] for a in asu]
    if mode in ("auto", "biomt") and biomt:
        A = Assembly()
        A.coords = np.concatenate([b[2] for b in biomt])
        A.chain_idx = np.concatenate([np.full(len(b[2]), b[0]) for b in biomt])
        A.sub_idx = np.concatenate([np.full(len(b[2]), i) for i, b in enumerate(biomt)])
        A.cap = np.full(len(A.coords), -1)
        uniq = {tuple(np.round(b[3], 3).ravel()): b[3] for b in biomt}
        A.rots, A.sites = list(uniq.values()), {}
        A.axes = axes_from_ops(A.rots)
        A.info = f"BIOMT assembly, {len(biomt)} subunits"
        return A, labels
    if mode == "biomt":
        sys.exit("--mode biomt: no REMARK 350 operators (>=20) found.")

    if mode in ("auto", "icosahedral"):
        ca = np.concatenate([v for _, v in asu])
        score, setting, contact, clash = fit_icosahedral(ca)
        ok = contact >= 0.25 and clash < 0.02
        if args.setting != "auto":
            setting = int(args.setting)
        if ok or mode == "icosahedral":
            if not ok:
                log(f"WARNING: weak icosahedral fit (contacts {contact:.0%}, clashes {clash:.1%}); "
                    "the input may not be in a standard 222 orientation.")
            ops = icosahedral_group(setting)
            c, ci = [], []
            for oi, R in enumerate(ops):
                for i, (_, v) in enumerate(asu):
                    c.append(v @ R.T)
                    ci.append(np.full(len(v), i))
            A = Assembly()
            A.coords, A.chain_idx = np.concatenate(c), np.concatenate(ci)
            A.sub_idx = np.concatenate([np.full(len(x), i) for i, x in enumerate(c)])
            A.cap = np.full(len(A.coords), -1)
            A.rots, A.sites = ops, {}
            A.axes = axes_from_ops(ops)
            A.info = (f"icosahedral expansion (222 setting {setting}), {len(asu)} chain(s) x 60 = "
                      f"{60 * len(asu)} subunits")
            log(f"icosahedral fit: setting {setting}, contacts {contact:.0%}, clashes {clash:.1%}")
            return A, labels
        log(f"Input does not look like an ASU in standard icosahedral orientation (inter-copy contacts "
            f"{contact:.0%}, clashes {clash:.1%}); building a T={args.T} lattice model instead. "
            "(Force the symmetry expansion with --mode icosahedral.)")

    templates = [v for _, v in asu]
    A = build_lattice(templates, args)
    return A, labels


# ------------------------------------------------------------------- extra structural proteins
PART_IDS = ["portal", "tail", "tapemeasure", "tailfiber", "decoration"]  # part id = index + 1
PART_COLORS = {"portal": "#8C564B", "tail": "#59A14F", "tapemeasure": "#E7298A",
               "tailfiber": "#9467BD", "decoration": "#FF9F1C"}
PART_LABELS = {"portal": "portal", "tail": "tail tube/sheath", "tapemeasure": "tape measure",
               "tailfiber": "tail fibers", "decoration": "decoration"}
DECO_N = {"5fold": 5, "penton": 5, "hexon": 6, "3fold": 3, "quasi3": 3, "2fold": 2}


def load_part(spec, what):
    """FILE.pdb or FILE.pdb:A,B  -> list of (n,3) CA arrays (one per chain)."""
    path, sel = spec, None
    if not os.path.isfile(spec) and ":" in spec:
        path, c = spec.rsplit(":", 1)
        sel = {x for x in c.split(",") if x}
    if not os.path.isfile(path):
        sys.exit(f"--{what}: file not found: {spec}")
    chains, _ = parse_pdb(path)
    if sel:
        chains = OrderedDict((c, v) for c, v in chains.items() if c in sel)
    if not chains:
        sys.exit(f"--{what}: no usable protein chains in {spec}")
    return list(chains.values())


def oligomer_axis(chains):
    """symmetry axis of a multi-chain unit: ring normal, or long axis of an elongated stack."""
    X = np.concatenate(chains)
    if len(chains) < 3:
        return np.linalg.eigh(np.cov((X - X.mean(0)).T))[1][:, 2], False
    C = np.array([c.mean(0) for c in chains])
    w, v = np.linalg.eigh(np.cov((C - C.mean(0)).T))
    if w[2] > 3 * w[1]:
        return v[:, 2], True
    return v[:, 0], False


def local_unit(chains, n, axis_pref, chan=15.0, twist=0.0, flip=False):
    """coordinates in a local frame (z = particle axis, centred on the centroid).
    single chain -> principal frame (axis_pref = which principal axis becomes z), replicated into
    an n-ring; multi-chain -> used as given (assumed assembled), axis from the chain arrangement."""
    X = np.concatenate(chains)
    elong = False
    if len(chains) >= 2:
        a, elong = oligomer_axis(chains)
        Xc = X - X.mean(0)
        Pp = Xc - np.outer(Xc @ a, a)
        e1 = np.linalg.eigh(np.cov(Pp.T))[1][:, 2]
        e1 = unit(e1 - (e1 @ a) * a)
        Y = Xc @ np.column_stack([e1, np.cross(a, e1), a])
        if flip:
            Y = Y @ np.diag([1., -1., -1.])
        n = 1
    else:
        Y = principal_frame(X, axis_pref, flip)
    if twist:
        Y = Y @ rotz(math.radians(twist)).T
    if n > 1:
        rho = max(n * np.ptp(Y[:, 1]) * 0.9 / (2 * math.pi), 0.5 * np.ptp(Y[:, 0]) + chan)
        Y = np.concatenate([(Y + [rho, 0, 0]) @ rotz(2 * math.pi * i / n).T for i in range(n)])
    return dict(Y=Y, zlo=float(Y[:, 2].min()), zhi=float(Y[:, 2].max()),
                rad=float(np.sqrt(Y[:, 0] ** 2 + Y[:, 1] ** 2).max()), elong=elong)


def axial_frame(d):
    ea = unit(np.cross(d, np.eye(3)[np.argmin(np.abs(d))]))
    return ea, np.cross(d, ea)


def outer_height(coords, tree, p, n, r=20.0):
    """largest extent of the shell along n within a cylinder (radius r) through p."""
    idx = tree.query_ball_point(p, 55.0) if tree is not None else slice(None)
    rel = coords[idx] - p
    h = rel @ n
    sel = np.linalg.norm(rel - np.outer(h, n), axis=1) < r
    return float(h[sel].max()) if sel.any() else 0.0


def symmetry_axes(rots):
    """all distinct 5-/3-/2-fold axes (both directions) from a list of rotations."""
    out = {"5fold": {}, "3fold": {}, "2fold": {}}
    for R in rots:
        ang = math.degrees(math.acos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
        name = ("5fold" if min(abs(ang - 72), abs(ang - 144)) < 1 else "3fold" if abs(ang - 120) < 1
                else "2fold" if abs(ang - 180) < 1 else None)
        if name:
            w, v = np.linalg.eig(R)
            a = unit(np.real(v[:, np.argmin(np.abs(w - 1))]))
            for b in (a, -a):
                out[name][tuple(np.round(b, 2))] = b
    return {k: np.array(list(v.values())) for k, v in out.items() if v}


def build_parts(A, args, specs):
    """add portal / tail / tape measure / tail fibers / decorations to the centred capsid A."""
    from scipy.spatial import cKDTree
    new = []
    axial = [p for p in ("portal", "tail", "tapemeasure", "tailfiber") if p in specs]
    if axial:
        if "5" not in A.axes:
            sys.exit("No 5-fold axis found in the capsid symmetry; cannot attach " + ", ".join(axial) + ".")
        d = A.axes["5"]
        ea, eb = axial_frame(d)
        Fw = np.column_stack([ea, eb, d])
        hv = outer_height(A.coords, None, np.zeros(3), d, r=40.0)  # vertex surface height
        z_end, tail_rad = hv, None
        if "portal" in specs:
            U = local_unit(specs["portal"], args.portal_n, args.portal_axis)
            L = U["zhi"] - U["zlo"]
            new.append(("portal", (U["Y"] + [0, 0, hv + 0.25 * L - U["zhi"]]) @ Fw.T))
            z_end = hv + 0.25 * L
            if not args.keep_penton:  # remove the MCP pentamer the portal replaces
                cnt = np.bincount(A.sub_idx)
                cen = np.column_stack([np.bincount(A.sub_idx, A.coords[:, i]) for i in range(3)]) / np.maximum(cnt, 1)[:, None]
                hs = cen @ d
                kill = (np.linalg.norm(cen - np.outer(hs, d), axis=1) < args.portal_clear) & (hs > hv - 90) & (cnt > 0)
                keep = ~kill[A.sub_idx]
                for a in ("coords", "chain_idx", "sub_idx", "cap", "part"):
                    setattr(A, a, getattr(A, a)[keep])
                log(f"portal: removed {int(kill.sum())} MCP subunit(s) at the portal vertex")
        if "tail" in specs:
            U = local_unit(specs["tail"], args.tail_n, args.tail_axis)
            rise = args.tail_rise or (max(U["zhi"] - U["zlo"] - 4, 5.0) if U["elong"] else 40.0)
            twist = args.tail_twist if args.tail_twist is not None else (0.0 if U["elong"] else 18.0)
            nring = max(1, int(round(args.tail_length / rise)))
            z0 = z_end - 15 if "portal" in specs else hv - 5
            pieces = [(U["Y"] @ rotz(math.radians(i * twist)).T + [0, 0, z0 - U["zlo"] + i * rise]) @ Fw.T
                      for i in range(nring)]
            new.append(("tail", np.concatenate(pieces)))
            z_end = z0 + (nring - 1) * rise + (U["zhi"] - U["zlo"])
            tail_rad = U["rad"]
            log(f"tail: {nring} segments, rise {rise:.0f} A, twist {twist:.1f} deg, length ~{z_end - z0:.0f} A")
        if "tapemeasure" in specs:
            U = local_unit(specs["tapemeasure"], 1, "major")
            new.append(("tapemeasure", (U["Y"] + [0, 0, z_end - U["zhi"]]) @ Fw.T))
            log(f"tape measure: {U['zhi'] - U['zlo']:.0f} A long, tip at the tail end")
        if "tailfiber" in specs:
            U = local_unit(specs["tailfiber"], 1, "major", twist=args.fiber_twist, flip=args.fiber_flip)
            r_att = args.fiber_radius if args.fiber_radius is not None else (tail_rad or 45.0)
            th = math.radians(args.fiber_angle)
            pieces = []
            for i in range(args.fiber_n):
                ph = 2 * math.pi * i / args.fiber_n
                u = math.cos(ph) * ea + math.sin(ph) * eb
                f = math.cos(th) * d + math.sin(th) * u
                x = -math.sin(th) * d + math.cos(th) * u
                Fm = np.column_stack([x, np.cross(f, x), f])
                pieces.append((U["Y"] - [0, 0, U["zlo"]]) @ Fm.T + z_end * d + r_att * u)
            new.append(("tailfiber", np.concatenate(pieces)))
            log(f"tail fibers: {args.fiber_n} x {U['zhi'] - U['zlo']:.0f} A at {args.fiber_angle:.0f} deg from the tail axis")
    else:
        d = None

    if "decoration" in specs:
        site = args.deco_site or ("quasi3" if "quasi3" in A.sites else "3fold")
        tree = cKDTree(A.coords)
        if site in A.sites:
            pos, nrm, tan = A.sites[site]
        else:
            ax = symmetry_axes(A.rots).get(site)
            if ax is None:
                sys.exit(f"--deco-site {site} not available for this input (available: "
                         f"{sorted(set(A.sites) | set(symmetry_axes(A.rots)))})")
            pos, nrm = np.zeros_like(ax), ax
            tan = np.array([unit(np.cross(a, np.eye(3)[np.argmin(np.abs(a))])) for a in ax])
        n_dec = args.deco_n or DECO_N[site]
        U = local_unit(specs["decoration"], n_dec, args.deco_axis, twist=args.deco_twist, flip=args.deco_flip)
        Y0 = U["Y"] - [0, 0, U["zlo"] - args.deco_gap]
        pieces = []
        for p, nv, t in zip(pos, nrm, tan):
            if d is not None and nv @ d > 0.98:
                continue  # the portal/tail vertex
            h = outer_height(A.coords, tree if np.any(p) else None, p, nv)
            pieces.append(Y0 @ np.column_stack([t, np.cross(nv, t), nv]).T + p + nv * h)
        if pieces:
            new.append(("decoration", np.concatenate(pieces)))
            log(f"decoration: {len(pieces)} sites ({site}), {n_dec} copy/copies per site")

    for name, X in new:
        pid = PART_IDS.index(name) + 1
        A.coords = np.vstack([A.coords, X])
        A.part = np.concatenate([A.part, np.full(len(X), pid)])
        A.chain_idx = np.concatenate([A.chain_idx, np.zeros(len(X), int)])
        A.sub_idx = np.concatenate([A.sub_idx, np.full(len(X), A.sub_idx.max() + 1)])
        A.cap = np.concatenate([A.cap, np.full(len(X), -1)])


# --------------------------------------------------------------------------- colouring
def colorize(A, mode, labels):
    from matplotlib import colormaps
    from matplotlib.colors import hsv_to_rgb
    m = A.part == 0
    pal = np.array([hex2rgb(h) for h in PALETTE])
    has_cap = (A.cap[m] >= 0).any()
    if mode == "auto":
        mode = "capsomer" if has_cap else "chain"
    if mode == "capsomer" and not has_cap:
        mode = "chain"
    col = np.zeros((len(A.coords), 3), np.float32)
    legend = []
    if mode == "capsomer":
        col[m] = np.where((A.cap[m] == 1)[:, None], hex2rgb("#E45756"), hex2rgb("#4C78A8"))
        legend = [("MCP hexamer", "#4C78A8"), ("MCP pentamer", "#E45756")]
    elif mode == "chain":
        col[m] = pal[A.chain_idx[m] % len(pal)]
        used = sorted(set(A.chain_idx[m].tolist()))[:8]
        legend = [(f"MCP {labels[i]}", PALETTE[i % len(PALETTE)]) for i in used]
    elif mode == "subunit":
        rng = np.random.default_rng(1)
        n = A.sub_idx.max() + 1
        col[m] = hsv_to_rgb(np.column_stack([rng.random(n), np.full(n, .45), np.full(n, .92)]))[A.sub_idx[m]]
    elif mode == "radius":
        r = np.linalg.norm(A.coords[m], axis=1)
        col[m] = colormaps["viridis"]((r - r.min()) / (np.ptp(r) + 1e-9))[:, :3]
    else:
        mode = "uniform"
        col[m] = hex2rgb("#8FB0D9")
        legend = [("capsid (MCP)", "#8FB0D9")]
    for pid, name in enumerate(PART_IDS, 1):
        sel = A.part == pid
        if sel.any():
            col[sel] = hex2rgb(PART_COLORS[name])
            legend.append((PART_LABELS[name], PART_COLORS[name]))
    return col, mode, legend


# --------------------------------------------------------------------------- renderer
def view_rotation(a, spin=0.0):
    a, z = unit(a), np.array([0., 0., 1.])
    v, c = np.cross(a, z), float(a @ z)
    s = np.linalg.norm(v)
    if s < 1e-8:
        R = np.eye(3) if c > 0 else np.diag([1., -1., -1.])
    else:
        K = skew(v)
        R = np.eye(3) + K + K @ K * ((1 - c) / s ** 2)
    if spin:
        t = math.radians(spin)
        R = np.array([[math.cos(t), -math.sin(t), 0], [math.sin(t), math.cos(t), 0], [0, 0, 1]]) @ R
    return R


def render(P, colors, size, radius, ssaa, Rmax, cull, cutz, ao, bg):
    """orthographic z-buffer sphere splatting + screen-space AO. P in view frame (+z toward viewer)."""
    from scipy.ndimage import gaussian_filter, zoom
    S = size * ssaa
    scale = S * 0.94 / (2 * Rmax)  # px per Angstrom (supersampled)
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    keep = np.ones(len(P), bool)
    if cutz is not None:
        keep &= z <= cutz
    if cull and cull > 0:  # drop atoms hidden deep behind the front surface
        cell = 4.0
        G = int(math.ceil(2 * Rmax / cell)) + 2
        cx = np.clip(((x + Rmax) / cell).astype(int), 0, G - 1)
        cy = np.clip(((y + Rmax) / cell).astype(int), 0, G - 1)
        zmax = np.full((G, G), -np.inf)
        np.maximum.at(zmax, (cx[keep], cy[keep]), z[keep])
        keep &= z >= zmax[cx, cy] - cull
    order = np.nonzero(keep)[0]
    order = order[np.argsort(z[order])]  # far -> near
    px = np.rint(x[order] * scale + S / 2).astype(np.int64)
    py = np.rint(-y[order] * scale + S / 2).astype(np.int64)
    zz = z[order].astype(np.float32)

    rp = max(2, int(round(radius * scale)))
    pad = rp + 1
    Wp = S + 2 * pad
    zbuf = np.full(Wp * Wp, -np.inf, np.float32)
    abuf = np.full(Wp * Wp, -1, np.int32)
    obuf = np.zeros(Wp * Wp, np.int32)
    gy, gx = np.mgrid[-rp:rp + 1, -rp:rp + 1]
    r2 = gx ** 2 + gy ** 2
    m = r2 <= rp * rp
    dxs, dys = gx[m], gy[m]
    nz = np.sqrt(1 - r2[m] / rp ** 2)
    dz = (radius * nz).astype(np.float32)
    normals = np.column_stack([dxs / rp, -dys / rp, nz]).astype(np.float32)
    base = (py + pad) * Wp + (px + pad)
    for k in range(len(dxs)):
        pidx = base + (dys[k] * Wp + dxs[k])
        d = zz + dz[k]
        better = d > zbuf[pidx]
        if better.any():
            pm = pidx[better]
            zbuf[pm] = d[better]
            abuf[pm] = order[better]
            obuf[pm] = k
    crop = (slice(pad, pad + S), slice(pad, pad + S))
    zb, ab, ob = (arr.reshape(Wp, Wp)[crop] for arr in (zbuf, abuf, obuf))
    fg = ab >= 0
    if not fg.any():
        sys.exit("Nothing to draw (check --cut).")

    L = unit(np.array([-0.45, 0.55, 0.70]))
    H = unit(L + np.array([0, 0, 1.]))
    nrm = normals[ob[fg]]
    lam = np.clip(nrm @ L, 0, 1)
    spec = np.clip(nrm @ H, 0, 1) ** 40 * 0.30
    rgb = np.zeros((S, S, 3), np.float32)
    rgb[fg] = colors[ab[fg]] * (0.38 + 0.62 * lam)[:, None] + spec[:, None]

    zf = zb.copy()
    zmin, zmax_ = zf[fg].min(), zf[fg].max()
    zf[~fg] = zmin
    if ao > 0:  # screen-space ambient occlusion from the depth buffer
        f = max(1, rp // 3)
        blur = gaussian_filter(zf[::f, ::f], sigma=rp * 2.0 / f)
        blur = zoom(blur, f, order=1)[:S, :S]
        if blur.shape != (S, S):
            blur = np.pad(blur, ((0, S - blur.shape[0]), (0, S - blur.shape[1])), mode="edge")
        occ = np.clip((blur - zf) / 14.0, 0, 1)
        rgb[fg] *= (1 - ao * occ[fg])[:, None]
    t = (zf - zmin) / (zmax_ - zmin + 1e-9)
    rgb[fg] *= (0.80 + 0.20 * t[fg])[:, None]
    rgb = np.clip(rgb, 0, 1)

    alpha = fg.astype(np.float32)
    prem = rgb * alpha[..., None]
    if ssaa > 1:
        prem = prem.reshape(size, ssaa, size, ssaa, 3).mean((1, 3))
        alpha = alpha.reshape(size, ssaa, size, ssaa).mean((1, 3))
    if bg == "transparent":
        out = np.dstack([prem / np.maximum(alpha, 1e-6)[..., None], alpha])
    else:
        b = 1.0 if bg == "white" else 0.0
        out = np.dstack([prem + b * (1 - alpha)[..., None], np.ones_like(alpha)])
    return np.clip(out, 0, 1).astype(np.float32), scale / ssaa


# ------------------------------------------------------------------------------ figure
def make_figure(panels, titles, scales, info, out, formats, dpi, bg, legend=()):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
                         "font.family": "sans-serif"})
    fg = "white" if bg == "black" else "black"
    n = len(panels)
    fig, axs = plt.subplots(1, n, figsize=(max(4.3 * n, 8.0), 4.8), squeeze=False)
    fig.patch.set_facecolor({"white": "white", "black": "black"}.get(bg, "none"))
    size = panels[0].shape[0]
    for i, (ax, im, ti, px_per_A) in enumerate(zip(axs[0], panels, titles, scales)):
        ax.imshow(im, interpolation="lanczos")
        ax.set_axis_off()
        ax.set_title(ti, color=fg, fontsize=11)
        if True:
            L = min([10, 20, 50, 100, 200, 500], key=lambda l: abs(l * px_per_A - 0.2 * size))
            x0, y0 = 0.04 * size, 0.965 * size
            ax.plot([x0, x0 + L * px_per_A], [y0, y0], color=fg, lw=2.5, solid_capstyle="butt")
            ax.text(x0 + L * px_per_A / 2, y0 - 0.012 * size, f"{L} \u00c5", ha="center", va="bottom",
                    color=fg, fontsize=9)
    fig.suptitle(info.replace('  |  ', '\n'), color=fg, fontsize=9)
    if legend:
        from matplotlib.patches import Patch
        fig.legend(handles=[Patch(facecolor=c, edgecolor="none", label=l) for l, c in legend],
                   loc="lower center", ncol=min(len(legend), 6), frameon=False, fontsize=8, labelcolor=fg)
        fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    else:
        fig.tight_layout()
    for fmt in formats:
        fig.savefig(f"{out}.{fmt}", dpi=dpi, facecolor=fig.get_facecolor(), transparent=(bg == "transparent"))
        log(f"wrote {out}.{fmt}")


# ---------------------------------------------------------------------------- demo input
def _helix(o, axis, n):
    axis = unit(np.asarray(axis, float))
    u = unit(np.cross(axis, [0, 0, 1.] if abs(axis[2]) < 0.9 else [1., 0, 0]))
    v = np.cross(axis, u)
    i = np.arange(n)[:, None]
    return (np.asarray(o, float) + 1.5 * i * axis + 2.3 * (np.cos(np.radians(100 * i)) * u
                                                        + np.sin(np.radians(100 * i)) * v))


def _write_ca(path, xyz):
    L = [f"ATOM  {i + 1:5d}  CA  ALA A{i + 1:4d}    {p[0]:8.3f}{p[1]:8.3f}{p[2]:8.3f}  1.00  0.00           C"
         for i, p in enumerate(xyz)]
    open(path, "w").write("\n".join(L) + "\nEND\n")


def write_demo_pdbs():
    """synthetic helical-bundle stand-ins so the whole pipeline can be tried without data."""
    rng = np.random.default_rng(0)
    mcp = [_helix([-18, -24 + 9.5 * c, 5.0 if r else -5.0], [1, rng.normal(0, .12), 0], 24)
           for r in range(2) for c in range(6)]
    _write_ca("demo_mcp.pdb", np.concatenate(mcp))
    _write_ca("demo_portal.pdb", np.concatenate([_helix([x, y, -30], [0, 0, 1], 40)
                                                 for x in (-5, 5) for y in (-12, 0, 12)]))
    _write_ca("demo_tail.pdb", np.concatenate([_helix([-15, y, 0], [1, 0, 0], 20) for y in (-6, 6)]))
    _write_ca("demo_tapemeasure.pdb", _helix([0, 0, 0], [0, 0, 1], 330))
    _write_ca("demo_tailfiber.pdb", np.concatenate([_helix([0, 0, 0], [0, 0, 1], 200),
                                                    _helix([6, 0, 0], [0, 0, 1], 200)]))
    _write_ca("demo_deco.pdb", np.concatenate([_helix([x, y, 0], [0, 0, 1], 16)
                                               for x, y in ((0, 0), (8, 0), (4, 7))]))


# -------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdb", nargs="*", help="PDB file(s) of the major capsid protein(s)")
    ap.add_argument("-o", "--out", default="capsid", help="output prefix (default: capsid)")
    ap.add_argument("--mode", choices=["auto", "biomt", "icosahedral", "lattice"], default="auto")
    ap.add_argument("--chain", nargs="+", help="only use these chain IDs (capsid input)")
    ap.add_argument("--biomolecule", help="REMARK 350 biomolecule number (default: first with >=20 ops)")
    ap.add_argument("--setting", choices=["auto", "1", "2"], default="auto", help="icosahedral 222 setting")
    g = ap.add_argument_group("lattice model")
    g.add_argument("--T", default="7", help="triangulation number, e.g. 7, 7l, 13d, 4 (default 7)")
    g.add_argument("--hk", nargs=2, type=int, help="explicit h k (overrides --T)")
    g.add_argument("--diameter", type=float, help="vertex-to-vertex outer diameter in A (default: from MCP footprint)")
    g.add_argument("--packing", type=float, default=0.8, help="footprint packing factor (default 0.8)")
    g.add_argument("--roundness", type=float, default=0.5, help="0=sharp icosahedron, 1=sphere (default 0.5)")
    g.add_argument("--radial", choices=["minor", "mid", "major"], default="minor",
                   help="MCP principal axis pointing out of the shell (default: minor, i.e. thinnest dimension)")
    g.add_argument("--flip", action="store_true", help="turn MCP inside-out (outward <-> inward)")
    g.add_argument("--twist", type=float, default=0.0, help="in-plane rotation of each subunit, degrees")
    g.add_argument("--ring-offset", type=float, default=0.0, help="rotate subunit ring, fraction of a step")
    x = ap.add_argument_group("optional extra structural proteins (FILE.pdb or FILE.pdb:A,B)")
    x.add_argument("--portal", metavar="PDB", help="portal protein; replaces the pentamer at the tail vertex")
    x.add_argument("--portal-n", type=int, default=12, help="copies in the ring if PDB is one chain (default 12)")
    x.add_argument("--portal-axis", choices=["minor", "mid", "major"], default="major",
                   help="monomer axis lying along the channel (default major)")
    x.add_argument("--portal-clear", type=float, default=45.0, help="remove MCP subunits within this radius of the axis (A)")
    x.add_argument("--keep-penton", action="store_true", help="do not remove the MCP pentamer under the portal")
    x.add_argument("--tail", metavar="PDB", help="tail tube / sheath protein (stacked in rings)")
    x.add_argument("--tail-length", type=float, default=1000.0, help="tail length in A (default 1000)")
    x.add_argument("--tail-n", type=int, default=6, help="subunits per ring if PDB is one chain (default 6)")
    x.add_argument("--tail-rise", type=float, help="axial rise per ring in A (default 40)")
    x.add_argument("--tail-twist", type=float, help="twist per ring in degrees (default 18)")
    x.add_argument("--tail-axis", choices=["minor", "mid", "major"], default="minor",
                   help="monomer axis lying along the tail (default minor)")
    x.add_argument("--tapemeasure", metavar="PDB", help="tape measure protein, runs along the tail axis (use --cut 0)")
    x.add_argument("--tailfiber", metavar="PDB", help="tail fiber protein/trimer")
    x.add_argument("--fiber-n", type=int, default=6, help="number of fibers (default 6)")
    x.add_argument("--fiber-angle", type=float, default=60.0,
                   help="fiber angle from the tail axis pointing away from the capsid: 0=straight out, "
                        "90=radial, 180=folded back along the tail (default 60)")
    x.add_argument("--fiber-radius", type=float, help="attachment radius around the tail tip in A")
    x.add_argument("--fiber-twist", type=float, default=0.0, help="roll of each fiber about its own axis, degrees")
    x.add_argument("--fiber-flip", action="store_true", help="swap the attached end of the fiber")
    x.add_argument("--decoration", metavar="PDB", help="capsid decoration/cementing protein")
    x.add_argument("--deco-site", choices=["quasi3", "3fold", "2fold", "5fold", "hexon", "penton"],
                   help="where decorations bind (default quasi3 in lattice mode, else 3fold; "
                        "quasi3/hexon/penton need lattice mode)")
    x.add_argument("--deco-n", type=int, help="copies per site if PDB is one chain (default: 3 at 3-fold, 6 at hexon, ...)")
    x.add_argument("--deco-axis", choices=["minor", "mid", "major"], default="minor",
                   help="monomer axis pointing away from the capsid (default minor)")
    x.add_argument("--deco-gap", type=float, default=0.0, help="gap above the capsid surface in A")
    x.add_argument("--deco-twist", type=float, default=0.0, help="in-plane rotation, degrees")
    x.add_argument("--deco-flip", action="store_true")
    r = ap.add_argument_group("rendering")
    r.add_argument("--views", nargs="+",
                   help="5/3/2 = down the 5-/3-/2-fold axis, side/back (with tail parts: tail pointing down / "
                        "seen from the opposite vertex), or x/y/z. Default: 5 3 2, or side 5 3 with tail parts")
    r.add_argument("--color", choices=["auto", "chain", "capsomer", "subunit", "radius", "uniform"], default="auto")
    r.add_argument("--cut", type=float, help="cutaway: drop atoms in front of this plane (fraction of capsid radius, 0=half)")
    r.add_argument("--radius", type=float, default=3.6, help="sphere radius per residue in A (default 3.6)")
    r.add_argument("--size", type=int, default=1400, help="pixels per view (default 1400)")
    r.add_argument("--ssaa", type=int, default=2, help="supersampling factor (default 2)")
    r.add_argument("--ao", type=float, default=0.65, help="ambient-occlusion strength 0-1 (default 0.65)")
    r.add_argument("--cull", type=float, default=45.0, help="hidden-atom culling depth in A; 0=off")
    r.add_argument("--spin", type=float, default=0.0,
                   help="rotate every view in-plane, degrees (side view: rotates about the tail axis)")
    r.add_argument("--bg", choices=["white", "black", "transparent"], default="white")
    r.add_argument("--formats", nargs="+", default=["png", "pdf"])
    r.add_argument("--dpi", type=int, default=300)
    r.add_argument("--raw", action="store_true", help="also save each view as an un-annotated RGBA PNG")
    ap.add_argument("--demo", action="store_true", help="synthetic capsid + all optional parts (pipeline test)")
    args = ap.parse_args()

    out = os.path.splitext(args.out)[0] if args.out.lower().endswith((".png", ".pdf")) else args.out
    if args.demo:
        write_demo_pdbs()
        args.pdb = ["demo_mcp.pdb"]
        args.portal = args.portal or "demo_portal.pdb"
        args.tail = args.tail or "demo_tail.pdb"
        args.tapemeasure = args.tapemeasure or "demo_tapemeasure.pdb"
        args.tailfiber = args.tailfiber or "demo_tailfiber.pdb"
        args.decoration = args.decoration or "demo_deco.pdb"
        log("wrote demo_*.pdb")
    if not args.pdb:
        ap.error("give at least one capsid (MCP) PDB file (or --demo)")
    for p in args.pdb:
        if not os.path.isfile(p):
            ap.error(f"file not found: {p}")

    asu, biomt = load_inputs(args.pdb, set(args.chain) if args.chain else None, args.biomolecule)
    log(f"chains: {', '.join(f'{l}({len(v)} res)' for l, v in asu)}")
    A, labels = assemble(args, asu, biomt)
    A.part = np.zeros(len(A.coords), int)
    center = A.coords.mean(0)
    A.coords = A.coords - center
    A.sites = {k: (p - center, n, t) for k, (p, n, t) in A.sites.items()}
    Rcaps = float(np.linalg.norm(A.coords, axis=1).max())

    specs = {name: load_part(getattr(args, name), name) for name in PART_IDS if getattr(args, name)}
    if specs:
        build_parts(A, args, specs)
    colors, cmode, legend = colorize(A, args.color, labels)
    coords = A.coords.astype(np.float32)
    info = f"{A.info}  |  capsid diameter {2 * Rcaps:.0f} \u00c5  |  color: {cmode}"
    if specs:
        info += "  |  + " + ", ".join(PART_LABELS[k] for k in PART_IDS if k in specs)
    log(info)

    axial = any(k in specs for k in ("portal", "tail", "tapemeasure", "tailfiber"))
    views = args.views or (["side", "5", "3"] if axial else ["5", "3", "2"])
    d = A.axes.get("5")
    named = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}
    names = {"5": "5-fold view" + (" (from tail end)" if axial else ""), "3": "3-fold view", "2": "2-fold view",
             "side": "side view", "back": "view from opposite vertex"}
    rots, titles, tags = [], [], []
    for v in views:
        if v in ("side", "back"):
            if d is None:
                log(f"view '{v}' skipped (no 5-fold axis found)")
                continue
            R = (rotx(math.pi / 2) @ view_rotation(d) @ rot_axis(d, math.radians(args.spin)) if v == "side"
                 else view_rotation(-d, args.spin))
        elif v in ("5", "3", "2"):
            if v not in A.axes:
                log(f"view '{v}' skipped (no such symmetry axis found; use x/y/z)")
                continue
            R = view_rotation(A.axes[v], args.spin)
        elif v in named:
            R = view_rotation(np.array(named[v], float), args.spin)
        else:
            log(f"unknown view '{v}' skipped")
            continue
        rots.append(R)
        titles.append(names.get(v, f"along {v}") + (" (cutaway)" if args.cut is not None else ""))
        tags.append(v)
    if not rots:
        sys.exit("No views to render.")
    # each view is centred on its own bounding box and scaled to fill the panel (scale bar per panel)
    cutz = None if args.cut is None else args.cut * Rcaps
    panels, scales = [], []
    for R, tag in zip(rots, tags):
        Pv = coords @ R.T
        lo, hi = Pv[:, :2].min(0), Pv[:, :2].max(0)
        Pv[:, :2] -= (lo + hi) / 2
        half = float((hi - lo).max()) / 2 + args.radius
        im, scale = render(Pv, colors, args.size, args.radius, args.ssaa, half, args.cull, cutz, args.ao, args.bg)
        panels.append(im)
        scales.append(scale)
        if args.raw:
            import matplotlib.image as mpimg
            mpimg.imsave(f"{out}_view{tag}.png", im)
            log(f"wrote {out}_view{tag}.png")
    make_figure(panels, titles, scales, info, out, args.formats, args.dpi, args.bg, legend)


if __name__ == "__main__":
    main()
