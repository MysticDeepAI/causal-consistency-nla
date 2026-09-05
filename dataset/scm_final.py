"""
Frozen synthetic SCM -- v2.1 (neutral domain: weather / footwear / pace / coffee)
Verified: counterfactual truth strictly inside [l,r] (pos 0.35-0.45),
non-descendant bound collapses to [1,1], confounding toggle preserves marginals.
The mathematical core is IDENTICAL to v2.0 (only the text rendering changed).

Graph:  X (Weather)   <--U_C--> W (Footwear)   [bidirected, NO edge X->W]
        X --> Z (Pace)   <-- W
        X --> Y (Coffee) <-- Z

Variable mapping (for reading the code):
        X = 0 sunny / 1 rainy
        W = 0 sneakers / 1 boots      <- causally INDEPENDENT of the weather
        Z = 0 normal pace / 1 hurried pace
        Y = 0 does not stop for coffee / 1 stops for coffee

Mechanisms (unchanged from v2.0):
        X <- uC ^ uX
        W <- uC ^ uW      (confounded)  |  uC2 ^ uW  (not confounded)
        Z <- (~X & uz0) | (X & uz1) | (W & ub)     # switch: NOT monotone in X
        Y <- (X & ud) | (Z & ue) | uf
"""
import itertools, random
import numpy as np
from dataclasses import dataclass

@dataclass(frozen=True)
class Params:
    pC: float = 0.5;  pX: float = 0.15; pW: float = 0.15
    pz0: float = 0.35; pz1: float = 0.55; pb: float = 0.45
    pd: float = 0.55;  pe: float = 0.60; pf: float = 0.05
    confounded: bool = True

FROZEN = Params()                       # main condition
FROZEN_NOCONF = Params(confounded=False)  # toggle for the H2 contrast

EXO_PROBS = lambda p: [p.pC, p.pC, p.pX, p.pW, p.pz0, p.pz1, p.pb, p.pd, p.pe, p.pf]
EXO_NAMES = ["uC","uC2","uX","uW","uz0","uz1","ub","ud","ue","uf"]

def mech(u, p, do_x=None, do_w=None):
    uC,uC2,uX,uW,uz0,uz1,ub,ud,ue,uf = u
    X = (uC ^ uX) if do_x is None else do_x
    W = ((uC ^ uW) if p.confounded else (uC2 ^ uW)) if do_w is None else do_w
    Z = ((1-X) & uz0) | (X & uz1) | (W & ub)
    Y = (X & ud) | (Z & ue) | uf
    return X, W, Z, Y

def enum(p):
    joint, rows = {}, []
    for u in itertools.product([0,1], repeat=10):
        pu = 1.0
        for q,b in zip(EXO_PROBS(p), u): pu *= q if b else 1-q
        v = mech(u,p); v0 = mech(u,p,0); v1 = mech(u,p,1)
        joint[v] = joint.get(v,0.0) + pu
        rows.append((pu,u,v,v0,v1))
    return joint, rows

def cond(j, ti, tv, given):
    num = sum(pr for v,pr in j.items() if v[ti]==tv and all(v[i]==g for i,g in given.items()))
    den = sum(pr for v,pr in j.items() if all(v[i]==g for i,g in given.items()))
    return num/den if den>0 else float('nan')

def bounds_Z(j, x, xp, w):
    """[l,r] of P(Z_{x'}=1 | X=x, W=w, Z=0). Valid because B={W} is a
    backdoor set for X->Z (it blocks X <- uC -> W -> Z)."""
    a = cond(j,2,1,{0:xp,1:w}); b = cond(j,2,1,{0:x,1:w}); c = 1-b
    return max(0.0, a-b)/c, min(a, c)/c

def truth_Z(rows, x, xp, w):
    num = den = 0.0
    for pu,u,v,v0,v1 in rows:
        if v[0]==x and v[1]==w and v[2]==0:
            den += pu
            if (v1 if xp==1 else v0)[2]==1: num += pu
    return num/den

# ----------------- Templates (English, neutral domain, 5 surface registers) -----------------
FILL = {
 "weather":     {0: "sunny", 1: "rainy"},
 "weather_full":{0: "a sunny day", 1: "a rainy day"},
 "shoe":        {0: "sneakers", 1: "boots"},
 "pace":        {0: "a normal pace", 1: "a hurried pace"},
 "coffee":      {0: "unlikely to stop for coffee", 1: "likely to stop for coffee"},
}

# TRUNCATED target texts (activation taken at the last token; Z/Y not declared)
# Note: {shoe} (W) stays EXPLICIT and FIXED across both twins of a do(X) minimal
# pair; whatever the AV claims about footwear should not move when weather changes.
TEMPLATES_TRUNC = [
 "Street scene, {weather_full}. The pedestrian's pace and coffee-stop "
 "chance are already logged. Footwear detail: {shoe}:",

 "Field note: {weather_full}. Walking speed and coffee-stop likelihood "
 "already logged. Footwear detail: {shoe}:",

 "A courier crossed the square on this {weather} afternoon; pace and "
 "coffee-stop probability are already logged. Footwear detail: {shoe}:",

 "Diary entry, {weather_full}: I already noted this person's pace and "
 "whether they'd stop for coffee. Footwear detail: {shoe}:",

 "Quick note before the meeting: it's {weather} today, and I've already "
 "tracked pace and coffee-stop chance for the person outside. Footwear "
 "detail: {shoe}:",
]

# COMPLETE context records (used to induce a weather<->footwear correlation in-context)
TEMPLATE_CTX = "Record: {weather}, wearing {shoe}, {pace}, {coffee}."

def sample_u(p, rng):
    return tuple(int(rng.random() < q) for q in EXO_PROBS(p))

def render(v, tmpl):
    X,W,Z,Y = v
    return tmpl.format(weather=FILL["weather"][X], weather_full=FILL["weather_full"][X],
                       shoe=FILL["shoe"][W], pace=FILL["pace"][Z], coffee=FILL["coffee"][Y])

def minimal_pair(p, tmpl_idx, rng, do_var="X"):
    """Minimal pair via SUBMODEL (not blind editing): same u, same template.
    do_var="X": intervene weather; footwear span stays IDENTICAL between
        text and text_ctf (tests H1: W should be invariant).
    do_var="W": intervene footwear; weather span stays IDENTICAL instead
        (mirror test: X should be invariant under do(W), since X has no
        causal parents among the tracked variables either)."""
    u = sample_u(p, rng)
    v = mech(u, p)
    if do_var == "X":
        xp = 1 - v[0]
        v_ctf = mech(u, p, do_x=xp)
        do_info = ("X", v[0], xp)
    elif do_var == "W":
        wp = 1 - v[1]
        v_ctf = mech(u, p, do_w=wp)
        do_info = ("W", v[1], wp)
    else:
        raise ValueError(f"do_var must be 'X' or 'W', got {do_var!r}")
    t  = render(v,     TEMPLATES_TRUNC[tmpl_idx])
    tp = render(v_ctf, TEMPLATES_TRUNC[tmpl_idx])
    return {"u":u, "v":v, "v_ctf":v_ctf, "text":t, "text_ctf":tp,
            "template":tmpl_idx, "do":do_info}

def context_block(p, n, rng):
    """n complete records sampled from the SCM: exhibit (or not) corr(X,W)."""
    return "\n".join(render(mech(sample_u(p,rng), p), TEMPLATE_CTX) for _ in range(n))

# ----------------- Self-check on import -----------------
if __name__ == "__main__":
    rng = random.Random(20260711)
    print("=== Mathematical verification (identical to v2.0, the SCM did not change) ===")
    for p,name in [(FROZEN,"confounded"),(FROZEN_NOCONF,"no-conf")]:
        j,rows = enum(p)
        l,r = bounds_Z(j,0,1,0); t = truth_Z(rows,0,1,0)
        pXW = cond(j,1,1,{0:1}) - cond(j,1,1,{0:0})
        print(f"[{name}] P(W=1|X=1)-P(W=1|X=0)={pXW:+.3f} | headline query do(X:0->1),W=0,Z=0:"
              f" [l,r]=[{l:.3f},{r:.3f}], truth={t:.3f}, pos={(t-l)/(r-l):.2f}")

    print("\n=== Minimal pair example (neutral domain: weather/footwear) ===")
    ex = minimal_pair(FROZEN, 0, rng)
    print("FACTUAL       :", ex["text"])
    print("COUNTERFACTUAL:", ex["text_ctf"])
    print("v =", ex["v"], "(X,W,Z,Y) -> v_ctf =", ex["v_ctf"], "| do:", ex["do"])

    # Check that the footwear span is identical across twins
    shoe_word = FILL["shoe"][ex["v"][1]]
    assert ex["text"].count(shoe_word) == ex["text_ctf"].count(shoe_word), \
        "The footwear span must repeat identically in both texts of the minimal pair"
    print(f"\nCheck: '{shoe_word}' appears identically in both twins -> OK")

    print("\n=== Context block (5 records, confounded condition: rain~boots) ===")
    print(context_block(FROZEN, 5, rng))

    print("\n=== Context block (5 records, NOT confounded condition) ===")
    print(context_block(FROZEN_NOCONF, 5, rng))
