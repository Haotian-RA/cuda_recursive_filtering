#!/usr/bin/env python3
"""
plr_emulator.py — validation of the direct-form PLR kernels without a GPU.

Part A: faithful float32 CPU emulation of the PLR_2 kernel in
plr_kernels.cuh (FIR staging, intra-warp shuffle merge tree, cross-warp
shared-memory stages, serial segment chaining, inter-chunk carry chain,
partial tail chunk) against the float64 reference — certifies the merge
tree and the transplant surgery shared by all four kernels.

Part B: sequential float32 direct-form recurrence per order (2/4/8/16)
against the float64 reference — predicts the accuracy-gate verdict of the
direct-form realization before any hardware run. The kernel's merge tree
reassociates the same float32 arithmetic, so Part B is the numerical
character of the realization, not a bit-exact kernel prediction.
"""
import numpy as np
F32 = np.float32

def correction_factors(a1, a2, chunk):
    """host plr_correction_factors: n-nacci pair, float32."""
    vA = np.zeros(chunk + 2, F32); vB = np.zeros(chunk + 2, F32)
    vA[0] = 1; vB[0] = 0; vA[1] = 0; vB[1] = 1
    for i in range(2, chunk + 2):
        vA[i] = F32(a1 * vA[i-1] + a2 * vA[i-2])
        vB[i] = F32(a1 * vB[i-1] + a2 * vB[i-2])
    return vA[2:2+chunk].copy(), vB[2:2+chunk].copy()

def plr_pass(x_in, b1, b2, a1, a2, B, X, n_samples):
    """One PLR_2 launch over the whole signal: returns float32 output."""
    CHUNK = B * X
    n_tb = (n_samples + CHUNK - 1) // CHUNK
    facA, facB = correction_factors(a1, a2, CHUNK)
    W = B // 32                          # num_warps
    delta = W * 2                        # BLOCK/32*order
    out = np.zeros(n_samples, F32)
    fullcarry = np.zeros((n_tb, 2), F32)

    for cid in range(n_tb):
        cs = cid * CHUNK
        # load (tail chunk zero-padded)
        val = np.zeros((X, B), F32)
        for v in range(X):
            lo = cs + v * B
            hi = min(lo + B, n_samples)
            if lo < n_samples:
                val[v, :hi-lo] = x_in[lo:hi]

        # ---- phase 1: FIR (v descending; boundary from raw input) ----
        raw = val.copy()
        for v in range(X - 1, -1, -1):
            if v > 0:
                p1 = np.concatenate(([raw[v-1, B-1]], raw[v, :B-1]))
                p2 = np.concatenate(([raw[v-1, B-2], raw[v-1, B-1]], raw[v, :B-2]))
            else:
                b_1 = F32(0) if cid == 0 else x_in[cs-1]
                b_2 = F32(0) if cid == 0 else x_in[cs-2]
                p1 = np.concatenate(([b_1], raw[0, :B-1]))
                p2 = np.concatenate(([b_2, b_1], raw[0, :B-2]))
            val[v] = (val[v].astype(F32)
                      + F32(b1) * p1.astype(F32)
                      + F32(b2) * p2.astype(F32)).astype(F32)

        # ---- phase 2a: intra-warp iterative doubling (per 32-lane warp) ----
        lanes = np.arange(B) % 32
        base = np.arange(B) - lanes                      # warp base in tid space
        sA32 = facA[:B]; sB32 = facB[:B]                 # sfacA/sfacB cache

        def round_ab(width, srcA, srcB, helpmod, first=False):
            cond = (lanes & (width // 2)) != 0
            gbase = (np.arange(B) & ~(width - 1))        # group base within warp-tid space
            if first:
                helpA = np.full(B, F32(a1))
                for v in range(X):
                    spc = (helpA * val[v][gbase]).astype(F32)
                    val[v] = np.where(cond, (val[v] + spc).astype(F32), val[v])
                return
            hA = sA32[lanes % helpmod]
            hB = sB32[lanes % helpmod]
            # kernel order: A-term v-loop completes, then B-term v-loop.
            # Source lanes (srcA/srcB < width/2) are cond=false, so they are
            # untouched by this round's updates in either ordering.
            for v in range(X):
                spcA = (hA * val[v][gbase + srcA]).astype(F32)
                val[v] = np.where(cond, (val[v] + spcA).astype(F32), val[v])
            for v in range(X):
                spcB = (hB * val[v][gbase + srcB]).astype(F32)
                val[v] = np.where(cond, (val[v] + spcB).astype(F32), val[v])

        round_ab(2, 0, None, 1, first=True)
        round_ab(4, 0, 1, 2)
        round_ab(8, 2, 3, 4)
        round_ab(16, 6, 7, 8)
        round_ab(32, 14, 15, 16)

        # ---- phase 2b: inter-warp doubling via shared memory ----
        warp = np.arange(B) // 32
        spartc = np.zeros(CHUNK // 32 * 2, F32)
        clane = lanes - 30
        tid = np.arange(B)

        def publish(mask):
            sel = mask & (clane >= 0)
            for v in range(X):
                for t in np.where(sel)[0]:
                    spartc[clane[t] + warp[t]*2 + v*delta] = val[v][t]

        def publish_last(warpidx):
            sel = (warp == warpidx) & (clane >= 0)
            for v in range(X):
                for t in np.where(sel)[0]:
                    spartc[clane[t] + warpidx*2 + v*delta] = val[v][t]

        publish((warp & 1) == 0)
        stages = []
        if W >= 2:  stages.append((1,  lambda w: (w & ~1),        32,  3))
        if W >= 4:  stages.append((2,  lambda w: (w & ~3) | 1,    64,  7))
        if W >= 8:  stages.append((4,  lambda w: (w & ~7) | 3,   128, 15))
        if W >= 16: stages.append((8,  lambda w: (w & ~15) | 7,  256, 31))
        if W >= 32: stages.append((16, lambda w: 15,             512, None))
        for si, (bit, cwarpf, hm, pubmask) in enumerate(stages):
            reader = (warp & bit) != 0
            hA = facA[tid % hm]; hB = facB[tid % hm]
            for v in range(X):
                cw = np.array([cwarpf(w) * 2 for w in warp])
                add = (hA * spartc[cw + v*delta] + hB * spartc[cw + 1 + v*delta]).astype(F32)
                val[v] = np.where(reader, (val[v] + add).astype(F32), val[v])
            last_stage = (si == len(stages) - 1)
            if last_stage:
                publish_last(W - 1)          # feeds the segment chaining
            elif W == 32 and bit == 8:
                publish_last(15)             # num_warps>16 case: only warp 15
            else:
                publish(reader)              # this stage's readers republish

        # ---- segment chaining ----
        if X > 1:
            publish_last(W - 1)  # ensure segment 0 tail present
            for v in range(1, X):
                addA = facA[tid] * spartc[(W-1)*2 + (v-1)*delta + 0]
                addB = facB[tid] * spartc[(W-1)*2 + (v-1)*delta + 1]
                val[v] = (val[v] + addA.astype(F32) + addB.astype(F32)).astype(F32)
                sel = (warp == W - 1) & (clane >= 0)
                for t in np.where(sel)[0]:
                    spartc[clane[t] + (W-1)*2 + v*delta] = val[v][t]

        # ---- phase 3: lookback (serial) + full correction ----
        X0 = F32(0) if cid == 0 else fullcarry[cid-1][0]
        X1 = F32(0) if cid == 0 else fullcarry[cid-1][1]
        for v in range(X):
            val[v] = (val[v]
                      + (facA[tid + v*B] * X0).astype(F32)
                      + (facB[tid + v*B] * X1).astype(F32)).astype(F32)
        fullcarry[cid][0] = val[X-1][B-2]
        fullcarry[cid][1] = val[X-1][B-1]

        for v in range(X):
            lo = cs + v * B
            hi = min(lo + B, n_samples)
            if lo < n_samples:
                out[lo:hi] = val[v][:hi-lo]
    return out



def butter_ba_norm(order):
    """Direct-form taps, b0-normalized, recurrence sign (matches ref_generate)."""
    from scipy.signal import butter
    b, a = butter(N=order, Wn=0.2, btype='lowpass', output='ba')
    return (b[1:] / b[0]).astype(np.float64), (-a[1:]).astype(np.float64), float(b[0])


def reference_f64(x_f32, order):
    """float64 sosfilt of the float32-normalized sections — identical to
    ref_generate.py / reference.bin."""
    from scipy.signal import butter, sosfilt
    sos = butter(N=order, Wn=0.2, btype='lowpass', output='sos')
    sos_n = sos.copy()
    for s in range(sos.shape[0]):
        sos_n[s, 0:3] = sos[s, 0:3] / sos[s, 0]
    sos_f32 = sos_n.astype(np.float32)
    return sosfilt(sos_f32.astype(np.float64), x_f32.astype(np.float64))


def direct_form_f32(x, bt, at):
    """Sequential float32 direct form: t[n] = x[n] + sum b_j x[n-j];
    y[n] = t[n] + sum a_j y[n-j]. Every operation rounded to float32."""
    k = len(bt)
    b = bt.astype(F32); a = at.astype(F32)
    xh = np.zeros(k, F32); yh = np.zeros(k, F32)
    out = np.zeros(len(x), F32)
    for n in range(len(x)):
        t = F32(x[n])
        for j in range(k):
            t = F32(t + F32(b[j] * xh[j]))
        y = t
        for j in range(k):
            y = F32(y + F32(a[j] * yh[j]))
        xh[1:] = xh[:-1]; xh[0] = F32(x[n])
        yh[1:] = yh[:-1]; yh[0] = y
        out[n] = y
    return out


if __name__ == '__main__':
    REL_TOL = 1e-4

    # ---- Part A: exact tree emulation of PLR_2 (order 2) ----
    print("Part A - PLR_2 merge-tree emulation (float32), order 2")
    print(f"{'GPU':>8} {'B':>5} {'x':>3} {'chunks':>7} | {'rel err':>10}")
    ok = True
    bt, at, g = butter_ba_norm(2)
    b1, b2 = F32(bt[0]), F32(bt[1])
    a1, a2 = F32(at[0]), F32(at[1])
    for gpu, B in (('RTX3060', 512), ('GTX1070', 1024)):
        for X in (1, 4, 9, 10):
            n = 3 * B * X + B // 2            # includes a partial tail chunk
            i = np.arange(n, dtype=np.float64)
            x = (g * np.cos(2.0 * np.pi * 120.0 * i / 1000.0)).astype(F32)
            ref = reference_f64(x, 2)
            got = plr_pass(x, b1, b2, a1, a2, B, X, n)
            rel = float(np.max(np.abs(got.astype(np.float64) - ref))
                        / np.max(np.abs(ref)))
            flag = '' if rel < REL_TOL else '   <<< FAIL'
            if rel >= REL_TOL: ok = False
            print(f"{gpu:>8} {B:>5} {X:>3} {3:>5}+t | {rel:>10.2e}{flag}")

    # ---- Part B: sequential float32 direct form per order ----
    print()
    print("Part B - sequential float32 direct form vs float64 reference")
    print(f"{'order':>6} {'n':>7} | {'rel err':>10} | verdict (gate = 1e-4)")
    n = 1 << 16
    i = np.arange(n, dtype=np.float64)
    for order in (2, 4, 8, 16):
        bt, at, g = butter_ba_norm(order)
        x = (g * np.cos(2.0 * np.pi * 120.0 * i / 1000.0)).astype(F32)
        ref = reference_f64(x, order)
        with np.errstate(over='ignore', invalid='ignore'):
            got = direct_form_f32(x, bt, at)
        finite = np.isfinite(got).all()
        if finite:
            rel = float(np.max(np.abs(got.astype(np.float64) - ref))
                        / np.max(np.abs(ref)))
            verdict = 'PASS' if rel < REL_TOL else 'FAIL'
        else:
            rel = float('inf'); verdict = 'FAIL (non-finite output)'
        if verdict != 'PASS' and order == 2: ok = False
        print(f"{order:>6} {n:>7} | {rel:>10.2e} | {verdict}")

    print()
    print("Part A verdict:", "ALL PASS" if ok else "FAILURES PRESENT")
    print("(Part B FAILs at high order are the expected finding of the "
          "direct-form lane, not emulator errors.)")
