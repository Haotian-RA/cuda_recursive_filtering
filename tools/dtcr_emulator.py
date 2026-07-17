#!/usr/bin/env python3
"""
dtcr_emulator.py — faithful float32 CPU emulation of the DTCR kernels in
dtcr_kernels.cuh, used to validate them without a GPU.

Checks performed by `python3 dtcr_emulator.py`:
  1. DTCR_32 loop form at (B=32, N=32/64/128) matches a float64 sequential
     cascade of the same Butterworth SOS (chunk carries + section handoffs
     exercised over multiple chunks).
  2. DTCR_64 loop form at (B=64, N=64) matches the same reference.
The hand-unrolled kernels are token-identical expansions of the loop bodies
(verified at build time), so these results cover them as well.
"""
import numpy as np
F32 = np.float32

def butter_norm_sos(n_sections):
    from scipy.signal import butter
    sos = butter(N=2*n_sections, Wn=0.2, btype='lowpass', output='sos')
    g = float(np.prod(sos[:, 0]))
    sos_n = sos.copy()
    for s in range(sos.shape[0]):
        sos_n[s, 0:3] = sos[s, 0:3] / sos[s, 0]
    return sos_n.astype(np.float32), g

def fma(a, b, c): return F32(F32(a) * F32(b) + F32(c))

def shfl_up(v, d):
    o = v.copy()
    if d > 0:
        o[d:] = v[:-d]
    return o

def host_factors(s, B, N):
    """DTCR host-side factor computation (float32, mirrors dtcr_kernels.cuh)."""
    LOG2 = N.bit_length() - 1
    b1, b2 = F32(s[1]), F32(s[2])
    a1, a2 = F32(-s[4]), F32(-s[5])
    L = LOG2 + 1
    arr = lambda: np.full(L, np.nan, F32)
    f, e, fde, hh, g, p, q, d, c = (arr() for _ in range(9))
    f[0] = F32(-a2); e[0] = F32(-a1); fde[0] = F32(f[0] / e[0])
    hh[0] = f[0]; g[0] = e[0]; p[1] = f[0]; q[1] = e[0]
    for n in range(1, L):
        f[n] = F32(f[n-1] * f[n-1])
        e[n] = F32(2 * f[n-1] - e[n-1] * e[n-1])
        fde[n] = F32(f[n] / e[n])
        hh[n] = F32(-e[n-1] * hh[n-1])
        g[n] = F32(f[n-1] - e[n-1] * g[n-1])
        d[n] = F32(-f[n-1] * f[n-1] / e[n-1])
        c[n] = F32(e[n-1] - f[n-1] / e[n-1])
        if n > 1:
            q[n] = F32(-e[n-1] * q[n-1])
            p[n] = F32(f[n-1] - e[n-1] * p[n-1])
    # decoupled first-round coefficients
    dc = np.zeros(6, F32)
    dc[0] = F32(f[0]*b1 - e[0]*b2)          # cb
    dc[1] = F32(b1 - e[0])                   # db
    dc[2] = F32(b2 * f[0])                   # fb
    dc[3] = F32(b2 + f[0] - b1*e[0])         # eb
    dc[4] = F32(b2 - e[0]*b1)                # hb
    dc[5] = F32(-e[0]*b2)                    # gb
    # block filtering factors over B lanes
    h0 = np.zeros(B, F32); h0[0] = F32(1); h0[1] = F32(-e[LOG2])
    for l in range(2, B):
        h0[l] = F32(-e[LOG2]*h0[l-1] - f[LOG2]*h0[l-2])
    hb2 = np.zeros(B, F32); hb1 = np.zeros(B, F32)
    he2 = np.zeros(B, F32); he1 = np.zeros(B, F32)
    for l in range(B):
        hb2[l] = F32(hh[LOG2]*h0[l]); he1[l] = F32(q[LOG2]*h0[l])
        tmp = F32(0) if l == 0 else h0[l-1]
        hb1[l] = F32(f[LOG2]*tmp + g[LOG2]*h0[l])
        he2[l] = F32(f[LOG2]*tmp + p[LOG2]*h0[l])
    return dict(b1=b1, b2=b2, e=e, fde=fde, cr_h=hh, cr_g=g, cr_p=p, cr_q=q,
                cr_d=d, cr_c=c, h0=h0, hb2=hb2, hb1=hb1, he2=he2, he1=he1,
                cb=dc[0], db=dc[1], fb=dc[2], eb=dc[3], hbc=dc[4], gbc=dc[5],
                LOG2=LOG2)

def dtcr(x, sos, B, N, two_warp_64):
    """Emulate DTCR_32 (two_warp_64=False) or DTCR_64 (True), loop form."""
    NS = len(sos)
    fac = [host_factors(s, B, N) for s in sos]
    HALF = N // 2; QUART = HALF // 2
    CHUNK = B * N
    nch = len(x) // CHUNK
    y = np.zeros_like(x)
    fullc = np.zeros((nch, NS, 2), F32)

    for cid in range(nch):
        cs = cid * CHUNK
        inn = x[cs:cs+CHUNK].reshape(B, N).copy()   # in[ti][bi]
        reg = np.zeros((2, B, HALF), F32)
        xi1 = np.zeros((2, B), F32); xi2 = np.zeros((2, B), F32)
        xi3 = np.zeros((2, B), F32); xi4 = np.zeros((2, B), F32)
        xcv = np.zeros((2, B), F32); tmpv = np.zeros((2, B), F32)
        yi1v = np.zeros((2, B), F32); yi2v = np.zeros((2, B), F32)

        for sec in range(NS):
            K = fac[sec]
            LOG2 = K['LOG2']
            # -------- Phase 1 boundary setup --------
            inn_snap = inn.copy()                    # values before this section's writes
            for ty in (0, 1):
                for tx in range(B):
                    if ty == 0:
                        if tx == 0:
                            xi3[ty][tx] = F32(0); xi4[ty][tx] = F32(0)
                            if cid == 0:
                                xi1[ty][tx] = F32(0); xi2[ty][tx] = F32(0)
                            elif sec == 0:
                                xi1[ty][tx] = x[cs-1]; xi2[ty][tx] = x[cs-2]
                        else:
                            xi1[ty][tx] = inn_snap[tx-1][N-1]
                            xi2[ty][tx] = inn_snap[tx-1][N-2]
                            xi3[ty][tx] = inn_snap[tx-1][N-3]
                            xi4[ty][tx] = inn_snap[tx-1][N-4]
                    else:
                        xi1[ty][tx] = inn_snap[tx][0]
                        if tx == 0:
                            xi4[ty][tx] = F32(0)
                            if cid == 0:
                                xi2[ty][tx] = F32(0); xi3[ty][tx] = F32(0)
                            elif sec == 0:
                                xi2[ty][tx] = x[cs-1]; xi3[ty][tx] = x[cs-2]
                        else:
                            xi2[ty][tx] = inn_snap[tx-1][N-1]
                            xi3[ty][tx] = inn_snap[tx-1][N-2]
                            xi4[ty][tx] = inn_snap[tx-1][N-3]
            # -------- first block + fused FIR/CR pass --------
            for ty in (0, 1):
                for tx in range(B):
                    if ty == 0:
                        if tx == 0:
                            h1, h2, h3 = K['b1'], K['b2'], F32(0)
                        else:
                            h1, h2, h3 = K['db'], K['eb'], K['cb']
                    else:
                        h1 = K['db']
                        if tx == 0:
                            h2, h3 = K['hbc'], K['gbc']
                        else:
                            h2, h3 = K['eb'], K['cb']
                    _xc = inn_snap[tx][ty]
                    _x1, _x2, _x3, _x4 = xi1[ty][tx], xi2[ty][tx], xi3[ty][tx], xi4[ty][tx]
                    reg[ty][tx][0] = fma(h1, _x1, fma(h2, _x2, fma(h3, _x3, fma(K['fb'], _x4, _xc))))
                    _x4 = _x2; _x3 = _x1; _x2 = _xc
                    for n in range(1, HALF):
                        _xc = inn_snap[tx][2*n + ty]
                        _x1 = inn_snap[tx][2*n + ty - 1]
                        reg[ty][tx][n] = fma(K['db'], _x1, fma(K['eb'], _x2, fma(K['cb'], _x3, fma(K['fb'], _x4, _xc))))
                        _x4 = _x2; _x3 = _x1; _x2 = _xc
                    xcv[ty][tx] = _xc
                    xi1[ty][tx], xi2[ty][tx], xi3[ty][tx], xi4[ty][tx] = _x1, _x2, _x3, _x4
            # -------- remaining CR rounds --------
            if two_warp_64:
                # in[tx][ty] = in_reg[HALF-1] published after the fused pass
                exch = np.array([[reg[0][tx][HALF-1], reg[1][tx][HALF-1]]
                                 for tx in range(B)], F32)
            for r in range(1, LOG2):
                step = 1 << r; sub = step >> 1; off = sub - 1
                if not two_warp_64:
                    for ty in (0, 1):
                        sh = shfl_up(reg[ty][:, HALF-1].copy(), 1)
                        sh[0] = F32(0)
                        for tx in range(B):
                            _x2 = sh[tx]
                            for n in range(0, HALF, step):
                                reg[ty][tx][n+off] = fma(-K['fde'][r], _x2, reg[ty][tx][n+off])
                                _x2 = reg[ty][tx][n+off+sub]
                                reg[ty][tx][n+off+sub] = fma(-K['e'][r], reg[ty][tx][n+off], _x2)
                else:
                    # DTCR_64: neighbor value through shared in[tx][ty]
                    # (published by the previous round or the fused pass)
                    for ty in (0, 1):
                        for tx in range(B):
                            _x2 = F32(0) if tx == 0 else exch[tx-1][ty]
                            for n in range(0, HALF, step):
                                reg[ty][tx][n+off] = fma(-K['fde'][r], _x2, reg[ty][tx][n+off])
                                _x2 = reg[ty][tx][n+off+sub]
                                reg[ty][tx][n+off+sub] = fma(-K['e'][r], reg[ty][tx][n+off], _x2)
                    exch = np.array([[reg[0][tx][HALF-1], reg[1][tx][HALF-1]] for tx in range(B)], F32)
            # -------- block filtering (per parity side) --------
            yiv = np.zeros((2, B), F32)
            if not two_warp_64:
                for ty in (0, 1):
                    col = reg[ty][:, HALF-1].copy()
                    for tx in range(B):
                        yiv[ty][tx] = F32(K['h0'][0] * col[tx])
                    for n in range(1, B):
                        sh = shfl_up(col, n)
                        for tx in range(B):
                            v = sh[tx] if tx >= n else F32(0)
                            yiv[ty][tx] = fma(K['h0'][n], v, yiv[ty][tx])
            else:
                exch2 = np.array([[reg[0][tx][HALF-1], reg[1][tx][HALF-1]] for tx in range(B)], F32)
                for ty in (0, 1):
                    for tx in range(B):
                        acc = F32(K['h0'][0] * reg[ty][tx][HALF-1])
                        for n in range(1, B):
                            v = exch2[tx-n][ty] if tx >= n else F32(0)
                            acc = fma(K['h0'][n], v, acc)
                        yiv[ty][tx] = acc
            # -------- lookback (serial: predecessors complete) --------
            if cid == 0:
                X0, X1 = F32(0), F32(0)
            else:
                X0, X1 = fullc[cid-1][sec]
            # -------- terminal correction + fullcarry --------
            for ty in (0, 1):
                he2c = K['he2'] if ty == 0 else K['hb2']
                he1c = K['he1'] if ty == 0 else K['hb1']
                for tx in range(B):
                    reg[ty][tx][HALF-1] = fma(-he1c[tx], X1, fma(-he2c[tx], X0, yiv[ty][tx]))
            fullc[cid][sec][0] = reg[0][B-1][HALF-1]
            fullc[cid][sec][1] = reg[1][B-1][HALF-1]
            for ty in (0, 1):
                for tx in range(B):
                    inn[tx][N-2+ty] = reg[ty][tx][HALF-1]
            # -------- deepest companion level --------
            if not two_warp_64:
                y2s = {ty: shfl_up(reg[ty][:, HALF-1].copy(), 2-ty) for ty in (0, 1)}
                y1s = {ty: shfl_up(reg[ty][:, HALF-1].copy(), 1+ty) for ty in (0, 1)}
            for ty in (0, 1):
                for tx in range(B):
                    if not two_warp_64:
                        _y2, _y1 = y2s[ty][tx], y1s[ty][tx]
                    if tx == 0:
                        tmpv[ty][tx] = X1
                        if ty == 0:
                            h2b = K['cr_p'][LOG2-1]; h1b = K['cr_q'][LOG2-1]
                            xcv[ty][tx] = X0; xi1[ty][tx] = X1; xi2[ty][tx] = X0
                        else:
                            h2b = K['cr_h'][LOG2-1]; h1b = K['cr_g'][LOG2-1]
                            xcv[ty][tx] = X1; xi2[ty][tx] = X1; xi3[ty][tx] = X0
                        _y2, _y1 = X0, X1
                    else:
                        if not two_warp_64:
                            if ty == 0:
                                h2b = K['cr_d'][LOG2]; h1b = K['cr_c'][LOG2]
                                tmpv[ty][tx] = _y1
                            else:
                                h2b = K['cr_c'][LOG2]; h1b = K['cr_d'][LOG2]
                                tmpv[ty][tx] = _y2
                            if tx == 1:
                                if ty == 0: _y2 = X0
                                else:       _y1 = X1
                            xcv[ty][tx] = tmpv[ty][tx]
                        else:
                            tmpv[ty][tx] = inn[tx-1][N-2+ty]
                            xcv[ty][tx] = tmpv[ty][tx]
                            if ty == 0:
                                h2b = K['cr_d'][LOG2]; h1b = K['cr_c'][LOG2]
                                _y1 = tmpv[ty][tx]
                                _y2 = X0 if tx == 1 else inn[tx-2][N-2]
                            else:
                                h2b = K['cr_c'][LOG2]; h1b = K['cr_d'][LOG2]
                                _y2 = inn[tx-1][N-1]
                                _y1 = X1 if tx == 1 else inn[tx-2][N-1]
                    reg[ty][tx][QUART-1] = fma(-h1b, _y1, fma(-h2b, _y2, reg[ty][tx][QUART-1]))
                    yi1v[ty][tx], yi2v[ty][tx] = _y1, _y2
            for ty in (0, 1):
                for tx in range(B):
                    inn[tx][HALF-2+ty] = reg[ty][tx][QUART-1]
            # -------- back-substitution rounds (loop form) --------
            zi1 = np.zeros((2, B), F32); zi2 = np.zeros((2, B), F32)
            for r in range(LOG2-2, 0, -1):
                P = HALF >> r; stride = 2 << r; sub = 1 << r; sub2 = sub >> 1
                if not two_warp_64:
                    y2sh = {ty: shfl_up(reg[ty][:, HALF-1-sub].copy(), 1) for ty in (0, 1)}
                else:
                    snap = inn.copy()
                for n in range(P):
                    for ty in (0, 1):
                        for tx in range(B):
                            if n == 0:
                                if tx == 0:
                                    if ty == 0:
                                        h2b = K['cr_p'][r]; h1b = K['cr_q'][r]
                                    else:
                                        h2b = K['cr_h'][r]; h1b = K['cr_g'][r]
                                    _y2 = X0 if not two_warp_64 else yi2v[ty][tx]
                                else:
                                    h2b = K['cr_d'][r+1]; h1b = K['cr_c'][r+1]
                                    if not two_warp_64:
                                        _y2 = y2sh[ty][tx]
                                    else:
                                        _y2 = snap[tx-1][N - stride - 2 + ty]
                                zi2[ty][tx] = _y2
                                zi1[ty][tx] = tmpv[ty][tx]
                                if tx == 0:
                                    yi2v[ty][tx] = _y2
                            elif n == 1:
                                h2b = K['cr_d'][r+1]; h1b = K['cr_c'][r+1]
                                zi2[ty][tx] = xcv[ty][tx]
                                zi1[ty][tx] = reg[ty][tx][sub-1]
                            else:
                                h2b = K['cr_d'][r+1]; h1b = K['cr_c'][r+1]
                                zi2[ty][tx] = zi1[ty][tx]
                                zi1[ty][tx] = reg[ty][tx][sub*n-1]
                            dst = sub*n + sub2 - 1
                            reg[ty][tx][dst] = fma(-h1b, zi1[ty][tx], fma(-h2b, zi2[ty][tx], reg[ty][tx][dst]))
                            inn[tx][stride*n + sub - 2 + ty] = reg[ty][tx][dst]
        y[cs:cs+CHUNK] = inn.reshape(-1)
    return y

def reference(x, sos, dtype=np.float64):
    y = np.asarray(x, dtype)
    for s in sos:
        b0, b1_, b2_, _, a1_, a2_ = [dtype(v) for v in s]
        out = np.zeros_like(y)
        x1 = x2 = y1_ = y2_ = dtype(0)
        for n in range(len(y)):
            v = b0*y[n] + b1_*x1 + b2_*x2 - a1_*y1_ - a2_*y2_
            x2, x1 = x1, y[n]; y2_, y1_ = y1_, v
            out[n] = v
        y = out
    return y

if __name__ == '__main__':
    print(f"{'kernel':>8} {'B':>4} {'N':>4} {'order':>6} | {'rel err':>10}")
    ok = True
    for NS in (1, 2, 4, 8):
        sos, g = butter_norm_sos(NS)
        for (B, N, tw) in ((32, 32, False), (32, 64, False), (32, 128, False), (64, 64, True)):
            CH = B * N
            i = np.arange(3 * CH, dtype=np.float64)
            x = (g * np.cos(2.0 * np.pi * 120.0 * i / 1000.0)).astype(F32)
            ref = reference(x, [s.astype(np.float64) for s in sos])
            got = dtcr(x, sos, B, N, tw)
            rel = float(np.max(np.abs(got.astype(np.float64) - ref)) / np.max(np.abs(ref)))
            name = 'DTCR_64' if tw else 'DTCR_32'
            flag = '' if rel < 1e-4 else '   <<< FAIL'
            if rel >= 1e-4: ok = False
            print(f"{name:>8} {B:>4} {N:>4} {2*NS:>6} | {rel:>10.2e}{flag}")
    print("ALL PASS" if ok else "FAILURES PRESENT")
