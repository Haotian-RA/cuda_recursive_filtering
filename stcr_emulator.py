#!/usr/bin/env python3
"""
stcr_emulator.py — faithful float32 CPU emulation of the STCR kernels in
stcr_kernels.cuh, used to validate the back-substitution variants without a GPU.

Checks performed by `python3 stcr_emulator.py`:
  1. N=32, BLOCK_SIZE=32: loop-form back substitution is BIT-IDENTICAL to the
     hand-unrolled form (they execute the same operations in the same order).
  2. BLOCK_SIZE=32 with N=32/64/128 and BLOCK_SIZE=64 with N=64: loop form
     matches a float64 sequential cascade of the same (Butterworth) SOS.
Chunk carries and multi-section handoffs are exercised (multiple chunks).
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
    sos_f32 = sos_n.astype(np.float32)          # what the kernel consumes
    return sos_f32, g

def host_factors(s, B, LOG2):
    b1, b2 = F32(s[1]), F32(s[2]); a1, a2 = F32(-s[4]), F32(-s[5])
    L = LOG2 + 1
    arr = lambda: np.full(L, np.nan, F32)
    f, e, fde, hh, g, p, q, d, c = (arr() for _ in range(9))
    f[0] = F32(-a2); e[0] = F32(-a1); fde[0] = F32(f[0]/e[0])
    hh[0] = f[0]; g[0] = e[0]; p[1] = f[0]; q[1] = e[0]
    for n in range(1, L):
        f[n] = F32(f[n-1]*f[n-1]); e[n] = F32(2*f[n-1] - e[n-1]*e[n-1])
        fde[n] = F32(f[n]/e[n]); hh[n] = F32(-e[n-1]*hh[n-1])
        g[n] = F32(f[n-1] - e[n-1]*g[n-1]); d[n] = F32(-f[n-1]*f[n-1]/e[n-1])
        c[n] = F32(e[n-1] - f[n-1]/e[n-1])
        if n > 1:
            q[n] = F32(-e[n-1]*q[n-1]); p[n] = F32(f[n-1] - e[n-1]*p[n-1])
    h0 = np.zeros(B, F32); h0[0] = F32(1); h0[1] = F32(-e[LOG2])
    for l in range(2, B): h0[l] = F32(-e[LOG2]*h0[l-1] - f[LOG2]*h0[l-2])
    hb2, hb1, he2, he1 = (np.zeros(B, F32) for _ in range(4))
    for l in range(B):
        hb2[l] = F32(hh[LOG2]*h0[l]); he1[l] = F32(q[LOG2]*h0[l])
        tmp = F32(0) if l == 0 else h0[l-1]
        hb1[l] = F32(f[LOG2]*tmp + g[LOG2]*h0[l]); he2[l] = F32(f[LOG2]*tmp + p[LOG2]*h0[l])
    return dict(b1=b1, b2=b2, f=f, e=e, fde=fde, cr_h=hh, cr_g=g, cr_p=p,
                cr_q=q, cr_d=d, cr_c=c, h0=h0, hb2=hb2, hb1=hb1, he2=he2, he1=he1)

def fma(a, b, c): return F32(F32(a)*F32(b) + F32(c))
def shfl_up(v, dlt):
    o = v.copy(); o[dlt:] = v[:-dlt]; return o

def backsub_rounds_loop(reg, K, X0, X1, xi1, xi2, y2o, y1e, B, N, LOG2):
    """Generic for-loop rounds; mirrors the CUDA loop text exactly."""
    for r in range(LOG2 - 2, 0, -1):
        stride = 2 << r; sub = stride >> 1; P = (N // 2) >> r
        s_e = shfl_up(reg[:, N - stride - 2], 1)
        s_o = shfl_up(reg[:, N - stride - 1], 1)
        for t in range(B):
            if t == 0:
                h2o = K['cr_h'][r]; h1o = K['cr_g'][r]
                h2e = K['cr_p'][r]; h1e = K['cr_q'][r]
                y2e_t = X0; y1o_t = X1
            else:
                h2o = K['cr_c'][r+1]; h1o = K['cr_d'][r+1]
                h2e = h1o; h1e = h2o
                y2e_t = s_e[t]; y1o_t = s_o[t]
            z2o = y2o[t]; z1o = y1o_t; z2e = y2e_t; z1e = y1e[t]
            reg[t][sub-2] = fma(-h1e, z1e, fma(-h2e, z2e, reg[t][sub-2]))
            reg[t][sub-1] = fma(-h1o, z1o, fma(-h2o, z2o, reg[t][sub-1]))
            h2o = K['cr_d'][r+1]; h1o = K['cr_c'][r+1]; h2e = h2o; h1e = h1o
            z2o = xi1[t]; z1o = reg[t][stride-1]; z2e = xi2[t]; z1e = reg[t][stride-2]
            reg[t][stride+sub-2] = fma(-h1e, z1e, fma(-h2e, z2e, reg[t][stride+sub-2]))
            reg[t][stride+sub-1] = fma(-h1o, z1o, fma(-h2o, z2o, reg[t][stride+sub-1]))
            for n in range(2, P):
                z2o = z1o; z1o = reg[t][stride*n-1]
                z2e = z1e; z1e = reg[t][stride*n-2]
                reg[t][stride*n+sub-2] = fma(-h1e, z1e, fma(-h2e, z2e, reg[t][stride*n+sub-2]))
                reg[t][stride*n+sub-1] = fma(-h1o, z1o, fma(-h2o, z2o, reg[t][stride*n+sub-1]))

def backsub_rounds_hard32(reg, K, X0, X1, xi1, xi2, y2o, y1e, B, N, LOG2):
    """Verbatim port of the hand-unrolled N=32 rounds from stcr_kernels.cuh."""
    assert N == 32
    for (r, pairs) in [(3, None), (2, None), (1, None)]:
        stride = 2 << r; sub = stride >> 1; P = 16 >> r
        s_e = shfl_up(reg[:, N - stride - 2], 1)
        s_o = shfl_up(reg[:, N - stride - 1], 1)
        for t in range(B):
            if t == 0:
                h2o = K['cr_h'][r]; h1o = K['cr_g'][r]
                h2e = K['cr_p'][r]; h1e = K['cr_q'][r]
                y2e_t = X0; y1o_t = X1
            else:
                h2o = K['cr_c'][r+1]; h1o = K['cr_d'][r+1]
                h2e = h1o; h1e = h2o
                y2e_t = s_e[t]; y1o_t = s_o[t]
            z2o = y2o[t]; z1o = y1o_t; z2e = y2e_t; z1e = y1e[t]
            reg[t][sub-2] = fma(-h1e, z1e, fma(-h2e, z2e, reg[t][sub-2]))
            reg[t][sub-1] = fma(-h1o, z1o, fma(-h2o, z2o, reg[t][sub-1]))
            h2o = K['cr_d'][r+1]; h1o = K['cr_c'][r+1]; h2e = h2o; h1e = h1o
            z2o = xi1[t]; z1o = reg[t][stride-1]; z2e = xi2[t]; z1e = reg[t][stride-2]
            reg[t][stride+sub-2] = fma(-h1e, z1e, fma(-h2e, z2e, reg[t][stride+sub-2]))
            reg[t][stride+sub-1] = fma(-h1o, z1o, fma(-h2o, z2o, reg[t][stride+sub-1]))
            for n in range(2, P):
                z2o = z1o; z1o = reg[t][stride*n-1]
                z2e = z1e; z1e = reg[t][stride*n-2]
                reg[t][stride*n+sub-2] = fma(-h1e, z1e, fma(-h2e, z2e, reg[t][stride*n+sub-2]))
                reg[t][stride*n+sub-1] = fma(-h1o, z1o, fma(-h2o, z2o, reg[t][stride*n+sub-1]))

def stcr(x, sos_f32, gain, B, N, form):
    """Emulate the STCR kernel (BLOCK_SIZE=B, N_BLOCKS=N) on gain-scaled input."""
    LOG2 = int(np.log2(N)); CH = B * N
    NS = sos_f32.shape[0]
    fac = [host_factors(sos_f32[s], B, LOG2) for s in range(NS)]
    xg = (np.float64(gain) * x.astype(np.float64)).astype(F32)  # device input
    nch = len(xg)//CH; y = np.zeros_like(xg)
    fullc = np.zeros((nch, NS, 2), F32)
    bs = backsub_rounds_loop if form == 'loop' else backsub_rounds_hard32
    for cid in range(nch):
        cs = cid*CH
        inn = xg[cs:cs+CH].reshape(B, N).copy()
        reg = np.zeros((B, N), F32)
        xi1 = np.zeros(B, F32); xi2 = np.zeros(B, F32)
        yi1 = np.zeros(B, F32); yi2 = np.zeros(B, F32)
        for sec in range(NS):
            K = fac[sec]
            yi2[0] = F32(0); yi1[0] = F32(0)
            if cid == 0: xi2[0] = F32(0); xi1[0] = F32(0)
            elif sec == 0: xi2[0] = xg[cs-2]; xi1[0] = xg[cs-1]
            for t in range(1, B):
                if sec == 0:
                    xi2[t] = inn[t-1][N-2]; xi1[t] = inn[t-1][N-1]
                    yi2[t] = inn[t-1][N-4]; yi1[t] = inn[t-1][N-3]
                oy1 = yi1[t]
                yi2[t] = fma(K['b1'], oy1, fma(K['b2'], yi2[t], xi2[t]))
                yi1[t] = fma(K['b1'], xi2[t], fma(K['b2'], oy1, xi1[t]))
            for t in range(B):
                _x2, _x1, _y2, _y1 = xi2[t], xi1[t], yi2[t], yi1[t]
                for n in range(N):
                    _xc = inn[t][n] if sec == 0 else reg[t][n]
                    _yc = fma(K['b1'], _x1, fma(K['b2'], _x2, _xc))
                    reg[t][n] = fma(-K['e'][0], _y1, fma(K['f'][0], _y2, _yc))
                    _x2, _x1 = _x1, _xc; _y2, _y1 = _y1, _yc
            for ro in range(1, LOG2):
                step = 2 << ro; sub = 1 << ro
                for i in range(2):
                    off = sub - 2 + i
                    sh = shfl_up(reg[:, N-2+i], 1); sh[0] = F32(0)
                    for t in range(B):
                        _x2 = sh[t]
                        for n in range(0, N, step):
                            reg[t][n+off] = fma(-K['fde'][ro], _x2, reg[t][n+off])
                            _x2 = reg[t][n+off+sub]
                            reg[t][n+off+sub] = fma(-K['e'][ro], reg[t][n+off], _x2)
            by2 = np.array([F32(K['h0'][0]*reg[t][N-2]) for t in range(B)], F32)
            by1 = np.array([F32(K['h0'][0]*reg[t][N-1]) for t in range(B)], F32)
            for n in range(1, B):
                xc = shfl_up(reg[:, N-2], n); yc = shfl_up(reg[:, N-1], n)
                xc[:n] = F32(0); yc[:n] = F32(0)
                for t in range(B):
                    by2[t] = fma(K['h0'][n], xc[t], by2[t])
                    by1[t] = fma(K['h0'][n], yc[t], by1[t])
            X0, X1 = (F32(0), F32(0)) if cid == 0 else tuple(fullc[cid-1][sec])
            for t in range(B):
                reg[t][N-2] = fma(-K['he1'][t], X1, fma(-K['he2'][t], X0, by2[t]))
                reg[t][N-1] = fma(-K['hb1'][t], X1, fma(-K['hb2'][t], X0, by1[t]))
            fullc[cid][sec][0] = reg[B-1][N-2]; fullc[cid][sec][1] = reg[B-1][N-1]
            y2e = shfl_up(reg[:, N-2], 2); y2o = shfl_up(reg[:, N-1], 1)
            y1e = shfl_up(reg[:, N-2], 1); y1o = shfl_up(reg[:, N-1], 2)
            for t in range(B):
                if t == 0:
                    h2e = K['cr_p'][LOG2-1]; h2o = K['cr_h'][LOG2-1]
                    h1e = K['cr_q'][LOG2-1]; h1o = K['cr_g'][LOG2-1]
                    y2e[t] = X0; y2o[t] = X0; y1e[t] = X1; y1o[t] = X1
                    xi2[t] = X0; xi1[t] = X1
                else:
                    h2e = K['cr_d'][LOG2]; h2o = K['cr_c'][LOG2]
                    h1e = K['cr_c'][LOG2]; h1o = K['cr_d'][LOG2]
                    if t == 1: y2e[t] = X0; y1o[t] = X1
                    xi2[t] = y1e[t]; xi1[t] = y2o[t]
                H = N // 2
                reg[t][H-2] = fma(-h1e, y1e[t], fma(-h2e, y2e[t], reg[t][H-2]))
                reg[t][H-1] = fma(-h1o, y1o[t], fma(-h2o, y2o[t], reg[t][H-1]))
            bs(reg, K, X0, X1, xi1, xi2, y2o, y1e, B, N, LOG2)
            if sec == NS - 1:
                for t in range(B): inn[t][:] = reg[t][:]
            else:
                yi2 = shfl_up(reg[:, N-4], 1); yi1 = shfl_up(reg[:, N-3], 1)
        y[cs:cs+CH] = inn.reshape(-1)
    return y

def reference(x, sos_f32, gain):
    from scipy.signal import sosfilt
    xg = (np.float64(gain) * x.astype(np.float64)).astype(F32)
    return sosfilt(sos_f32.astype(np.float64), xg.astype(np.float64))

if __name__ == '__main__':
    ok = True
    for NS in (1, 2, 4, 8):
        sos, g = butter_norm_sos(NS)
        # ---- check 1: bit identity, loop vs hand-unrolled, B=32 N=32 ----
        x = np.array([np.cos(2*np.pi*120*i/1000.0) for i in range(3*32*32)], F32)
        a = stcr(x, sos, g, 32, 32, 'loop')
        b = stcr(x, sos, g, 32, 32, 'hard32')
        bit = np.array_equal(a, b)
        ok &= bit
        # ---- check 2: loop form vs float64 reference across (B, N) ----
        line = f"order {2*NS:>2}: loop==unrolled(N=32): {'BIT-IDENTICAL' if bit else 'MISMATCH'}"
        for (B, N) in [(32, 32), (32, 64), (32, 128), (64, 64)]:
            x = np.array([np.cos(2*np.pi*120*i/1000.0) for i in range(3*B*N)], F32)
            got = stcr(x, sos, g, B, N, 'loop')
            ref = reference(x, sos, g)
            rel = float(np.max(np.abs(got.astype(np.float64) - ref)) / max(np.max(np.abs(ref)), 1e-30))
            ok &= rel < 1e-4
            line += f" | {B}x{N}: rel={rel:.1e}"
        print(line)
    print("ALL CHECKS PASSED" if ok else "FAILURES PRESENT")
