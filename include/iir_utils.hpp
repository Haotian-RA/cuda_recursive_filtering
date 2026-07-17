#ifndef IIR_UTILS_HPP
#define IIR_UTILS_HPP

// Host-only utilities for direct-form-to-SOS conversion.
// Contains:
//   - polyRoots: Durand-Kerner polynomial root finder
//   - tf2sos:    direct-form (b, a) -> second-order sections
//
// All root finding and pairing runs internally in double precision (the
// filter type T stays float): float Durand-Kerner could never reach the
// old 1e-14 threshold, so the iteration always burned its full budget and
// order-16 polynomials were fragile. Results are cast to T only in the
// final SOS coefficients. Marked `inline` so this header can be included
// in multiple translation units without linker errors.

#include <vector>
#include <array>
#include <complex>
#include <cmath>
#include <algorithm>

typedef float T;
using Complex  = std::complex<T>;       // kept for any external users
using ComplexD = std::complex<double>;  // internal precision for root finding


inline std::vector<ComplexD> polyRoots(const std::vector<T>& coeffs) {
    int n = (int)coeffs.size() - 1;
    if (n <= 0) return {};

    std::vector<double> c(n);
    for (int i = 0; i < n; i++)
        c[i] = (double)coeffs[i + 1] / (double)coeffs[0];

    std::vector<ComplexD> roots(n);
    for (int i = 0; i < n; i++)
        roots[i] = std::polar(0.4, 2.0 * M_PI * i / n + 0.1);

    for (int iter = 0; iter < 1000; iter++) {
        double maxDelta = 0;
        for (int i = 0; i < n; i++) {
            ComplexD p(1.0, 0);
            for (int j = 0; j < n; j++)
                p = p * roots[i] + c[j];

            ComplexD denom(1.0, 0);
            for (int j = 0; j < n; j++)
                if (i != j) denom *= (roots[i] - roots[j]);

            ComplexD delta = p / denom;
            roots[i] -= delta;
            maxDelta = std::max(maxDelta, std::abs(delta));
        }
        if (maxDelta < 1e-12) break;   // reachable in double: early exit fires
    }
    return roots;
}


inline std::vector<std::array<T, 6>> tf2sos(
    const std::vector<T>& b,
    const std::vector<T>& a)
{
    std::vector<ComplexD> zeros = polyRoots(b);
    std::vector<ComplexD> poles = polyRoots(a);

    double k = (double)b[0] / (double)a[0];
    int nSections = (std::max(zeros.size(), poles.size()) + 1) / 2;

    while (zeros.size() < 2 * nSections) zeros.push_back(0.0);
    while (poles.size() < 2 * nSections) poles.push_back(0.0);

    auto makePairs = [](std::vector<ComplexD> roots) {  // Note: copy, not const ref
        std::vector<std::pair<ComplexD, ComplexD>> pairs;
        std::vector<bool> used(roots.size(), false);

        // For real-coefficient polynomials, force conjugate pairing
        // by matching each root with the closest to its conjugate
        for (size_t i = 0; i < roots.size(); i++) {
            if (used[i]) continue;
            if (std::abs(roots[i].imag()) > 1e-10) {
                ComplexD target = std::conj(roots[i]);
                int bestJ = -1;
                double bestDist = 1e30;
                for (size_t j = i + 1; j < roots.size(); j++) {
                    if (!used[j]) {
                        double dist = std::abs(roots[j] - target);
                        if (dist < bestDist) {
                            bestDist = dist;
                            bestJ = j;
                        }
                    }
                }
                if (bestJ >= 0 && bestDist < 0.1) {  // Generous tolerance
                    // Average to ensure exact conjugate pair
                    double realPart = (roots[i].real() + roots[bestJ].real()) / 2;
                    double imagPart = (std::abs(roots[i].imag()) + std::abs(roots[bestJ].imag())) / 2;
                    pairs.push_back({ComplexD(realPart, imagPart), ComplexD(realPart, -imagPart)});
                    used[i] = used[bestJ] = true;
                }
            }
        }

        // Pair remaining real roots
        std::vector<int> remaining;
        for (size_t i = 0; i < roots.size(); i++)
            if (!used[i]) remaining.push_back(i);
        for (size_t i = 0; i + 1 < remaining.size(); i += 2)
            pairs.push_back({roots[remaining[i]], roots[remaining[i + 1]]});

        return pairs;
    };

    auto zeroPairs = makePairs(zeros);
    auto polePairs = makePairs(poles);

    std::vector<std::array<T, 6>> sos;
    for (int s = 0; s < nSections; s++) {
        auto& zp = zeroPairs[s];
        auto& pp = polePairs[s];

        ComplexD sumZ = zp.first + zp.second;
        ComplexD prodZ = zp.first * zp.second;
        ComplexD sumP = pp.first + pp.second;
        ComplexD prodP = pp.first * pp.second;

        double g = (s == 0) ? k : 1.0;

        // Cast to filter precision T only here, in the final coefficients.
        sos.push_back({
            (T)g,
            (T)(-g * sumZ.real()),
            (T)(g * prodZ.real()),
            (T)1.0,
            (T)(-sumP.real()),
            (T)(prodP.real())
        });
    }

    return sos;
}

#endif // IIR_UTILS_HPP
