#!/bin/bash

# run_accuracy.sh — accuracy test for the chosen kernel across all supported
# filter orders.
#
# For each N_SECTIONS in {1, 2, 4, 8}:
#   1. Regenerate reference.bin via ref_generate.py
#   2. Compile test_accuracy.cu with the current config
#   3. Run ./main, which verifies the kernel output against reference.bin

set -u

# Run everything relative to the repo root (where this script lives);
# all generated artifacts (reference.bin, filter_taps.hpp, binary) go to build/.
cd "$(dirname "$0")"
mkdir -p build


# ==========================================================================
# Configuration
# ==========================================================================
export N_SAMPLES_LOG2=25
export BLOCK_SIZE=32
export N_BLOCKS=64

# Which kernel to test. Available: PH (STCR, DTCR, PLR to come)
KERNEL=PLR

# GPU profile for gpu_specs.hpp: RTX3060 (default) or GTX1070
GPU=RTX3060

if [ "$GPU" = "GTX1070" ]; then
    GPU_FLAG="-DGPU_GTX1070"
else
    GPU_FLAG=""
fi

ORDERS=(1 2 4 8)

# STCR back substitution: 0 = generic for-loop kernels (default),
# 1 = manually unrolled kernels (spill-proof; STCR and DTCR, supported
#     combinations: 32x32, 32x64, 32x128, 64x64)
HANDUNROLLED=1

if [ "$HANDUNROLLED" = "1" ]; then
    UNROLL_FLAG="-DSTCR_HANDUNROLLED -DDTCR_HANDUNROLLED"
else
    UNROLL_FLAG=""
fi

# PLR only: values-per-thread ceiling, per filter order ('auto' or integer).
# 'auto' runs an ascending compile-time spill calibration (x = 2, 4, 6, ...;
# at the first spill it fine-steps back by 1), prints a registers-vs-x table
# and the selected maximum. Copy the printed number over 'auto' here to
# freeze it for the measurement campaign. An explicit integer is always
# honored as-is (diagnostics still printed by the main compile).
PLR_X_MAX_O2=auto
PLR_X_MAX_O4=auto
PLR_X_MAX_O8=auto
PLR_X_MAX_O16=auto



echo "=========================================================="
echo "$KERNEL accuracy test"
echo "=========================================================="
echo "  KERNEL         = $KERNEL"
echo "  GPU profile    = $GPU"
echo "  HANDUNROLLED   = $HANDUNROLLED"
echo "  N_SAMPLES_LOG2 = $N_SAMPLES_LOG2  (batch = 2^$N_SAMPLES_LOG2)"
echo "  BLOCK_SIZE     = $BLOCK_SIZE"
echo "  N_BLOCKS       = $N_BLOCKS"
echo "  Orders tested  =$(for n in "${ORDERS[@]}"; do printf ' %d' $((2*n)); done)"
echo ""


PASSED=()
FAILED=()

for N_SECTIONS in "${ORDERS[@]}"; do
    export N_SECTIONS
    ORDER=$((2 * N_SECTIONS))
    echo "----- Order $ORDER (N_SECTIONS=$N_SECTIONS) -----"

    # Step 1: Generate reference
    python3 tools/ref_generate.py --quiet \
        --output build/reference.bin --header build/filter_taps.hpp
    if [ $? -ne 0 ]; then
        echo "  reference generation FAILED"
        FAILED+=("order $ORDER (ref)")
        continue
    fi

    # ---------------- PLR_X_MAX resolution (PLR only) ----------------
    PLR_FLAG=""
    if [ "$KERNEL" = "PLR" ]; then
        case $ORDER in
            2)  PLR_X_SEL=$PLR_X_MAX_O2 ;;
            4)  PLR_X_SEL=$PLR_X_MAX_O4 ;;
            8)  PLR_X_SEL=$PLR_X_MAX_O8 ;;
            16) PLR_X_SEL=$PLR_X_MAX_O16 ;;
        esac
        if [ "$PLR_X_SEL" = "auto" ]; then
            echo "  PLR_X_MAX=auto: ascending spill calibration (order $ORDER)"
            plr_calib() {
                nvcc -O2 -arch=native -w -Xptxas -v \
                     -Iinclude -Ibuild \
                     $GPU_FLAG -DKERNEL_PLR -DPLR_X_MAX=$1 \
                     -DN_SAMPLES_LOG2=$N_SAMPLES_LOG2 \
                     -DN_SECTIONS=$N_SECTIONS \
                     -DBLOCK_SIZE=$BLOCK_SIZE \
                     -DN_BLOCKS=$N_BLOCKS \
                     -c src/test_accuracy.cu -o /tmp/plr_calib.o > /tmp/plr_calib.log 2>&1 \
                    || return 2
                local blk=$(grep -A5 "Compiling entry function.*PLR_" /tmp/plr_calib.log)
                CAL_REGS=$(echo "$blk" | grep -o "[0-9]* registers" | head -1)
                CAL_SPILL=$(echo "$blk" | grep -o "[0-9]* bytes spill stores" | grep -o "^[0-9]*" | head -1)
                CAL_SPILL=${CAL_SPILL:-0}
                [ "$CAL_SPILL" -eq 0 ]
            }
            best=""
            for x in 2 4 6 8 10 12 14 16 18 20; do
                plr_calib $x; rc=$?
                if [ $rc -eq 2 ]; then
                    echo "    x=$x : compile FAILED"; tail -5 /tmp/plr_calib.log; break
                elif [ $rc -eq 0 ]; then
                    echo "    x=$x : OK    ($CAL_REGS, 0 spill bytes)"
                    best=$x
                else
                    echo "    x=$x : SPILL ($CAL_REGS, $CAL_SPILL spill bytes)"
                    fm1=$((x - 1))
                    if [ $fm1 -ge 1 ] && { [ -z "$best" ] || [ $fm1 -gt $best ]; }; then
                        if plr_calib $fm1; then
                            echo "    x=$fm1 : OK    ($CAL_REGS, 0 spill bytes)"
                            best=$fm1
                        else
                            echo "    x=$fm1 : SPILL ($CAL_REGS, $CAL_SPILL spill bytes)"
                        fi
                    fi
                    break
                fi
            done
            if [ -z "$best" ]; then
                echo "    WARNING: no spill-free x found; using x=1"
                best=1
            fi
            PLR_X_SEL=$best
            echo "  PLR_X_MAX selected: $PLR_X_SEL  (copy this number over 'auto' for the campaign)"
        fi
        PLR_FLAG="-DPLR_X_MAX=$PLR_X_SEL"
        echo "  PLR_X_MAX in effect: $PLR_X_SEL"
    fi

    # Step 2: Compile
    nvcc -O2 -arch=native -w \
         -Iinclude -Ibuild \
         $GPU_FLAG \
         $UNROLL_FLAG \
         $PLR_FLAG \
         -DKERNEL_$KERNEL \
         -DN_SAMPLES_LOG2=$N_SAMPLES_LOG2 \
         -DN_SECTIONS=$N_SECTIONS \
         -DBLOCK_SIZE=$BLOCK_SIZE \
         -DN_BLOCKS=$N_BLOCKS \
         -o build/main src/test_accuracy.cu
    if [ $? -ne 0 ]; then
        echo "  compilation FAILED"
        FAILED+=("order $ORDER (compile)")
        continue
    fi

    # Step 3: Run
    ./build/main
    if [ $? -eq 0 ]; then
        PASSED+=("order $ORDER")
    else
        FAILED+=("order $ORDER (accuracy)")
    fi

    echo ""
done


# ==========================================================================
# Summary
# ==========================================================================
echo "=========================================================="
echo "Summary"
echo "=========================================================="
echo "  Passed: ${#PASSED[@]} / ${#ORDERS[@]}"
if [ ${#PASSED[@]} -gt 0 ]; then
    for entry in "${PASSED[@]}"; do
        echo "    $entry"
    done
fi
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "  Failed:"
    for entry in "${FAILED[@]}"; do
        echo "    $entry"
    done
    exit 1
fi

echo ""
echo "All orders passed."
