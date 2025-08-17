from optimizer import find_best_strategy
from matmul import Simulate
import des_simulator as ds
from hardware import HW

# M,K;K,N
DIM_STRS = [
    "456,4096;4096,7424",
    "1096,4096;4096,4096",
    "1114,12288;12288,8064",
    "10,15616;15616,4224",
    "1051,15488;15488,12160",
    "1054,8192;8192,128",
    "1032,9472;9472,13568",
    "1920,1920;1920,9088",
    "101,1408;1408,256",
    "1196,1152;1152,14080",
    "1617,8704;8704,7296",
    "1026,2176;2176,1536",
    "1002,6912;6912,1536",
    "556,640;640,384",
    "1915,6016;6016,6400",
    "268,384;384,3072",
    "456,4096;4096,7424",
    "1097,2432;2432,8448",
    "1431,5888;5888,5632",
]

FORMATS  = ['fp16', 'fp16']   # [dtype A, dtype B]

def parse_dim(s: str):
    # "M,K;K,N" -> [M, N, K]
    left, right = s.split(';')
    m, k1 = [int(x) for x in left.split(',')]
    k2, n = [int(x) for x in right.split(',')]
    assert k1 == k2, f"K mismatch in {s}"
    return [m, n, k1]

def report_once(raw_mnk, strat, total_cycles, totals, union_out2, union_out12, totals_ops):
    print("="*72)
    print("Problem size (M,N,K):", raw_mnk, " dtypes:", FORMATS)
    print("L2 tile:", strat.L2_mnk_values[0])
    print("L1 tile:", strat.L1_mnk_values[0])
    print("L0 tile:", strat.L0_mnk_values[0])
    print("DFF tile:", strat.DFF_mnk_values)

    # Overall makespan
    total_ms = total_cycles / HW.CLOCK_FREQ * 1e3
    total_s  = total_cycles / HW.CLOCK_FREQ
    print(f"Total cycles = {total_cycles:.1f}")
    print(f"Total time   = {total_ms:.6f} ms  ({total_s:.9f} s)")

    # AIC tick 口径
    aic_ms = total_cycles / HW.AIC_TICK_FREQ * 1e3
    aic_s  = total_cycles / HW.AIC_TICK_FREQ
    print(f"Total time (AIC tick) = {aic_ms:.6f} ms  ({aic_s:.9f} s)")

    # aicore 并集口径
    print(f"AICore union time (exclude OUT2) = {union_out2 / HW.CLOCK_FREQ * 1e3:.6f} ms  ({union_out2 / HW.CLOCK_FREQ:.9f} s)")
    print(f"AICore union time (exclude OUT2, AIC tick) = {union_out2 / HW.AIC_TICK_FREQ * 1e3:.6f} ms  ({union_out2 / HW.AIC_TICK_FREQ:.9f} s)")

    # 各阶段 cycles（区间并集）
    for p, v in totals.items():
        print(f"{p.name} cycles = {v:.1f}")

    m_cycles = totals[ds.Pipeline.M]
    m_ms     = m_cycles / HW.CLOCK_FREQ * 1e3
    m_ms_aic = m_cycles / HW.AIC_TICK_FREQ * 1e3
    print(f"M stage (Cube) = {m_cycles:.1f} cycles  ≈ {m_ms:.6f} ms  |  {m_ms_aic:.6f} ms (AIC tick)")

    # 细粒度模块（OpType）用时（区间并集）
    print("\n--- OpType union time (ms) ---")
    for op, cyc in totals_ops.items():
        print(f"{op.name:14s}: {cyc / HW.CLOCK_FREQ * 1e3:.6f}")

if __name__ == '__main__':
    summary = []  # (dim_str, union_out2_ms)
    from des_simulator import Pipeline as P

    for dim_str in DIM_STRS:
        RAW_MNK = parse_dim(dim_str)
        strat, _ = find_best_strategy(RAW_MNK, FORMATS)
        sim = Simulate(strat)
        total_cycles, totals, totals_ops = sim.calculate_pipelined_cycles_with_breakdown()

        # 基于区间并集的 aicore 窗口
        union_out2  = ds.LAST_SIM.union_cycles_of({P.M, P.MTE1, P.MTE2, P.FIX, P.OUT1})  # 排除 OUT2
        union_out12 = ds.LAST_SIM.union_cycles_of({P.M, P.MTE1, P.MTE2, P.FIX})          # 排除 OUT1+OUT2

        report_once(RAW_MNK, strat, total_cycles, totals, union_out2, union_out12, totals_ops)

        summary.append((dim_str, union_out2 / HW.CLOCK_FREQ * 1e3))

    print("\n" + "="*72)
    print("Summary: AICore union time (exclude OUT2) per dimension (ms)")
    for dim_str, ms in summary:
        print(f"{ms*1000:.6f}")
        