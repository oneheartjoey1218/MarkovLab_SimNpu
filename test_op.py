from optimizer import find_best_strategy
from matmul import Simulate
import des_simulator as ds
from hardware import HW

RAW_MNK  = [1051, 12160, 15488]      # [M, N, K]
FORMATS  = ['fp16', 'fp16']      # [dtype A, dtype B]

def report_once(strat, total_cycles, totals, union_out2, union_out12):
    print("Problem size (M,N,K):", RAW_MNK, " dtypes:", FORMATS)
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

    # 各阶段 cycles
    for p, v in totals.items():
        print(f"{p.name} cycles = {v:.1f}")

    m_cycles = totals[ds.Pipeline.M]
    m_ms     = m_cycles / HW.CLOCK_FREQ * 1e3
    m_ms_aic = m_cycles / HW.AIC_TICK_FREQ * 1e3
    print(f"M stage (Cube) = {m_cycles:.1f} cycles  ≈ {m_ms:.6f} ms  |  {m_ms_aic:.6f} ms (AIC tick)")

if __name__ == '__main__':
    strat, _ = find_best_strategy(RAW_MNK, FORMATS)
    sim = Simulate(strat)
    total_cycles, totals = sim.calculate_pipelined_cycles_with_breakdown()

    # 基于区间并集的 aicore 窗口
    from des_simulator import Pipeline as P
    # aicore：排除 OUT2，保留 OUT1
    union_out2  = ds.LAST_SIM.union_cycles_of({P.M, P.MTE1, P.MTE2, P.FIX, P.OUT1})
    # 另一种口径：排除 OUT1+OUT2
    union_out12 = ds.LAST_SIM.union_cycles_of({P.M, P.MTE1, P.MTE2, P.FIX})

    report_once(strat, total_cycles, totals, union_out2, union_out12)
