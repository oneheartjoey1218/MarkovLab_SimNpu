from matmul import MatMul_Strategy, Simulate
from hardware import HW

def best_tile(cap_bytes, tb=2, raw_dims=(float('inf'),)*3, max_dim=2048):
    """
    cap_bytes 缓存容量
    tb 元素字节数 (fp16=2 / fp32=4)
    raw_dims 原始矩阵 (M,N,K) 尺寸上界
    max_dim 搜索维度总体上限
    """
    M_max, N_max, K_max = raw_dims
    cap_elems = cap_bytes // tb # 可容纳的元素数
    best_blk, best_size = None, 0

    for m in range(8, min(max_dim, M_max)+1, 8):
        for k in range(8, min(max_dim, K_max)+1, 8):
            for n in range(8, min(max_dim, N_max)+1, 8):
                size = m*k + k*n + m*n # A+B+C占用元素
                if size <= cap_elems:
                    vol = m * n * k
                    if vol > best_size:
                        best_blk, best_size = (m, k, n), vol
    return best_blk

def find_best_strategy(raw_mnk, storage_formats, dataflow_mode='best', time_limit_s=10):
    tb = 2 if 'fp16' in storage_formats else 4

    # 传入 raw_mnk 作为维度上界，避免 tile 超尺寸
    L2_tile  = best_tile(HW.L2_CAPACITY,  tb, raw_dims=raw_mnk)
    L1_tile  = best_tile(HW.L1_CAPACITY,  tb, raw_dims=raw_mnk)
    L0_tile  = best_tile(HW.L0A_CAPACITY, tb, raw_dims=raw_mnk)
    DFF_tile = best_tile(HW.ABDFF_CAPACITY, tb, raw_dims=raw_mnk)

    strat = MatMul_Strategy(dataflow_mode, raw_mnk, storage_formats, option=None)
    strat.L2_mnk_values = [L2_tile]
    strat.L1_mnk_values = [L1_tile]
    strat.L0_mnk_values = [L0_tile]
    strat.DFF_mnk_values = list(DFF_tile)

    # 默认设为 inner product + row-major
    for i in (1, 2, 3, 4):
        setattr(strat, f"L{i}_block_strategy", 'inner')
        setattr(strat, f"L{i}_storage_formats", (0, 0))

    sim = Simulate(strat)
    total_cycles = sim.calculate_pipelined_cycles()
    frequency_hz = HW.CLOCK_FREQ
    total_time_s = total_cycles / frequency_hz
    return strat, total_time_s