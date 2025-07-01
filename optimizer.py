from ortools.sat.python import cp_model
from hardware import HW
from matmul import MatMul_Strategy, Simulate

def enum_blocks(max_M: int, max_K: int, max_N: int, cap_bytes: int, tb: int):
    """
    枚举所有满足
        m*k*tb ≤ cap_bytes,
        k*n*tb ≤ cap_bytes,
        m*n*tb ≤ cap_bytes
    的 (m,k,n) 三元组。
    """
    out = []
    # 步长设置为每层最小访问粒度除以字节数
    step = max(HW.MIN_ACCESS['L2'] // tb, 1)
    for m in range(step, max_M + 1, step):
        for k in range(step, max_K + 1, step):
            if m * k * tb > cap_bytes:
                continue
            for n in range(step, max_N + 1, step):
                if k * n * tb > cap_bytes or m * n * tb > cap_bytes:
                    continue
                out.append((m, k, n))
    return out

def find_best_strategy(raw_mnk, storage_formats, dataflow_mode, time_limit_s=10):
    """
    在给定 raw_mnk=[M0,N0,K0]、存储格式、dataflow_mode 下，
    用 CP-SAT 枚举 + 回调仿真 搜索最优 MatMul_Strategy。
    """
    M0, N0, K0 = raw_mnk
    tb = 2 if 'fp16' in storage_formats else 4

    # 1) 生成每层的候选块列表
    caps = {
        1: HW.L2_CAPACITY,   # L2Cache
        2: HW.L1_CAPACITY,    # L1
        3: HW.L0A_CAPACITY,   # L0 (取 L0A)
        4: HW.ABDFF_CAPACITY  # DFF 寄存器层
    }
    blocks = {
        i: enum_blocks(M0, K0, N0, caps[i], tb)
        for i in (1, 2, 3, 4)
    }

    # 2) 构建 CP-SAT 模型
    model = cp_model.CpModel()

    # 每层选择哪个三元组
    idx = {
        i: model.NewIntVar(0, len(blocks[i]) - 1, f"idx{i}")
        for i in (1, 2, 3, 4)
    }
    # 内积/外积 和 存储格式决策
    SI = {i: model.NewBoolVar(f"SI{i}") for i in (1, 2, 3, 4)}
    SO = {i: model.NewBoolVar(f"SO{i}") for i in (1, 2, 3, 4)}
    ZL = {i: model.NewBoolVar(f"ZL{i}") for i in (1, 2, 3, 4)}
    ZR = {i: model.NewBoolVar(f"ZR{i}") for i in (1, 2, 3, 4)}

    # 3) 容量约束已在枚举时保证

    # 4) SI+SO ≥ 1
    for i in (1, 2, 3, 4):
        model.Add(SI[i] + SO[i] >= 1)

    # 5) 用 AddElement 将 idx 解码为 (m,k,n)
    Mi = {}
    Ki = {}
    Ni = {}
    for i in (1, 2, 3, 4):
        ms = [b[0] for b in blocks[i]]
        ks = [b[1] for b in blocks[i]]
        ns = [b[2] for b in blocks[i]]
        Mi[i] = model.NewIntVar(min(ms), max(ms), f"M_{i}")
        Ki[i] = model.NewIntVar(min(ks), max(ks), f"K_{i}")
        Ni[i] = model.NewIntVar(min(ns), max(ns), f"N_{i}")
        model.AddElement(idx[i], ms, Mi[i])
        model.AddElement(idx[i], ks, Ki[i])
        model.AddElement(idx[i], ns, Ni[i])

    # 6) 层间尺寸单调约束：当 SI_i=0（外积）时，m_i ≤ m_{i-1}；同理可加对 k,n 的约束
    for i in (2, 3, 4):
        model.Add(Mi[i] <= Mi[i-1]).OnlyEnforceIf(SI[i].Not())
        model.Add(Ki[i] <= Ki[i-1]).OnlyEnforceIf(SI[i].Not())
        model.Add(Ni[i] <= Ni[i-1]).OnlyEnforceIf(SI[i].Not())

    # 7) 回调函数：每有可行解就评估一次仿真
    best_time = float('inf')
    best_strat = None

    class SearchCallback(cp_model.CpSolverSolutionCallback):
        def __init__(self):
            super().__init__()

        def OnSolutionCallback(self):
            nonlocal best_time, best_strat
            # 读取当前方案
            sel_idx = {i: self.Value(idx[i]) for i in (1, 2, 3, 4)}
            blk_sel = {i: blocks[i][sel_idx[i]] for i in (1, 2, 3, 4)}

            # 构造自定义策略
            strat = MatMul_Strategy(dataflow_mode, raw_mnk, storage_formats, option=None)
            strat.L2_mnk_values       = [blk_sel[1]]
            strat.L1_mnk_values       = [blk_sel[2]]
            strat.L0_mnk_values       = [blk_sel[3]]
            strat.DFF_mnk_values      = list(blk_sel[4])
            # 注入内/外积和存储格式
            for i in (1, 2, 3, 4):
                setattr(strat, f"L{i}_block_strategy",
                        'inner' if self.Value(SI[i]) else 'outer')
                setattr(strat, f"L{i}_storage_formats",
                        (self.Value(ZL[i]), self.Value(ZR[i])))

            # 仿真评估
            sim = Simulate(strat)
            t = sim.calculate_cycles()
            if t < best_time:
                best_time = t
                best_strat = strat

    # 8) 启动求解器
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.SearchForAllSolutions(model, SearchCallback())

    return best_strat, best_time
