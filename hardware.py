CUBE_MACS_PER_CYCLE = 4096  # 16×16×16 FP16 MAC

# 各级缓存/缓冲区 容量 (bytes)
L2_CAPACITY   = 32 * 1024**2
L1_CAPACITY   = 1  * 1024**2
LOA_CAPACITY  = 64 * 1024
LOB_CAPACITY  = 64 * 1024
LOC_CAPACITY  = 256 * 1024
UB_CAPACITY   = 256 * 1024
SB_CAPACITY   = 16  * 1024

# 最小访存粒度 (bytes)
MIN_ACCESS = {
    'DRAM': 1,
    'L2'  : 32,
    'L1'  : 32,
    'LOA' : 512,
    'LOB' : 128,
    'LOC' : 512,
    'UB'  : 32,
    'SB'  : 2,
}

# 访存带宽
IO_BW = {
    'DRAM→L2': 1024,
    'L2→L1'  : MIN_ACCESS['L2'],
    'L1→LOA' : MIN_ACCESS['LOA'],
    'L1→LOB' : MIN_ACCESS['LOB'],
    'LOA→L1' : MIN_ACCESS['LOA'],
    'LOB→L1' : MIN_ACCESS['LOB'],
    'L1→LOC' : MIN_ACCESS['LOC'],
    'LOC→L1' : MIN_ACCESS['LOC'],
    'LOC→UB' : MIN_ACCESS['UB'],
    'UB→LOC' : MIN_ACCESS['UB'],
}