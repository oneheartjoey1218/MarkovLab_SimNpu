from matmul import *

# Cube 计算单元：16×16×16 FP16 矩阵乘，每周期 4096 MAC
CUBE_MACS_PER_CYCLE = 4096

# 最小访存粒度 (bytes)
MIN_ACCESS = {
    'L1'  : 32,
    'LO'  : 512,   # LOA/LOB 合并
    'LOC' : 512,
}

# 每层的缓存容量
L2_CAPACITY = 32 * 1024**2 # L2
L1_CAPACITY = 1 * 1024**2 # L1
LOA_CAPACITY = 64 * 1024 # LOA
LOB_CAPACITY = 64 * 1024 # LOB
LOC_CAPACITY = 256 * 1024 # LOC 缓冲
UB_CAPACITY  = 256 * 1024 # Unify Buffer
SB_CAPACITY  = 16 * 1024 # Scalar Buffer

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


        
def main():
    # 测试用例
    pass

if __name__ == "__main__":
    main()