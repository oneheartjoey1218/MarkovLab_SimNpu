from dataclasses import dataclass

@dataclass(frozen=True)
class HardwareSpec:
    def __init__(self):
        # Cube 计算单元：16×16×16 FP16 矩阵乘，每周期 4096 MAC
        self.CUBE_MACS_PER_CYCLE = 4096

        # AI Core个数
        self.AI_CORE_COUNT = 24

        # 最小访存粒度 (bytes)
        self.MIN_ACCESS = {
            'Chip': 2,
            'L2': 2,
            'L1': 32,
            'LO': 512,       # LOA/LOB 合并
            'LOC': 512,
            'UB': 32,        # Unify Buffer
            'ABDFF': 512,    # A/B DFF 寄存器【推测是这样的，不过这个具体数值取决于华为那边的矩阵计算的padding相关信息，还得找他们问】
            'AccumDFF': 512  # Accum DFF 寄存器
        }

        # 每层的缓存容量
        self.MEM_CAPACITY = 128 * 1024**3 # Chip，910C为128GB【数据来自网络，不保证准确】
        self.L2_CAPACITY = 32 * 1024**2   # L2
        self.L1_CAPACITY = 1 * 1024**2    # L1
        self.LOA_CAPACITY = 64 * 1024     # LOA
        self.LOB_CAPACITY = 64 * 1024     # LOB
        self.LOC_CAPACITY = 256 * 1024    # LOC 缓冲
        self.UB_CAPACITY = 256 * 1024     # Unify Buffer
        self.SB_CAPACITY = 16 * 1024      # Scalar Buffer
        self.ABDFF_CAPACITY = 512         # A/B DFF 寄存器
        self.AccumDFF_CAPACITY = 512      # Accum DFF 寄存器【实际大小未知，还得去找华为要】

        # IO 带宽
        self.IO_BW = { # 【这里应该加一个DRAM→L1 的带宽】【网上号称昇腾的主存是HBM，带宽达到1.2TB/s，但是哪一段的带宽不得而知】
            'DRAM→L2': 1024,
            'L2→L1': self.MIN_ACCESS['L1'],
            'L1→LOA': self.MIN_ACCESS['LO'],
            'L1→LOB': self.MIN_ACCESS['LO'],
            'LOA→L1': self.MIN_ACCESS['LO'],
            'LOB→L1': self.MIN_ACCESS['LO'],
            'L1→LOC': self.MIN_ACCESS['LOC'],
            'LOC→L1': self.MIN_ACCESS['LOC'],
            'LOC→UB': self.MIN_ACCESS['LOC'],
            'UB→LOC': self.MIN_ACCESS['LOC'],
        }

HW = HardwareSpec()
