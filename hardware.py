class HardwareSpec:
    def __init__(self):
        # Cube 计算单元：16×16×16 FP16 矩阵乘，每周期 4096 MAC
        self.CUBE_MACS_PER_CYCLE = 4096

        # AI Core个数
        self.AI_CORE_COUNT = 24

        # 最小访存粒度 (bytes)
        self.MIN_ACCESS = {
            'Chip': 2,
            'L2': 512,
            'L1': 32,
            'L0A': 512,
            'L0B': 512,
            'L0C': 512,
            'UB': 32,        # Unify Buffer
            'SB': 2,        # Scalar Buffer
            'ABDFF': 512,    # A/B DFF 寄存器【推测是这样的，不过这个具体数值取决于华为那边的矩阵计算的padding相关信息，还得找他们问】
            'AccumDFF': 512  # Accum DFF 寄存器
        }
        self.MIN_ACCESS['DRAM'] = self.MIN_ACCESS['L2']
        self.MIN_ACCESS['EXT'] = self.MIN_ACCESS['Chip']

        # 每层的缓存容量
        self.MEM_CAPACITY = 64 * 1024**3 # Chip，910C为128GB【数据来自网络，不保证准确】
        self.L2_CAPACITY = 192 * 1024**2   # L2
        self.L1_CAPACITY = 1 * 1024**2    # L1
        self.L0A_CAPACITY = 64 * 1024     # L0A
        self.L0B_CAPACITY = 64 * 1024     # L0B
        self.L0C_CAPACITY = 256 * 1024    # L0C 缓冲
        self.UB_CAPACITY = 256 * 1024     # Unify Buffer
        self.SB_CAPACITY = 16 * 1024      # Scalar Buffer
        self.ABDFF_CAPACITY = 512         # A/B DFF 寄存器
        self.AccumDFF_CAPACITY = 512      # Accum DFF 寄存器【实际大小未知，还得去找华为要】

        # IO 带宽
        #还需要profiling数据！
        # 占位带宽bytes/cycle, 现在全都不知道
        self.EXT_DRAM_BW = 512    # 外部存储 - 芯片 DRAM
        self.DRAM_EXT_BW = 512    # 芯片 DRAM - 外部存储
        self.DRAM_L2_BW  = 1024   # 芯片 DRAM - L2 Cache
        self.L2_DRAM_BW  = 1024   # L2 Cache - 芯片 DRAM
        #DRAM→L2 UB→DRAM DRAM→L1（第一次可能不会触发缓存）
        self.IO_BW = { # 【这里应该加一个DRAM→L1 的带宽】【网上号称昇腾的主存是HBM，带宽达到1.2TB/s，但是哪一段的带宽不得而知】
            'L0C→L2':  self.MIN_ACCESS['L0C'], # 先占位
            'L2→L0C':  self.MIN_ACCESS['L0C'], # 占位
            'DRAM→L1': self.DRAM_L2_BW, # 占位
            'EXT→DRAM': self.EXT_DRAM_BW,
            'DRAM→EXT': self.DRAM_EXT_BW,
            'DRAM→L2': self.DRAM_L2_BW,
            'L2→DRAM': self.L2_DRAM_BW,
            'L2→L1': self.MIN_ACCESS['L1'],
            'L1→L0A': self.MIN_ACCESS['L0A'],
            'L1→L0B': self.MIN_ACCESS['L0B'],
            'L0A→L1': self.MIN_ACCESS['L0A'],
            'L0B→L1': self.MIN_ACCESS['L0B'],
            'L1→L0C': self.MIN_ACCESS['L0C'],
            'L0C→L1': self.MIN_ACCESS['L0C'],
            'L0C→UB': self.MIN_ACCESS['L0C'],
            'UB→L0C': self.MIN_ACCESS['L0C'],
            'AccumDFF→L0C': self.MIN_ACCESS['AccumDFF'],
            'UB→L1': self.MIN_ACCESS['L1']
        }
        
        # 暂时未知的超参数
        # L2 Cache 相联度 
        self.L2_ASSOCIATIVITY = 8
        # L2 输入输出容量划分比例
        self.L2_INPUT_RATIO = 0.8
        

HW = HardwareSpec()
