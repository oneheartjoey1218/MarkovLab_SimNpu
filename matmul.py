from init import (
    L2_CAPACITY,
    L1_CAPACITY,
    LOA_CAPACITY,
    LOB_CAPACITY,
    LOC_CAPACITY,
)

from modules import (
    Device,
    ComputeModule,
    IOModule,
    MemoryModule
)

_device = Device(
    compute=ComputeModule(),
    io=IOModule(),
    memory=MemoryModule()
)

# 划分矩阵乘法的策略
def split_blocks(blocks, max_elems):
    """
    对每个 (M,N,K) 二分拆分，直到 M*N*K <= max_elems。
    优先沿最大维度二分。
    """
    out = []
    for M, N, K in blocks:
        if M * N * K <= max_elems:
            out.append((M, N, K))
        else:
            if M >= N and M >= K:
                m2 = M // 2
                out += split_blocks([(m2, N, K), (M-m2, N, K)], max_elems)
            elif N >= K:
                n2 = N // 2
                out += split_blocks([(M, n2, K), (M, N-n2, K)], max_elems)
            else:
                k2 = K // 2
                out += split_blocks([(M, N, k2), (M, N, K-k2)], max_elems)
    return out


class MatMul_Strategy:
    def __init__(self, dataflow_mode, raw_mnk_values, raw_storage_formats):
        """
        Initialize the strategy with dataflow mode, raw MNK values, and raw storage formats.
        """
        self.dataflow_mode = dataflow_mode
        self.raw_mnk_values = raw_mnk_values  # 原始矩阵MNK值
        self.raw_storage_formats = raw_storage_formats  # 原始矩阵存储格式
        self.elem_bytes = 2 if 'fp16' in raw_storage_formats else 4

        # 暂时初始化其余属性
        # self.raw_block_strategy = None
        self.chip_mnk_values = None
        # self.chip_storage_formats = None
        # self.chip_block_strategy = None
        self.L2_mnk_values = None
        # self.L2_storage_formats = None
        # self.L2_block_strategy = None
        self.L1_mnk_values = None
        # self.L1_storage_formats = None
        # self.L1_block_strategy = None
        self.L0_mnk_values = None
        # self.L0_storage_formats = None
        # self.L0_block_strategy = None

        # 调用策略生成函数，补全剩余属性
        self.generate_strategy()

    def generate_strategy(self):
        """
        分块
        1. Chip无需拆分
        2. L2 - L2_CAPACITY
        3. L1 - L1_CAPACITY
        4. L0(L0=LOA/LOB) - LOx_CAPACITY
        """
        
        # 1) Chip
        self.chip_mnk_values = list(self.raw_mnk_values)
        
        # 2) L2
        max_L2 = L2_CAPACITY // self.elem_bytes
        self.L2_mnk_values = split_blocks(self.chip_mnk_values, max_L2)

        # 3) L1
        max_L1 = L1_CAPACITY // self.elem_bytes
        self.L1_mnk_values = []
        for block in self.L2_mnk_values:
            self.L1_mnk_values += split_blocks([block], max_L1)
            
        # 4) L0 (使用 LOA + LOB 容量中较小者)
        max_LOA = LOA_CAPACITY // self.elem_bytes
        max_LOB = LOB_CAPACITY // self.elem_bytes
        max_LO  = min(max_LOA, max_LOB)
        self.L0_mnk_values = []
        for block in self.L1_mnk_values:
            self.L0_mnk_values += split_blocks([block], max_LO)

    def calculate_cycles(self):
        # 本层无计算
        return 0
    
    
class Simulate:
    """
    从原始矩阵分块到芯片上
    """
    def __init__(self, M, N, K, M_tile, N_tile, K_tile, block_strategy, storage_format, next_layer: 'Chip_tile' = None):
        
        # 决策变量
        self.M_tile = M_tile
        self.N_tile = N_tile
        self.K_tile = K_tile
        self.block_strategy = block_strategy # 分块策略；内积/外积
        self.storage_format = storage_format # 存储格式；行主序/列主序

        # 约束
        self.M = M # 原始矩阵大小M
        self.N = N # 原始矩阵大小N
        self.K = K # 原始矩阵大小K
        # 最外层没有容量限制

        # 下层结构
        self.next_layer = next_layer  # next_layer是Chip_tile实例

        # 向上回传数据
        self.K_reduction_cycle_count = None
        self.K_reduction_io_count = None
        self.M_K_io_cycle_count = None
        self.K_N_io_cycle_count = None
        self.M_N_io_cycle_count = None
        self.compute_cycle_count = None

    def calculate_cycles(self):
        """
        Calculate the cycles for the current layer based on the next layer's metrics.
        """
        total = 0.0
        if self.next_layer and isinstance(self.next_layer, Chip_tile):
            self.next_layer.calculate_cycles()
            # 仿真逻辑
        
        # Off-chip to L2
        '''
        size_A = self.M_tile * self.K_tile * 2
        size_B = self.K_tile * self.N_tile * 2
        total += _device.io.load(size_A, 'DRAM', 'L2')
        total += _device.io.load(size_B, 'DRAM', 'L2')
        
        # IO
        self.M_K_io_cycle_count = size_A
        self.K_N_io_cycle_count = size_B
        return total
        '''
class Chip_tile:
    """
    从芯片分块到L2上
    """
    def __init__(self, size, M, N, K, min_load_size, M_tile, N_tile, K_tile, block_strategy, storage_format, next_layer: 'L2_tile' = None):
        # 决策变量
        self.M_tile = M_tile
        self.N_tile = N_tile
        self.K_tile = K_tile
        self.block_strategy = block_strategy # 分块策略；内积/外积
        self.storage_format = storage_format # 存储格式；行主序/列主序

        # 约束
        self.size = size # 芯片主存结构存储容量
        self.min_load_size = min_load_size # 芯片主存最小访存大小
        self.M = M
        self.N = N
        self.K = K

        # 下层结构
        self.next_layer = next_layer  # next_layer是L2_tile实例

        # 向上回传数据
        self.K_reduction_cycle_count = None
        self.K_reduction_io_count = None
        self.M_K_io_cycle_count = None
        self.K_N_io_cycle_count = None
        self.M_N_io_cycle_count = None
        self.compute_cycle_count = None

    def calculate_cycles(self):
        """
        Calculate the cycles for the current layer based on the next layer's metrics.
        """
        total = 0.0
        if self.next_layer and isinstance(self.next_layer, L2_tile):
            self.next_layer.calculate_cycles()
            # 仿真逻辑
            
        # L2 to L1
        '''
        size_A = self.M_tile * self.K_tile * 2
        size_B = self.K_tile * self.N_tile * 2
        total += _device.io.load(size_A, 'L2', 'L1')
        total += _device.io.load(size_B, 'L2', 'L1')
        return total
        '''
        
class L2_tile:
    """
    从L2分块到每个L1上，910C有24个L1_tile
    """
    def __init__(self, size, M, N, K, min_load_size, M_tile, N_tile, K_tile, block_strategy, storage_format, L1_num, next_layers: 'list' = None):
        # 决策变量
        self.M_tile = M_tile
        self.N_tile = N_tile
        self.K_tile = K_tile
        self.block_strategy = block_strategy # 分块策略；内积/外积
        self.storage_format = storage_format # 存储格式；行主序/列主序

        # 约束
        self.size = size # L2-buffer结构存储容量
        self.min_load_size = min_load_size # L2-buffer最小访存大小
        self.M = M
        self.N = N
        self.K = K
        
        # 下层结构
        self.L1_num = L1_num # L1_num是L1_tile的数量
        self.next_layers = next_layers  # next_layer是一系列L1_tile实例

        # 向上回传数据
        self.K_reduction_cycle_count = None
        self.K_reduction_io_count = None
        self.M_K_io_cycle_count = None
        self.K_N_io_cycle_count = None
        self.M_N_io_cycle_count = None
        self.compute_cycle_count = None

    def calculate_cycles(self):
        """
        Calculate the cycles for the current layer based on the next layer's metrics.
        注意L2 buffer是有一个L2-cache在旁边，这个我们没法主动控制，但是在写代码时要把cache带来的影响考虑进去
        """
        total = 0
        if self.next_layer and isinstance(self.next_layer, L1_tile):
            self.next_layer.calculate_cycles()
            # 仿真逻辑
        
        # L1 to L0
        '''
        size_A = self.M_tile * self.K_tile * 2
        size_B = self.K_tile * self.N_tile * 2
        total += _device.io.load(size_A, 'L1', 'LO')
        total += _device.io.load(size_B, 'L1', 'LO')
        return total
        '''


class L1_tile:
    """
    从L1分块到L0上。注意这只是L1 buffer结构，unified buffer可能需要新开一个类
    """
    def __init__(self, size, min_load_size, M, N, K, M_tile, N_tile, K_tile, block_strategy, storage_format, L0_id, next_layers: 'list' = None):
        # 决策变量
        self.M_tile = M_tile
        self.N_tile = N_tile
        self.K_tile = K_tile
        self.block_strategy = block_strategy # 分块策略；内积/外积
        self.storage_format = storage_format # 存储格式；行主序/列主序

        # 约束
        self.size = size # L1 buffer结构存储容量
        self.min_load_size= min_load_size # L1 buffer最小访存大小
        self.M = M
        self.N = N
        self.K = K
        
        # 下层结构
        self.L0_id = L0_id # L0_id是L0_A、L0_B、L0_C的标识符
        self.next_layers = next_layers  # next_layers是L0_tile实例

        # 向上回传数据
        self.K_reduction_cycle_count = None
        self.K_reduction_io_count = None
        self.M_K_io_cycle_count = None
        self.K_N_io_cycle_count = None
        self.M_N_io_cycle_count = None
        self.compute_cycle_count = None

    def calculate_cycles(self):
        """
        Calculate the cycles for the current layer based on the next layer's metrics.
        """
        total = 0.0
        if self.next_layer and isinstance(self.next_layer, L0_tile):
            self.next_layer.calculate_cycles()
            # 仿真逻辑
        
        # Compute L0 and Write Back
        size_C = self.M_tile * self.N_tile * 2
        total += _device.compute.compute(self.M_tile, self.N_tile, self.K_tile)
        total += _device.io.store(size_C, 'LO', 'L1')
        return total


class L0_tile:
    """
    L0上的结构运算。注意，由于L0_A、L0_B、L0_C是分开的，所以L0_tile的实例化需要三个不同的实例来表示。
    """
    def __init__(self, size, M, N, K, min_load_size_1, min_load_size_2, storage_format):
        # 约束
        self.size = size # L0结构存储容量
        self.min_load_size_1 = min_load_size_1 # L0的最小访存大小
        self.min_load_size_2 = min_load_size_2 # L0的最小访存大小
        self.M = M
        self.N = N
        self.K = K        
        self.storage_format = storage_format # 存储格式，在本层还有点用

        # 向上回传数据
        self.K_reduction_cycle_count = None
        self.K_reduction_io_count = None
        self.M_K_io_cycle_count = None
        self.K_N_io_cycle_count = None
        self.M_N_io_cycle_count = None
        self.compute_cycle_count = None

    def calculate_cycles(self):
        """
        Calculate the cycles for the current layer.
        """
        total = 0
        size_A = self.M * self.K * 2
        size_B = self.K * self.N * 2
        size_C = self.M * self.N * 2
        
        # 1) LO -> SM
        cycles += _device.io.load(size_A, 'LO', 'UB')
        cycles += _device.io.load(size_B, 'LO', 'UB')
        
        # 2) Compute
        cycles += _device.compute.compute(self.M, self.N, self.K)
        
        # 3) UB -> LO (写回)
        cycles += _device.io.store(size_C, 'UB', 'LO')
        return cycles

