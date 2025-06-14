from modules import (
    Device,
    ComputeModule,
    IOModule,
    MemoryModule
)

from hardware import HardwareSpec, HW # 全局硬件实例

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
    def __init__(self, dataflow_mode, raw_mnk_values, raw_storage_formats, option, chip_type:'HardwareSpec' = HW):
        """
        Initialize the strategy with dataflow mode, raw MNK values, and raw storage formats.
        """
        self.dataflow_mode = dataflow_mode
        self.raw_mnk_values = raw_mnk_values  # 原始矩阵MNK值，数据格式为列表：[M, N, K]
        self.raw_storage_format = raw_storage_formats  # 原始矩阵存储格式，数据格式为字符串列表：[左矩阵格式,右矩阵格式]，两个格式的选择为0和1（或者False和True，True表示特殊处理格式）
        self.chip_type = chip_type  # 芯片类型，910C/910B/910A等
        self.elem_bytes = 2 if 'fp16' in raw_storage_formats else 4 # 这行是哪来的？我不知道

        # 暂时初始化其余属性
        self.chip_mnk_values = None
        self.chip_storage_formats = None
        self.chip_block_strategy = None # 分块策略，包括内积法（沿M轴和N轴切分）、外积法（沿K轴切分）、内外积结合（都切分，得算两次）
        
        self.L1_mnk_values = None       # L1与UB通用
        self.L1_storage_formats = None
        self.L1_block_strategy = None
        
        self.L0_mnk_values = None       # L0A/B与L0C通用
        self.L0_storage_formats = None
        self.L0_block_strategy = None

        self.DFF_mnk_values = None
        self.DFF_storage_formats = None # ABDFF与AccumDFF通用
        self.DFF_block_strategy = None

        # 调用策略生成函数，补全剩余属性
        if option == 'ascend':
            # Ascend策略
            self.ascend_strategy()
        elif option == 'best':
            # 最佳策略搜索
            self.generate_strategy()

    def ascend_strategy(self):
        """
        Ascend策略生成函数，先实现这个，用简单的逻辑给上面的其余属性赋值
        """
        pass # 其他的还没算好具体的值
        self.DFF_mnk_values = [16,16,16]
        raise NotImplementedError("Ascend strategy generation is not implemented yet.")

    def generate_strategy(self):
        """
        最佳分块策略
        1. Chip无需拆分
        2. L2 - L2_CAPACITY
        3. L1 - L1_CAPACITY
        4. L0(L0=LOA/LOB) - LOx_CAPACITY
        """
        
        # 1) Chip
        self.chip_mnk_values = list(self.raw_mnk_values)
        
        # 2) L2
        max_L2 = self.chip_type.L2_CAPACITY // self.elem_bytes
        self.L2_mnk_values = split_blocks(self.chip_mnk_values, max_L2)

        # 3) L1
        max_L1 = self.chip_type.L1_CAPACITY // self.elem_bytes
        self.L1_mnk_values = []
        for block in self.L2_mnk_values:
            self.L1_mnk_values += split_blocks([block], max_L1)
            
        # 4) L0 (使用 LOA + LOB 容量中较小者)
        max_LOA = self.chip_type.LOA_CAPACITY // self.elem_bytes
        max_LOB = self.chip_type.LOB_CAPACITY // self.elem_bytes
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
    def __init__(self, strategy: 'MatMul_Strategy' = None):#, next_layer: 'Chip_tile' = None):
        """
            strategy先在外部调用matmul_strategy生成，生成策略后传入Simulate类的实例中
        """
        # 下层结构初始化
        self.next_layer = Chip_tile(strategy=strategy) # next_layer是Chip_tile实例，把策略传递进去

        # 决策变量
        self.M_tile = strategy.chip_mnk_values[0]
        self.N_tile = strategy.chip_mnk_values[1]
        self.K_tile = strategy.chip_mnk_values[2]
        self.block_strategy = strategy.chip_block_strategy # 分块策略；内积/外积
        self.storage_formats = strategy.chip_storage_formats # 存储格式；行主序/列主序

        # 约束
        self.M = strategy.raw_mnk_values[0] # 原始矩阵大小M
        self.N = strategy.raw_mnk_values[1] # 原始矩阵大小N
        self.K = strategy.raw_mnk_values[2] # 原始矩阵大小K
        # 最外层没有容量限制

        # 向上回传数据
        self.K_reduction_cycle_count = None # 结果矩阵拼接耗时
        #self.K_reduction_io_count = None   # 停用
        self.M_K_io_cycle_count = None      # 左矩阵传入耗时
        self.K_N_io_cycle_count = None      # 右矩阵传入耗时
        self.M_N_io_cycle_count = None      # 结果矩阵传出耗时
        self.compute_cycle_count = None     # 内层计算耗时        

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
    从芯片分块到 L1 上，910C有24个L1_tile和24个UB_tile。这里的逻辑要仔细想清楚，包括往L1的io读时间、从UB回来的io写时间、L2 cache机制等因素
    """
    def __init__(self, strategy: 'MatMul_Strategy' = None):
        # 下层结构数量与初始化
        self.L1_num = strategy.chip_type.AI_CORE_COUNT     # L1_num是L1_tile的数量，来自device信息超参数
        self.next_layers_L1 = [
            L1_tile(id=i, strategy=strategy) for i in range(self.L1_num)
        ] # next_layers是L1_tile实例的列表，数量为24个（910C有24个AI Core），并把策略传递进去
        self.next_layers_UB = [
            UB_tile(id=i, strategy=strategy) for i in range(self.L1_num)
        ]# 还要初始化UB_tile，数量与L1_tile一致
        
        # 决策变量
        self.M_tile = strategy.L1_mnk_values[0]
        self.N_tile = strategy.L1_mnk_values[1]
        self.K_tile = strategy.L1_mnk_values[2]
        self.block_strategy = strategy.L1_block_strategy   # 分块策略；内积/外积
        self.storage_formats = strategy.L1_storage_formats # 存储格式；行主序/列主序

        # 约束
        self.size = strategy.chip_type.MEM_CAPACITY                # 芯片主存结构存储容量
        self.min_load_size = strategy.chip_type.MIN_ACCESS['Chip'] # 芯片主存最小访存大小
        self.M = strategy.chip_mnk_values[0] # 芯片层矩阵大小M
        self.N = strategy.chip_mnk_values[1] # 芯片层矩阵大小N
        self.K = strategy.chip_mnk_values[2] # 芯片层矩阵大小K

        # 向上回传数据
        self.K_reduction_cycle_count = None # 结果矩阵拼接耗时
        #self.K_reduction_io_count = None
        self.M_K_io_cycle_count = None      # 左矩阵传入耗时
        self.K_N_io_cycle_count = None      # 右矩阵传入耗时
        self.M_N_io_cycle_count = None      # 结果矩阵传出耗时
        self.compute_cycle_count = None     # 内层计算耗时

    def calculate_cycles(self):
        """
        Calculate the cycles for the current layer based on the next layer's metrics.
        需要考虑分块策略（在最前面用self.block_strategy区分计算路径）
        需要考虑 L1_tile 的数量
        需要考虑cache hit和cache miss的情况
        """
        total = 0.0
        if self.next_layer and isinstance(self.next_layer, L1_tile):
            self.next_layer.calculate_cycles()
            # 仿真逻辑
            
        # Chip to L1
        '''
        size_A = self.M_tile * self.K_tile * 2
        size_B = self.K_tile * self.N_tile * 2
        total += _device.io.load(size_A, 'L2', 'L1')
        total += _device.io.load(size_B, 'L2', 'L1')
        return total
        '''

# 从这里开始，是Chip→L1→L0→ABDFF的分块结构，负责数据向矩阵计算Core的搬入
class L1_tile:
    """
    从 L1 分块到 L0 上。注意一个L1对应两个L0，因此在写约束的时候注意要把约束除以2；L2 cache机制也要考虑，不过不要和Chip层重复计算
    """
    def __init__(self, id, strategy: 'MatMul_Strategy' = None):
        # 本层结构编号
        self.L1_id = id
        
        # 下层结构初始化
        self.L0_num = 2                 # L0_num是L0_tile的数量，理论上也来自超参数，不过这边就直接在这初始化了
        self.next_layers = [
            L0_tile(id=i, strategy=strategy) for i in range(self.L0_num)
        ] # next_layers是L0_tile实例的列表，这里只有L0A和L0B共2个实例，并把策略传递进去
        
        # 决策变量
        self.M_tile = strategy.L0_mnk_values[0]
        self.N_tile = strategy.L0_mnk_values[1]
        self.K_tile = strategy.L0_mnk_values[2]
        self.block_strategy = strategy.L0_block_strategy   # 分块策略；内积/外积
        self.storage_formats = strategy.L0_storage_formats # 存储格式；行主序/列主序

        # 约束
        self.size = strategy.chip_type.L1_CAPACITY              # L1 buffer结构存储容量
        self.min_load_size= strategy.chip_type.MIN_ACCESS['L1'] # L1 buffer最小访存大小
        self.M = strategy.L1_mnk_values[0] # L1 buffer层矩阵大小M
        self.N = strategy.L1_mnk_values[1] # L1 buffer层矩阵大小N
        self.K = strategy.L1_mnk_values[2] # L1 buffer层矩阵大小K        

        # 向上回传数据
        self.K_reduction_cycle_count = None # 结果矩阵拼接耗时
        #self.K_reduction_io_count = None
        self.M_K_io_cycle_count = None      # 左矩阵传入耗时
        self.K_N_io_cycle_count = None      # 右矩阵传入耗时
        self.M_N_io_cycle_count = None      # 结果矩阵传出耗时
        self.compute_cycle_count = None     # 内层计算耗时

    def calculate_cycles(self):
        """
        Calculate the cycles for the current layer based on the next layer's metrics.
        需要考虑分块策略（在最前面用self.block_strategy区分计算路径）
        """
        total = 0.0
        if self.next_layer and isinstance(self.next_layer, L0_tile):
            self.next_layer.calculate_cycles()
            # 仿真逻辑
        
        # L1 to L0
        size_C = self.M_tile * self.N_tile * 2
        total += _device.compute.compute(self.M_tile, self.N_tile, self.K_tile)
        total += _device.io.store(size_C, 'LO', 'L1')
        return total

class L0_tile:
    """
    从 L0 分块到寄存器上。注意，由于L0_A/B、L0_C是分开的，L0_C需要单开一个类
    """
    def __init__(self, id, strategy: 'MatMul_Strategy' = None):
        # 本层结构编号
        self.L0_id = id
        
        # 下层结构初始化
        self.next_layer = AB_DFF_tile(strategy=strategy) # next_layer是DFF_tile实例，把策略传递进去

        # 决策变量
        self.M_tile = strategy.DFF_mnk_values[0]
        self.N_tile = strategy.DFF_mnk_values[1]
        self.K_tile = strategy.DFF_mnk_values[2]
        self.block_strategy = strategy.DFF_block_strategy   # 分块策略；内积/外积【弄清楚分块策略对应的层的名称】
        self.storage_formats = strategy.DFF_storage_formats # 存储格式；行主序/列主序
        
        # 约束
        self.size = strategy.chip_type.LOA_CAPACITY              # L0结构存储容量（L0A和L0B是一致的）
        self.min_load_size = strategy.chip_type.MIN_ACCESS['LO'] # L0的最小访存大小
        self.M = strategy.L0_mnk_values[0] # L0层矩阵大小M
        self.N = strategy.L0_mnk_values[1] # L0层矩阵大小N
        self.K = strategy.L0_mnk_values[2] # L0层矩阵大小K

        # 向上回传数据
        self.K_reduction_cycle_count = None # 结果矩阵拼接耗时
        #self.K_reduction_io_count = None
        self.M_K_io_cycle_count = None      # 左矩阵传入耗时
        self.K_N_io_cycle_count = None      # 右矩阵传入耗时
        self.M_N_io_cycle_count = None      # 结果矩阵传出耗时
        self.compute_cycle_count = None     # 内层计算耗时

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

class AB_DFF_tile:
    """
    寄存器上的结构运算。注意在计算compute_cycle_count的时候，注意在AB_DFF_tile与Accum_DFF_tile之间的不要重复计算
    """
    def __init__(self, strategy: 'MatMul_Strategy' = None):
        # 约束
        self.size = strategy.chip_type.ABDFF_CAPACITY               # DFF结构存储容量
        self.min_load_size = strategy.chip_type.MIN_ACCESS['ABDFF'] # DFF的最小访存大小
        self.M = strategy.DFF_mnk_values[0] # DFF层矩阵大小M
        self.N = strategy.DFF_mnk_values[1] # DFF层矩阵大小N
        self.K = strategy.DFF_mnk_values[2] # DFF层矩阵大小K
        self.storage_formats = strategy.DFF_storage_formats # 存储格式，在本层可能有点用

        # 向上回传数据
        self.K_reduction_cycle_count = None # 结果矩阵拼接耗时
        #self.K_reduction_io_count = None
        self.M_K_io_cycle_count = None      # 左矩阵传入耗时
        self.K_N_io_cycle_count = None      # 右矩阵传入耗时
        self.M_N_io_cycle_count = None      # 结果矩阵传出耗时
        self.compute_cycle_count = None     # 内层计算耗时

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

# 从这里开始，是Chip←UB←L0C→AccumDFF的分块结构，负责数据从矩阵计算Core的搬出
class UB_tile:
    """
    从 UnifyBuffer 分块到 L0C 上
    """
    def __init__(self, id, strategy: 'MatMul_Strategy' = None):
        # 本层结构编号
        self.UB_id = id
        
        # 下层结构初始化
        self.next_layer = L0C_tile(strategy=strategy) # next_layer是L0C_tile实例，把策略传递进去
        
        # 决策变量，决策变量与L1_tile相同
        self.M_tile = strategy.L0_mnk_values[0]
        self.N_tile = strategy.L0_mnk_values[1]
        self.K_tile = strategy.L0_mnk_values[2]
        self.block_strategy = strategy.L0_block_strategy   # 分块策略；内积/外积
        self.storage_formats = strategy.L0_storage_formats # 存储格式；行主序/列主序

        # 约束，但是约束与L1_tile不同
        self.size = strategy.chip_type.UB_CAPACITY              # L1 buffer结构存储容量
        self.min_load_size= strategy.chip_type.MIN_ACCESS['UB'] # L1 buffer最小访存大小
        self.M = strategy.L1_mnk_values[0] # L1 buffer层矩阵大小M【这仨是与L1_tile共用的】
        self.N = strategy.L1_mnk_values[1] # L1 buffer层矩阵大小N
        self.K = strategy.L1_mnk_values[2] # L1 buffer层矩阵大小K        

        # 向上回传数据
        self.K_reduction_cycle_count = None # 结果矩阵拼接耗时
        #self.K_reduction_io_count = None
        self.M_K_io_cycle_count = None      # 左矩阵传入耗时
        self.K_N_io_cycle_count = None      # 右矩阵传入耗时
        self.M_N_io_cycle_count = None      # 结果矩阵传出耗时
        self.compute_cycle_count = None     # 内层计算耗时

    def calculate_cycles(self):
        """
        Calculate the cycles for the current layer based on the next layer's metrics.
        需要考虑分块策略（在最前面用self.block_strategy区分计算路径）
        """
        total = 0.0
        if self.next_layer and isinstance(self.next_layer, L0_tile):
            self.next_layer.calculate_cycles()
            # 仿真逻辑
        
        # L1 to L0
        size_C = self.M_tile * self.N_tile * 2
        total += _device.compute.compute(self.M_tile, self.N_tile, self.K_tile)
        total += _device.io.store(size_C, 'LO', 'L1')
        return total

class L0C_tile:
    """
    从L0分块到寄存器上。注意，由于L0_A/B、L0_C是分开的，L0_C需要单开一个类
    """
    def __init__(self, strategy: 'MatMul_Strategy' = None):
        # 下层结构初始化
        self.next_layer = Accum_DFF_tile(strategy=strategy) # next_layer是Accum_DFF_tile实例，把策略传递进去

        # 决策变量
        self.M_tile = strategy.DFF_mnk_values[0]
        self.N_tile = strategy.DFF_mnk_values[1]
        self.K_tile = strategy.DFF_mnk_values[2]
        self.block_strategy = strategy.DFF_block_strategy   # 分块策略；内积/外积【弄清楚分块策略对应的层的名称】
        self.storage_formats = strategy.DFF_storage_formats # 存储格式；行主序/列主序
        
        # 约束
        self.size = strategy.chip_type.LOC_CAPACITY               # L0结构存储容量（L0A和L0B是一致的）
        self.min_load_size = strategy.chip_type.MIN_ACCESS['LOC'] # L0的最小访存大小
        self.M = strategy.L0_mnk_values[0] # L0层矩阵大小M
        self.N = strategy.L0_mnk_values[1] # L0层矩阵大小N
        self.K = strategy.L0_mnk_values[2] # L0层矩阵大小K

        # 向上回传数据
        self.K_reduction_cycle_count = None # 结果矩阵拼接耗时
        #self.K_reduction_io_count = None
        self.M_K_io_cycle_count = None      # 左矩阵传入耗时
        self.K_N_io_cycle_count = None      # 右矩阵传入耗时
        self.M_N_io_cycle_count = None      # 结果矩阵传出耗时
        self.compute_cycle_count = None     # 内层计算耗时

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

class Accum_DFF_tile:
    """
    寄存器上的结构运算
    """
    def __init__(self, strategy: 'MatMul_Strategy' = None):
        # 约束
        self.size = strategy.chip_type.AccumDFF_CAPACITY               # DFF结构存储容量
        self.min_load_size = strategy.chip_type.MIN_ACCESS['AccumDFF'] # DFF的最小访存大小
        self.M = strategy.DFF_mnk_values[0] # DFF层矩阵大小M
        self.N = strategy.DFF_mnk_values[1] # DFF层矩阵大小N
        self.K = strategy.DFF_mnk_values[2] # DFF层矩阵大小K
        self.storage_formats = strategy.DFF_storage_formats # 存储格式，在本层可能有点用

        # 向上回传数据
        self.K_reduction_cycle_count = None # 结果矩阵拼接耗时
        #self.K_reduction_io_count = None
        self.M_K_io_cycle_count = None      # 左矩阵传入耗时
        self.K_N_io_cycle_count = None      # 右矩阵传入耗时
        self.M_N_io_cycle_count = None      # 结果矩阵传出耗时
        self.compute_cycle_count = None     # 内层计算耗时

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