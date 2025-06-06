class MatMul_Strategy:
    def __init__(self, dataflow_mode, raw_mnk_values, raw_storage_formats):
        """
        Initialize the strategy with dataflow mode, raw MNK values, and raw storage formats.
        """
        self.dataflow_mode = dataflow_mode
        self.raw_mnk_values = raw_mnk_values  # 原始矩阵MNK值
        self.raw_storage_formats = raw_storage_formats  # 原始矩阵存储格式

        # 暂时初始化其余属性
        self.raw_block_strategy = None
        self.chip_mnk_values = None
        self.chip_storage_formats = None
        self.chip_block_strategy = None
        self.L2_mnk_values = None
        self.L2_storage_formats = None
        self.L2_block_strategy = None
        self.L1_mnk_values = None
        self.L1_storage_formats = None
        self.L1_block_strategy = None
        self.L0_mnk_values = None
        self.L0_storage_formats = None
        self.L0_block_strategy = None

        # 调用策略生成函数，补全剩余属性
        self.generate_strategy()

    def generate_strategy(self):
        """
        Generate the computation strategy for all layers.
        """
        # 策略生成函数逻辑
        pass

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
        if self.next_layer and isinstance(self.next_layer, Chip_tile):
            self.next_layer.calculate_cycles()
            # 仿真逻辑
            pass

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
        if self.next_layer and isinstance(self.next_layer, L2_tile):
            self.next_layer.calculate_cycles()
            # 仿真逻辑
            pass


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
        if self.next_layer and isinstance(self.next_layer, L1_tile):
            self.next_layer.calculate_cycles()
            # 仿真逻辑
            pass


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
        if self.next_layer and isinstance(self.next_layer, L0_tile):
            self.next_layer.calculate_cycles()
            # 仿真逻辑
            pass


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
        # 仿真逻辑
        pass
