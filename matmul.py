from modules import (
    Device,
    ComputeModule,
    IOModule,
    MemoryModule,
    L2_CACHE_MGR
)

from hardware import HardwareSpec, HW # 全局硬件实例

_device = Device(
    compute=ComputeModule(),
    io=IOModule(),
    memory=MemoryModule()
)

from math import ceil
import numpy as np
import copy

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
    L2_mnk_values       = None
    L1_mnk_values       = None
    L0_mnk_values       = None
    DFF_mnk_values      = None

    L2_block_strategy   = None
    L1_block_strategy   = None
    L0_block_strategy   = None
    DFF_block_strategy  = None

    L2_storage_formats  = None
    L1_storage_formats  = None
    L0_storage_formats  = None
    DFF_storage_formats = None
    
    def __init__(self, dataflow_mode, raw_mnk_values, raw_storage_formats, option=None, chip_type:'HardwareSpec' = HW):
        """
        Initialize the strategy with dataflow mode, raw MNK values, and raw storage formats.
        """
        self.dataflow_mode = dataflow_mode
        self.raw_mnk_values = raw_mnk_values  # 原始矩阵MNK值，数据格式为列表：[M, N, K]
        self.raw_storage_format = raw_storage_formats  # 原始矩阵存储格式，数据格式为字符串列表：[左矩阵格式,右矩阵格式]，两个格式的选择为0和1（或者False和True，True表示特殊处理格式）
        self.chip_type = chip_type  # 芯片类型，910C/910B/910A等
        self.elem_bytes = 2 if 'fp16' in raw_storage_formats else 4 # 这行是哪来的？我不知道

        # 暂时初始化其余属性
        self.chip_mnk_values = list(self.raw_mnk_values)  # 芯片分块的MNK值，数据格式为列表：[M, N, K]
        self.chip_storage_formats = list(self.raw_storage_format) # 字符串列表：[左矩阵格式,右矩阵格式]
        self.chip_block_strategy = None # 分块策略，包括内积法（沿M轴和N轴切分）、外积法（沿K轴切分）、内外积结合（都切分，得算两次）
        if option is None:
            return
        
        self.L1_mnk_values = None       # L1与UB通用
        self.L1_storage_formats = None # 字符串列表：[左矩阵格式,右矩阵格式]
        self.L1_block_strategy = None
        
        self.L0_mnk_values = None       # L0A/B与L0C通用
        self.L0_storage_formats = None # 单个元素：左矩阵格式 或 右矩阵格式，分别用（0，1）表示（常规，特殊）
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
        4. L0(L0=L0A/L0B) - L0x_CAPACITY
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
            
        # 4) L0 (使用 L0A + L0B 容量中较小者)
        max_L0A = self.chip_type.L0A_CAPACITY // self.elem_bytes
        max_L0B = self.chip_type.L0B_CAPACITY // self.elem_bytes
        max_L0  = min(max_L0A, max_L0B)
        self.L0_mnk_values = []
        for block in self.L1_mnk_values:
            self.L0_mnk_values += split_blocks([block], max_L0)
            
        # DFF
        self.DFF_mnk_values = list(self.L0_mnk_values[0])  # DFF与L0通用

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
        # 保存 strategy 对象!!!
        self.strategy = strategy

        # 下层结构初始化
        self.next_layer = Chip_tile(strategy=strategy) # next_layer是Chip_tile实例，把策略传递进去

        # 决策变量
        self.M_tile = strategy.chip_mnk_values[0]
        self.N_tile = strategy.chip_mnk_values[1]
        self.K_tile = strategy.chip_mnk_values[2]
        self.block_strategy = strategy.chip_block_strategy # 分块策略；内积/外积
        self.storage_formats = strategy.chip_storage_formats # 存储格式；行主序/列主序

        # 约束 # 注意，这里同时要分左右矩阵，所以存储约束需要除以2
        self.size = strategy.chip_type.MEM_CAPACITY                   # 芯片主存结构存储容量
        # self.min_load_size = strategy.chip_type.MIN_ACCESS['Chip']  # 芯片主存最小访存大小【疑问：这里有没有最小访存大小？】
        self.M = strategy.raw_mnk_values[0] # 原始矩阵大小M
        self.N = strategy.raw_mnk_values[1] # 原始矩阵大小N
        self.K = strategy.raw_mnk_values[2] # 原始矩阵大小K

        # 向上回传数据
        self.K_reduction_latency = None # 结果矩阵拼接耗时
        #self.K_reduction_io_count = None   # 停用
        self.M_K_io_latency = None      # 左矩阵传入耗时
        self.K_N_io_latency = None      # 右矩阵传入耗时
        self.M_N_io_latency = None      # 结果矩阵传出耗时
        self.compute_latency = None     # 内层计算耗时
        self.mem_alloc_read_latency = None   # 内存分配耗时
        self.mem_alloc_write_latency = None  # 内存分配耗时
       
    # 按流水线方式计算周期数，占位 
    def calculate_pipelined_cycles(self) -> int:
        self.calculate_cycles()
        scheduler = PipelineScheduler(self)
        return scheduler.simulate()


    def calculate_cycles(self):
        """
        计算当前层的执行周期数
        """
        # 1. 加载查找表(LUT)
        if self.look_up_table is None:
            self.load_look_up_table()

        # 2. 验证缓存容量
        self.validate_cache_capacity()

         # 3. 计算完整分块数量和余数
        M_l2_t = self.M // self.M_tile  # M维度完整分块数量
        N_l2_t = self.N // self.N_tile  # N维度完整分块数量
        K_l2_t = self.K // self.K_tile  # K维度完整分块数量
        M_remain = self.M % self.M_tile  # M维度剩余元素数量
        N_remain = self.N % self.N_tile  # N维度剩余元素数量
        K_remain = self.K % self.K_tile  # K维度剩余元素数量

        # 4. 创建分块数组（直接使用ceil计算总块数）
        self.tiles = np.empty([
            ceil(self.M / self.M_tile),  # M维度总块数
            ceil(self.N / self.N_tile),  # N维度总块数
            ceil(self.K / self.K_tile),  # K维度总块数
        ], dtype=Chip_tile)

        # 5. 初始化所有分块(处理各种边界情况)
        # numpy的广播机制实现批量初始化！
        # 5.1 初始化完整分块
        if M_l2_t * N_l2_t * K_l2_t != 0:
            self.tiles[:M_l2_t, :N_l2_t, :K_l2_t] = self.create_chip_tile(
                self.M_tile, self.N_tile, self.K_tile
            )
        
        # 5.2 处理M维度的边界分块
        if M_remain != 0:
            self.tiles[-1, :N_l2_t, :K_l2_t] = self.create_chip_tile(
                M_remain, self.N_tile, self.K_tile
            )
        
        # 5.3 处理N维度的边界分块  
        if N_remain != 0:
            self.tiles[:M_l2_t, -1, :K_l2_t] = self.create_chip_tile(
                self.M_tile, N_remain, self.K_tile
            )
        
        # 5.4 处理K维度的边界分块
        if K_remain != 0:
            self.tiles[:M_l2_t, :N_l2_t, -1] = self.create_chip_tile(
                self.M_tile, self.N_tile, K_remain
            )
        
        # 5.5 处理M和N维度的边界分块
        if M_remain * N_remain != 0:
            self.tiles[-1, -1, :K_l2_t] = self.create_chip_tile(
                M_remain, N_remain, self.K_tile
            )
        
        # 5.6 处理M和K维度的边界分块
        if M_remain * K_remain != 0:
            self.tiles[-1, :N_l2_t, -1] = self.create_chip_tile(
                M_remain, self.N_tile, K_remain
            )
        
        # 5.7 处理N和K维度的边界分块
        if N_remain * K_remain != 0:
            self.tiles[:M_l2_t, -1, -1] = self.create_chip_tile(
                self.M_tile, N_remain, K_remain
            )
        
        # 5.8 处理M、N和K三个维度的边界分块
        if M_remain * N_remain * K_remain != 0:
            self.tiles[-1, -1, -1] = self.create_chip_tile(
                M_remain, N_remain, K_remain
            )


        # 6. 初始化总周期数(包含第一个分块的加载时间)
        total_latency = 0
        total_latency += (
            self.tiles[0, 0, 0].M_K_io_latency + self.tiles[0, 0, 0].K_N_io_latency
        )

        # 7. 记录前一个分块的位置
        previous_m = 0
        previous_n = 0
        previous_k = 0

        # 8. 按照指定的循环顺序遍历所有分块
        for m, n, k in self.generate_tile_loops(
            ceil(self.M / self.M_tile),  # M方向总块数
            ceil(self.N / self.N_tile),  # N方向总块数
            ceil(self.K / self.K_tile),  # K方向总块数
            self.chip_loop_order,        # 循环顺序配置
        ):
            # 跳过第一个分块(已在第6部分处理)
            if m == 0 and n == 0 and k == 0:
                continue

            # 获取当前分块和前一个分块的引用
            current_tile = self.tiles[m, n, k]
            previous_tile = self.tiles[previous_m, previous_n, previous_k]

            # 9. 计算当前分块的读取延迟（数据加载时间）
            if m == previous_m and k == previous_k:
                # 情况1：仅N维度变化 → 只需加载新的B矩阵(K_N)
                current_tile_read_latency = current_tile.K_N_io_latency
            elif n == previous_n and k == previous_k:
                # 情况2：仅M维度变化 → 只需加载新的A矩阵(M_K)
                current_tile_read_latency = current_tile.M_K_io_latency
            else:
                # 情况3：其他情况 → 需要加载完整的A和B矩阵
                current_tile_read_latency = current_tile.M_K_io_latency + current_tile.K_N_io_latency
            
            # 特殊处理：当K>0且跨M/N分块时，需要加载部分和矩阵
            if k > 0 and not (m == previous_m and n == previous_n):
                current_tile_read_latency += current_tile.M_N_io_latency  # C矩阵加载时间

            # 10. 计算前一个分块的计算延迟
            previous_tile_compute_latency = previous_tile.compute_latency
            if k > 0:  # 如果是K维度的非第一个分块
                previous_tile_compute_latency += previous_tile.K_reduction_latency

            # 11. 计算前一个分块的写回延迟
            if m == previous_m and n == previous_n:
                # 情况1：仍在同一个输出分块 → 无需写回中间结果
                previous_tile_write_latency = 0
            else:
                # 情况2：跨输出分块 → 必须写回当前部分和
                previous_tile_write_latency = previous_tile.M_N_io_latency

            # 12. 累加周期数（关键路径计算）
            if self.is_double_buffering:
                # 双缓冲模式：允许加载与计算重叠
                total_latency += max(
                    current_tile_read_latency,       # 当前分块加载时间
                    previous_tile_compute_latency    # 前一分块计算时间
                ) + previous_tile_write_latency      # 前一分块写回时间
            else:
                # 非双缓冲模式：顺序执行各阶段
                total_latency += (
                    current_tile_read_latency +      # 串行加载
                    previous_tile_compute_latency +  # 串行计算
                    previous_tile_write_latency      # 串行写回
                )

            # 更新前一个分块位置
            previous_m = m
            previous_n = n
            previous_k = k

        # 13. 处理最后一个分块的计算和写回
        total_latency += (
            self.tiles[-1, -1, -1].M_N_io_latency +  # 最终结果写回时间
            self.tiles[-1, -1, -1].compute_latency    # 最后一个分块计算时间
        )

        # 14. 处理K维度的归约尾端
        if previous_k > 0:  # 如果K方向有多个分块
            total_latency += ceil(self.tiles[-1, -1, -1].K_reduction_latency)

        # 15. 返回总周期数
        return total_latency
    
    '''验证缓存容量是否足够'''
    def validate_cache_capacity(self):
        # 计算所需的缓存空间
        required_cache = (
            self.M_tile * self.N_tile + 
            self.N_tile * self.K_tile + 
            self.M_tile * self.K_tile
        ) * self.strategy.elem_bytes
        
        # 获取可用的缓存容量 L2_CAPACITY!
        available_cache = HW.L2_CAPACITY
        assert required_cache <= available_cache, (
                f"缓存容量不足: 所需空间 {required_cache} 字节，"
                f"可用空间 {available_cache} 字节"
            )
        
        '''# 检查是否启用双缓冲 这里需要is_double_buffering这个变量！！
        is_double_buffering = self.block_strategy == "double_buffer"
        
        # 验证缓存容量
        if is_double_buffering:
            assert required_cache <= available_cache // 2, (
                f"缓存容量不足: 所需空间 {required_cache} 字节，"
                f"双缓冲模式下可用空间 {available_cache // 2} 字节"
            )
        else:
            assert required_cache <= available_cache, (
                f"缓存容量不足: 所需空间 {required_cache} 字节，"
                f"可用空间 {available_cache} 字节"
            )
            '''
    
    '''加载之前计算的性能查找表'''
    def load_look_up_table(self):
        # 这里需要根据实际情况加载查找表
        # 假设我们已经有了查找表，这里简化处理
        self.look_up_table = {}

    
    #传入参数待修改！！！！！！！！！！！！！！！！！！！！！
    '创建一个新的Chip_tile分块！'
    def create_chip_tile(self, M, N, K):
    
        return Chip_tile(
            M=M,
            N=N,
            K=K,
            M_tile=M,
            N_tile=N,
            K_tile=K,
            next_layer=None,  # 根据实际情况设置
            block_strategy=self.block_strategy,
            storage_format=self.storage_formats
        )

    '''生成遍历分块的循环顺序'''
    def generate_tile_loops(self, M_tiles, N_tiles, K_tiles, loop_order):
        # 根据分块策略确定循环顺序
        if loop_order == "mnk":
            for m in range(M_tiles):
                for n in range(N_tiles):
                    for k in range(K_tiles):
                        yield m, n, k
        elif loop_order == "mkn":
            for m in range(M_tiles):
                for k in range(K_tiles):
                    for n in range(N_tiles):
                        yield m, n, k
        elif loop_order == "nmk":
            for n in range(N_tiles):
                for m in range(M_tiles):
                    for k in range(K_tiles):
                        yield m, n, k
        elif loop_order == "nkm":
            for n in range(N_tiles):
                for k in range(K_tiles):
                    for m in range(M_tiles):
                        yield m, n, k
        elif loop_order == "kmn":
            for k in range(K_tiles):
                for m in range(M_tiles):
                    for n in range(N_tiles):
                        yield m, n, k
        elif loop_order == "knm":
            for k in range(K_tiles):
                for n in range(N_tiles):
                    for m in range(M_tiles):
                        yield m, n, k
        else:
            # 默认使用mnk顺序
            for m in range(M_tiles):
                for n in range(N_tiles):
                    for k in range(K_tiles):
                        yield m, n, k



class PipelineTileState:
    """
    追踪每个 tile 在六个流水阶段（0=OUT2,1=OUT1,2=FIX,3=MTE2,4=MTE1,5=M）的状态。
    """
    def __init__(self, coord):
        self.coord = coord
        self.current_stage = 0
        self.remaining_cycles = 0
        self.completed_stages = set()

    def is_stage_completed(self, stage_id):
        return stage_id in self.completed_stages

    def start_stage(self, stage_id, duration):
        self.current_stage = stage_id
        self.remaining_cycles = duration

    def step(self):
        if self.remaining_cycles > 0:
            self.remaining_cycles -= 1
            if self.remaining_cycles == 0:
                self.completed_stages.add(self.current_stage)
                return True
        return False


class PipelineScheduler:
    """
    支持 24-core 并行的六层流水线调度器
    """
    def __init__(self, sim):
        # sim: Simulate 实例
        self.sim = sim
        self.chip_tiles = sim.tiles
        self.states     = {}
        self.cycle      = 0
        # Sflag延迟，目前未知保留为0
        self.flag_delay = getattr(sim, 'flag_delay', 0)
        # 记录每层上一次调度的 tile 坐标，用于 A/B 重用判断
        self.prev_coord = [None] * 6

        M, N, K = sim.M, sim.N, sim.K
        mt, nt, kt = sim.M_tile, sim.N_tile, sim.K_tile
        self.M_tiles = ceil(M / mt)
        self.N_tiles = ceil(N / nt)
        self.K_tiles = ceil(K / kt)

        for m in range(self.M_tiles):
            for n in range(self.N_tiles):
                for k in range(self.K_tiles):
                    self.states[(m, n, k)] = PipelineTileState((m, n, k))

        self.layer_busy     = [[] for _ in range(6)]
        self.parallel_limit = {3: 24, 4: 24, 5: 24}

    def simulate(self):
        while not self._all_done():
            self._advance_cycle()
        return self.cycle

    def _advance_cycle(self):
        self._progress_running_stages()
        self._try_launch_new_stages()
        self.cycle += 1

    def _progress_running_stages(self):
        for stage_id in range(6):
            done = []
            for coord in self.layer_busy[stage_id]:
                if self.states[coord].step():
                    done.append(coord)
            for coord in done:
                self.layer_busy[stage_id].remove(coord)

    def _try_launch_new_stages(self):
        for coord, state in self.states.items():
            stage = state.current_stage

            # 本层资源检查
            if stage in (0, 1, 2):
                if self.layer_busy[stage]:
                    continue
            else:
                if len(self.layer_busy[stage]) >= self.parallel_limit[stage]:
                    continue

            if not self._stage_ready(coord, stage):
                continue

            latency = self._get_stage_latency(coord, stage)
            if latency is None:
                continue

            state.start_stage(stage, latency)
            self.layer_busy[stage].append(coord)
            # 更新本层上一次调度的 tile，用于下一次重用判断
            self.prev_coord[stage] = coord

    def _stage_ready(self, coord, stage_id):
        m, n, k = coord
        state = self.states[coord]

        if stage_id == 0:   # OUT2
            return True
        elif stage_id == 1: # OUT1
            return state.is_stage_completed(0)
        elif stage_id == 2: # FIX
            return state.is_stage_completed(5) and not self.layer_busy[1]
        elif stage_id == 3: # MTE2
            return state.is_stage_completed(0)
        elif stage_id == 4: # MTE1
            if not state.is_stage_completed(3):
                return False
            if k > 0:
                prev = self.states[(m, n, k - 1)]
                if not prev.is_stage_completed(2):
                    return False
            return True
        elif stage_id == 5: # M
            return state.is_stage_completed(4)
        return False

    def _get_stage_latency(self, coord, stage_id):
        m, n, k = coord
        chip_tile = self.chip_tiles[m, n, k]
        A = chip_tile.M * chip_tile.K * chip_tile.elem_bytes
        B = chip_tile.K * chip_tile.N * chip_tile.elem_bytes
        C = chip_tile.M * chip_tile.N * chip_tile.elem_bytes

        # 重用判断：根据上次调度的 tile 坐标
        prev = self.prev_coord[stage_id]
        if prev is None:
            load_A = load_B = 1
        else:
            pm, pn, pk = prev
            load_A = 1 if (n != pn or k != pk) else 0
            load_B = 1 if (m != pm or k != pk) else 0

        if stage_id == 0: # OUT2
            base = load_A * _device.io.load(A, 'EXT', 'DRAM') + load_B * _device.io.load(B, 'EXT', 'DRAM')
        elif stage_id == 1: # OUT1
            base = _device.io.store(C, 'L2', 'DRAM')
        elif stage_id == 2: # FIX
            base = _device.io.store(C, 'L0C', 'L2')
        elif stage_id == 3: # MTE2
            base = load_A * (_device.io.load(A, 'DRAM', 'L2') + _device.io.load(A, 'L2', 'L1')) + load_B * (_device.io.load(B, 'DRAM', 'L2') + _device.io.load(B, 'L2', 'L1'))
        elif stage_id == 4: # MTE1
            t1 = max(load_A * _device.io.load(A, 'L1', 'L0A'), load_B * _device.io.load(B, 'L1', 'L0B'))
            if k > 0:
                prev_fix = self.states[(m, n, k - 1)]
                if not prev_fix.is_stage_completed(2):
                    return None
            t2 = _device.io.store(C, 'AccumDFF', 'L0C')
            base = t1 + t2
        elif stage_id == 5: # M
            base = _device.compute.compute(chip_tile.M, chip_tile.N, chip_tile.K)
        else:
            return None

        return base + self.flag_delay

    def _all_done(self):
        return all(5 in s.completed_stages for s in self.states.values())


                        
class Chip_tile:
    """
    从芯片分块到 L1 上，910C有24个L1_tile和24个UB_tile。这里的逻辑要仔细想清楚，包括往L1的io读时间、从UB回来的io写时间、L2 cache机制等因素
    """
    def __init__(self, base_address = None, strategy: 'MatMul_Strategy' = None, M=None, N=None, K=None):
        # 保存 strategy 对象!!!
        self.strategy = strategy
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
        self.base_address = base_address # 基地址，用于计算每个L1_tile的地址

        # 约束 # 注意，这里同时要分左右矩阵，所以存储约束需要除以2
        self.size = strategy.chip_type.L1_CAPACITY                  # 芯片L1结构存储容量
        self.min_load_size = strategy.chip_type.MIN_ACCESS['Chip']  # 主存最小访存大小
        # self.M = strategy.chip_mnk_values[0] # 芯片层矩阵大小M
        # self.N = strategy.chip_mnk_values[1] # 芯片层矩阵大小N
        # self.K = strategy.chip_mnk_values[2] # 芯片层矩阵大小K
        self.M = M # 芯片层矩阵大小M
        self.N = N # 芯片层矩阵大小N
        self.K = K # 芯片层矩阵大小K
        
        # 硬件约束
        self.core_count = strategy.chip_type.AI_CORE_COUNT
        #换self.l2_bandwidth_per_cycle = strategy.chip_type.L2_BANDWIDTH 
        self.DRAM_to_l2_bandwidth = strategy.chip_type.IO_BW['DRAM→L2']
        self.clock_freq = strategy.chip_type.CLOCK_FREQ
        self.vector_flops_per_cycle = strategy.chip_type.VECTOR_FLOPS_PER_CYCLE
        #这些先不要，目前认为没有，用elem_bytes!!!
        # self.systolic_input_word_size = strategy.chip_type.SYSTOLIC_INPUT_WORD_SIZE
        # self.systolic_output_word_size = strategy.chip_type.SYSTOLIC_OUTPUT_WORD_SIZE
        #芯片间数据传输速度chiplet_module.io_module.bandwidth
        #获取到 外部 到 DRAM 的带宽值！
        self.io_bandwidth = strategy.chip_type.IO_BW['OUTSIDE→DRAM']
        self.vector_flops = None #该参数待定！！！
        # 向上回传数据
        # vector_flops_per_cycle是什么,等于CUBE_MACS_PER_CYCLE吗???要改成vector_flops； chip_tile_bandwidth是什么???应该是主存到chip_tile的带宽
        self.K_reduction_latency = ceil(
            self.M * self.N / self.vector_flops
        ) + 2 * ceil(
            self.M * self.N * self.strategy.elem_bytes / self.chip_tile_bandwidth
        )
        self.K_reduction_io_count = 2 * self.M * self.N * self.strategy.elem_bytes
        self.M_K_io_latency = self._simulate_chip_tile_io_latency(self.M, self.K)
        self.K_N_io_latency = self._simulate_chip_tile_io_latency(self.K, self.N)
        self.M_N_io_latency = self._simulate_chip_tile_io_latency(self.M, self.N)
        self.compute_latency = self._simulate_chip_tile_compute_latency()
        self.mem_alloc_read_latency = None   # 内存分配耗时
        self.mem_alloc_write_latency = None  # 内存分配耗时
        
    def calculate_cycles(self):
        core_cycles = []
        size_A = self.M_tile * self.K_tile * self.strategy.elem_bytes
        size_B = self.K_tile * self.N_tile * self.strategy.elem_bytes
        size_C = self.M_tile * self.N_tile * self.strategy.elem_bytes
        for core in range(self.L1_num):
            # 1) DRAM-L2-L1，传入真实基地址
            addr_A, addr_B = self.base_addresses[core]
            t1 = L2_CACHE_MGR.read(core, addr_A, size_A) + L2_CACHE_MGR.read(core, addr_B, size_B)
            # 2) L1-L0A/L0B
            t2 = max(_device.io.load(size_A, 'L1', 'L0A'), 
                     _device.io.load(size_B, 'L1', 'L0B'))
            # 3) Cube-AccumDFF-L0C
            t3 = _device.compute.compute(self.M_tile, self.N_tile, self.K_tile)
            t3 += _device.io.store(size_C, 'AccumDFF', 'L0C')   
            # 4) L0C-UB
            t4 = _device.io.load(size_C, 'L0C', 'UB')
            # 5) 缓存本次 C 矩阵块到 L2 输出段
            t5 = L2_CACHE_MGR.write(core, size_C)
            core_cycles.append(t1 + t2 + t3 + t4 + t5)
        # 等所有 core 的块都累积到 L2 了，下一步统一 flush
        for core in range(self.L1_num):
            t_flush = L2_CACHE_MGR.flush(core)
            core_cycles[core] += t_flush
        return core_cycles
        
    def _simulate_chip_tile_io_latency(self, M: int, N: int) -> int:
        """计算IO传输周期数"""
        #io_bandwidth 芯片间数据传输速度待确定？？？
        return ceil(
            M * N * self.strategy.elem_bytes / (
                self.io_bandwidth 
            )
        )

    def _simulate_chip_tile_compute_latency(self) -> int:
        """核心计算周期模拟"""
        # 1. L1分块参数初始化
        M_l1_t = self.M // self.M_tile # M方向完整分块数  
        N_l1_t = self.N // self.N_tile 
        K_l1_t = self.K // self.K_tile 
        M_remain = self.M % self.M_tile # M方向剩余元素
        N_remain = self.N % self.N_tile 
        K_remain = self.K % self.K_tile 
        
        # 2. 创建L1分块数组
        l1_tiles = np.empty(
            [ceil(self.M / self.M_tile ), ceil(self.N / self.N_tile ), ceil(self.K / self.K_tile )],
            dtype=L1_tile
        )
        
        # 边界分块初始化（8种情况）
        if M_l1_t * N_l1_t * K_l1_t != 0:
            l1_tiles[:M_l1_t, :N_l1_t, :K_l1_t] = self._create_l1_tile(
                self.M_tile , self.N_tile , self.K_tile 
            )
        if M_remain != 0:
            l1_tiles[-1, :N_l1_t, :K_l1_t] = self._create_l1_tile(
                M_remain, self.N_tile , self.K_tile 
            )
        if N_remain != 0:
            l1_tiles[:M_l1_t, -1, :K_l1_t] = self._create_l1_tile(
                self.M_tile , N_remain, self.K_tile 
            )
        if K_remain != 0:
            l1_tiles[:M_l1_t, :N_l1_t, -1] = self._create_l1_tile(
                self.M_tile , self.N_tile , K_remain
            )
        if M_remain * N_remain != 0:
            l1_tiles[-1, -1, :K_l1_t] = self._create_l1_tile(
                M_remain, N_remain, self.K_tile 
            )
        if M_remain * K_remain != 0:
            l1_tiles[-1, :N_l1_t, -1] = self._create_l1_tile(
                M_remain, self.N_tile , K_remain
            )
        if N_remain * K_remain != 0:
            l1_tiles[:M_l1_t, -1, -1] = self._create_l1_tile(
                self.M_tile , N_remain, K_remain
            )
        if M_remain * N_remain * K_remain != 0:
            l1_tiles[-1, -1, -1] = self._create_l1_tile(
                M_remain, N_remain, K_remain
            )
        
        # 3. 数据量统计矩阵
        ## 初始化三个矩阵分别记录A(M×K)、B(K×N)、C(M×N)的数据量
        M_K_tile_size = np.zeros(
            [ceil(self.M / self.M_tile ), ceil(self.K / self.K_tile )], dtype=int
        )
        # 填充每个分块的实际数据量（考虑边界）
        # 完整分块
        M_K_tile_size[:M_l1_t, :K_l1_t] = self.M_tile  * self.K_tile 
        # M边界
        if M_remain > 0: 
            M_K_tile_size[-1, :K_l1_t] = M_remain * self.K_tile 
        # K边界
        if K_remain > 0: 
            M_K_tile_size[:M_l1_t, -1] = self.M_tile  * K_remain
        # M，K边界
        if M_remain > 0 and K_remain > 0: 
            M_K_tile_size[-1, -1] = M_remain * K_remain
        
        K_N_tile_size = np.zeros(
            [ceil(self.K / self.K_tile ), ceil(self.N / self.N_tile )], dtype=int
        )
        K_N_tile_size[:K_l1_t, :N_l1_t] = self.K_tile  * self.N_tile 
        if K_remain > 0: 
            K_N_tile_size[-1, :N_l1_t] = K_remain * self.N_tile 
        if N_remain > 0: 
            K_N_tile_size[:K_l1_t, -1] = self.K_tile  * N_remain
        if K_remain > 0 and N_remain > 0: 
            K_N_tile_size[-1, -1] = K_remain * N_remain
        
        M_N_tile_size = np.zeros(
            [ceil(self.M / self.M_tile ), ceil(self.N / self.N_tile )], dtype=int
        )
        M_N_tile_size[:M_l1_t, :N_l1_t] = self.M_tile  * self.N_tile 
        if M_remain > 0: 
            M_N_tile_size[-1, :N_l1_t] = M_remain * self.N_tile 
        if N_remain > 0: 
            M_N_tile_size[:M_l1_t, -1] = self.M_tile  * N_remain
        if M_remain > 0 and N_remain > 0: 
            M_N_tile_size[-1, -1] = M_remain * N_remain
        
        # 4. 任务调度核心逻辑
        total_cycle = 0
        # 记录前一批次A矩阵的加载位置
        prev_read_M_K = np.zeros([ceil(self.M/self.M_tile ), ceil(self.K/self.K_tile )], dtype=bool)
        # 记录前一批次B矩阵的加载位置
        prev_read_K_N = np.zeros([ceil(self.K/self.K_tile ), ceil(self.N/self.N_tile )], dtype=bool)
        # 记录前一批次C矩阵的加载位置
        prev_read_M_N = np.zeros([ceil(self.M/self.M_tile ), ceil(self.N/self.N_tile )], dtype=bool)
        # 记录前一批次C矩阵的存储位置
        prev_write_M_N = np.zeros([ceil(self.M/self.M_tile ), ceil(self.N/self.N_tile )], dtype=bool)
        prev_compute_latency = 0 # 前一批次计算周期
        active_tile_list = [] # 用于临时存储当前批次待处理的L1分块
        
        # 分块循环调度
        for m, n, k in self.generate_tile_loops(
            ceil(self.M/self.M_tile ),
            ceil(self.N/self.N_tile ),
            ceil(self.K/self.K_tile ),
            self.l1_loop_order
        ):
            active_tile_list.append((m, n, k, l1_tiles[m, n, k]))
            
            # 批次触发条件
            # 条件1：判断是否到达最后一个分块
            if (m == ceil(self.M/self.M_tile )-1 and 
                n == ceil(self.N/self.N_tile )-1 and 
                k == ceil(self.K/self.K_tile )-1):
                pass # 强制触发处理
            # 条件2：判断是否攒够一个完整批次,不够再加1！
            elif len(active_tile_list) < self.core_count:
                continue # 继续积累分块
            #确保当前批次分块数不超过物理计算核心数
            assert len(active_tile_list) <= self.core_count
                
            # 初始化当前批次的数据加载标记矩阵
            current_read_M_K = np.zeros_like(prev_read_M_K)
            current_read_K_N = np.zeros_like(prev_read_K_N)
            current_read_M_N = np.zeros_like(prev_read_M_N)
            current_write_M_N = np.zeros_like(prev_write_M_N)
            current_compute_latency = 0
            
            # 处理当前批次每个分块
            '''数据标记：通过布尔矩阵记录当前批次需要加载的A/B/C矩阵分块位置  
                •  归约开销：当temp_k>0时，增加部分和累加的向量化操作周期  
                •  瓶颈计算：取所有分块计算周期的最大值（模拟最慢核心决定批次完成时间）'''
            for idx in range(len(active_tile_list)):
                # 获取当前分块信息
                temp_m, temp_n, temp_k, temp_l1_tile = active_tile_list[idx]

                # 标记需要加载的数据分块
                current_read_M_K[temp_m, temp_k] = 1 # 标记A(M×K)分块
                current_read_K_N[temp_k, temp_n] = 1 # 标记B(K×N)分块
                current_read_M_N[temp_m, temp_n] = (temp_k > 0) # 仅当k>0时加载部分和C
                current_write_M_N[temp_m, temp_n] = 1 # 所有分块需写回结果
                
                # 计算分块周期（含归约）
                temp_l1_tile_compute_latency = temp_l1_tile.compute_latency
                if temp_k > 0:
                    temp_l1_tile_compute_latency += ceil(
                        temp_l1_tile.M * temp_l1_tile.N / self.vector_flops#！！！
                    )
                current_compute_latency = max(current_compute_latency, temp_l1_tile_compute_latency)
            
            # 数据依赖分析
            '''增量加载：~previous_batch_Read_*排除已缓存数据

                    写回优化：仅写回不被下批次复用的部分和（~current_batch_Read_M_N）

                    冲突避免：~(previous + Write)防止读后写（RAW） hazard'''
            curr_M_K_read_count = np.sum(
                (current_read_M_K * (~prev_read_M_K)) * M_K_tile_size
            )
            curr_K_N_read_count = np.sum(
                (current_read_K_N * (~prev_read_K_N)) * K_N_tile_size
            )
            #假设在一个分块矩阵乘法的计算中，每次只计算一部分结果，然后把这些结果累加到结果矩阵 C 上。
            # 在计算下一个分块之前，需要读取结果矩阵 C 中对应的部分，和当前计算得到的结果相加。
            # 这种情况下，就会出现读取矩阵 C 的操作！！！???这块需要确定到底有没有！！！
            curr_M_N_read_count = np.sum(
                (current_read_M_N * (~(prev_read_M_N + prev_write_M_N))) * M_N_tile_size
            )
            prev_M_N_write_count = np.sum(
                (prev_write_M_N * (~current_read_M_N)) * M_N_tile_size
            )
            
            # 流水线周期计算
            '''流水线原理：

                    重叠执行：当前批次加载与前批次计算并行（max取两者较长者）

                    写回串行：必须等前批次计算完成才能写回

                    带宽计算：数据量 ÷ L2带宽（考虑输入/输出字宽差异）'''
            # # 计算当前批次加载周期
            # current_batch_read_latency = ceil(
            #     (curr_M_K_read_count + curr_K_N_read_count + curr_M_N_read_count) * 
            #     self.strategy.elem_bytes / self.DRAM_to_l2_bandwidth
            # )
            # 计算前批次写回周期
            # previous_batch_write_latency = ceil(
            #     prev_M_N_write_count * self.strategy.elem_bytes / self.DRAM_to_l2_bandwidth
            # )
            # total_latency += max(
            #     current_batch_read_latency,  # 当前批次加载时间
            #     prev_compute_latency                # 前批次计算时间
            #     ) + previous_batch_write_latency # 前批次写回时间


            #???按这个代码的逻辑这里core的id应该不需要，2人的代码需要选取一个逻辑
            addr_A, addr_B = self.base_addresses[core]
            addr_C = self.base_addresses[core]#???暂时假设一下

            size_A = curr_M_K_read_count * self.strategy.elem_bytes
            size_B = curr_K_N_read_count * self.strategy.elem_bytes
            size_C = curr_M_N_read_count * self.strategy.elem_bytes
            # 1) DRAM-L2-L1，传入真实基地址
            T1 = L2_CACHE_MGR.read(core, addr_A, size_A) 
            + L2_CACHE_MGR.read(core, addr_B, size_B)+L2_CACHE_MGR.read(core, addr_C, size_C)
            # 2) L1-L0A/L0B
            T2 = max(_device.io.load(size_A, 'L1', 'L0A'), 
                     _device.io.load(size_B, 'L1', 'L0B'))
        
            current_batch_read_latency=T1+T2
            # 上批次的compute
            # 3) Cube-AccumDFF-L0C
            T3 = _device.io.store(size_C, 'AccumDFF', 'L0C') 
            # 4) L0C-UB
            T4 = _device.io.load(size_C, 'L0C', 'UB')
            # 5) 缓存本次 C 矩阵块到 L2 输出段
            T5 = L2_CACHE_MGR.write(core, size_C)
            previous_batch_write_latency=T3+T4+T5
            
            total_latency += max(
                current_batch_read_latency,  # 当前批次加载时间
                prev_compute_latency                # 前批次计算时间
                ) + previous_batch_write_latency # 前批次写回时间
            

            # 状态更新
            # 保存当前批次状态用于下次迭代
            prev_compute_latency = current_compute_latency
            prev_read_M_K = copy.deepcopy(current_read_M_K)
            prev_read_K_N = copy.deepcopy(current_read_K_N)
            prev_read_M_N = copy.deepcopy(current_read_M_N)
            prev_write_M_N = copy.deepcopy(current_write_M_N)
            # 清空当前批次
            active_tile_list = []
        
        # 尾部处理 可能也要修改???
        total_latency += (
            prev_compute_latency +
            ceil(
                np.sum(prev_write_M_N * M_N_tile_size) * self.strategy.elem_bytes /
                self.DRAM_to_l2_bandwidth
        ))
        #???统一 flush的操作，还需要修改
        t_flush = L2_CACHE_MGR.flush(core)
        total_latency += t_flush
        
        
        return total_latency

    def _create_l1_tile(self, M, N, K) -> 'L1_tile':
        """创建L1分块实例"""
        return L1_tile(
            id=0, 
            strategy=self.strategy,
            M=M,
            N=N,
            K=K
        )

    def generate_tile_loops(self, M_tiles, N_tiles, K_tiles, loop_order):
        """生成分块遍历顺序"""
        if loop_order == "mnk":
            for m in range(M_tiles):
                for n in range(N_tiles):
                    for k in range(K_tiles):
                        yield m, n, k
        elif loop_order == "mkn":
            for m in range(M_tiles):
                for k in range(K_tiles):
                    for n in range(N_tiles):
                        yield m, n, k
        elif loop_order == "nmk":
            for n in range(N_tiles):
                for m in range(M_tiles):
                    for k in range(K_tiles):
                        yield m, n, k
        elif loop_order == "nkm":
            for n in range(N_tiles):
                for k in range(K_tiles):
                    for m in range(M_tiles):
                        yield m, n, k
        elif loop_order == "kmn":
            for k in range(K_tiles):
                for m in range(M_tiles):
                    for n in range(N_tiles):
                        yield m, n, k
        elif loop_order == "knm":
            for k in range(K_tiles):
                for n in range(N_tiles):
                    for m in range(M_tiles):
                        yield m, n, k
        else:
            # 默认使用mnk顺序
            for m in range(M_tiles):
                for n in range(N_tiles):
                    for k in range(K_tiles):
                        yield m, n, k


# 从这里开始，是Chip→L1→L0→ABDFF的分块结构，负责数据向矩阵计算Core的搬入
class L1_tile:
    """
    从 L1 分块到 L0 上。注意一个L1对应两个L0，因此在写约束的时候注意要把约束除以2；L2 cache机制也要考虑，不过不要和Chip层重复计算
    """
    def __init__(self, id, strategy: 'MatMul_Strategy' = None, M=None, N=None, K=None):
        # 本层结构编号
        self.L1_id = id
        self.strategy=strategy
        
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
        self.size = strategy.chip_type.L0A_CAPACITY              # L0 buffer结构存储容量
        self.min_load_size= strategy.chip_type.MIN_ACCESS['L1']  # L1 buffer最小访存大小
          # 注意，由于在L1层关心L0层容量，所以这里没有用if来区分L0A和L0B，后面需要修改一下
          # 同时，由于这里分块只关系L0A和L0B，所以没有考虑L0C的约束，需要另作考虑
        # self.M = strategy.L1_mnk_values[0] # L1 buffer层矩阵大小M
        # self.N = strategy.L1_mnk_values[1] # L1 buffer层矩阵大小N
        # self.K = strategy.L1_mnk_values[2] # L1 buffer层矩阵大小K      
        self.M = M # L1 buffer层矩阵大小M
        self.N = N # L1 buffer层矩阵大小N
        self.K = K # L1 buffer层矩阵大小K  

        # 向上回传数据
        self.K_reduction_latency = None # 结果矩阵拼接耗时
        #self.K_reduction_io_count = None
        self.M_K_io_latency = None      # 左矩阵传入耗时
        self.K_N_io_latency = None      # 右矩阵传入耗时
        self.M_N_io_latency = None      # 结果矩阵传出耗时
        self.compute_latency = None     # 内层计算耗时
        self.mem_alloc_read_latency = None   # 内存分配耗时
        self.mem_alloc_write_latency = None  # 内存分配耗时

    def calculate_cycles(self):
        """
        Calculate the cycles for the current layer based on the next layer's metrics.
        需要考虑分块策略（在最前面用self.block_strategy区分计算路径）
        """
        # L1 to L0A/B
        size_A = self.M_tile * self.K_tile * self.strategy.elem_bytes
        size_B = self.K_tile * self.N_tile * self.strategy.elem_bytes
        if size_A > HW.L0A_CAPACITY:
            raise ValueError(f"L0A overflow: {size_A} > {HW.L0A_CAPACITY}")
        if size_B > HW.L0B_CAPACITY:
            raise ValueError(f"L0B overflow: {size_B} > {HW.L0B_CAPACITY}")
        cyc_A = _device.io.load(size_A, 'L1', 'L0A')
        cyc_B = _device.io.load(size_B, 'L1', 'L0B')
        total = max(cyc_A, cyc_B)
        
        # L0A/B 递归
        for l0 in self.next_layers:
            total += l0.calculate_cycles()
        
        return total

class L0_tile:
    """
    从 L0 分块到寄存器上。注意，由于L0_A/B、L0_C是分开的，L0_C需要单开一个类
    """
    def __init__(self, id, strategy: 'MatMul_Strategy' = None):
        # 本层结构编号
        self.L0_id = id
        self.strategy = strategy
        
        # 下层结构初始化
        self.next_layer = AB_DFF_tile(strategy=strategy) # next_layer是DFF_tile实例，把策略传递进去

        # 决策变量
        self.M_tile = strategy.DFF_mnk_values[0]
        self.N_tile = strategy.DFF_mnk_values[1]
        self.K_tile = strategy.DFF_mnk_values[2]
        self.block_strategy = strategy.DFF_block_strategy   # 分块策略；内积/外积【弄清楚分块策略对应的层的名称】
        self.storage_formats = strategy.DFF_storage_formats # 存储格式；行主序/列主序
        
        # 约束
        self.size = strategy.chip_type.ABDFF_CAPACITY              # L0结构存储容量（L0A和L0B是一致的）
        self.key = 'L0A' if self.L0_id == 0 else 'L0B' # L0A和L0B的key
        self.min_load_size = strategy.chip_type.MIN_ACCESS[self.key] # L0的最小访存大小
        self.M = strategy.L0_mnk_values[0] # L0层矩阵大小M
        self.N = strategy.L0_mnk_values[1] # L0层矩阵大小N
        self.K = strategy.L0_mnk_values[2] # L0层矩阵大小K

        # 向上回传数据
        self.K_reduction_latency = None # 结果矩阵拼接耗时
        #self.K_reduction_io_count = None
        self.M_K_io_latency = None      # 左矩阵传入耗时
        self.K_N_io_latency = None      # 右矩阵传入耗时
        self.M_N_io_latency = None      # 结果矩阵传出耗时
        self.compute_latency = None     # 内层计算耗时
        self.mem_alloc_read_latency = None   # 内存分配耗时
        self.mem_alloc_write_latency = None  # 内存分配耗时

    def calculate_cycles(self):
        """
        包含：
        1. L0A/L0B to Cube 寄存器应该没有IO cycle的开销
        2. Cube上运算 MxK  x KxN 的矩阵乘
        3. Accum DFF 寄存器的累加 同1
        """
        # Cube 计算
        cube_cycles = _device.compute.compute(self.M_tile, self.N_tile, self.K_tile)
        # 从 AccumDFF 写回L0C
        size_C = self.M_tile * self.N_tile * self.strategy.elem_bytes
        io_cycles = _device.io.store(size_C, 'AccumDFF', 'L0C')
        total = cube_cycles + io_cycles
        return total


class AB_DFF_tile:
    """
    寄存器上的结构运算。注意在计算compute_latency的时候，注意在AB_DFF_tile与Accum_DFF_tile之间的不要重复计算
    """
    def __init__(self, strategy: 'MatMul_Strategy' = None):
        # 保存 strategy 对象!!!
        self.strategy = strategy
        # 约束
        self.size = strategy.chip_type.ABDFF_CAPACITY               # DFF结构存储容量【通常来说应该要用更下层的容量做约束，但这已经是最底层了，我就先写着了】
        self.min_load_size = strategy.chip_type.MIN_ACCESS['ABDFF'] # DFF的最小访存大小
        self.M = strategy.DFF_mnk_values[0] # DFF层矩阵大小M
        self.N = strategy.DFF_mnk_values[1] # DFF层矩阵大小N
        self.K = strategy.DFF_mnk_values[2] # DFF层矩阵大小K
        self.storage_formats = strategy.DFF_storage_formats # 存储格式，在本层可能有点用

        # 向上回传数据
        self.K_reduction_latency = None # 结果矩阵拼接耗时
        #self.K_reduction_io_count = None
        self.M_K_io_latency = None      # 左矩阵传入耗时
        self.K_N_io_latency = None      # 右矩阵传入耗时
        self.M_N_io_latency = None      # 结果矩阵传出耗时
        self.compute_latency = None     # 内层计算耗时
        self.mem_alloc_read_latency = None   # 内存分配耗时
        self.mem_alloc_write_latency = None  # 内存分配耗时

    def calculate_cycles(self):
        # 寄存器之间的数据搬运似乎不需要占用计算周期
        return 0

# 从这里开始，是Chip←UB←L0C→AccumDFF的分块结构，负责数据从矩阵计算Core的搬出
class UB_tile:
    """
    从 UnifyBuffer 分块到 L0C 上
    """
    def __init__(self, id, strategy: 'MatMul_Strategy' = None):
        # 保存 strategy 对象!!!
        self.strategy = strategy
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
        self.size = strategy.chip_type.L0C_CAPACITY             # L0C buffer结构存储容量
        self.min_load_size= strategy.chip_type.MIN_ACCESS['UB'] # L1 buffer最小访存大小
        self.M = strategy.L1_mnk_values[0] # L1 buffer层矩阵大小M【这仨是与L1_tile共用的】
        self.N = strategy.L1_mnk_values[1] # L1 buffer层矩阵大小N
        self.K = strategy.L1_mnk_values[2] # L1 buffer层矩阵大小K        

        # 向上回传数据
        self.K_reduction_latency = None # 结果矩阵拼接耗时
        #self.K_reduction_io_count = None
        self.M_K_io_latency = None      # 左矩阵传入耗时
        self.K_N_io_latency = None      # 右矩阵传入耗时
        self.M_N_io_latency = None      # 结果矩阵传出耗时
        self.compute_latency = None     # 内层计算耗时
        self.mem_alloc_read_latency = None   # 内存分配耗时
        self.mem_alloc_write_latency = None  # 内存分配耗时

    def calculate_cycles(self):
        """
        Calculate the cycles for the current layer based on the next layer's metrics.
        需要考虑分块策略（在最前面用self.block_strategy区分计算路径）
        """
        # 目前由 L0C 层执行写回，暂时赋0
        return 0

class L0C_tile:
    """
    从L0分块到寄存器上。注意，由于L0_A/B、L0_C是分开的，L0_C需要单开一个类
    """
    def __init__(self, strategy: 'MatMul_Strategy' = None):
        # 下层结构初始化
        self.next_layer = Accum_DFF_tile(strategy=strategy) # next_layer是Accum_DFF_tile实例，把策略传递进去
        self.strategy = strategy

        # 决策变量
        self.M_tile = strategy.DFF_mnk_values[0]
        self.N_tile = strategy.DFF_mnk_values[1]
        self.K_tile = strategy.DFF_mnk_values[2]
        self.block_strategy = strategy.DFF_block_strategy   # 分块策略；内积/外积【弄清楚分块策略对应的层的名称】
        self.storage_formats = strategy.DFF_storage_formats # 存储格式；行主序/列主序
        
        # 约束
        self.size = strategy.chip_type.AccumDFF_CAPACITY          # 寄存器结构存储容量
        self.min_load_size = strategy.chip_type.MIN_ACCESS['L0C'] # L0的最小访存大小
        self.M = strategy.L0_mnk_values[0] # L0层矩阵大小M
        self.N = strategy.L0_mnk_values[1] # L0层矩阵大小N
        self.K = strategy.L0_mnk_values[2] # L0层矩阵大小K

        # 向上回传数据
        self.K_reduction_latency = None # 结果矩阵拼接耗时
        #self.K_reduction_io_count = None
        self.M_K_io_latency = None      # 左矩阵传入耗时
        self.K_N_io_latency = None      # 右矩阵传入耗时
        self.M_N_io_latency = None      # 结果矩阵传出耗时
        self.compute_latency = None     # 内层计算耗时
        self.mem_alloc_read_latency = None   # 内存分配耗时
        self.mem_alloc_write_latency = None  # 内存分配耗时

    def calculate_cycles(self):
        """
        Calculate the cycles for the current layer.
        """
        # 结果矩阵字节数
        size_C = self.M_tile * self.N_tile * self.strategy.elem_bytes
        # 从L0C 到 UB
        loc_cycles = _device.io.store(size_C, 'L0C', 'UB')
        return loc_cycles

class Accum_DFF_tile:
    """
    寄存器上的结构运算
    """
    def __init__(self, strategy: 'MatMul_Strategy' = None):
        # 保存 strategy 对象!!!
        self.strategy = strategy
        # 约束
        self.size = strategy.chip_type.AccumDFF_CAPACITY               # DFF结构存储容量【通常来说应该要用更下层的容量做约束，但这已经是最底层了，我就先写着了】
        self.min_load_size = strategy.chip_type.MIN_ACCESS['AccumDFF'] # DFF的最小访存大小
        self.M = strategy.DFF_mnk_values[0] # DFF层矩阵大小M
        self.N = strategy.DFF_mnk_values[1] # DFF层矩阵大小N
        self.K = strategy.DFF_mnk_values[2] # DFF层矩阵大小K
        self.storage_formats = strategy.DFF_storage_formats # 存储格式，在本层可能有点用

        # 向上回传数据
        self.K_reduction_latency = None # 结果矩阵拼接耗时
        #self.K_reduction_io_count = None
        self.M_K_io_latency = None      # 左矩阵传入耗时
        self.K_N_io_latency = None      # 右矩阵传入耗时
        self.M_N_io_latency = None      # 结果矩阵传出耗时
        self.compute_latency = None     # 内层计算耗时
        self.mem_alloc_read_latency = None   # 内存分配耗时
        self.mem_alloc_write_latency = None  # 内存分配耗时

    def calculate_cycles(self):
        # 同AB_DFF
        return 0