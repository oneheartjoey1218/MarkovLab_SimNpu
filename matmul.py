from hardware import HardwareSpec, HW # 全局硬件实例

from modules import device as _device, L2_CACHE_MGR
from typing import List, Dict
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
        # 下层结构初始化
        self.strategy   = strategy
        self.elem_bytes = strategy.elem_bytes
        self.next_layer = Chip_tile(
            strategy = strategy,
            M        = strategy.chip_mnk_values[0],
            N        = strategy.chip_mnk_values[1],
            K        = strategy.chip_mnk_values[2],
        )
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

    def build_tiles(self):
        """
        构建 self.tiles（三维 Chip_tile 数组），并对每个 tile 调用 build_subtiles()。
        """
        from math import ceil
        import numpy as np

        # 1) 计算完整块数与边界余数
        M_l2 = self.M // self.M_tile
        N_l2 = self.N // self.N_tile
        K_l2 = self.K // self.K_tile
        M_rem = self.M % self.M_tile
        N_rem = self.N % self.N_tile
        K_rem = self.K % self.K_tile

        # 2) 分配空数组
        dims = [
            ceil(self.M / self.M_tile),
            ceil(self.N / self.N_tile),
            ceil(self.K / self.K_tile),
        ]
        self.tiles = np.empty(dims, dtype=object)

        # 3) 批量初始化完整分块
        block = self.create_chip_tile(self.M_tile, self.N_tile, self.K_tile)
        if M_l2 and N_l2 and K_l2:
            self.tiles[:M_l2, :N_l2, :K_l2] = block

        # 4) 各方向边界
        if M_rem:
            self.tiles[-1, :N_l2, :K_l2] = self.create_chip_tile(M_rem, self.N_tile, self.K_tile)
        if N_rem:
            self.tiles[:M_l2, -1, :K_l2] = self.create_chip_tile(self.M_tile, N_rem, self.K_tile)
        if K_rem:
            self.tiles[:M_l2, :N_l2, -1] = self.create_chip_tile(self.M_tile, self.N_tile, K_rem)
        if M_rem and N_rem:
            self.tiles[-1, -1, :K_l2] = self.create_chip_tile(M_rem, N_rem, self.K_tile)
        if M_rem and K_rem:
            self.tiles[-1, :N_l2, -1] = self.create_chip_tile(M_rem, self.N_tile, K_rem)
        if N_rem and K_rem:
            self.tiles[:M_l2, -1, -1] = self.create_chip_tile(self.M_tile, N_rem, K_rem)
        if M_rem and N_rem and K_rem:
            self.tiles[-1, -1, -1] = self.create_chip_tile(M_rem, N_rem, K_rem)

        # 5) 为每个 Chip_tile 生成 L1/L0 子块
        for chip in self.tiles.flatten():
            if chip is not None:
                chip.build_subtiles()
                    
    def load_look_up_table(self):
        # 这里需要根据实际情况加载查找表
        self.look_up_table = {}
        
    def validate_cache_capacity(self):
        # 计算所需的缓存空间
        required_cache = (
            self.M_tile * self.N_tile + 
            self.N_tile * self.K_tile + 
            self.M_tile * self.K_tile
        ) * self.elem_bytes
        
        # 获取可用的缓存容量 L2_CAPACITY!
        available_cache = HW.L2_CAPACITY
        assert required_cache <= available_cache, (
                f"缓存容量不足: 所需空间 {required_cache} 字节，"
                f"可用空间 {available_cache} 字节"
            )
            #传入参数待修改！！！！！！！！！！！！！！！！！！！！！
    '创建一个新的Chip_tile分块！'
    
    def create_chip_tile(self, M, N, K):
        return Chip_tile(
            strategy=self.strategy,
            M=M, N=N, K=K
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
                        
    def calculate_pipelined_cycles(self) -> float:
        """
        离散事件仿真 + 并行加载优化：
        批量 L1/L0 加载阶段改为并行（取最大时延）
        """
        # 1) 查表与缓存校验
        if not hasattr(self, 'look_up_table') or self.look_up_table is None:
            self.load_look_up_table()
        self.validate_cache_capacity()

        # 2) 多级分块
        self.build_tiles()

        # 3) 初始化 DES 调度器
        from des_simulator import PipelineSimulator, Pipeline, OpType, TileLevel
        core_cnt = self.strategy.chip_type.AI_CORE_COUNT
        parallel_limit = {
            Pipeline.OUT2: 1,
            Pipeline.OUT1: 1,
            Pipeline.FIX:  core_cnt,
            Pipeline.MTE2: core_cnt,
            Pipeline.MTE1: core_cnt,
            Pipeline.M:    core_cnt,
        }
        sim = PipelineSimulator(parallel_limit)

        # 回调：批量 L0 加载后调度计算
        def cb_compute_batch(evt_batch):
            for l0 in evt_batch.metadata:
                sim.add_event(
                    OpType.CUBE_GEMM, TileLevel.CUBE, Pipeline.M,
                    l0.compute_latency,
                    [evt_batch], on_complete=cb_write, metadata=l0
                )

        # 回调：L1-L0 并行加载
        def cb_l0(evt_l1):
            l0_list = evt_l1.metadata.l0_tiles
            # 并行加载：对每个 L0_tile 分别取 A/B 最大延迟，再对所有 L0_tile 取最大
            batch_dur = max((max(l0.load_A_latency, l0.load_B_latency) for l0 in l0_list))
            sim.add_event(
                OpType.L1_TO_L0AB, TileLevel.L0, Pipeline.MTE1,
                batch_dur, [evt_l1], on_complete=cb_compute_batch, metadata=l0_list
            )

        # 回调：写回链
        def cb_write(evt_m):
            l0 = evt_m.metadata
            e1 = sim.add_event(
                OpType.ACCUM_TO_L0C, TileLevel.L0, Pipeline.MTE1,
                l0.write_latency, [evt_m], metadata=l0
            )
            e2 = sim.add_event(
                OpType.L0C_TO_L2, TileLevel.L1, Pipeline.FIX,
                l0.write_latency, [e1], metadata=l0
            )
            e3 = sim.add_event(
                OpType.L2_TO_MEM, TileLevel.CHIP, Pipeline.OUT1,
                l0.write_latency, [e2], metadata=l0
            )
            sim.add_event(
                OpType.MEM_TO_EXT, TileLevel.CHIP, Pipeline.OUT2,
                l0.write_latency, [e3], metadata=l0
            )

        # 回调
        def cb_l1(evt_chip):
            for l1 in evt_chip.metadata.l1_tiles:
                # 并行加载 A/B 到 L1
                load_dur = max(l1.load_A_latency, l1.load_B_latency)
                sim.add_event(
                    OpType.MEM_TO_L2_L1, TileLevel.L1, Pipeline.MTE2,
                    load_dur, [evt_chip], on_complete=cb_l0, metadata=l1
                )

        # 4) 首批 Chip 级事件
        dim0, dim1, dim2 = self.tiles.shape
        for m in range(dim0):
            for n in range(dim1):
                for k in range(dim2):
                    chip = self.tiles[m, n, k]
                    if not chip:
                        continue
                    # 并行加载 A/B 至 chip
                    lat = max(chip.out2_A, chip.out2_B)
                    sim.add_event(
                        OpType.EXT_TO_MEM, TileLevel.CHIP, Pipeline.OUT2,
                        lat, [], on_complete=cb_l1, metadata=chip
                    )

        # 5) 运行并返回总周期
        return sim.run()


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
        M, N, K = sim.M, sim.N, sim.K
        mt, nt, kt = sim.M_tile, sim.N_tile, sim.K_tile
        self.M_tiles = ceil(M / mt)
        self.N_tiles = ceil(N / nt)
        self.K_tiles = ceil(K / kt)
        self.core_cnt = sim.strategy.chip_type.AI_CORE_COUNT
        self.row_grp   = max(1, min(self.M_tiles, self.core_cnt // 4))
        self.core_map  = self._build_core_map()
        # Sflag延迟，目前未知保留为0
        self.flag_delay = getattr(sim, 'flag_delay', 0)
        # 记录每层上一次调度的 tile 坐标，用于 A/B 重用判断
        self.prev_coord = [None] * 6



        for m in range(self.M_tiles):
            for n in range(self.N_tiles):
                for k in range(self.K_tiles):
                    self.states[(m, n, k)] = PipelineTileState((m, n, k))

        self.layer_busy     = [[] for _ in range(6)]
        ai = sim.strategy.chip_type.AI_CORE_COUNT
        self.parallel_limit = {3: ai, 4: ai, 5: ai}
    
    NEXT_STAGE = {          # 正确的流水顺序
        0: 3,   # OUT2  → MTE2
        3: 4,   # MTE2  → MTE1
        4: 5,   # MTE1  → M
        5: 2,   # M     → FIX
        2: 1,   # FIX   → OUT1
        1: 6    # OUT1  → 结束 (6 表示全部完成)
    }
    
    def _build_core_map(self):
        """
        将 M×N tile 网格划成 row_grp 行，每行用 4 个 core，
        行内再按 n 维 Round-Robin。返回 dict:
            {(m,n): core_id}
        """
        cores_per_row = self.core_cnt // self.row_grp # 4
        mapping = {}
        for m in range(self.M_tiles):
            base = (m % self.row_grp) * cores_per_row # 行基址
            for n in range(self.N_tiles):
                offset = n % cores_per_row # 同行内散列
                mapping[(m, n)] = base + offset
        return mapping
    
    def _assign_core(self, m, n, k):
        return self.core_map[(m, n)]
    
    def simulate(self):
        while not self._all_done():
            self._advance_cycle()
        for core_id in range(self.sim.strategy.chip_type.AI_CORE_COUNT):
            self.cycle += L2_CACHE_MGR.flush(core_id)
        return self.cycle

    def _advance_cycle(self):
        self._progress_running_stages()
        self._try_launch_new_stages()
        self.cycle += 1

    def _progress_running_stages(self):
        for stage_id in range(6):
            done = []
            for coord in list(self.layer_busy[stage_id]):
                state = self.states[coord]
                if state.step():          # step() 在剩余周期→0 时返回 True
                    done.append(coord)    # 收集已完成的 tile
            for coord in done:
                self.layer_busy[stage_id].remove(coord)
                state = self.states[coord]
                state.current_stage = self.NEXT_STAGE[stage_id]

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
            if latency == 0:
                state.completed_stages.add(stage)
                state.current_stage = self.NEXT_STAGE[stage]
                self.prev_coord[stage] = coord
                # 直接进入下一轮调度，不占用 layer_busy
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
            return state.is_stage_completed(2)
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
        tile      = self.chip_tiles[m, n, k]
        C_bytes = tile.M_tile * tile.N_tile * tile.elem_bytes
        '''
        chip_tile = self.chip_tiles[m, n, k]
        A = chip_tile.M * chip_tile.K * chip_tile.elem_bytes
        B = chip_tile.K * chip_tile.N * chip_tile.elem_bytes
        C = chip_tile.M * chip_tile.N * chip_tile.elem_bytes
        '''
        lut = getattr(self.sim, 'look_up_table', {})
        key = (tile.M_tile, tile.N_tile, tile.K_tile)
        
        # 重用判断：根据上次调度的 tile 坐标
        prev = self.prev_coord[stage_id]
        if prev is None:
            load_A = load_B = 1
        else:
            pm, pn, pk = prev
            load_A = 1 if (n != pn or k != pk) else 0
            load_B = 1 if (m != pm or k != pk) else 0

        if stage_id == 0: # OUT2
            # A
            if key in lut and 'out2_A' in lut[key]:
                a_lat = lut[key]['out2_A']
            else:
                a_lat = tile.out2_A
            # B
            if key in lut and 'out2_B' in lut[key]:
                b_lat = lut[key]['out2_B']
            else:
                b_lat = tile.out2_B
            base = load_A * a_lat + load_B * b_lat
            
        elif stage_id == 1: # OUT1
            core_id = self._assign_core(m, n, k)
            base = L2_CACHE_MGR.flush(core_id)
                
        elif stage_id == 2: # FIX
            core_id = self._assign_core(m, n, k)
            base = L2_CACHE_MGR.write(core_id, C_bytes)
            
        elif stage_id == 3:  # MTE2
            core_id = self._assign_core(m, n, k)
            stride  = HW.MIN_ACCESS['L2']
            a_addr  = ((m*self.N_tiles + n)*self.K_tiles + k) * stride
            b_addr  = ((n*self.M_tiles + m)*self.K_tiles + k) * stride
            a_lat   = L2_CACHE_MGR.read(core_id, a_addr,
                                        tile.M_tile*tile.K_tile*tile.elem_bytes)
            b_lat   = L2_CACHE_MGR.read(core_id, b_addr,
                                        tile.K_tile*tile.N_tile*tile.elem_bytes)
            base = load_A * a_lat + load_B * b_lat

            
        elif stage_id == 4: # MTE1
            # 前一个 K-depth 的 FIX 必须完成
            if k > 0 and not self.states[(m, n, k-1)].is_stage_completed(2):
                return None
            if key in lut and 'mte1' in lut[key]:
                base = lut[key]['mte1']
            else:
                base = tile.mte1
            
        elif stage_id == 5: # M
            if key in lut and 'compute' in lut[key]:
                base = lut[key]['compute']
            else:
                base = tile.compute
        else:
            return None
        return base + self.flag_delay
    
    def _all_done(self):
        return all(5 in s.completed_stages for s in self.states.values())


                        
class Chip_tile:
    """
    从芯片分块到 L1 上，910C有24个L1_tile和24个UB_tile。这里的逻辑要仔细想清楚，包括往L1的io读时间、从UB回来的io写时间、L2 cache机制等因素
    """
    def __init__(self, strategy: 'MatMul_Strategy', M: int, N: int, K: int):
        
        '''新写的流水线逻辑，其下的保留后续可能会用到'''
        # 保存策略和分块尺寸
        self.strategy   = strategy
        self.M_tile     = M
        self.N_tile     = N
        self.K_tile     = K
        self.elem_bytes = strategy.elem_bytes  # fp16=2, else=4
        
        # 2) 计算 A/B/C 三块的字节数
        A_bytes = self.M_tile * self.K_tile * self.elem_bytes
        B_bytes = self.K_tile * self.N_tile * self.elem_bytes
        C_bytes = self.M_tile * self.N_tile * self.elem_bytes
        
        # 3) OUT2 (EXT → DRAM)
        self.out2_A = _device.io.load(A_bytes, 'EXT', 'DRAM')
        self.out2_B = _device.io.load(B_bytes, 'EXT', 'DRAM')
        
        '''
        # 4) MTE2 (DRAM → L2 → L1)
        self.mte2_A = (
            _device.io.load(A_bytes, 'DRAM', 'L2') +
            _device.io.load(A_bytes, 'L2', 'L1')
        )
        self.mte2_B = (
            _device.io.load(B_bytes, 'DRAM', 'L2') +
            _device.io.load(B_bytes, 'L2', 'L1')
        )
        '''
        
        # 5) OUT1 (L2 → DRAM 写回 C)
        # self.out1_C = _device.io.store(C_bytes, 'L2', 'DRAM')
        
        # 6) FIX (L0C → L2 写回 C)
        self.fix_C = _device.io.store(C_bytes, 'L0C', 'L2')

        # 7) MTE1 (L1 → L0A/L0B + AccumDFF → L0C)
        latA = _device.io.load(A_bytes, 'L1', 'L0A')
        latB = _device.io.load(B_bytes, 'L1', 'L0B')
        latC = _device.io.store(C_bytes, 'AccumDFF', 'L0C')
        self.mte1 = max(latA, latB) + latC
        
        # 8) M (核心矩阵乘计算)
        self.compute = _device.compute.compute(M, N, K)
        '''
        新流水线逻辑至此
        '''
        self.l1_tiles: List[L1_tile] = []
        self.l0_tiles: Dict[int, List[L0_tile]] = {}
        
    def build_subtiles(self):
        """
        在每个 Chip_tile 上，按照 L1 缓存容量切分出 l1_tiles，
        并为每个 L1_tile 调用 build_subtiles() 生成对应的 L0_tile。
        """
        from matmul import split_blocks, L1_tile

        # L1 切分
        max_L1 = self.strategy.chip_type.L1_CAPACITY // self.elem_bytes
        self.l1_tiles = []
        for idx, (m1, n1, k1) in enumerate(
            split_blocks([(self.M_tile, self.N_tile, self.K_tile)], max_L1)
        ):
            l1 = L1_tile(id=idx, strategy=self.strategy, M=m1, N=n1, K=k1)
            # 生成 L0 级子块
            l1.build_subtiles()
            self.l1_tiles.append(l1)
            
        
        
class L1_tile:
    def __init__(self, id: int, strategy: 'MatMul_Strategy', M: int, N: int, K: int):
        self.id = id
        self.strategy   = strategy
        self.M_tile     = M
        self.N_tile     = N
        self.K_tile     = K
        self.elem_bytes = strategy.elem_bytes
        # DRAM->L2 + L2->L1 延迟计算
        
        # 计算A和B从DRAM到L2，再从L2到L1的传输延迟
        io_bw = strategy.chip_type.IO_BW # I/O带宽字典
        A_bytes = M * K * self.elem_bytes # A矩阵在该tile中的字节数
        B_bytes = K * N * self.elem_bytes # B矩阵在该tile中的字节数
        
        # 加载A和B所需时间
        self.load_A_latency = A_bytes / io_bw['DRAM→L2'] + A_bytes / io_bw['L2→L1']
        self.load_B_latency = B_bytes / io_bw['DRAM→L2'] + B_bytes / io_bw['L2→L1']
        
        # 初始化空的L0子块列表（后续通过切分填充）
        self.l0_tiles: List['L0_tile'] = []

    def build_subtiles(self):
        """
        为当前 L1_tile 切分 L0_tile 并填充 self.l0_tiles
        """
        from matmul import split_blocks, L0_tile # 导入L0_tile类与切分函数
        
        # 根据L0A和L0B容量，计算每个L0子tile可容纳的最大元素数
        max_L0A = self.strategy.chip_type.L0A_CAPACITY // self.elem_bytes
        max_L0B = self.strategy.chip_type.L0B_CAPACITY // self.elem_bytes
        max_L0  = min(max_L0A, max_L0B)
        
        self.l0_tiles = [] # 初始化 L0 子 tile 列表
        
        # 使用split_blocks对当前L1 tile的M×N×K进行切分，得到多个L0子块
        for idx, (m0, n0, k0) in enumerate(
            split_blocks([(self.M_tile, self.N_tile, self.K_tile)], max_L0)
        ):
            # 构建 L0_tile 并加入列表
            self.l0_tiles.append(
                L0_tile(id=idx, strategy=self.strategy, M=m0, N=n0, K=k0)
            )
    
class L0_tile:
    """"
    大部分和L1差不多
    """
    def __init__(self, id: int, strategy: 'MatMul_Strategy', M: int, N: int, K: int):
        self.id = id
        self.strategy   = strategy
        self.M_tile     = M
        self.N_tile     = N
        self.K_tile     = K
        self.elem_bytes = strategy.elem_bytes
        
        io_bw   = strategy.chip_type.IO_BW
        macs_pc = strategy.chip_type.CUBE_MACS_PER_CYCLE
        
        A_bytes = M * K * self.elem_bytes
        B_bytes = K * N * self.elem_bytes
        C_bytes = M * N * self.elem_bytes
        
        # A和B从L1到LA的传输延迟
        self.load_A_latency    = A_bytes / io_bw['L1→L0A']
        self.load_B_latency    = B_bytes / io_bw['L1→L0B']
        
        # GEMM计算延迟 = 总MAC次数 / 每周期可完成的MAC数
        self.compute_latency   = (M * N * K) / macs_pc
        
        # 写入L0C的延迟
        self.write_latency     = C_bytes / io_bw['AccumDFF→L0C'] 


class UB_tile:
    """
    Unify Buffer 级别 Tile 的占位实现。
    """
    def __init__(self, id, strategy=None, M=None, N=None, K=None):
        self.UB_id    = id
        self.M_tile   = M
        self.N_tile   = N
        self.K_tile   = K

    def calculate_cycles(self):
        return 0


class L0C_tile:
    """
    L0C（输出缓冲）级别 Tile 的占位实现。
    """
    def __init__(self, strategy=None, M=None, N=None, K=None):
        self.M_tile   = M
        self.N_tile   = N
        self.K_tile   = K

    def calculate_cycles(self):
        return 0


class Accum_DFF_tile:
    """
    累加寄存器级别 Tile 的占位实现。
    """
    def __init__(self, strategy=None, M=None, N=None, K=None):
        self.M_tile   = M
        self.N_tile   = N
        self.K_tile   = K

    def calculate_cycles(self):
        return 0
