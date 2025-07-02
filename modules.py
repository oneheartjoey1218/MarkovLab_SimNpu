import math
from hardware import HW

def align(size: int, granularity: int) -> int:
    """向上对齐到 granularity 的整数倍"""
    return math.ceil(size / granularity) * granularity

class ComputeModule:
    def compute(self, M: int, N: int, K: int) -> float:
        """执行 M×K × K×N 矩阵乘，返回周期数"""
        return (M * N * K) / HW.CUBE_MACS_PER_CYCLE
    
    # 新增：使用指令时长表的计算方法！
    def compute_with_time_table(self, M: int, N: int, K: int) -> float:
        """执行 M×K × K×N 矩阵乘，从指令时长表中获取时间，若表中无数据则使用原方法计算"""
        instruction_time = HW.INSTRUCTION_TIME['compute']
        if instruction_time is None:
            return self.compute(M, N, K)
        else:
            return instruction_time

class IOModule:
    def load(self, size: int, src: str, dst: str) -> float:
        """模拟 src->dst 的 DMA，返回周期"""
        key     = f"{src}→{dst}"
        bw      = HW.IO_BW[key]
        aligned = align(size, HW.MIN_ACCESS[dst])
        return aligned / bw

    def store(self, size_bytes: int, src: str, dst: str) -> float:
        """模拟 src->dst 的 DMA（写回），返回周期"""
        return self.load(size_bytes, src, dst)
    
    # 新增：使用指令时长表的加载方法！
    def load_with_time_table(self, size: int, src: str, dst: str) -> float:
        """模拟 src->dst 的 DMA，从指令时长表中获取时间，若表中无数据则使用原方法计算"""
        instruction_time = HW.INSTRUCTION_TIME['load']
        if instruction_time is None:
            return self.load(size, src, dst)
        else:
            return instruction_time
        
    # 新增：使用指令时长表的存储方法！
    def store_with_time_table(self, size_bytes: int, src: str, dst: str) -> float:
        """模拟 src->dst 的 DMA（写回），从指令时长表中获取时间，若表中无数据则使用原方法计算"""
        instruction_time = HW.INSTRUCTION_TIME['store']
        if instruction_time is None:
            return self.store(size_bytes, src, dst)
        else:
            return instruction_time

class MemoryModule:
    def alloc(self, size_bytes: int):
        """模拟内存分配，不计周期，仅返回指针占位符"""
        return object()

    def free(self, ptr):
        """模拟内存释放"""
        pass

class Device:
    """聚合 Compute / IO / Memory 三大模块"""
    def __init__(self,
                 compute: ComputeModule,
                 io:      IOModule,
                 memory:  MemoryModule):
        self.compute = compute
        self.io      = io
        self.memory  = memory
        
class SetAssociativeCache:
    def __init__(self, capacity: int, block_size: int, assoc: int):
        self.block_size = block_size
        self.assoc = assoc
        # 组数 = (容量 // 块大小) // 关联度
        self.num_sets = (capacity // block_size) // assoc
        # 每组初始化空列表
        self.sets = {i: [] for i in range(self.num_sets)}
        
    def access(self, address: int) -> bool:
        # 按地址访问：True=命中, False=未命中 且插入新块
        block_no = address // self.block_size
        idx = block_no % self.num_sets
        way = self.sets[idx]
        if block_no in way:
            way.remove(block_no)
            way.append(block_no)
            return True
        if len(way) >= self.assoc:
            way.pop(0)
        way.append(block_no)
        return False
    
class InputOutputL2Cache:
    """按输入/输出划分的 L2 Cache容量, 替换逻辑相同"""
    def __init__(self,
                 total_capacity: int,
                 input_ratio: float,
                 block_size: int,
                 assoc: int):
        # 按比例划分容量
        input_cap  = int(total_capacity * input_ratio)
        output_cap = total_capacity - input_cap
        # 分别构建两段 Set-Associative Cache
        self.input_cache  = SetAssociativeCache(input_cap,  block_size, assoc)
        self.output_cache = SetAssociativeCache(output_cap, block_size, assoc)
        self.block_size   = block_size
        # 用于累积来自 L0C 的各个部分写回大小
        self.pending_writes: list[int] = []

    def read(self, address: int, size: int) -> float:
        total = 0.0
        lines = (size + self.block_size - 1) // self.block_size
        for i in range(lines):
            addr = address + i * self.block_size
            if self.input_cache.access(addr):
                # L2 命中路径：L2→L1
                total += device.io.load(self.block_size, 'L2', 'L1')
            else:
                # L2 未命中：DRAM→L2 + L2→L1
                total += device.io.load(self.block_size, 'DRAM', 'L2')
                total += device.io.load(self.block_size, 'L2',   'L1')
        return total
    
    def write(self, size: int) -> float:
            """
            缓存写回：只把本次块大小累积到 pending_writes，不立刻发 DRAM
            """
            self.pending_writes.append(size)
            return 0.0

    def flush(self) -> float:
            """
            一次性把所有 pending_writes 拼成一个连续大块，通过 L2->DRAM DMA 写回，清空 pending_writes
            """
            if not self.pending_writes:
                return 0.0
            total_size = sum(self.pending_writes)
            # 对齐到 MIN_ACCESS['L2']
            from modules import align
            aligned = align(total_size, HW.MIN_ACCESS['L2'])
            # 发起一次大 DMA
            cycles = device.io.store(aligned, 'L2', 'DRAM')       
            self.pending_writes.clear()
            return cycles

'''
# 旧L2Cache实现，暂时不删以防万一
class L2Cache:
    """单核 L2 Cache，集合关联命中模型"""
    def __init__(self):
        self.cache = SetAssociativeCache(
            capacity = HW.L2_CAPACITY,
            block_size = HW.MIN_ACCESS['L2'],
            assoc = HW.L2_ASSOCIATIVITY
        )

    def read(self, address: int, size: int) -> float:
        total = 0.0
        lines = (size + self.cache.block_size - 1) // self.cache.block_size
        for i in range(lines):
            # 真实地址 = 基地址 + 块内偏移
            addr = address + i * self.cache.block_size
            if self.cache.access(addr):
                # 命中：L2-L1
                total += device.io.load(self.cache.block_size, 'L2', 'L1')
            else:
                # 未命中：DRAM-L2 + L2-L1
                total += device.io.load(self.cache.block_size, 'DRAM', 'L2')
                total += device.io.load(self.cache.block_size, 'L2', 'L1')
        return total

    def write(self, size: int) -> float:
        # 分配策略
        return 0.0
'''

class L2CacheManager:
    """L2 Cache 管理器，按 core_id 分配分段管理的 L2 Cache"""
    def __init__(self, num_cores: int):
        self.caches = {
            i: InputOutputL2Cache(
                total_capacity = HW.L2_CAPACITY,
                input_ratio    = HW.L2_INPUT_RATIO,
                block_size     = HW.MIN_ACCESS['L2'],
                assoc          = HW.L2_ASSOCIATIVITY
            )
            for i in range(num_cores)
        }

    def read(self, core_id: int, address: int, size: int) -> float:
        return self.caches[core_id].read(address, size)

    def write(self, core_id: int, size: int) -> float:
        return self.caches[core_id].write(size)
    
    def flush(self, core_id: int) -> float:
        # 触发 core_id 对应 L2Cache 的一次性拼接写回
        return self.caches[core_id].flush()
    
# 全局多核 L2 Cache 管理器
L2_CACHE_MGR = L2CacheManager(HW.AI_CORE_COUNT)   

# 全局 device 实例
device = Device(
    compute=ComputeModule(),
    io=IOModule(),
    memory=MemoryModule()
)
