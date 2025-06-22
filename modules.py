import math
from hardware import HW

def align(size: int, granularity: int) -> int:
    """向上对齐到 granularity 的整数倍"""
    return math.ceil(size / granularity) * granularity

class ComputeModule:
    def compute(self, M: int, N: int, K: int) -> float:
        """执行 M×K × K×N 矩阵乘，返回周期数"""
        return (M * N * K) / HW.CUBE_MACS_PER_CYCLE

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
    
class L2CacheManager:
    # L2 Cache 管理器，按 core_id 分配单核 Cache
    def __init__(self, num_cores: int):
        self.caches = {i: L2Cache() for i in range(num_cores)}

    def read(self, core_id: int, address: int, size: int) -> float:
        return self.caches[core_id].read(address, size)

    def write(self, core_id: int, size: int) -> float:
        return self.caches[core_id].write(size)
    
# 全局多核 L2 Cache 管理器
L2_CACHE_MGR = L2CacheManager(HW.AI_CORE_COUNT)   

# 全局 device 实例
device = Device(
    compute=ComputeModule(),
    io=IOModule(),
    memory=MemoryModule()
)
