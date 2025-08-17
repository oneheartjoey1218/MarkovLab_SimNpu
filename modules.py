import math
from hardware import HW

def align(size: int, granularity: int) -> int:
    """向上对齐到 granularity 的整数倍"""
    return math.ceil(size / granularity) * granularity

class ComputeModule:
    def _eff_from_curve(self, x: float) -> float:
        # 使用 HW.GFLOPS_EFF_CURVE 做分段线性插值，最终乘 COMPUTE_EFF_BIAS 并封顶到 1.0
        pts = getattr(HW, 'GFLOPS_EFF_CURVE', [(128,0.98),(16,0.95),(1,0.80),(0,0.40)])
        pts = sorted(pts, key=lambda t: t[0], reverse=True)
        bias = getattr(HW, 'COMPUTE_EFF_BIAS', 1.10)

        if x >= pts[0][0]:
            return min(1.0, pts[0][1] * bias)

        for (x1,y1),(x2,y2) in zip(pts, pts[1:]):
            if x <= x1 and x >= x2:
                if x1 == x2:
                    return min(1.0, y1 * bias)
                t = (x - x2) / (x1 - x2)
                val = y2 + t*(y1 - y2)
                return min(1.0, val * bias)

        return min(1.0, pts[-1][1] * bias)

    def compute(self, M: int, N: int, K: int) -> float:
        if getattr(HW, "ALIGN_COMPUTE_16", False):
            def ceil16(x: int) -> int: return (x + 15) // 16 * 16
            Mm = ceil16(M); Nn = ceil16(N); Kk = ceil16(K)
        else:
            Mm, Nn, Kk = M, N, K

        tiles_m = max(1, Mm // 16)
        tiles_n = max(1, Nn // 16)
        occ = tiles_m * tiles_n

        eff = self._eff_from_curve(occ)
        eff = max(0.05, min(1.0, eff))
        effective_macs = HW.CUBE_MACS_PER_CORE * eff
        return (Mm * Nn * Kk) / effective_macs

class IOModule:
    def _uplift_factor(self, key: str, aligned: int) -> float:
        # 原有的长突发提升逻辑
        rules = {
            'DRAM→L2':   [(2*1024*1024, 1.25), (8*1024*1024, 1.50)],
            'L2→DRAM':   [(2*1024*1024, 1.20), (8*1024*1024, 1.40)],
            'DRAM→EXT':  [(2*1024*1024, 1.40), (8*1024*1024, 1.80)],
            'EXT→DRAM':  [(2*1024*1024, 1.40), (8*1024*1024, 1.80)],
            'L2→L1':     [(512*1024,     1.10), (2*1024*1024, 1.25)],
        }
        uplift = 1.0
        for th, fac in rules.get(key, []):
            if aligned >= th:
                uplift = fac
        return uplift

    def _bw_eff_from_curve(self, mb: float) -> float:
        pts = getattr(HW, 'MEM_MB_EFF_CURVE',
                      [(128,0.98),(32,0.95),(8,0.90),(1,0.75),(0,0.50)])
        pts = sorted(pts, key=lambda t: t[0], reverse=True)
        bias = getattr(HW, 'MEM_EFF_BIAS', 1.20)

        if mb >= pts[0][0]:
            return min(1.0, pts[0][1] * bias)

        for (x1,y1),(x2,y2) in zip(pts, pts[1:]):
            if mb <= x1 and mb >= x2:
                if x1 == x2:
                    return min(1.0, y1 * bias)
                t = (mb - x2) / (x1 - x2)
                val = y2 + t*(y1 - y2)
                return min(1.0, val * bias)

        return min(1.0, pts[-1][1] * bias)

    def load(self, size: int, src: str, dst: str) -> float:
        """模拟 src->dst 的 DMA，返回周期"""
        key     = f"{src}→{dst}"
        bw      = HW.IO_BW[key]
        aligned = align(size, HW.MIN_ACCESS[dst])
        uplift  = self._uplift_factor(key, aligned)
        mb_eff  = self._bw_eff_from_curve(aligned / (1024*1024))
        return aligned / (bw * uplift * mb_eff)

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
        # number of cache lines
        lines = (size + self.block_size - 1) // self.block_size

        # Per-line costs
        l2_to_l1 = device.io.load(self.block_size, 'L2', 'L1')
        dram_to_l2 = device.io.load(self.block_size, 'DRAM', 'L2')

        use_db = getattr(HW, 'MTE2_DOUBLE_BUFFER', True)
        miss_cost = max(dram_to_l2, l2_to_l1) if use_db else (dram_to_l2 + l2_to_l1)

        # If a fixed hit rate is provided, bypass set-associative simulation for speed/stability
        if hasattr(HW, 'L2_FIXED_HIT_RATE') and HW.L2_FIXED_HIT_RATE is not None:
            hit_rate = float(HW.L2_FIXED_HIT_RATE)
            if hit_rate < 0.0: hit_rate = 0.0
            if hit_rate > 1.0: hit_rate = 1.0
            hit_lines = int(lines * hit_rate)
            miss_lines = lines - hit_lines
            return hit_lines * l2_to_l1 + miss_lines * miss_cost

        # Otherwise, fall back to set-associative access modeling
        total = 0.0
        for i in range(lines):
            addr = address + i * self.block_size
            if self.input_cache.access(addr):
                total += l2_to_l1
            else:
                total += miss_cost
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
