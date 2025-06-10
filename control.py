from init import IO_BW, CUBE_MACS_PER_CYCLE, MIN_ACCESS
import math

def align(size: int, granularity: int) -> int:
    """向上对齐到 granularity 的整数倍"""
    return math.ceil(size / granularity) * granularity

class ComputeModule:
    def compute(self, M: int, N: int, K: int) -> float:
        #执行 MxK * KxN 矩阵乘，返回周期数
        macs = M * N * K
        return macs / CUBE_MACS_PER_CYCLE

class IOModule:
    def load(self, size: int, src: str, dst: str) -> float:
        #模拟 src->dst 的 DMA，返回周期
        # 把原始的 src 和 dst 转换为 IO_BW 的键并根据键名找到对应的带宽
        key = f"{src}→{dst}"
        bw = IO_BW[key]
        # 把要传输的字节数对齐到最小访问粒度并计算周期
        aligned = align(size, MIN_ACCESS[dst])
        return aligned / bw

    # 逻辑同上，写回
    def store(self, size_bytes: int, src: str, dst: str) -> float:
        """模拟 src->dst 的 DMA，返回周期"""
        return self.load(size_bytes, src, dst)
    
class MemoryModule:
    def alloc(self, size_bytes: int):
        """模拟内存分配，不计周期，返回指针占位符"""
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