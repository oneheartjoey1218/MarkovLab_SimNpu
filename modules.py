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

# 全局 device 实例
device = Device(
    compute=ComputeModule(),
    io=IOModule(),
    memory=MemoryModule()
)
