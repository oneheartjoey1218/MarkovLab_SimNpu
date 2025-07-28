from enum import Enum
from dataclasses import dataclass
import heapq
from typing import List, Dict, Callable, Any

# 定义各个流水线阶段的类
class Pipeline(Enum):
    OUT2 = "OUT2"   # 外部 → MEM
    OUT1 = "OUT1"   # L2 → MEM → 外部
    FIX  = "FIX"    # L0C → L2
    MTE2 = "MTE2"   # MEM → L2 → L1
    MTE1 = "MTE1"   # L1 → L0A/L0B
    M    = "M"      # GEMM 计算阶段
    
# 定义数据流向的类
class OpType(Enum):
    EXT_TO_MEM    = "外部→MEM"
    MEM_TO_L2_L1  = "MEM→L2→L1"
    L1_TO_L0AB    = "L1→L0A/L0B"
    L0AB_TO_DFF   = "L0A/B→A/B_DFF"
    CUBE_GEMM     = "Cube GEMM"
    ACCUM_TO_L0C  = "Accum_DFF→L0C"
    L0C_TO_L2     = "L0C→L2"
    L2_TO_MEM     = "L2→MEM"
    MEM_TO_EXT    = "MEM→外部"

# 事件的 tile 层级
class TileLevel(Enum):
    CHIP = "chip_tile"
    L1   = "L1_tile"
    L0   = "L0_tile"
    CUBE = "cube_tile"

# 流水线事件的数据结构
@dataclass
class PipelineEvent:
    op: OpType # 操作类型
    level: TileLevel # 所在层级
    pipeline: Pipeline # 所属流水线
    duration: float # 周期数
    dependencies: List["PipelineEvent"] # 前置依赖
    on_complete: Callable[["PipelineEvent"], None] # 完成后的回调函数
    metadata: Any

    def __post_init__(self):
        self.start_cycle: float = 0.0 # 开始周期
        self.end_cycle: float = 0.0 # 结束周期，就是开始+duration

# 离散事件仿真器类
class PipelineSimulator:
    def __init__(self, parallel_limit: Dict[Pipeline, int]):
        self.current_cycle: float = 0.0 # 当前全局周期
        self.event_queue: List[(float, int, PipelineEvent)] = [] # 最小堆队列
        self.pipeline_busy: Dict[Pipeline, List[PipelineEvent]] = {p: [] for p in Pipeline} # 事件列表，流水线分开
        self.parallel_limit: Dict[Pipeline, int] = parallel_limit # 各流水线并发限制
        self._counter: int = 0

    # 添加一个事件并计算开始时间
    def add_event(
        self,
        op: OpType,
        level: TileLevel,
        pipeline: Pipeline,
        duration: float,
        dependencies: List[PipelineEvent] = None,
        on_complete: Callable[[PipelineEvent], None] = None,
        metadata: Any = None,
    ) -> PipelineEvent:
        deps = dependencies or [] # 前置依赖项
        max_dep_end = max((dep.end_cycle for dep in deps), default=0.0) # 找到所有依赖的结束周期中的最大值
        busy = self.pipeline_busy[pipeline] # 当前流水线上的已有的事件
        limit = self.parallel_limit[pipeline] # 并发限制
        pipeline_free = 0.0
        
        # 当前流水线已满的情况下，寻找最早的空闲时间
        if len(busy) >= limit:
            pipeline_free = max(evt.end_cycle for evt in busy)
            
        start = max(max_dep_end, pipeline_free, self.current_cycle) # 开始时间
        
        evt = PipelineEvent(op, level, pipeline, duration, deps, on_complete, metadata) # 创建事件
        evt.start_cycle = start
        evt.end_cycle = start + duration
        
        busy.append(evt) # 加入busy列表
        self._counter += 1
        
        # 放入事件队列并按结束时间排序
        heapq.heappush(self.event_queue, (evt.end_cycle, self._counter, evt))
        return evt


    # 仿真启动器
    def run(self, max_cycles: float = float('inf')) -> float:
        while self.event_queue and self.current_cycle < max_cycles:
            end_cycle, _, evt = heapq.heappop(self.event_queue) # 取出最早完成的事件
            self.current_cycle = end_cycle # 推进当前周期
            
            # 如果是 chip_tile 且是最终写回阶段，打印完成信息，测试用
            if evt.level == TileLevel.CHIP and evt.op == OpType.MEM_TO_EXT:
                meta_id = getattr(evt.metadata, 'id', None)
                print(f"Chip_tile ID={meta_id} 完成 at 周期 {evt.end_cycle:.1f}")
                
            # 从busy列表删除
            self.pipeline_busy[evt.pipeline].remove(evt)
            
            # 触发后续事件
            if evt.on_complete:
                evt.on_complete(evt)
        print(f"*** Total cycles: {self.current_cycle:.1f} ***") # 总周期数
        return self.current_cycle
