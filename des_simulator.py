from enum import Enum
from dataclasses import dataclass
import heapq
from typing import List, Dict, Callable, Any, Tuple
from hardware import HW

LAST_SIM = None
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
    L0C_TO_ACCUM  = "L0C→Accum_DFF"
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
        self.sim = None

# 离散事件仿真器类
class PipelineSimulator:
    def __init__(self, parallel_limit: Dict[Pipeline, int], verbose: bool = False):
        self.current_cycle: float = 0.0
        self.event_queue: List[(float, int, PipelineEvent)] = []
        self.pipeline_busy: Dict[Pipeline, List[PipelineEvent]] = {p: [] for p in Pipeline}
        self.parallel_limit: Dict[Pipeline, int] = parallel_limit
        self._counter: int = 0
        self.verbose = verbose
        self.intervals = {p: [] for p in Pipeline}

    # 计算给定流水线集合的并集时长（cycles）
    def union_cycles_of(self, include):
        iv = []
        for p in include:
            iv.extend(self.intervals.get(p, []))
        if not iv:
            return 0.0
        iv.sort(key=lambda x: x[0])
        merged = []
        cur_s, cur_e = iv[0]
        for s, e in iv[1:]:
            if s <= cur_e:
                cur_e = max(cur_e, e)
            else:
                merged.append((cur_s, cur_e))
                cur_s, cur_e = s, e
        merged.append((cur_s, cur_e))
        return sum(e - s for s, e in merged)
    
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
        deps = dependencies or []
        max_dep_end = max((dep.end_cycle for dep in deps), default=0.0)
        limit = self.parallel_limit[pipeline]

        # 每条流水线一个最小堆，记录并发中的“结束时间”
        if not hasattr(self, '_pipe_end_heap'):
            self._pipe_end_heap = {p: [] for p in Pipeline}
        heap = self._pipe_end_heap[pipeline]

        # 候选开始时间：不早于依赖/当前时间
        start = max(max_dep_end, self.current_cycle)

        # 释放在 start 之前已结束的占用
        while heap and heap[0] <= start:
            import heapq
            heapq.heappop(heap)

        # 并发已满 → 推迟到最早结束时刻
        if len(heap) >= limit:
            earliest_end = heap[0]
            if earliest_end > start:
                start = earliest_end
                import heapq
                heapq.heappop(heap)

        evt = PipelineEvent(op, level, pipeline, duration, deps, on_complete, metadata)
        evt.start_cycle = start
        evt.end_cycle   = start + duration

        import heapq
        heapq.heappush(heap, evt.end_cycle)

        # 仅记录时间区间（做并集统计），不再堆积所有事件对象以省内存
        self.intervals[pipeline].append((evt.start_cycle, evt.end_cycle))

        # 事件计数 + 上限保护
        self._counter += 1
        if self._counter > getattr(HW, 'MAX_EVENTS', 3000000):
            raise RuntimeError(f'Event cap exceeded: {self._counter}')

        heapq.heappush(self.event_queue, (evt.end_cycle, self._counter, evt))
        return evt



    # 仿真启动器
    def run(self, max_cycles: float = float('inf')) -> float:
        import heapq
        while self.event_queue and self.current_cycle < max_cycles:
            end_cycle, _, evt = heapq.heappop(self.event_queue)
            self.current_cycle = end_cycle
            if self.verbose and evt.level == TileLevel.CHIP and evt.op == OpType.MEM_TO_EXT:
                meta_id = getattr(evt.metadata, 'id', None)
                print(f"Chip_tile ID={meta_id} 完成 at 周期 {evt.end_cycle:.1f}")
            # 防守式移除，避免大列表 O(N) 失败导致卡顿
            try:
                self.pipeline_busy[evt.pipeline].remove(evt)
            except ValueError:
                pass
            if evt.on_complete:
                evt.on_complete(evt)
        if self.verbose:
            print(f"*** Total cycles: {self.current_cycle:.1f} ***")
        return self.current_cycle


    def run_with_breakdown(self, max_cycles: float = float('inf')):
        """
        运行仿真，返回总周期, 各流水线的忙碌周期
        忙碌时间用事件时间区间的并集
        """
        total_cycles = self.run(max_cycles)

        def union_length(intervals):
            if not intervals:
                return 0.0
            iv = sorted(intervals, key=lambda x: x[0])
            merged = []
            cur_s, cur_e = iv[0]
            for s, e in iv[1:]:
                if s <= cur_e:
                    cur_e = max(cur_e, e)
                else:
                    merged.append((cur_s, cur_e))
                    cur_s, cur_e = s, e
            merged.append((cur_s, cur_e))
            return sum(e - s for s, e in merged)
        totals = {p: union_length(self.intervals[p]) for p in Pipeline}
        global LAST_SIM
        LAST_SIM = self

        return total_cycles, totals

    

