class HardwareSpec:
    def __init__(self):
        self.AI_CORE_COUNT = 24
        self.CUBE_MACS_PER_CORE = 4096
        # For backward compatibility: use per-core number here
        self.CUBE_MACS_PER_CYCLE = self.CUBE_MACS_PER_CORE
        self.CHIP_MACS_PER_CYCLE = self.CUBE_MACS_PER_CORE * self.AI_CORE_COUNT
        self.CLOCK_FREQ = 1.85e9  # Hz
        # 每个物理周期包含的 AIC 内部 tick 数（与 profiling 对齐）
        self.TICKS_PER_CYCLE = 6
        # AIC tick 的等效频率
        self.AIC_TICK_FREQ = self.CLOCK_FREQ * self.TICKS_PER_CYCLE
        self.ALIGN_COMPUTE_16 = False

        self.MIN_ACCESS = {
            'Chip': 2,
            'L2': 512,
            'L1': 32,
            'L0A': 512,
            'L0B': 512,
            'L0C': 512,
            'UB': 32,
            'SB': 2,
            'ABDFF': 512,
            'AccumDFF': 512
        }
        self.MIN_ACCESS['DRAM'] = self.MIN_ACCESS['L2']
        self.MIN_ACCESS['EXT']  = self.MIN_ACCESS['Chip']

        # Capacities
        self.MEM_CAPACITY      = 64 * 1024**3
        self.L2_CAPACITY       = 192 * 1024**2
        self.L1_CAPACITY       = 1   * 1024**2
        self.L0A_CAPACITY      = 64  * 1024
        self.L0B_CAPACITY      = 64  * 1024
        self.L0C_CAPACITY      = 256 * 1024
        self.UB_CAPACITY       = 256 * 1024
        self.SB_CAPACITY       = 16  * 1024
        self.ABDFF_CAPACITY    = 512
        self.AccumDFF_CAPACITY = 512

        # I/O bandwidth (bytes/cycle)
        def tbps_to_bpc(tbps: float) -> float:
            return (tbps * 1e12) / self.CLOCK_FREQ
        def gbps_to_bpc(gbps: float) -> float:
            return (gbps * 1e9) / self.CLOCK_FREQ

        l2_to_l1_bpc   = tbps_to_bpc(4.07)
        dram_to_l2_bpc = tbps_to_bpc(1.35)
        l1_to_l0a_bpc  = gbps_to_bpc(220.0)
        l1_to_l0b_bpc  = gbps_to_bpc(440.0)

        self.IO_BW = {
            'DRAM→L2': float(dram_to_l2_bpc),
            'L2→L1'  : float(l2_to_l1_bpc),
            'L1→L0A' : float(l1_to_l0a_bpc),
            'L1→L0B' : float(l1_to_l0b_bpc),
            'AccumDFF→L0C': 210.0,

            'L0C→L2' : 86.0,
            'L2→DRAM': float(dram_to_l2_bpc),
            'DRAM→EXT': 32.0,
            'EXT→DRAM': 32.0,

            'L2→L0C' : 110.0,
            'L0A→L1' : float(self.MIN_ACCESS['L0A']),
            'L0B→L1' : float(self.MIN_ACCESS['L0B']),
            'L1→L0C' : float(self.MIN_ACCESS['L0C']),
            'L0C→L1' : 20.0,
            'L0C→UB' : float(self.MIN_ACCESS['L0C']),
            'UB→L0C' : float(self.MIN_ACCESS['L0C']),
            'UB→L1'  : float(self.MIN_ACCESS['L1']),
        }

        # L2 cache policy
        self.L2_ASSOCIATIVITY = 8
        self.L2_INPUT_RATIO   = 0.8
        self.L2_FIXED_HIT_RATE = None
HW = HardwareSpec()
