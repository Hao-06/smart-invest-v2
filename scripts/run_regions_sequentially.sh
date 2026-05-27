#!/bin/bash
# 顺序跑 All Weather 各区间 —— 每区间独立进程，30 分钟超时
# 避免一个区间卡死影响别的

cd "$(dirname "$0")/.."

# 区间 ID 列表（参考 test_meta_strategy.py 的 REGIONS）
# 0 = 2023 全年（已跑）
# 1 = 2024 Q1
# 2 = 2024 Q2-Q3
# 3 = 9.24（已跑）
# 4 = 2025 全年
# 5 = 2026 至今
REGIONS_TO_RUN=(5)

mkdir -p /tmp/allw_runs

for r in "${REGIONS_TO_RUN[@]}"; do
    echo ""
    echo "================================================================"
    echo "  开始跑区间 $r (`date '+%H:%M:%S'`)"
    echo "================================================================"
    log_file="/tmp/allw_runs/region_${r}.log"

    # 用 perl 实现 timeout（macOS 没 gnu timeout）
    perl -e '
        $pid = fork();
        if ($pid == 0) {
            exec @ARGV;
            exit 1;
        }
        $start = time();
        while (time() - $start < 1800) {  # 30 min
            $kid = waitpid($pid, 1);  # WNOHANG
            if ($kid == $pid) { exit($? >> 8); }
            sleep 5;
        }
        # timeout
        kill 9, $pid;
        print STDERR "[runner] Region timed out after 30 min, killed\n";
        exit 124;
    ' python3 scripts/test_meta_strategy.py --regions "$r" > "$log_file" 2>&1

    exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "  ✓ 区间 $r 完成 (`date '+%H:%M:%S'`)"
        # 提取关键结果
        grep -E "区间：|Regime 分布|平均权重|平均总仓位|Meta-Agent  累计|沪深 300|超额" "$log_file" | tail -8
    else
        echo "  ✗ 区间 $r 失败/超时 (exit=$exit_code)"
    fi
done

echo ""
echo "================================================================"
echo "  全部区间跑完 (`date '+%H:%M:%S'`)"
echo "================================================================"
