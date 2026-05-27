#!/bin/bash
# V2 6 区间验证（独立进程 + 90 分钟超时；PIT universe 方法论加固版）
cd "$(dirname "$0")/.."
mkdir -p /tmp/v2_runs

declare -a REGIONS=(
    "2023 全年:2023-01-09:2023-12-29"
    "2024 Q1:2024-01-02:2024-03-29"
    "2024 Q2-Q3:2024-04-01:2024-08-30"
    "2024 9.24:2024-09-02:2024-11-29"
    "2025 全年:2025-01-02:2025-12-31"
    "2026 H1:2026-01-02:2026-05-18"
)

for entry in "${REGIONS[@]}"; do
    IFS=':' read -r name start end <<< "$entry"
    echo ""
    echo "==== 跑 $name ($start ~ $end) ===="
    log_file="/tmp/v2_runs/${name// /_}.log"
    perl -e '
        $pid = fork();
        if ($pid == 0) { exec @ARGV; exit 1; }
        $start = time();
        while (time() - $start < 5400) {
            $kid = waitpid($pid, 1);
            if ($kid == $pid) { exit($? >> 8); }
            sleep 5;
        }
        kill 9, $pid;
        exit 124;
    ' python3 scripts/test_live_agent_v2.py --start "$start" --end "$end" --name "$name" \
        --out "runs/v2_region_${name// /_}.json" > "$log_file" 2>&1
    exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "  ✓ $name 完成"
        grep -E "推荐日数|超额|累计|期末|V2 累计" "$log_file" | tail -5
    else
        echo "  ✗ $name 失败 (exit=$exit_code)"
    fi
done
