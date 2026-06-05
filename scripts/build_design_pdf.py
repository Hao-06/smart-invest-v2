#!/usr/bin/env python3
"""从 ``docs/作品设计书.md`` 重新生成 HTML（保留自定义样式）+ PDF。

链路：Markdown --(pandoc)--> body 片段 --(套回保留的 <head> 样式)--> HTML
      --(Chrome headless 打印)--> PDF

设计书的 ``<head>`` 含两个 ``<style>``：① pandoc 默认排版；② 自定义 @page A4 +
页脚页码 + PingFang 字体。本脚本只用 pandoc 重新渲染正文 body，**保留整段 head**，
因此改 md 后样式（含页码/字体/页边距）保持不变。评委可一键复现 PDF。

用法：
    python3 scripts/build_design_pdf.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MD = ROOT / "docs" / "作品设计书.md"
HTML = ROOT / "docs" / "作品设计书.html"
PDF = ROOT / "docs" / "作品设计书.pdf"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def main() -> int:
    # 1. pandoc：md → body 片段（不加 -s，只输出正文 HTML）
    body = subprocess.run(
        ["pandoc", str(MD), "-t", "html"],
        capture_output=True, text=True, check=True,
    ).stdout

    # 2. 保留现有 HTML 的 <head>（含两个 style 块），只替换 <body> 内容
    old = HTML.read_text()
    head = old[: old.index("<body>") + len("<body>")]
    HTML.write_text(head + "\n" + body + "\n</body>\n</html>\n")

    # 3. Chrome headless → PDF
    subprocess.run(
        [CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer",
         f"--print-to-pdf={PDF}", f"file://{HTML}"],
        capture_output=True,
    )
    print(f"✓ 已从 {MD.name} 重新生成 HTML + PDF")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
