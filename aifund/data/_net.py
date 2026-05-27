"""网络环境设置：强制数据请求走直连，绕过本机代理。

两件事：
1. **绕过代理**：AkShare 的数据源（东方财富、腾讯财经等）均为境内站点，
   本应直连访问。若本机配置了代理（如 Clash / V2Ray），requests 会默认
   继承代理环境变量，反而把境内请求绕到境外节点，导致连接被重置或超时。
2. **全局 socket 超时**：AkShare 内部 requests 调用很多没设 timeout，
   极端情况下（服务端 hang 着不响应）能让进程阻塞数十分钟。在 socket 层
   设全局 60 秒超时，所有 HTTP 调用都受其约束 —— 即使单接口异常也只会
   慢 60 秒，不会拖死整个回测/决策流程。

导入 `aifund.data` 包时即自动生效。
"""
from __future__ import annotations

import os
import socket

_PROXY_VARS = (
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "ftp_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "FTP_PROXY",
)

#: 全局 socket 超时（秒）—— 兜底防止某个接口 hang 死整个进程
#: 降到 30 秒（之前 60 秒），因为 SSL read 卡死过 16 分钟的实测案例
_DEFAULT_SOCKET_TIMEOUT = 30.0

_done = False


def setup_direct_connection() -> None:
    """清除代理环境变量，强制 requests 直连；同时设全局 socket 超时。幂等。"""
    global _done
    if _done:
        return
    for var in _PROXY_VARS:
        os.environ.pop(var, None)
    # NO_PROXY=* 让 requests 对所有 host 跳过代理，
    # 同时屏蔽 macOS 系统级代理设置的自动探测。
    os.environ["NO_PROXY"] = "*"
    os.environ["no_proxy"] = "*"
    # **强制**设全局 socket 超时（不再检查 is None）—— 防御性，兜底单接口 hang 死
    # SSL read 在 30 秒后会抛 timeout 异常，触发 _try_sources 的 retry 或回退
    socket.setdefaulttimeout(_DEFAULT_SOCKET_TIMEOUT)
    _done = True


# 导入即生效
setup_direct_connection()
