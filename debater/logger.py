"""日志配置

集中管理 tool call 和 LLM 调用日志，便于排查 UI 无响应问题。
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from typing import Optional


def _setup_logger(name: str, log_path: str, max_bytes: int = 5 * 1024 * 1024) -> logging.Logger:
    """通用日志配置
    
    文件大小超过 max_bytes 自动轮转，保留 3 个备份
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # 避免重复添加 handler
    if logger.handlers:
        return logger
    
    formatter = logging.Formatter(
        fmt="[%(asctime)s] %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # 文件 handler：轮转，默认 5MB 一个文件，保留 3 个备份
    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=max_bytes,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    # 同时输出到 stderr，方便 Streamlit 命令行查看
    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    
    return logger


def setup_tool_logger() -> logging.Logger:
    """配置 tool call 执行日志"""
    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "tool_calls.log")
    return _setup_logger("debater.tools", log_path)


def setup_llm_logger() -> logging.Logger:
    """配置 LLM 调用日志"""
    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "llm_calls.log")
    return _setup_logger("debater.llm", log_path)


def setup_session_logger() -> logging.Logger:
    """配置会话级日志（轮次边界、角色状态、prompt/响应等）"""
    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "session.log")
    return _setup_logger("debater.session", log_path, max_bytes=10 * 1024 * 1024)


# 单例
tool_logger = setup_tool_logger()
llm_logger = setup_llm_logger()
session_logger = setup_session_logger()


import re


# 敏感信息脱敏模式
_SENSITIVE_PATTERNS = [
    # API Keys
    (re.compile(r'sk-[a-zA-Z0-9]{20,}'), 'sk-***REDACTED***'),
    (re.compile(r'api[_-]?key\s*[:=]\s*["\']?[a-zA-Z0-9_-]{10,}["\']?', re.IGNORECASE), 'api_key=***REDACTED***'),
    (re.compile(r'password\s*[:=]\s*["\']?[^\s"\']+["\']?', re.IGNORECASE), 'password=***REDACTED***'),
    (re.compile(r'secret\s*[:=]\s*["\']?[^\s"\']+["\']?', re.IGNORECASE), 'secret=***REDACTED***'),
    (re.compile(r'token\s*[:=]\s*["\']?[a-zA-Z0-9_-]{10,}["\']?', re.IGNORECASE), 'token=***REDACTED***'),
    # 邮箱
    (re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'), '***EMAIL***'),
    # 手机号（中国大陆）
    (re.compile(r'1[3-9]\d{9}'), '***PHONE***'),
]


def _sanitize(text: Optional[str]) -> str:
    """脱敏：替换日志中的敏感信息"""
    if not isinstance(text, str):
        return ""
    for pattern, replacement in _SENSITIVE_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def log_prompt(logger: logging.Logger, label: str, system_prompt: str, user_prompt: str):
    """打印 prompt 摘要到日志
    
    打印长度和开头预览（截断 + 脱敏），避免敏感信息泄露和日志膨胀。
    """
    SYS_CUTOFF = 600
    USER_CUTOFF = 1200
    safe_sys = _sanitize(system_prompt[:SYS_CUTOFF])
    safe_user = _sanitize(user_prompt[:USER_CUTOFF])
    logger.info(f"{'='*60}")
    logger.info(f"[PROMPT] {label}")
    logger.info(f"{'='*60}")
    logger.info(f"system_prompt: len={len(system_prompt)} preview=\n{safe_sys}")
    if len(system_prompt) > SYS_CUTOFF:
        logger.info(f"... ({len(system_prompt) - SYS_CUTOFF} chars omitted) ...")
    logger.info(f"user_prompt:   len={len(user_prompt)} preview=\n{safe_user}")
    if len(user_prompt) > USER_CUTOFF:
        logger.info(f"... ({len(user_prompt) - USER_CUTOFF} chars omitted) ...")
    logger.info(f"{'='*60}")


def log_response(logger: logging.Logger, label: str, text: str):
    """打印响应到日志
    
    打印响应摘要（截断 + 脱敏），方便排查模型输出问题。
    """
    MAX_LOG_LEN = 2000  # 比原来 3000 更保守
    safe_text = _sanitize(text)
    logger.info(f"[RESPONSE] {label} | len={len(text)}")
    if len(safe_text) <= MAX_LOG_LEN:
        logger.info(f"FULL:\n{safe_text}")
    else:
        logger.info(f"HEAD:\n{safe_text[:MAX_LOG_LEN // 2]}")
        logger.info(f"... ({len(text) - MAX_LOG_LEN} chars omitted) ...")
        logger.info(f"TAIL:\n{safe_text[-MAX_LOG_LEN // 2:]}")
    logger.info(f"{'='*60}")


def log_event(logger: logging.Logger, label: str, **kwargs):
    """打印结构化事件到日志"""
    parts = [f"{k}={v!r}" for k, v in kwargs.items()]
    logger.info(f"[EVENT] {label} | {' | '.join(parts)}")
