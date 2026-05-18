"""日志配置

集中管理 tool call 和 LLM 调用日志，便于排查 UI 无响应问题。
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler


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


def log_prompt(logger: logging.Logger, label: str, system_prompt: str, user_prompt: str):
    """打印 prompt 摘要到日志
    
    打印长度和开头预览，方便排查 prompt 结构问题。
    """
    logger.info(f"{'='*60}")
    logger.info(f"[PROMPT] {label}")
    logger.info(f"{'='*60}")
    logger.info(f"system_prompt: len={len(system_prompt)} preview=\n{system_prompt[:400]}")
    logger.info(f"user_prompt:   len={len(user_prompt)} preview=\n{user_prompt[:800]}")
    logger.info(f"{'='*60}")


def log_response(logger: logging.Logger, label: str, text: str):
    """打印响应到日志
    
    打印完整响应（或长度+预览），方便排查模型输出问题。
    """
    logger.info(f"[RESPONSE] {label} | len={len(text)}")
    # 打印完整内容，但限制在 3000 字以内避免日志过大
    if len(text) <= 3000:
        logger.info(f"FULL:\n{text}")
    else:
        logger.info(f"HEAD:\n{text[:1500]}")
        logger.info(f"... ({len(text)-3000} chars omitted) ...")
        logger.info(f"TAIL:\n{text[-1500:]}")
    logger.info(f"{'='*60}")


def log_event(logger: logging.Logger, label: str, **kwargs):
    """打印结构化事件到日志"""
    parts = [f"{k}={v!r}" for k, v in kwargs.items()]
    logger.info(f"[EVENT] {label} | {' | '.join(parts)}")
