"""配置加载与管理"""

import os
from pathlib import Path
from typing import Dict, List, Any, Optional

import yaml
from dotenv import load_dotenv


# 加载 .env 文件
load_dotenv()


class ModelConfig:
    """单个模型的配置"""
    
    def __init__(self, config_dict: Dict[str, Any]):
        self.id = config_dict.get("id", "unknown")
        self.name = config_dict.get("name", self.id)
        self.provider = config_dict.get("provider", "anthropic")
        self.model_id = config_dict.get("model_id", self.id)
        self.api_key_env = config_dict.get("api_key_env")
        self._api_key = config_dict.get("api_key")
        self.base_url = config_dict.get("base_url")
        self.temperature = config_dict.get("temperature", 0.3)
        self.max_tokens = config_dict.get("max_tokens", 4096)
        self.role = config_dict.get("role", "proposer")
        self.extra_body = config_dict.get("extra_body")  # 模型特定的额外请求体参数
        self.critique_style = config_dict.get("critique_style", "standard")  # standard | adversarial
    
    @property
    def api_key(self) -> Optional[str]:
        # 优先读取直接配置的 api_key，其次读环境变量
        if self._api_key:
            return self._api_key
        if self.api_key_env:
            return os.environ.get(self.api_key_env)
        return None


class DebateConfig:
    """辩论配置"""
    
    def __init__(self, config_dict: Dict[str, Any]):
        self.consensus_threshold = config_dict.get("consensus_threshold", 0.85)
        self.min_confidence = config_dict.get("min_confidence", 0.70)


class AppConfig:
    """应用总配置"""
    
    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            # 默认查找项目根目录的 config.yaml
            root = Path(__file__).parent.parent
            config_path = root / "config.yaml"
        
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        
        # 加载模型配置
        self.models: Dict[str, ModelConfig] = {}
        for model_id, model_data in data.get("models", {}).items():
            model_data = dict(model_data)
            model_data["id"] = model_id
            self.models[model_id] = ModelConfig(model_data)
        
        # 加载辩论配置
        self.debate = DebateConfig(data.get("debate", {}))
        
        # 加载场景配置
        self.scenarios = data.get("scenarios", {})
    
    def get_proposer_models(self) -> Dict[str, ModelConfig]:
        """获取所有参与辩论的模型"""
        return {
            k: v for k, v in self.models.items()
            if v.role == "proposer"
        }
    
    def get_available_models(self) -> Dict[str, ModelConfig]:
        """获取所有可用的模型（可用于选择）"""
        return self.models
    
    def get_aggregator_model(self) -> Optional[ModelConfig]:
        """获取聚合器模型（如果没有指定，返回第一个模型）"""
        aggregators = [
            v for k, v in self.models.items()
            if v.role == "aggregator"
        ]
        if aggregators:
            return aggregators[0]
        # 默认用第一个模型作为聚合器
        return next(iter(self.models.values()), None)


# 全局配置实例（懒加载）
_config_instance: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """获取全局配置"""
    global _config_instance
    if _config_instance is None:
        _config_instance = AppConfig()
    return _config_instance


def reload_config(config_path: Optional[str] = None) -> AppConfig:
    """重新加载配置（用于 UI 动态切换）
    
    清除全局配置缓存并重新加载，返回新的配置实例。
    """
    global _config_instance
    _config_instance = AppConfig(config_path)
    return _config_instance


def list_config_files(config_dir: Optional[str] = None) -> List[str]:
    """列出指定目录下所有可用的配置文件（*.yaml）"""
    if config_dir is None:
        config_dir = Path(__file__).parent.parent
    else:
        config_dir = Path(config_dir)
    return sorted([str(p) for p in config_dir.glob("config*.yaml")])


def save_config(data: Dict[str, Any], config_path: Optional[str] = None) -> str:
    """保存配置到 YAML 文件
    
    Args:
        data: 配置数据字典
        config_path: 目标文件路径，默认覆盖当前加载的配置文件
    
    Returns:
        保存的文件路径
    """
    if config_path is None:
        cfg = get_config()
        # 尝试从全局实例推断路径（AppConfig 没有保存路径，默认用根目录 config.yaml）
        config_path = Path(__file__).parent.parent / "config.yaml"
    else:
        config_path = Path(config_path)
    
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
    
    return str(config_path)


def load_config_raw(config_path: Optional[str] = None) -> Dict[str, Any]:
    """加载原始配置字典（不解析为对象）"""
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config.yaml"
    else:
        config_path = Path(config_path)
    
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def copy_config(src_path: str, dst_name: str, config_dir: Optional[str] = None) -> str:
    """复制配置文件
    
    Args:
        src_path: 源配置文件路径
        dst_name: 新文件名（不含 .yaml 后缀）
        config_dir: 目标目录，默认与源文件同目录
    
    Returns:
        新文件路径
    """
    src = Path(src_path)
    if config_dir is None:
        config_dir = src.parent
    else:
        config_dir = Path(config_dir)
    
    dst = config_dir / f"{dst_name}.yaml"
    if dst.exists():
        raise FileExistsError(f"文件已存在: {dst}")
    
    import shutil
    shutil.copy2(src, dst)
    return str(dst)
