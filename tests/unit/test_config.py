"""测试 config.py — ModelConfig, DebateConfig, AppConfig"""

import pytest
import yaml

from debater.config import ModelConfig, DebateConfig, AppConfig


class TestModelConfig:
    def test_basic_properties(self):
        """应正确解析基本属性"""
        cfg = ModelConfig({
            "id": "kimi",
            "name": "Kimi",
            "provider": "anthropic",
            "model_id": "kimi-for-coding",
            "temperature": 0.5,
            "max_tokens": 2048,
            "role": "proposer",
        })
        assert cfg.id == "kimi"
        assert cfg.name == "Kimi"
        assert cfg.provider == "anthropic"
        assert cfg.model_id == "kimi-for-coding"
        assert cfg.temperature == 0.5
        assert cfg.max_tokens == 2048
        assert cfg.role == "proposer"

    def test_defaults(self):
        """默认值应正确"""
        cfg = ModelConfig({})
        assert cfg.id == "unknown"
        assert cfg.name == "unknown"
        assert cfg.provider == "anthropic"
        assert cfg.model_id == "unknown"
        assert cfg.temperature == 0.3
        assert cfg.max_tokens == 4096
        assert cfg.role == "proposer"
        assert cfg.api_key_env is None
        assert cfg.base_url is None

    def test_api_key_from_config(self):
        """直接配置的 api_key 应优先"""
        cfg = ModelConfig({"api_key": "direct-key"})
        assert cfg.api_key == "direct-key"

    def test_api_key_from_env(self, monkeypatch):
        """无直接配置时应从环境变量读取"""
        monkeypatch.setenv("TEST_API_KEY", "env-key")
        cfg = ModelConfig({"api_key_env": "TEST_API_KEY"})
        assert cfg.api_key == "env-key"

    def test_api_key_none_when_no_source(self):
        """无配置且无环境变量时应返回 None"""
        cfg = ModelConfig({})
        assert cfg.api_key is None


class TestDebateConfig:
    def test_defaults(self):
        """默认值应正确"""
        cfg = DebateConfig({})
        assert cfg.consensus_threshold == 0.85
        assert cfg.min_confidence == 0.70

    def test_custom_values(self):
        """应正确解析自定义值"""
        cfg = DebateConfig({"consensus_threshold": 0.9, "min_confidence": 0.8})
        assert cfg.consensus_threshold == 0.9
        assert cfg.min_confidence == 0.8


class TestAppConfig:
    def test_load_from_yaml(self, temp_dir):
        """应从 YAML 文件正确加载配置"""
        config = {
            "models": {
                "kimi": {
                    "name": "Kimi",
                    "provider": "anthropic",
                    "model_id": "kimi-for-coding",
                    "api_key": "test-key",
                    "role": "proposer",
                },
                "deepseek": {
                    "name": "DeepSeek",
                    "provider": "anthropic",
                    "model_id": "deepseek-v4-pro",
                    "role": "proposer",
                },
            },
            "debate": {
                "consensus_threshold": 0.9,
            },
            "scenarios": {
                "question_analysis": "问题分析",
            },
        }
        config_file = temp_dir / "config.yaml"
        config_file.write_text(yaml.dump(config))

        app_cfg = AppConfig(config_path=str(config_file))

        assert "kimi" in app_cfg.models
        assert "deepseek" in app_cfg.models
        assert app_cfg.models["kimi"].name == "Kimi"
        assert app_cfg.debate.consensus_threshold == 0.9
        assert app_cfg.scenarios["question_analysis"] == "问题分析"

    def test_get_proposer_models(self, temp_dir):
        """get_proposer_models 应只返回 proposer 角色"""
        config = {
            "models": {
                "a": {"role": "proposer"},
                "b": {"role": "aggregator"},
                "c": {"role": "proposer"},
            }
        }
        config_file = temp_dir / "config.yaml"
        config_file.write_text(yaml.dump(config))

        app_cfg = AppConfig(config_path=str(config_file))
        proposers = app_cfg.get_proposer_models()
        assert set(proposers.keys()) == {"a", "c"}

    def test_get_aggregator_model(self, temp_dir):
        """get_aggregator_model 应返回 aggregator 或第一个模型"""
        config = {
            "models": {
                "a": {"role": "proposer"},
                "b": {"role": "aggregator"},
            }
        }
        config_file = temp_dir / "config.yaml"
        config_file.write_text(yaml.dump(config))

        app_cfg = AppConfig(config_path=str(config_file))
        agg = app_cfg.get_aggregator_model()
        assert agg is not None
        assert agg.role == "aggregator"

    def test_get_aggregator_model_fallback(self, temp_dir):
        """无 aggregator 时应返回第一个模型"""
        config = {
            "models": {
                "a": {"role": "proposer"},
            }
        }
        config_file = temp_dir / "config.yaml"
        config_file.write_text(yaml.dump(config))

        app_cfg = AppConfig(config_path=str(config_file))
        agg = app_cfg.get_aggregator_model()
        assert agg is not None
        assert agg.role == "proposer"

    def test_get_aggregator_model_empty(self, temp_dir):
        """无模型时应返回 None"""
        config = {"models": {}}
        config_file = temp_dir / "config.yaml"
        config_file.write_text(yaml.dump(config))

        app_cfg = AppConfig(config_path=str(config_file))
        assert app_cfg.get_aggregator_model() is None
