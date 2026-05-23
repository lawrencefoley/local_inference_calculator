import json

from click.testing import CliRunner

from main import calculate_kv_cache_from_config, cli
from models import get_all_models


def test_help_command() -> None:
    result = CliRunner().invoke(cli, ["--help"])

    assert result.exit_code == 0
    assert "llmfit --context 4096" in result.output


def test_list_models_command() -> None:
    result = CliRunner().invoke(cli, ["--list-models"])

    assert result.exit_code == 0
    assert "AVAILABLE MODELS" in result.output
    assert "Usage: uv run llmfit --model <size>" in result.output


def test_model_catalog_splits_mixed_size_entries() -> None:
    qwen_coder_models = [model for model in get_all_models() if model.name == "qwen3-coder 30B"]

    assert len(qwen_coder_models) == 1
    assert qwen_coder_models[0].params_billion == 30


def test_calculate_kv_cache_from_config() -> None:
    kv_cache = calculate_kv_cache_from_config(
        {
            "num_hidden_layers": 32,
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
        }
    )

    assert kv_cache == 0.125


def test_vram_gb_command() -> None:
    result = CliRunner().invoke(cli, ["--vram-gb", "24", "--quantization", "int4"])

    assert result.exit_code == 0
    assert "MAX CONTEXT BY MODEL" in result.output
    assert "24 GB VRAM, INT4" in result.output
    assert "Max Context" in result.output


def test_config_json_command(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "_name_or_path": "example/model",
                "architectures": ["LlamaForCausalLM"],
                "num_hidden_layers": 32,
                "hidden_size": 4096,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "max_position_embeddings": 8192,
            }
        )
    )

    result = CliRunner().invoke(
        cli,
        ["--config-json", str(config_path), "--params-b", "7", "--context", "1024", "--summary", "none"],
    )

    assert result.exit_code == 0
    assert "Loaded model metadata from config.json" in result.output
    assert "KV cache: 0.1250 MB/token" in result.output


def test_config_json_vram_gb_command(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "_name_or_path": "example/model",
                "architectures": ["LlamaForCausalLM"],
                "num_hidden_layers": 32,
                "hidden_size": 4096,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "max_position_embeddings": 8192,
            }
        )
    )

    result = CliRunner().invoke(
        cli,
        ["--config-json", str(config_path), "--params-b", "7", "--vram-gb", "24", "--quantization", "int4"],
    )

    assert result.exit_code == 0
    assert "example/model" in result.output
    assert "8,192*" in result.output
