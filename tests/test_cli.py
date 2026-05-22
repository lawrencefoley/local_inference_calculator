import json

from click.testing import CliRunner

from main import calculate_kv_cache_from_config, cli


def test_help_command() -> None:
    result = CliRunner().invoke(cli, ["--help"])

    assert result.exit_code == 0
    assert "local-inference-calculator --context 4096" in result.output


def test_list_models_command() -> None:
    result = CliRunner().invoke(cli, ["--list-models"])

    assert result.exit_code == 0
    assert "AVAILABLE MODELS" in result.output
    assert "Usage: uv run local-inference-calculator --model <size>" in result.output


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
