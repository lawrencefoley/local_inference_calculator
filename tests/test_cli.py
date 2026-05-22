from click.testing import CliRunner

from main import cli


def test_help_command() -> None:
    result = CliRunner().invoke(cli, ["--help"])

    assert result.exit_code == 0
    assert "local-inference-calculator --context 4096" in result.output


def test_list_models_command() -> None:
    result = CliRunner().invoke(cli, ["--list-models"])

    assert result.exit_code == 0
    assert "AVAILABLE MODELS" in result.output
    assert "Usage: uv run local-inference-calculator --model <size>" in result.output
