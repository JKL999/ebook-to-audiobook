"""Tests for CLI interface."""

import pytest
from typer.testing import CliRunner
from pathlib import Path

# Import will be added once CLI is implemented
# from ebook2audio.cli import app

runner = CliRunner()

class TestCLI:
    """Test CLI commands."""
    
    @pytest.mark.skip(reason="CLI not yet implemented")
    def test_cli_help(self):
        """Test CLI help command."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "ebook2audio" in result.stdout
    
    @pytest.mark.skip(reason="CLI not yet implemented")
    def test_cli_version(self):
        """Test CLI version command."""
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "1.0.0" in result.stdout
    
    @pytest.mark.skip(reason="CLI not yet implemented")
    def test_cli_convert_command(self, temp_dir):
        """Test convert command."""
        input_file = temp_dir / "test.txt"
        input_file.write_text("Test content")
        output_file = temp_dir / "output.mp3"
        
        result = runner.invoke(app, [
            "convert",
            str(input_file),
            "--output", str(output_file),
            "--voice", "gtts_en_us"
        ])
        
        # Would need mocking for actual test
        assert result.exit_code == 0
    
    @pytest.mark.skip(reason="CLI not yet implemented")
    def test_cli_list_voices(self):
        """Test list-voices command."""
        result = runner.invoke(app, ["list-voices"])
        assert result.exit_code == 0
        assert "gtts_en_us" in result.stdout
    
    @pytest.mark.skip(reason="CLI not yet implemented")
    def test_cli_preview_voice(self, temp_dir):
        """Test preview-voice command."""
        output_file = temp_dir / "preview.mp3"
        
        result = runner.invoke(app, [
            "preview-voice",
            "gtts_en_us",
            "--text", "Hello world",
            "--output", str(output_file)
        ])
        
        # Would need mocking for actual test
        assert result.exit_code == 0