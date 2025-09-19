"""
Unit tests for mining CLI interface.
Tests argument parsing, command execution, and error handling.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import ANY, MagicMock, patch

import pytest

from generate_glossary.mining.main import (
    apply_logging_overrides,
    build_parser,
    monitor_job_if_requested,
    main,
)
from generate_glossary.mining.config import (
    ConfigError,
    FirecrawlConfig,
    MiningConfig,
    MiningModuleConfig,
    OutputConfig,
)
from generate_glossary.mining.utils import load_terms_from_file


class TestArgumentParser:
    """Test CLI argument parsing."""

    def test_build_parser_with_minimum_args(self) -> None:
        """Parser should accept required positional arguments."""
        parser = build_parser()
        args = parser.parse_args(['terms.txt', '-o', 'output.json'])

        assert args.terms_file == 'terms.txt'
        assert args.output == 'output.json'
        assert args.use_summary is None
        assert args.use_batch is None

    def test_build_parser_all_flags(self) -> None:
        """Parser should handle optional flags without error."""
        parser = build_parser()
        args = parser.parse_args([
            'terms.txt',
            '-o', 'output.json',
            '--batch-size', '20',
            '--max-urls', '5',
            '--max-age', '86400000',
            '--max-pages', '3',
            '--no-summary',
            '--no-batch',
            '--request-timeout', '45',
            '--retry-attempts', '4',
            '--retry-delay', '0.5',
            '--search-categories', 'research,academic',
            '--academic-domains', 'edu,org',
            '--config', 'config.yml',
            '--log-level', 'DEBUG',
            '--verbose',
            '--quiet',
            '--output-format', 'jsonl',
            '--webhook-url', 'https://example.com/hook',
            '--queue-job', 'job-123',
            '--save-metadata',
            '--include-source-urls',
        ])

        assert args.batch_size == 20
        assert args.max_urls == 5
        assert args.max_age == 86400000
        assert args.max_pages == 3
        assert args.use_summary is False
        assert args.use_batch is False
        assert args.request_timeout == 45
        assert args.retry_attempts == 4
        assert args.retry_delay == 0.5
        assert args.search_categories == 'research,academic'
        assert args.academic_domains == 'edu,org'
        assert args.config == 'config.yml'
        assert args.log_level == 'DEBUG'
        assert args.verbose is True
        assert args.quiet is True
        assert args.output_format == 'jsonl'
        assert args.webhook_url == 'https://example.com/hook'
        assert args.queue_job == 'job-123'
        assert args.save_metadata is True
        assert args.include_source_urls is True


class TestLoadTermsFromFile:
    """Test loading terms from file."""

    def test_load_terms_success(self) -> None:
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as handle:
            handle.write("machine learning\nneural networks\ndeep learning\n")
            temp_path = handle.name

        try:
            terms = load_terms_from_file(temp_path)
            assert terms == ["machine learning", "neural networks", "deep learning"]
        finally:
            Path(temp_path).unlink()

    def test_load_terms_empty_file(self) -> None:
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as handle:
            handle.write("")
            temp_path = handle.name

        try:
            with pytest.raises(ValueError, match="Terms file is empty"):
                load_terms_from_file(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_load_terms_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_terms_from_file('nonexistent.txt')

    def test_load_terms_ignores_comments(self) -> None:
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as handle:
            handle.write("# comment\nterm one\n# another\nterm two\n")
            temp_path = handle.name

        try:
            terms = load_terms_from_file(temp_path)
            assert terms == ['term one', 'term two']
        finally:
            Path(temp_path).unlink()


class TestApplyLoggingOverrides:
    """Test logging override helpers."""

    def _make_config(self) -> MiningModuleConfig:
        return MiningModuleConfig(
            firecrawl=FirecrawlConfig(api_key='test'),
            mining=MiningConfig(),
            output=OutputConfig(),
            logging={'level': 'INFO'},
        )

    def test_verbose_overrides_level(self) -> None:
        cfg = self._make_config()
        args = MagicMock(verbose=True, quiet=False, log_level=None)

        apply_logging_overrides(cfg, args)
        assert cfg.logging['level'] == 'DEBUG'

    def test_quiet_overrides_level(self) -> None:
        cfg = self._make_config()
        args = MagicMock(verbose=False, quiet=True, log_level=None)

        apply_logging_overrides(cfg, args)
        assert cfg.logging['level'] == 'ERROR'

    def test_explicit_log_level_wins(self) -> None:
        cfg = self._make_config()
        args = MagicMock(verbose=True, quiet=True, log_level='WARNING')

        apply_logging_overrides(cfg, args)
        assert cfg.logging['level'] == 'WARNING'


class TestMonitorJobIfRequested:
    """Test job monitoring helper."""

    @patch('generate_glossary.mining.main.monitor_queue_status')
    def test_monitor_job_executes_when_id_present(self, mock_monitor) -> None:
        client = object()
        mock_monitor.return_value = {'status': 'completed'}

        result = monitor_job_if_requested(client, 'job-123')

        mock_monitor.assert_called_once_with(client, 'job-123')
        assert result == {'status': 'completed'}

    @patch('generate_glossary.mining.main.monitor_queue_status')
    def test_monitor_job_skips_when_id_absent(self, mock_monitor) -> None:
        client = object()

        result = monitor_job_if_requested(client, None)

        mock_monitor.assert_not_called()
        assert result is None


class TestMainFunction:
    """Test the main CLI function."""

    @patch('generate_glossary.mining.main.save_mining_results')
    @patch('generate_glossary.mining.main.mine_concepts_simple')
    @patch('generate_glossary.mining.main.get_firecrawl_client')
    @patch('generate_glossary.mining.main.load_terms_from_file')
    @patch('generate_glossary.mining.main.configure_logging')
    @patch('generate_glossary.mining.main.override_with_cli_args')
    @patch('generate_glossary.mining.main.load_config')
    @patch('generate_glossary.mining.main.log_mining_operation')
    def test_main_success(
        self,
        mock_log_operation,
        mock_load_config,
        mock_override,
        mock_configure_logging,
        mock_load_terms,
        mock_get_client,
        mock_mine_concepts,
        mock_save_results,
    ) -> None:
        cfg = MiningModuleConfig(
            firecrawl=FirecrawlConfig(api_key='k'),
            mining=MiningConfig(),
            output=OutputConfig(),
            logging={'level': 'INFO'},
        )
        mock_load_config.return_value = cfg
        mock_override.return_value = cfg
        mock_load_terms.return_value = ['term one']
        mock_get_client.return_value = object()
        mock_mine_concepts.return_value = {'results': {}, 'statistics': {}}

        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / 'results.json'
            mock_save_results.return_value = str(output_file)

            argv = ['terms.txt', '-o', str(output_file)]
            with patch('sys.argv', ['mine-web', *argv]):
                exit_code = main()

            assert exit_code == 0

        mock_load_config.assert_called_once()
        mock_configure_logging.assert_called_once_with(cfg.logging)
        mock_load_terms.assert_called_once()
        mock_get_client.assert_called_once()
        mock_mine_concepts.assert_called_once()
        mock_save_results.assert_called_once()
        mock_log_operation.assert_any_call('startup', ANY)

    @patch('generate_glossary.mining.main.load_config', side_effect=ConfigError('config error'))
    def test_main_configuration_failure(self, mock_load_config) -> None:
        with patch('sys.argv', ['mine-web', 'terms.txt', '-o', 'output.json']):
            exit_code = main()

        assert exit_code == 1

    @patch('generate_glossary.mining.main.load_config')
    @patch('generate_glossary.mining.main.override_with_cli_args')
    @patch('generate_glossary.mining.main.configure_logging')
    @patch('generate_glossary.mining.main.load_terms_from_file', side_effect=FileNotFoundError('missing'))
    def test_main_missing_terms_file(
        self,
        mock_load_terms,
        mock_configure_logging,
        mock_override,
        mock_load_config,
    ) -> None:
        cfg = MiningModuleConfig(
            firecrawl=FirecrawlConfig(api_key='k'),
            mining=MiningConfig(),
            output=OutputConfig(),
            logging={'level': 'INFO'},
        )
        mock_load_config.return_value = cfg
        mock_override.return_value = cfg

        with patch('sys.argv', ['mine-web', 'missing.txt', '-o', 'output.json']):
            exit_code = main()

        assert exit_code == 1


# --- Integration smoke test -------------------------------------------------

def test_cli_entry_point_metadata(tmp_path: Path) -> None:
    """Ensure the CLI emits metadata files during successful runs."""
    from generate_glossary.mining.main import save_mining_results as main_save_results

    cfg = MiningModuleConfig(
        firecrawl=FirecrawlConfig(api_key='k'),
        mining=MiningConfig(),
        output=OutputConfig(),
        logging={'level': 'INFO'},
    )

    with patch('generate_glossary.mining.main.load_config', return_value=cfg),         patch('generate_glossary.mining.main.override_with_cli_args', return_value=cfg),         patch('generate_glossary.mining.main.configure_logging'),         patch('generate_glossary.mining.main.load_terms_from_file', return_value=['term']),         patch('generate_glossary.mining.main.get_firecrawl_client'),         patch('generate_glossary.mining.main.mine_concepts_simple', return_value={'results': {}, 'statistics': {}}),         patch('generate_glossary.mining.main.save_mining_results', wraps=main_save_results) as wrapped_save:

        output_dir = tmp_path / 'outputs'
        argv = ['terms.txt', '-o', str(output_dir)]
        with patch('sys.argv', ['mine-web', *argv]):
            exit_code = main()

        assert exit_code == 0
        wrapped_save.assert_called_once()

        metadata_files = sorted(tmp_path.glob('**/*.meta.json'))
        assert metadata_files, 'Expected metadata file to be generated'

        payload = json.loads(metadata_files[-1].read_text())
        assert 'statistics' in payload
