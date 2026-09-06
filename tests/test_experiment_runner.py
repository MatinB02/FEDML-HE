import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from Codes.path_utils import safe_path_component
from run_experiments import run_experiment_group, run_single_experiment


class ExperimentRunnerTests(unittest.TestCase):
    def test_safe_path_component_is_stable_and_collision_resistant(self):
        first = safe_path_component('attack/run:1')
        second = safe_path_component('attack?run:1')

        self.assertNotIn('/', first)
        self.assertNotIn(':', first)
        self.assertNotEqual(first, second)
        self.assertEqual(first, safe_path_component('attack/run:1'))

    def test_subprocess_output_streams_directly_to_log_files(self):
        config = {'exp_id': 'run/one', 'group': 'tests'}

        def fake_run(_cmd, **kwargs):
            self.assertNotIn('capture_output', kwargs)
            kwargs['stdout'].write(b'streamed stdout')
            kwargs['stderr'].write(b'streamed stderr')
            return SimpleNamespace(returncode=0)

        with tempfile.TemporaryDirectory() as directory:
            with patch('run_experiments.subprocess.run', side_effect=fake_run):
                result = run_single_experiment(config, Path(directory))

            self.assertEqual(result['status'], 'success')
            self.assertEqual(
                Path(result['stdout_path']).read_bytes(), b'streamed stdout'
            )
            self.assertEqual(
                Path(result['stderr_path']).read_bytes(), b'streamed stderr'
            )

    def test_duplicate_experiment_ids_are_rejected(self):
        configs = [
            {'exp_id': 'duplicate', 'group': 'tests'},
            {'exp_id': 'duplicate', 'group': 'tests'},
        ]
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory, 'duplicate_configs.json')
            config_path.write_text(json.dumps(configs), encoding='utf-8')

            with self.assertRaisesRegex(ValueError, 'must be unique'):
                run_experiment_group(str(config_path), dry_run=True)


if __name__ == '__main__':
    unittest.main()
