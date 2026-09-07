"""Coordinator boundary tests without importing heavyweight optional agents."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch


class CoordinatorTests(unittest.TestCase):
    def setUp(self):
        definitions = (
            ('agent.ingestor', 'run_ingestor_agent', {'status': 'success', 'media_type': 'image', 'ready_for_next_agent': True}),
            ('agent.provinence', 'run_provenance_agent', {'status': 'success', 'ready_for_next_agent': True}),
            ('agent.forensic', 'run_forensics_agent', {'status': 'success', 'ready_for_next_agent': True}),
            ('agent.crossReff', 'run_crossref_agent', {'status': 'completed', 'claims_checked': 0}),
        )
        modules = {}
        self.calls = []
        self.outputs = []
        for name, function, output in definitions:
            module = types.ModuleType(name)
            call = Mock(return_value=output)
            setattr(module, function, call)
            modules[name] = module
            self.calls.append(call)
            self.outputs.append(output)
        spec = importlib.util.spec_from_file_location('coordinator_under_test', Path(__file__).resolve().parents[1] / 'image_pipeline.py')
        self.pipeline = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, modules):
            spec.loader.exec_module(self.pipeline)

    def test_all_four_dictionaries_reach_engine(self):
        engine = self.pipeline.run_decision_engine
        with patch.object(self.pipeline, 'run_decision_engine', wraps=engine) as decision:
            result = self.pipeline.run_local_image_pipeline('example.jpg')
            decision.assert_called_once_with(*self.outputs)
        self.calls[1].assert_called_once_with(self.outputs[0])
        self.calls[2].assert_called_once_with(self.outputs[1])
        self.calls[3].assert_called_once_with(self.outputs[1], self.outputs[2])
        self.assertEqual(len(result['final_decision']['agent_summary']), 4)

    def test_ingestor_failure_is_displayable(self):
        self.calls[0].return_value = {'status': 'error', 'ready_for_next_agent': False}
        result = self.pipeline.run_local_image_pipeline('missing.jpg')
        self.assertEqual(result['final_decision']['category'], 'Needs Review')
        self.calls[1].assert_not_called()

    def test_provenance_failure_is_displayable(self):
        self.calls[1].return_value = None
        result = self.pipeline.run_local_image_pipeline('example.jpg')
        self.assertEqual(result['final_decision']['category'], 'Needs Review')
        self.calls[2].assert_not_called()

    def test_forensics_failure_skips_crossref_and_preserves_four_summaries(self):
        self.calls[2].return_value = {'status': 'error', 'ready_for_next_agent': False}
        result = self.pipeline.run_local_image_pipeline('example.jpg')
        self.calls[3].assert_not_called()
        self.assertEqual(result['final_decision']['category'], 'Needs Review')
        self.assertFalse(result['final_decision']['decision_state']['cross_reference_available'])
        self.assertEqual(len(result['final_decision']['agent_summary']), 4)
