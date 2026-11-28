"""Prompt-iteration provenance tests. All target results are temporary test fixtures."""
import json
from pathlib import Path
import tempfile
import unittest

from heerbench.config import load_config
from heerbench.dataset import Dataset
from heerbench.evaluation import generation_records, prepare_judges, prepare_validation
from heerbench.judge_fork import fork_judge_run
from heerbench.judge_validation import judge_configuration_digest
from heerbench.planning import create_plan
from heerbench.runner import Runner, load_run
from heerbench.store import Store
from heerbench.util import digest, project_root, read_jsonl, software_fingerprint, write_json


class JudgeForkTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='heerbench_fork_fixture_')
        self.root = Path(self.tmp.name)
        self.source = self.root/'source'
        repo = project_root()
        cfg = load_config(repo/'configs/pilot.yaml')
        ds = Dataset(cfg['dataset'])
        cfg['chain_ids'] = [next(c['chain_id'] for c in ds.chains if c['case_id'] == case)
                            for case in ('C1','C2')]
        create_plan(cfg, ds, self.source)
        store = Store(self.source)
        for plan in read_jsonl(self.source/'plan.jsonl'):
            store.add(plan)
            request = {'request_id':plan['request_id'], 'model':'UNIT_TEST_FIXTURE_ONLY',
                       'system':'UNIT TEST SYSTEM', 'prefix':'UNIT TEST SOURCE',
                       'task':'UNIT TEST QUESTION', 'max_output_tokens':64}
            store.update(plan['request_id'], 'succeeded', request=store.pack_request(request),
                result={'text':'UNIT TEST FIXTURE ONLY. '+ds.by_id[plan['question_id']]['expected_answer'],
                        'provider':'UNIT_TEST_FIXTURE_ONLY', 'fixture_not_research_data':True,
                        'usage':{'input_tokens':123,'output_tokens':12},
                        'cost':{'total_usd':'0.123'}})
        # Sentinel old judge rows/approval must never be inherited by the new run.
        store.add({'request_id':'old-judge-fixture','judge_role':'primary'}, 'judge')
        store.update('old-judge-fixture','failed',error={'kind':'UNIT_TEST_ONLY'})
        store.close()
        write_json(self.source/'judge_approval.json', {'decision':'UNIT_TEST_SENTINEL_NOT_REAL_APPROVAL'})
        self.prompt = self.root/'revised_judge.txt'
        self.prompt.write_text('UNIT TEST REVISED JUDGE INSTRUCTIONS ONLY.\n')
        self.out = self.root/'fork'

    def tearDown(self):
        self.tmp.cleanup()

    def test_preserves_answers_panel_and_source_but_resets_judging(self):
        source_runner = Runner(self.source)
        panel = self.root/'panel'
        try:
            prepare_validation(source_runner, panel, n=8, development_n=2)
            original_records = generation_records(source_runner)
            original_rows = source_runner.store.rows('generation')
        finally:
            source_runner.close()
        snapshots = {name:(self.source/name).read_bytes()
                     for name in ('config.frozen.json','manifest.json','plan.jsonl','judge_approval.json','journal.sqlite')}
        parent = json.loads((self.source/'manifest.json').read_text())
        fork_manifest = fork_judge_run(self.source, self.out, self.prompt)
        for name, content in snapshots.items():
            self.assertEqual((self.source/name).read_bytes(), content)
        self.assertEqual(fork_manifest['parent_run_id'], parent['run_id'])
        self.assertEqual(fork_manifest['source_config_sha256'], parent['config_sha256'])
        self.assertEqual(fork_manifest['software_sha256'], software_fingerprint())
        self.assertTrue(fork_manifest['reused_generation_only'])
        self.assertEqual(fork_manifest['purpose'], 'judge_development')
        self.assertFalse((self.out/'judge_approval.json').exists())
        fork_runner = Runner(self.out)
        try:
            self.assertEqual(generation_records(fork_runner), original_records)
            self.assertEqual(fork_runner.store.rows('generation'), original_rows)
            self.assertEqual(fork_runner.store.rows('judge'), [])
            self.assertEqual(fork_runner.cfg['judge_system'], self.prompt.read_text())
            with self.assertRaisesRegex(ValueError, 'held-out judge report'):
                prepare_judges(fork_runner)
            prepared = prepare_judges(fork_runner, validation_selection=panel/'selection.private.json')
            self.assertEqual(prepared['new_judge_requests'], 8)
            for row in fork_runner.store.rows('judge'):
                request = fork_runner.store.unpack_request(row['request'])
                self.assertEqual(request['system'], self.prompt.read_text())
            with self.assertRaises((ValueError, RuntimeError)):
                fork_runner.step('generation')
            with self.assertRaises((ValueError, RuntimeError)):
                fork_runner.recover([original_rows[0]['id']], 'UNIT TEST MUST NOT REGENERATE')
        finally:
            fork_runner.close()
        reuse = json.loads((self.out/'generation_reuse.json').read_text())
        self.assertEqual(reuse['new_target_generation_cost_usd'], '0')
        self.assertEqual(reuse['judge_results_copied'], 0)
        ledger = read_jsonl(self.out/'generation_reuse_ledger.jsonl')
        self.assertTrue(all(r['new_target_cost_usd'] == '0' and not r['is_new_target_observation'] for r in ledger))
        self.assertTrue(all(r['historical_cost']['total_usd'] == '0.123' for r in ledger))
        self.assertEqual(digest(json.loads((self.out/'config.frozen.json').read_text())), fork_manifest['config_sha256'])
        cfg = json.loads((self.out/'config.frozen.json').read_text())
        template = json.loads((self.out/'judge_approval.template.json').read_text())
        primary = next(m for m in cfg['judge_models'] if m['id'] == cfg['judge']['primary'])
        self.assertEqual(template['decision'], 'not_yet_reviewed')
        self.assertEqual(template['configuration_sha256'], judge_configuration_digest(
            self.prompt.read_text(), primary, cfg['dataset_fingerprint']))

    def test_pending_source_request_prevents_a_fork(self):
        store = Store(self.source)
        store.update(store.rows('generation')[0]['id'], 'pending')
        store.close()
        with self.assertRaisesRegex(ValueError, 'terminal'):
            fork_judge_run(self.source, self.out, self.prompt)
        self.assertFalse(self.out.exists())

    def test_existing_destination_and_inside_source_are_rejected(self):
        self.out.mkdir()
        (self.out/'keep.txt').write_text('Do not overwrite')
        with self.assertRaises(ValueError):
            fork_judge_run(self.source, self.out, self.prompt)
        with self.assertRaises(ValueError):
            fork_judge_run(self.source, self.source/'child', self.prompt)
        self.assertEqual((self.out/'keep.txt').read_text(), 'Do not overwrite')

    def test_fork_configuration_cannot_change_generator_parameters(self):
        fork_judge_run(self.source, self.out, self.prompt)
        cfg = json.loads((self.out/'config.frozen.json').read_text())
        cfg['generation']['max_output_tokens'] += 1
        write_json(self.out/'config.frozen.json', cfg)
        with self.assertRaisesRegex(ValueError, 'configuration'):
            load_run(self.out)


if __name__ == '__main__':
    unittest.main()
