"""End-to-end calibration integration with synthetic UNIT TEST FIXTURES only.

No provider is constructed or contacted. Fake labels and answers exist solely in
temporary test directories and are never publication data or genuine approval.
"""
import csv
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from heerbench.evaluation import (calibration_report, generation_records,
                                 prepare_judges, prepare_validation)
from heerbench.judge import JudgeRequest, SYSTEM_PROMPT, score_judgement
from heerbench.judge_validation import judge_configuration_digest
from heerbench.store import Store
from heerbench.util import digest, write_json

from test_judge import panel_fixture, valid_pass


class EvaluationIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='heerbench_test_fixture_')
        self.root = Path(self.tmp.name)
        q, sources, errors, chains, records = panel_fixture()
        local = [{'question_id': qid, 'local_error_id': qid+'-E1', 'text': row['error_conditions'][0]}
                 for qid, row in q.items()]
        dataset = SimpleNamespace(by_id=q, sources=sources, errors=errors,
                                  local_errors=local, chains=chains,
                                  fingerprint='UNIT_TEST_FIXTURE_CORPUS',
                                  for_question=lambda rows, qid: [r for r in rows if r['question_id'] == qid])
        cfg = {'mode':'pilot', 'is_mock':False, 'seed':20261127,
               'judge_system':SYSTEM_PROMPT+'\nUNIT TEST CONFIGURATION ONLY.\n',
               'judge':{'primary':'fixture-judge-1','secondary':'fixture-judge-2', 'secondary_fraction':.2},
               'judge_models':[{'id':'fixture-judge-1','model':'fixture-primary-version','max_output_tokens':8192},
                               {'id':'fixture-judge-2','model':'fixture-secondary-version','max_output_tokens':8192}]}
        self.runner = SimpleNamespace(root=self.root, cfg=cfg, dataset=dataset, store=Store(self.root))
        for r in records:
            row = {k:v for k,v in r.items() if k not in {'task_id','generation_status','text'}}
            row.update(request_id=r['task_id'], case_id=q[r['question_id']]['case_id'],
                       abf_stage=q[r['question_id']]['abf_stage'], parent_request_ids=[], parent_question_ids=[])
            self.runner.store.add(row, 'generation')
            self.runner.store.update(row['request_id'], 'succeeded',
                result={'text':r['text'], 'provider':'UNIT_TEST_FIXTURE_ONLY', 'fixture_not_research_data':True})
        self.panel = self.root/'panel'

    def tearDown(self):
        self.runner.store.close()
        self.tmp.cleanup()

    def _panel(self, n=200, development_n=50):
        prepare_validation(self.runner, self.panel, n=n, development_n=development_n, reviewers=2)
        return self.panel/'selection.private.json'

    def _fill_fixture_human_labels(self):
        """Test fixture labels only; production exports are never filled by software."""
        selection = json.loads((self.panel/'selection.private.json').read_text())
        heldout_id = next(a['blind_answer_id'] for a in selection['assignment'] if a['split'] == 'heldout')
        for filename in ('reviewer_1.csv', 'reviewer_2.csv', 'adjudicated.csv'):
            path = self.panel/filename
            with path.open(encoding='utf-8-sig', newline='') as stream:
                reader = csv.DictReader(stream)
                rows, fields = list(reader), reader.fieldnames
            changed = False
            for r in rows:
                r['verdict'] = 'satisfied' if r['kind'] == 'positive' else 'error_absent'
                r['comment'] = 'UNIT TEST FIXTURE ONLY - not a human annotation.'
                if filename == 'reviewer_2.csv' and r['blind_answer_id'] == heldout_id and r['kind'] == 'positive' and not changed:
                    r['verdict'] = 'not_satisfied'
                    changed = True
            with path.open('w', encoding='utf-8-sig', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)

    def _finish_fixture_judges(self, omit_repeat=None):
        for row in self.runner.store.rows('judge'):
            if row['plan']['judge_repetition'] == omit_repeat:
                continue
            contract = JudgeRequest(**json.loads(self.runner.store.get_blob(row['plan']['judge_contract_blob'])))
            judgement = valid_pass(contract)
            self.runner.store.update(row['id'], 'succeeded',
                result={'judgement':judgement, 'score':score_judgement(judgement, contract),
                        'fixture_not_research_data':True})

    def test_full_200_answer_panel_600_calls_report_and_bound_approval(self):
        selection_path = self._panel()
        selection = json.loads(selection_path.read_text())
        self.assertEqual(len(selection['development']), 50)
        self.assertEqual(len(selection['heldout']), 150)
        queued = prepare_judges(self.runner, validation_selection=selection_path, validation_repetitions=3)
        self.assertEqual(queued['new_judge_requests'], 600)
        self.assertEqual(prepare_judges(self.runner, validation_selection=selection_path)['new_judge_requests'], 0)
        contracts = []
        for r in self.runner.store.rows('judge'):
            req = self.runner.store.unpack_request(r['request'])
            self.assertEqual(req['system'], self.runner.cfg['judge_system'])
            payload = json.loads(req['task'])['evaluation']
            self.assertNotIn('model_id', payload)
            self.assertNotIn('condition', payload)
            contracts.append(r['plan']['judge_contract_blob'])
        self.assertEqual(len(set(contracts)), 24)
        self._fill_fixture_human_labels()
        self._finish_fixture_judges()
        report = calibration_report(self.runner, self.panel)
        self.assertEqual(report['heldout']['n_answers'], 150)
        self.assertEqual(report['development']['n_answers'], 50)
        self.assertEqual(report['repeatability']['repeats'], 3)
        self.assertEqual(report['repeatability']['n_answers'], 200)
        self.assertEqual(report['heldout']['by_kind']['positive']['agreement'], 1)
        self.assertLess(report['expert_agreement']['by_kind']['positive']['agreement'], 1)
        hashfile = json.loads((self.panel/'judge_approval.hashes.json').read_text())
        self.assertEqual(hashfile['heldout_report_sha256'], digest(report['heldout']))
        approval = json.loads((self.panel/'judge_approval.template.json').read_text())
        self.assertEqual(approval['configuration_sha256'], judge_configuration_digest(
            self.runner.cfg['judge_system'], self.runner.cfg['judge_models'][0], self.runner.dataset.fingerprint))
        approval.update(decision='approved', approved_by='UNIT TEST FIXTURE, NOT A PERSON',
                        approved_at_utc='2000-01-01T00:00:00Z', scope='UNIT TEST ONLY',
                        decision_rationale='Only exercises binding validation, not scientific acceptance.',
                        heldout_report_sha256=hashfile['heldout_report_sha256'])
        approval_path = self.root/'test_approval.json'
        write_json(approval_path, approval)
        self.runner.cfg['mode'] = 'full'
        queued = prepare_judges(self.runner, approval_file=approval_path,
                               heldout_report_file=self.panel/'judge_heldout_report.json')
        self.assertEqual(queued['new_judge_requests'], 224+45)
        roles = [r['plan']['judge_role'] for r in self.runner.store.rows('judge')]
        self.assertEqual(roles.count('primary'), 224)
        self.assertEqual(roles.count('secondary'), 45)
        self.assertEqual(roles.count('calibration'), 600)
        self.runner.cfg['judge_models'][0]['model'] = 'different-version'
        with self.assertRaisesRegex(ValueError, 'configuration'):
            prepare_judges(self.runner, approval_file=approval_path,
                           heldout_report_file=self.panel/'judge_heldout_report.json')

    def test_missing_entire_repeat_is_retained_as_missing(self):
        selection_path = self._panel(n=8, development_n=2)
        prepare_judges(self.runner, validation_selection=selection_path, validation_repetitions=3)
        self._fill_fixture_human_labels()
        self._finish_fixture_judges(omit_repeat=3)
        report = calibration_report(self.runner, self.panel)
        self.assertEqual(report['repeatability']['repeats'], 3)
        positive = report['repeatability']['by_kind']['positive']
        self.assertEqual(positive['n_items_with_missing_repeats'], positive['n_expected_items'])
        self.assertIsNone(positive['all_repeat_agreement_on_scorable_items'])

    def test_deleted_human_rows_cannot_improve_agreement_by_disappearing(self):
        selection_path = self._panel(n=8, development_n=2)
        prepare_judges(self.runner, validation_selection=selection_path)
        self._fill_fixture_human_labels()
        self._finish_fixture_judges()
        path = self.panel/'adjudicated.csv'
        with path.open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream)
            rows, fields = list(reader), reader.fieldnames
        with path.open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows[1:])
        with self.assertRaisesRegex(ValueError, 'expected.*row'):
            calibration_report(self.runner, self.panel)

    def test_changed_answer_or_assignment_is_rejected(self):
        selection_path = self._panel(n=8, development_n=2)
        selection = json.loads(selection_path.read_text())
        task = selection['development'][0]['task_id']
        self.runner.store.update(task, 'succeeded', result={'text':'A different returned answer.'})
        with self.assertRaisesRegex(ValueError, 'changed'):
            prepare_judges(self.runner, validation_selection=selection_path)

    def test_zero_repeats_and_incomplete_generation_are_rejected(self):
        selection_path = self._panel(n=8, development_n=2)
        with self.assertRaisesRegex(ValueError, 'positive integer'):
            prepare_judges(self.runner, validation_selection=selection_path, validation_repetitions=0)
        first = self.runner.store.rows('generation')[0]
        self.runner.store.update(first['id'], 'pending')
        with self.assertRaisesRegex(ValueError, 'Complete'):
            prepare_validation(self.runner, self.root/'other-panel', n=8, development_n=2)

    def test_returned_empty_answer_remains_eligible_and_can_be_prepared(self):
        first = self.runner.store.rows('generation')[0]
        self.runner.store.update(first['id'], 'succeeded', result={'text':'', 'provider':'UNIT_TEST_FIXTURE_ONLY'})
        self.assertEqual(len(generation_records(self.runner)), 224)
        prepared = prepare_judges(self.runner)
        self.assertEqual(prepared['new_judge_requests'], 224+45)
        candidates = [r for r in self.runner.store.rows('judge')
                      if r['plan']['generation_request_id'] == first['id']]
        self.assertTrue(candidates)
        for row in candidates:
            contract = json.loads(self.runner.store.get_blob(row['plan']['judge_contract_blob']))
            self.assertEqual(contract['payload']['candidate_answer'], '')


if __name__ == '__main__':
    unittest.main()
