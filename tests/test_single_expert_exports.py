"""Human review modes export honest blank references without leaking identities."""
from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import pytest

from heerbench.judge_validation import (
    export_validation_package, human_review_protocol, load_label_csv,
    select_validation_sample,
)
from test_judge import panel_fixture


def panel(n=8, development_n=2):
    questions, sources, errors, chains, records = panel_fixture()
    selection = select_validation_sample(records, questions, chains, n=n,
                                         development_n=development_n, seed=12)
    return selection, questions, sources, errors


def read_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write_rows(path, rows):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_default_sample_is_ten_development_and_fifty_heldout_answers():
    questions, _, _, chains, records = panel_fixture()
    selected = select_validation_sample(records, questions, chains)
    assert len(selected['development']) == 10
    assert len(selected['heldout']) == 50
    dev = {r['group_id'] for r in selected['assignment'] if r['split'] == 'development'}
    held = {r['group_id'] for r in selected['assignment'] if r['split'] == 'heldout'}
    assert dev.isdisjoint(held)


def test_single_expert_exports_only_one_blank_human_reference(tmp_path):
    selection, questions, sources, errors = panel()
    selection['human_review'] = human_review_protocol(1)
    # Calibration pooling can have prepared a separate run subdirectory first.
    (tmp_path / 'pooled_run').mkdir()
    paths = export_validation_package(selection, questions, sources, errors, tmp_path)
    assert 'reviewer_1.csv' in paths
    assert 'reviewer_2.csv' not in paths and 'adjudicated.csv' not in paths
    assert not (tmp_path / 'reviewer_2.csv').exists()
    assert not (tmp_path / 'adjudicated.csv').exists()
    rows = read_rows(paths['reviewer_1.csv'])
    assert rows and all(r['verdict'] == '' and r['comment'] == '' for r in rows)
    assert len({r['blind_answer_id'] for r in rows}) == 8
    metadata = json.loads((tmp_path / 'selection.json').read_text())
    assert metadata['human_review'] == human_review_protocol(1)
    readme = (tmp_path / 'README.md').read_text()
    assert 'single-expert comparison' in readme.lower()
    assert '8 generated answers' in readme
    assert '2 for\ndevelopment and 6 held out' in readme
    assert 'does not estimate judge\nrepeatability' in readme
    assert '200-answer' not in readme and '150\nanswers' not in readme


def test_blinded_split_lists_enable_development_without_private_metadata(tmp_path):
    selection, questions, sources, errors = panel()
    paths = export_validation_package(selection, questions, sources, errors, tmp_path)
    sets = {}
    for split in ('development', 'heldout'):
        ids = json.loads((tmp_path / (split + '_answer_ids.json')).read_text())
        assert set(ids) == {'split', 'blind_answer_ids'}
        assert ids['split'] == split
        sets[split] = set(ids['blind_answer_ids'])
        assert sets[split] == {r['blind_answer_id'] for r in selection['assignment'] if r['split'] == split}
    assert sets['development'].isdisjoint(sets['heldout'])
    assert len(sets['development']) == 2 and len(sets['heldout']) == 6
    for filename in ('reviewer_1.csv', 'answers_blinded.jsonl', 'answers_blinded.md',
                     'development_answer_ids.json', 'heldout_answer_ids.json'):
        text = Path(paths[filename]).read_text(encoding='utf-8-sig')
        assert 'model-one' not in text and 'model-two' not in text
        assert 'SECRET_MODEL' not in text and 'SECRET_CONDITION' not in text
        assert not any(r['task_id'] in text for r in selection['assignment'])


def test_two_experts_preserve_both_independent_sheets_and_blank_adjudication(tmp_path):
    selection, questions, sources, errors = panel()
    selection['human_review'] = human_review_protocol(2)
    paths = export_validation_package(selection, questions, sources, errors, tmp_path, reviewers=2)
    sheets = [read_rows(paths[name]) for name in ('reviewer_1.csv', 'reviewer_2.csv', 'adjudicated.csv')]
    assert sheets[0] == sheets[1] == sheets[2]
    assert all(r['verdict'] == '' for r in sheets[0])
    metadata = json.loads((tmp_path / 'selection.json').read_text())
    assert metadata['human_review'] == human_review_protocol(2)
    assert 'Each expert labels independently' in (tmp_path / 'README.md').read_text()


@pytest.mark.parametrize('value', [0, 3, True, False, 1.0, '1', None])
def test_invalid_human_reviewer_counts_cannot_mislabel_reference(value):
    with pytest.raises(ValueError, match='1 or 2'):
        human_review_protocol(value)


def test_declared_review_protocol_cannot_change_during_export(tmp_path):
    selection, questions, sources, errors = panel()
    selection['human_review'] = human_review_protocol(2)
    with pytest.raises(ValueError, match='differs'):
        export_validation_package(selection, questions, sources, errors, tmp_path, reviewers=1)
    assert list(tmp_path.iterdir()) == []
    selection['human_review'] = {**human_review_protocol(1), 'reviewers': True}
    with pytest.raises(ValueError, match='differs'):
        export_validation_package(selection, questions, sources, errors, tmp_path)


@pytest.mark.parametrize('protected', ['reviewer_1.csv', 'reviewer_2.csv', 'adjudicated.csv',
    'reviewer_3.csv', 'selection.private.json', 'selection.json', 'answers_blinded.jsonl',
    'README.md', 'development_answer_ids.json'])
def test_export_refuses_to_overwrite_any_existing_panel_artifact(tmp_path, protected):
    selection, questions, sources, errors = panel()
    sentinel = 'KEEP PRIOR HUMAN LABELS AND PROVENANCE'
    (tmp_path / protected).write_text(sentinel)
    with pytest.raises(FileExistsError, match='overwrite'):
        export_validation_package(selection, questions, sources, errors, tmp_path)
    assert (tmp_path / protected).read_text() == sentinel
    assert {p.name for p in tmp_path.iterdir()} == {protected}


def test_split_filtered_labels_do_not_validate_unfinished_heldout_rows(tmp_path):
    selection, questions, sources, errors = panel()
    paths = export_validation_package(selection, questions, sources, errors, tmp_path)
    dev_ids = {r['blind_answer_id'] for r in selection['assignment'] if r['split'] == 'development'}
    rows = read_rows(paths['reviewer_1.csv'])
    for row in rows:
        if row['blind_answer_id'] in dev_ids:
            row['verdict'] = 'satisfied' if row['kind'] == 'positive' else 'error_absent'
        else:
            row['verdict'] = 'UNFINISHED_HELDOUT_DO_NOT_READ'
    write_rows(paths['reviewer_1.csv'], rows)
    development = load_label_csv(paths['reviewer_1.csv'], blind_answer_ids=dev_ids)
    assert {r['blind_answer_id'] for r in development} == dev_ids
    assert all(r['verdict'] != 'UNFINISHED_HELDOUT_DO_NOT_READ' for r in development)
    with pytest.raises(ValueError, match='Invalid or unfinished'):
        load_label_csv(paths['reviewer_1.csv'])
    dev_row = next(r for r in rows if r['blind_answer_id'] in dev_ids)
    rows.append(copy.deepcopy(dev_row))
    write_rows(paths['reviewer_1.csv'], rows)
    with pytest.raises(ValueError, match='Duplicate'):
        load_label_csv(paths['reviewer_1.csv'], blind_answer_ids=dev_ids)
