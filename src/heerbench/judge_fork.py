"""Reuse immutable target answers in a fresh, explicitly separate judge study.

Only generation journal rows and their referenced blobs are copied. No generator
is called, no judge results are inherited, and the original run is read-only.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import tempfile

from .runner import load_run
from .judge_validation import judge_configuration_digest
from .store import Store
from .util import canonical, digest, software_fingerprint, write_json, write_jsonl, verified_blob_bytes


def _blob_ids(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key.endswith('_blob'):
                if not isinstance(item, str) or not re.fullmatch(r'[0-9a-f]{64}', item):
                    raise ValueError('Invalid content-addressed blob identifier in source journal')
                yield item
            else:
                yield from _blob_ids(item)
    elif isinstance(value, list):
        for item in value:
            yield from _blob_ids(item)


def fork_judge_run(source: str | Path, out: str | Path,
                   prompt_file: str | Path) -> dict:
    """Create a judge-development run while preserving exact target answers.

    All source generation rows must be terminal and match the frozen source plan.
    IDs, plan, attempts, request bodies, responses and original cost records remain
    unchanged. A separate reuse ledger distinguishes historical costs from new
    charges. The runner prohibits generation/recovery in the resulting fork.
    """
    source = Path(source).resolve()
    out = Path(out).resolve()
    prompt_file = Path(prompt_file).resolve()
    if out == source or out.is_relative_to(source):
        raise ValueError('The judge fork must be outside the original run directory')
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError('Judge fork destination must be new or empty')
    system = prompt_file.read_text(encoding='utf-8')
    if not system.strip():
        raise ValueError('Judge instructions may not be empty')
    _, source_cfg, _, plan = load_run(source)
    parent = json.loads((source/'manifest.json').read_text())
    journal = source/'journal.sqlite'
    if not journal.is_file():
        raise ValueError('A judge fork requires a completed generation journal')
    connection = sqlite3.connect(journal.as_uri()+'?mode=ro', uri=True)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute('BEGIN')
        raw_rows = [dict(r) for r in connection.execute(
            "SELECT * FROM tasks WHERE kind='generation' ORDER BY rowid")]
    finally:
        connection.close()
    by_id = {r['request_id']: r for r in plan}
    if len(raw_rows) != len(plan) or {r['id'] for r in raw_rows} != set(by_id):
        raise ValueError('Generation journal does not cover the complete frozen source plan')
    decoded = []
    for raw in raw_rows:
        row = {**raw, **{k: json.loads(raw[k]) if raw[k] is not None else None
                         for k in ('plan', 'request', 'result', 'error')}}
        if row['plan'] != by_id[row['id']]:
            raise ValueError('A source generation record differs from its frozen plan')
        if row['state'] not in {'succeeded', 'failed', 'blocked'}:
            raise ValueError('All source generation requests must be terminal before a judge fork')
        if row['state'] == 'succeeded' and (not isinstance(row['result'], dict)
                or not isinstance(row['result'].get('text'), str)):
            raise ValueError('A successful source generation has no string answer')
        decoded.append(row)
    generation_hash = digest(decoded)
    blobs = sorted(set(_blob_ids(decoded)))
    blob_paths = {}
    for blob in blobs:
        path = source/'blobs'/blob[:2]/(blob+'.txt')
        try:
            if not path.resolve().is_relative_to(source) or not path.is_file():
                raise ValueError("Missing or outside source run")
            verified_blob_bytes(path, blob)
        except (OSError, ValueError) as exc:
            raise ValueError(f"Source generation blob is missing or corrupted: {blob} ({exc})") from exc
        blob_paths[blob] = path
    provenance = {
        'purpose':'judge_development', 'parent_run_id':parent['run_id'],
        'source_run_path':str(source), 'source_config_sha256':parent['config_sha256'],
        'source_plan_sha256':parent['plan_sha256'],
        'source_software_sha256':parent['software_sha256'],
        'source_generation_sha256':generation_hash,
        'reused_generation_only':True, 'new_target_generation_requests':0,
    }
    cfg = copy.deepcopy(source_cfg)
    cfg.update(judge_system=system, judge_prompt_file=str(prompt_file),
               reused_generation_only=True, purpose='judge_development',
               calibration_only=True, generation_reuse=provenance)
    fingerprint = digest(cfg)
    manifest = {**parent, **provenance, 'run_id':fingerprint[:20],
                'config_sha256':fingerprint, 'software_sha256':software_fingerprint(),
                'created_utc':datetime.now(timezone.utc).isoformat(),
                'judge_prompt_sha256':digest(system), 'calibration_only':True,
                'scope':'Judge development and validation on unchanged frozen target answers',
                'target_response_reuse_note':'Copied target responses are the original observations, not additional repetitions.'}
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=out.name+'.staging-', dir=out.parent))
    store = None
    try:
        write_json(temporary/'config.frozen.json', cfg)
        write_json(temporary/'manifest.json', manifest)
        # Preserve even the byte formatting of the original immutable request plan.
        shutil.copyfile(source/'plan.jsonl', temporary/'plan.jsonl')
        for blob, source_path in blob_paths.items():
            target = temporary/'blobs'/blob[:2]/(blob+'.txt')
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(verified_blob_bytes(source_path, blob))
        store = Store(temporary)
        columns = ('id','kind','plan','state','attempts','request','result','error','updated')
        store.db.executemany(
            'INSERT INTO tasks('+','.join(columns)+') VALUES('+','.join('?' for _ in columns)+')',
            [tuple(r[k] for k in columns) for r in raw_rows])
        store.db.commit()
        store.event(None, 'generation_records_reused', {
            **provenance, 'count':len(raw_rows), 'new_target_generation_cost_usd':'0',
            'note':'Historical target costs remain attached to copied observations and were incurred in the source run.'})
        store.export()
        store.close()
        store = None
        write_json(temporary/'generation_reuse.json', {
            **provenance, 'records_copied':len(raw_rows), 'generation_blobs_copied':len(blobs),
            'new_target_generation_cost_usd':'0', 'judge_results_copied':0,
            'approvals_copied':0,
            'note':'Reused target outputs and validation labels retain their original IDs. This fork adds judge calls only.'})
        primary = next(m for m in cfg['judge_models'] if m['id'] == cfg['judge']['primary'])
        write_json(temporary/'judge_approval.template.json', {
            'decision':'not_yet_reviewed', 'approved_by':'', 'approved_at_utc':'',
            'scope':'', 'decision_rationale':'',
            'configuration_sha256':judge_configuration_digest(system, primary, cfg['dataset_fingerprint']),
            'heldout_report_sha256':'',
            'note':'New unapproved template for this fork. No earlier approval applies to the revised judge prompt.'})
        write_jsonl(temporary/'generation_reuse_ledger.jsonl', [
            {'request_id':r['id'], 'source_run_id':parent['run_id'], 'state':r['state'],
             'source_response_sha256':digest(r['result']),
             'historical_usage':(r['result'] or {}).get('usage'),
             'historical_cost':(r['result'] or {}).get('cost'),
             'new_target_cost_usd':'0', 'is_new_target_observation':False}
            for r in decoded])
        (temporary/'JUDGE_FORK.md').write_text(
            '# Judge development fork\n\n'
            'Target answers are copied unchanged from the recorded source run. They are not new '
            'model repetitions. Original target token and cost records are historical. This fork '
            'incurs no target generation cost. New judge calls have their own journal entries.\n\n'
            'Reuse the original selection.private.json and human-label files. Tune only on '
            'development labels. Keep held-out labels and agreement hidden until the candidate '
            'judge is frozen. Do not use the same held-out results to tune and validate a prompt.\n\n'
            'No old judge results or validation approvals have been copied. Generation dispatch '
            'and generation recovery are disabled for this run.\n', encoding='utf-8')
        if out.exists():
            out.rmdir()  # Fails safely if another process has populated the destination.
        os.replace(temporary, out)
    except BaseException:
        if store is not None:
            store.close()
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest
