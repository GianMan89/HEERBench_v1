from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .config import load_config, load_catalog, resolve_model_ids
from .dataset import Dataset
from .planning import create_plan
from .runner import Runner, preflight, estimate_run, load_run
from .evaluation import prepare_judges, prepare_validation, calibration_report, export_analysis
from .util import project_root, write_json


def _model_options(p):
    group = p.add_mutually_exclusive_group()
    group.add_argument('--models', nargs='+', help='Catalogue IDs or quoted display names')
    group.add_argument('--group', help='Model group, e.g. default, extended or pilot')


def _study_options(p, *, models=False):
    p.add_argument('run', nargs='?', type=Path, help='Existing frozen segment (legacy direct mode)')
    p.add_argument('--study', help='Incremental study name')
    p.add_argument('--runs-root', type=Path, default=Path('runs'))
    if models:
        p.add_argument('--models', nargs='+', help='Only these target models in the study')


def parser():
    p = argparse.ArgumentParser(description='HEERBench reproducible ABF experiments')
    sub = p.add_subparsers(dest='command', required=True)
    v = sub.add_parser('validate', help='Verify embedded dataset hashes and graph')
    v.add_argument('--dataset', type=Path)
    models = sub.add_parser('models', help='List model IDs, groups, availability and deployment details')
    models.add_argument('--group')
    models.add_argument('--available', action='store_true')
    models.add_argument('--catalog', type=Path)
    plan = sub.add_parser('plan', help='Freeze one isolated run without API calls')
    plan.add_argument('--config', type=Path, default=project_root() / 'configs/main.yaml')
    plan.add_argument('--out', required=True, type=Path)
    _model_options(plan)
    plan.add_argument('--pilot', action='store_true', default=None)
    plan.add_argument('--repetitions', type=int)
    pre = sub.add_parser('preflight', help='Offline capacity and optional key-file checks')
    _study_options(pre, models=True)
    pre.add_argument('--credentials', action='store_true')
    est = sub.add_parser('estimate', help='Estimate generation and judge costs without API calls')
    _study_options(est)
    est.add_argument('--expected-output-tokens', type=int, default=1000)
    est.add_argument('--expected-judge-output-tokens', type=int, default=1200)
    est.add_argument('--calibration-answers', type=int, default=60,
                     help='Number of answers in the optional calibration cost forecast')
    est.add_argument('--calibration-repetitions', type=int, default=1,
                     help='Primary-judge calls per answer in the optional calibration cost forecast')
    run = sub.add_parser('run', help='Add/resume selected models and repetitions without rerunning completed answers')
    _study_options(run)
    _model_options(run)
    run.add_argument('--config', type=Path)
    run.add_argument('--pilot', action='store_true', default=None)
    run.add_argument('--repetitions', type=int, help='Total desired repetitions per selected model, not extra repetitions')
    run.add_argument('--plan-only', action='store_true', help='Prepare immutable segments without API calls')
    run.add_argument('--max-requests', type=int)
    judge = sub.add_parser('judge', help='Prepare/resume judging of selected completed models')
    _study_options(judge, models=True)
    judge.add_argument('--max-requests', type=int)
    judge.add_argument('--approval', type=Path)
    judge.add_argument('--heldout-report', type=Path)
    prep = sub.add_parser('judge-prepare', help='Freeze blinded judge requests and the secondary sample')
    _study_options(prep, models=True)
    prep.add_argument('--selection', type=Path)
    prep.add_argument('--repetitions', type=int, default=1, help='Repeated judging for calibration only')
    prep.add_argument('--approval', type=Path)
    prep.add_argument('--heldout-report', type=Path)
    val = sub.add_parser('calibration-prepare', help='Export blinded expert worksheets, pooling separate pilot models if needed')
    _study_options(val)
    val.add_argument('--out', required=True, type=Path)
    val.add_argument('--n', type=int, default=60)
    val.add_argument('--development-n', type=int, default=10)
    val.add_argument('--reviewers', type=int, choices=[1, 2], default=1, help='One expert reference or two independent experts with adjudication')
    val.add_argument('--group-by', choices=['component', 'question', 'scenario'], default='component')
    cal = sub.add_parser('calibration-report', help='Compare completed expert and judge labels')
    cal.add_argument('run', type=Path)
    cal.add_argument('--panel', type=Path, required=True)
    cal.add_argument('--split', choices=['development', 'heldout', 'all'], default='all')
    ana = sub.add_parser('analyse', help='Export results for one frozen segment')
    ana.add_argument('run', type=Path)
    ana.add_argument('--out', type=Path)
    ana.add_argument('--bootstrap', type=int, default=10000)
    ana.add_argument('--plots', action='store_true')
    report = sub.add_parser('report', help='Consolidate compatible runs, preserving separate protocol groups')
    report.add_argument('--all-runs', action='store_true')
    report.add_argument('--runs-root', type=Path, default=Path('runs'))
    report.add_argument('--study')
    report.add_argument('--out', required=True, type=Path)
    report.add_argument('--bootstrap', type=int, default=10000)
    report.add_argument('--plots', action='store_true')
    status = sub.add_parser('status', help='Read durable run/study state without changing journals')
    _study_options(status, models=True)
    rec = sub.add_parser('recover', help='Explicitly recover failed or uncertain transport attempts')
    rec.add_argument('run', type=Path)
    rec.add_argument('--request-id', action='append', required=True)
    rec.add_argument('--reason', required=True)
    adopt = sub.add_parser('adopt-batch', help='Attach a confirmed provider job after an interrupted submission')
    adopt.add_argument('run', type=Path)
    adopt.add_argument('--submission-id', required=True)
    adopt.add_argument('--provider-batch-id', required=True)
    adopt.add_argument('--reason', required=True)
    fork = sub.add_parser('fork-judge', help='Keep target answers fixed while developing a new judge prompt')
    fork.add_argument('run', type=Path)
    fork.add_argument('--prompt', type=Path, required=True)
    fork.add_argument('--out', type=Path, required=True)
    demo = sub.add_parser('demo', help='Zero-cost offline pipeline check with unscorable mock answers')
    demo.add_argument('--out', type=Path, required=True)
    return p


def _paths(args):
    if args.run:
        if args.study or getattr(args, 'models', None):
            raise ValueError('Choose a frozen run path or --study with optional --models')
        return [args.run], None
    if not args.study:
        raise ValueError('Provide a run path or --study')
    from .studies import study_segments
    ids = resolve_model_ids(load_catalog(), args.models) if getattr(args, 'models', None) else None
    paths = [Path(s) for s in study_segments(args.runs_root, args.study, ids)]
    if not paths:
        raise ValueError('No segments match the selected study/models')
    return paths, ids


def _estimate(path, args):
    from .cost_estimation import forecast_judges
    result = estimate_run(path, args.expected_output_tokens)
    _, cfg, ds, plan = load_run(path)
    result['judging'] = forecast_judges(cfg, ds, plan,
        expected_candidate_tokens=args.expected_output_tokens,
        expected_judge_output_tokens=args.expected_judge_output_tokens,
        calibration_answers=args.calibration_answers,
        calibration_repetitions=args.calibration_repetitions)
    write_json(path / 'cost_estimate.json', result)
    return result


def _status(path, ids=None):
    from .aggregation import read_run_snapshot
    from collections import Counter
    snap = read_run_snapshot(path)
    selected = lambda rows: [r for r in rows if ids is None or r['plan']['model_id'] in ids]
    generation = selected(snap['generation'])
    judges = selected(snap['judges'])
    # A plan that has not been opened by Runner still has all requests pending.
    known = {r['id'] for r in generation}
    planned = [r for r in snap['plan'] if ids is None or r['model_id'] in ids]
    states = Counter(r['state'] for r in generation)
    missing = sum(r['request_id'] not in known for r in planned)
    if missing:
        states['pending'] += missing
    batches = []
    journal = Path(path).resolve() / 'journal.sqlite'
    if journal.exists():
        import sqlite3
        db = sqlite3.connect(journal.as_uri() + '?mode=ro', uri=True)
        try:
            for local_id, state, handle in db.execute('SELECT id,state,handle FROM batches ORDER BY rowid'):
                batches.append({'local_submission_id': local_id, 'state': state,
                                'provider_batch_id': (json.loads(handle) if handle else {}).get('batch_id')})
        finally:
            db.close()
    return {'segment': str(path), 'generation': dict(states),
            'judge': dict(Counter(r['state'] for r in judges)), 'planned_requests': len(planned), 'batches': batches}


def _judge_study(args, paths, ids):
    prepared = []
    for path in paths:
        r = Runner(path)
        try:
            if args.command == 'judge-prepare' or args.study:
                prepared.append({'segment': str(path), **prepare_judges(r,
                    validation_selection=getattr(args, 'selection', None),
                    validation_repetitions=getattr(args, 'repetitions', 1),
                    approval_file=args.approval, heldout_report_file=args.heldout_report,
                    model_ids=ids)})
        finally:
            r.close()
    if args.command == 'judge-prepare':
        return {'prepared': prepared}
    if args.study:
        from .studies import execute_study
        models, repetitions = set(), {}
        for path in paths:
            _, cfg, _, plan = load_run(path)
            for row in plan:
                mid = row['model_id']
                if ids is None or mid in ids:
                    models.add(mid)
                    repetitions.setdefault(mid, set()).add(row['repetition'])
        selection = {'study': args.study, 'requested_models': sorted(models),
                     'requested_repetitions': {k: sorted(v) for k, v in repetitions.items()},
                     'segments': [str(p) for p in paths]}
        return {**execute_study(selection, kind='judge', max_requests=args.max_requests), 'prepared': prepared}
    r = Runner(paths[0])
    try:
        return r.step('judge', args.max_requests)
    finally:
        r.close()


def main(argv=None):
    args = parser().parse_args(argv)
    runner = None
    try:
        if getattr(args, 'max_requests', None) is not None and args.max_requests < 1:
            raise ValueError('max-requests must be positive')
        if args.command == 'validate':
            result = Dataset(args.dataset or project_root() / 'data/heerbench_v05').validate()
        elif args.command == 'models':
            cat = load_catalog(args.catalog)
            ids = resolve_model_ids(cat, group=args.group) if args.group else list(cat['models'])
            result = {'groups': cat.get('groups', {}), 'default_judges': cat.get('default_judges', {}), 'models': []}
            for mid in ids:
                m = cat['models'][mid]
                if args.available and not m.get('enabled', True):
                    continue
                result['models'].append({'id': mid, **{k: m.get(k) for k in (
                    'display_name','model','kind','enabled','availability','architecture','precision',
                    'max_context_tokens','max_input_tokens','base_url','parameters','api_key_file','pricing_valid_until')}})
        elif args.command == 'plan':
            cfg = load_config(args.config, models=args.models, model_group=args.group,
                              pilot=args.pilot, repetitions=args.repetitions)
            result = create_plan(cfg, Dataset(cfg['dataset']), args.out)
        elif args.command == 'run':
            if args.run:
                if args.study or args.models or args.group or args.config or args.pilot or args.repetitions or args.plan_only:
                    raise ValueError('Frozen run paths resume their original configuration. Use --study to select/add models or repetitions.')
                runner = Runner(args.run)
                result = runner.step(max_requests=args.max_requests)
            else:
                from .studies import prepare_study, execute_study
                config_path = args.config or project_root() / ('configs/pilot.yaml' if args.pilot else 'configs/main.yaml')
                cfg = load_config(config_path, models=args.models, model_group=args.group,
                                  pilot=args.pilot, repetitions=args.repetitions)
                study = args.study or ('pilot' if cfg['mode'] == 'pilot' else 'main')
                selected = prepare_study(cfg, args.runs_root, study)
                result = selected if args.plan_only else execute_study(selected, max_requests=args.max_requests)
                result['plan_only'] = args.plan_only
        elif args.command == 'report':
            from .aggregation import consolidate_runs
            if not args.all_runs and not args.study:
                raise ValueError('Choose --all-runs or --study to define the report scope')
            result = consolidate_runs(args.runs_root, args.out, all_runs=args.all_runs,
                study=args.study, bootstrap=args.bootstrap, plots=args.plots)
        elif args.command in {'preflight','estimate','status','judge','judge-prepare','calibration-prepare'}:
            paths, ids = _paths(args)
            if args.command == 'preflight':
                checks = [{'segment': str(path), **preflight(path, args.credentials, model_ids=ids)} for path in paths]
                result = ({k: v for k,v in checks[0].items() if k != 'segment'} if args.run else
                          {'ready': all(c['ready'] for c in checks), 'segments': checks})
            elif args.command == 'estimate':
                from decimal import Decimal
                estimates = [{'segment': str(path), **_estimate(path, args)} for path in paths]
                complete = all(e['generation_total_usd'] is not None and e['judging']['scoring_total_usd'] is not None for e in estimates)
                result = {'segments': estimates, 'planning_total_usd': str(sum(
                    Decimal(e['generation_total_usd']) + Decimal(e['judging']['scoring_total_usd']) for e in estimates)) if complete else None,
                    'note': 'Full planned generation plus scoring, not remaining cost. Optional calibration is separate and performed once per judge validation, not once per segment.'}
            elif args.command == 'status':
                states = [_status(path, ids) for path in paths]
                result = ({k: states[0][k] for k in ('generation','judge','batches')} if args.run else {'segments': states})
            elif args.command in {'judge','judge-prepare'}:
                result = _judge_study(args, paths, ids)
            else:
                if args.study:
                    from .study_calibration import prepare_study_calibration
                    bundle = prepare_study_calibration(paths, args.out / 'pooled_run')
                    runner = Runner(bundle['run'])
                else:
                    bundle = None
                    runner = Runner(paths[0])
                result = {'panel': prepare_validation(runner, args.out, args.n, args.development_n, args.group_by, reviewers=args.reviewers)}
                if bundle:
                    result['pooled_run'] = bundle
        elif args.command == 'fork-judge':
            from .judge_fork import fork_judge_run
            result = fork_judge_run(args.run, args.out, args.prompt)
        elif args.command == 'demo':
            cfg = load_config(project_root() / 'configs/demo.yaml')
            create_plan(cfg, Dataset(cfg['dataset']), args.out)
            runner = Runner(args.out)
            runner.step()
            prepare_judges(runner)
            runner.step('judge')
            result = {'is_mock': True, 'status': runner.status(),
                      'reports': export_analysis(runner, args.out / 'reports', bootstrap=100)}
        else:
            runner = Runner(args.run)
            if args.command == 'calibration-report':
                result = calibration_report(runner, args.panel, split=args.split)
            elif args.command == 'analyse':
                result = export_analysis(runner, args.out or args.run / 'reports', args.bootstrap, args.plots)
            elif args.command == 'recover':
                runner.recover(args.request_id, args.reason)
                result = runner.status()
            elif args.command == 'adopt-batch':
                runner.adopt_batch(args.submission_id, args.provider_batch_id, args.reason)
                result = runner.status()
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    except (ValueError, FileNotFoundError, RuntimeError, KeyError) as exc:
        print(f'HEERBench: {exc}', file=sys.stderr)
        raise SystemExit(2) from None
    finally:
        if runner:
            runner.close()
