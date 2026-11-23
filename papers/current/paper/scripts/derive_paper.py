#!/usr/bin/env python3
"""Reconcile included observations and regenerate manuscript outputs.

Run in the pinned analysis environment. The source artifact is read-only; generated
tables, figures and the claim evidence map are written to the selected paper tree.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import re
from pathlib import Path
import statistics


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def tex(value):
    return str(value).replace('_', r'\_').replace('%', r'\%').replace('&', r'\&')


def table(path, headings, rows, long=False):
    environment = 'longtable' if long else 'tabular'
    lines = [r'\begin{' + environment + '}{' + 'l' * len(headings) + '}',
             r'\toprule', ' & '.join(headings) + r' \\', r'\midrule']
    if long:
        lines += [r'\endhead']
    lines += [' & '.join(map(tex, row)) + r' \\' for row in rows]
    lines += [r'\bottomrule', r'\end{' + environment + '}']
    path.write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--paper', type=Path, required=True)
    args = parser.parse_args()
    e, a, p = args.evidence.resolve(), args.artifact.resolve(), args.paper.resolve()
    build, figures = p / 'build', p / 'figures'
    build.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    artifact_manifest = load(a / 'manifest_v6_1.json')
    for name, record in artifact_manifest['files'].items():
        assert sha(a / name) == record['sha256'], name
    claims = {c['claim_id']: c for c in map(json.loads, (e / '04_CLAIM_EVIDENCE_REGISTRY.jsonl').read_text().splitlines())}
    used = []
    macros = {}

    def bind(name, value, places=None):
        macros[name] = f'{value:.{places}f}' if places is not None else str(value)

    def check(claim_id, value, location, row_ids, *, paper_value=None,
              transform='identity', paper_unit=None, paper_denominator=None,
              analysis_unit='See manuscript study definition'):
        c = claims[claim_id]
        expected = c['value']
        if isinstance(value, (float, int)):
            assert math.isclose(value, expected, abs_tol=0.00000051), (claim_id, value, expected)
        else:
            assert value == expected, (claim_id, value, expected)
        used.append({'manuscript_location': location, 'claim_id': claim_id,
                     'source_path': c['source']['path'], 'source_sha256': c['source']['sha256'],
                     'row_ids': '|'.join(map(str, row_ids)), 'value': json.dumps(value, sort_keys=True),
                     'manuscript_value':json.dumps(value if paper_value is None else paper_value,sort_keys=True),
                     'sign_transform':transform,
                     'unit': c['unit'], 'cohort': c['cohort'], 'evidence_class': c['evidence_class'],
                     'estimand':c['estimand'],'source_denominator':c['denominator'],
                     'manuscript_unit':c['unit'] if paper_unit is None else paper_unit,
                     'manuscript_denominator':c['denominator'] if paper_denominator is None else paper_denominator,
                     'analysis_unit':analysis_unit,
                     'interval':json.dumps(c['interval'],sort_keys=True),
                     'sign_convention':c['sign_convention'],
                     'scope_tags':json.dumps(c['scope_tags']),
                     'limitation_tags':json.dumps(c['limitation_tags'])})
        return value

    cohort=load(e / 'data/v6_1/e1/study_rows.json')
    protocol_path=e / 'data/v6_1/e1/protocol/protocol-e1.json'
    protocol=load(protocol_path)
    e1summary=load(e / 'data/v6_1/e1/summary.json')
    source_inventory=load(a / 'provenance/source-inventory.json')
    protocol_record=source_inventory['files']['evidence/data/v6_1/e1/protocol/protocol-e1.json']
    registered_protocol=source_inventory['registered_protocol_sha256']['E1']
    registered_cohort_freeze=source_inventory['e1_cohort_freeze_sha256']
    assert cohort['protocol_sha256']==registered_protocol
    assert sha(protocol_path)==protocol_record['distribution_sha256']
    assert cohort['protocol_sha256']==e1summary['protocol_sha256']
    assert cohort['cohort_freeze_sha256']==registered_cohort_freeze
    assert cohort['cohort_freeze_sha256']==e1summary['cohort_freeze_sha256']
    e1=cohort['episodes']
    planned={r['episode_id']:r for r in protocol['episodes']}
    assert {r['episode_id'] for r in e1}==set(planned)
    for r in e1:
        assert all(r[key]==planned[r['episode_id']][key] for key in ['arm','block','family','repeat_id','task_id'])
        valid=[c['ordinal'] for c in r['candidates'] if c['build_valid'] and c['worker_status']=='COMPLETED']
        assert r['selected_ordinal']==max(valid)
        assert r['first_build_valid_ordinal']==min(valid)
        assert len(r['candidates'])<=protocol['revision_budget']
        assert r['wall_s']<=protocol['wall_budget_s']
        assert {d['draw_id'] for d in r['heldout_selected_draws']}=={f'heldout-{seed}' for seed in protocol['heldout_draws_sealed']['seeds']}
    assert len(e1) == 54 and len({r['episode_id'] for r in e1}) == 54
    arms = ['A0', 'A1', 'A2']
    labels = ['CAD diagnostics', 'Aggregate feedback', 'Structured witnesses']
    outcomes, repairs = [], []
    for arm, label in zip(arms, labels):
        rows = [r for r in e1 if r['arm'] == arm]
        assert len(rows) == 18 and len({r['task_id'] for r in rows}) == 9
        counts = {k: sum(r['heldout_selected_final'] == k for r in rows)
                  for k in ['PASS', 'FAIL', 'UNKNOWN', 'CRASH', 'NO_CANDIDATE']}
        check(f'V6.1-E1-{arm}-outcomes', counts, 'Main tab:repair; Supplement episode table', [r['episode_id'] for r in rows])
        repaired = sum(r['heldout_first_final'] == 'FAIL' and r['heldout_selected_final'] == 'PASS' for r in rows)
        check(f'V6.1-E1-{arm}-first-fail-final-pass', repaired, 'Main tab:repair', [r['episode_id'] for r in rows])
        outcomes.append([label, len(rows), counts['PASS'], counts['FAIL'], sum(counts[k] for k in ['UNKNOWN', 'CRASH', 'NO_CANDIDATE']), repaired])
        bind({'A0':'EOneCadPass','A1':'EOneAggregatePass','A2':'EOneStructuredPass'}[arm],counts['PASS'])
        repairs.append(repaired)
    table(build / 'e1_main.tex', ['Feedback', '$n$', 'Pass', 'Fail', 'Other', 'Repairs'], outcomes)
    detailed = []
    for r in e1:
        candidate = next((c for c in r['candidates'] if c['ordinal'] == r['selected_ordinal']), None)
        assert candidate is not None and candidate['build_valid']
        params = candidate['params']
        detailed.append([r['episode_id'], r['submitted_revisions'], r['first_build_valid_ordinal'],
                         r['selected_ordinal'], r['heldout_first_final'], r['heldout_selected_final'],
                         '/'.join(str(params.get(k, '-')) for k in ['l1', 'l2', 'time_scale'])])
    table(build / 'e1_episodes.tex', ['Episode', 'Sent', 'First', 'Selected', 'First test', 'Final test', 'Parameters'], detailed, True)
    tasks = sorted({r['task_id'] for r in e1})
    task_contrasts = []
    for task in tasks:
        means = {arm: statistics.mean(r['heldout_selected_final'] == 'PASS' for r in e1 if r['task_id'] == task and r['arm'] == arm) for arm in arms}
        task_contrasts.append(means['A2'] - means['A1'])
    check('V6.1-E1-primary_A2_minus_A1', statistics.mean(task_contrasts), 'Abstract; Results; Figure 3', tasks)
    for key,prefix in [('primary_A2_minus_A1','EOnePrimary'),('secondary_A2_minus_A0','EOneStructuredCad'),('secondary_A1_minus_A0','EOneAggregateCad')]:
        metric=e1summary[key]
        bind(prefix,metric['estimate'],6)
        bind(prefix+'Rounded',metric['estimate'],3)
        bind(prefix+'Low',metric['interval'][0],6)
        bind(prefix+'High',metric['interval'][1],6)
        bind(prefix+'HighRounded',metric['interval'][1],3)
    for arm in arms:
        bind({'A0':'TokensCad','A1':'TokensAggregate','A2':'TokensStructured'}[arm],f"{e1summary['provider_usage'][arm]['input_tokens_sum_where_reported']:,}")
        check(f'V6.1-E1-{arm}-transitions',e1summary['completed_repair_transitions_protocol_literal'][arm],'Transition-definition disclosure',tasks)
        check(f'V6.1-E1-{arm}-transitions-buildvalid',e1summary['completed_repair_transitions'][arm],'Transition-definition disclosure',tasks)

    e0 = load(e / 'data/v6_1/e0/confirmatory-A/summary.json')
    keys = ['correct_accept', 'correct_reject', 'false_accept', 'false_reject', 'abstain', 'crash']
    e0rows = []
    checker_names = {'old_sampled': 'Sampled screen', 'v6_centerline': 'Centerline witness',
                     'v61_scoped': 'Feature-local checker', 'v61_global': 'Whole-part projection'}
    for key, label in checker_names.items():
        counts = e0['confusion_totals']['all_geometry'][key]['total']
        check(f'V6.1-E0-{key}-confusion', counts, 'Main tab:checker; Supplement checker tables', ['all 260 geometry case IDs'])
        assert sum(counts.values()) == e0['geometry_case_count']
        e0rows.append([label, 'Feature' if key == 'v61_scoped' else 'Global'] + [counts[k] for k in keys])
    table(build / 'e0_main.tex', ['Checker', 'Truth', 'CA', 'CR', 'FA', 'FR', 'U', 'Crash'], e0rows)
    families = []
    family_labels={'off_axis_control':'Off-axis valid control','off_axis_local_web':'Off-axis local web',
                   'pilot_axis':'Axial pilot holes','v61_bottom_skin_invalid':'Thin bottom skin',
                   'v61_drill_point':'Drill-point apex','v61_feature_valid':'Valid feature',
                   'v61_near_threshold':'Near-threshold wall','v61_non_feature_thin_plate':'Non-feature thin plate',
                   'v61_oblique_rim':'Oblique rim','v61_offaxis_web':'Off-axis web',
                   'v61_radial_sidewall_invalid':'Thin radial sidewall','v61_rotation_tessellation':'Rotation/tessellation',
                   'v61_through_hole':'Through hole'}
    assert set(family_labels)==set(e0['confusion_by_family'])
    for family, cells in sorted(e0['confusion_by_family'].items()):
        for key in checker_names:
            cell = cells[key]
            families.append([family_labels[family], checker_names[key], cell['n']] + [cell['total'][k] for k in keys])
    table(build / 'e0_families.tex', ['Family', 'Checker', '$n$', 'CA', 'CR', 'FA', 'FR', 'U', 'Crash'], families, True)

    with (a / 'data/operation_trials.csv').open() as stream:
        operation = list(csv.DictReader(stream))
    e2rows, e2plot, paired_counts = [], [], []
    for profile, label in [('reference', 'Reference approach'), ('historical_fast', 'Fast approach')]:
        rows = [r for r in operation if r['phase'] == 'holdout' and r['control_profile'] == profile]
        assert len(rows) == 200, (profile, collections.Counter(r['phase'] for r in operation))
        cells = {design: {r['disturbance_id']: r for r in rows if r['design_id'] == design} for design in ['original', 'historical_v4_lead_in']}
        assert set(cells['original']) == set(cells['historical_v4_lead_in']) and len(cells['original']) == 100
        values = []
        for design in cells:
            successes = sum(r['outcome'] == 'PASS' for r in cells[design].values())
            check(f'V6-E2-{profile}-{design}', successes, 'Main tab:fixture and fig:paired', list(cells[design]))
            values.append(successes)
            bind(('ETwoReference' if profile=='reference' else 'ETwoFast')+('Original' if design=='original' else 'Revised'),successes)
        e2rows.append([label, f'{values[0]}/100', f'{values[1]}/100', f'{(values[0]-values[1])/100:+.3f}'])
        check(f'V6-E2-{profile}-contrast',(values[1]-values[0])/100,
              'Main tab:fixture; manuscript uses original minus revised',list(cells['original']),
              paper_value=(values[0]-values[1])/100,transform='negate source revised-minus-original contrast',
              paper_unit='success proportion',paper_denominator=len(cells['original']),analysis_unit='paired disturbance draw')
        e2plot.append(values)
        paired_counts.append(collections.Counter((cells['original'][seed]['outcome']=='PASS',cells['historical_v4_lead_in'][seed]['outcome']=='PASS') for seed in cells['original']))
    table(build / 'e2_main.tex', ['Profile', 'Original', 'Revised', 'Original minus revised'], e2rows)
    fast_counts=paired_counts[1]
    (build/'paired_caption.tex').write_text(r'\newcommand{\PlacementPairSentence}{'+
        f"The fast profile has {fast_counts[True,True]} both-pass, "
        f"{fast_counts[False,True]} revised-only, {fast_counts[True,False]} original-only "
        f"and {fast_counts[False,False]} both-fail draws."+'}\n')

    with (a / 'data/production_runs.csv').open() as stream:
        production = list(csv.DictReader(stream))
    assert len(production) == 720 and all(r['check_status'] == 'PASS' for r in production)
    quartets = collections.defaultdict(dict)
    for r in production:
        key = (r['scenario_id'], r['paired_seed'])
        assert r['condition'] not in quartets[key]
        quartets[key][r['condition']] = float(r['good_units'])
    assert len(quartets) == 180 and all(set(v) == {'M00', 'M01', 'M10', 'M11'} for v in quartets.values())
    per_scenario = collections.defaultdict(list)
    for (scenario, seed), q in quartets.items():
        per_scenario[scenario].append([q['M10']-q['M00'], q['M01']-q['M00'], q['M11']-q['M00'], q['M11']-q['M10']-q['M01']+q['M00']])
    e3rows = []
    for scenario, values in sorted(per_scenario.items()):
        assert len(values) == 30
        e3rows.append([scenario, len(values)] + [f'{statistics.mean(v[i] for v in values):.4f}' for i in range(4)])
    for i, name in enumerate(['mechanical', 'scheduling', 'combined', 'interaction']):
        estimate = statistics.mean(statistics.mean(v[i] for v in values) for values in per_scenario.values())
        check(f'V6-E3-{name}', estimate, 'Abstract; production results; fig:production', [f'{s}:{seed}' for s, seed in quartets],
              paper_unit='good units per 1200 s run',paper_denominator=len(quartets),
              analysis_unit='180 scenario-seed quartets; 30 per fixed scenario; source720 inventories individual runs')
        bind('EThree'+name.title(),estimate,6)
    table(build / 'e3_scenarios.tex', ['Scenario', 'Paired seeds', 'Mechanical', 'Scheduling', 'Combined', 'Interaction'], e3rows)
    # Match the frozen producer's rounded t975(29)=2.045 lookup, rather
    # than silently substituting the artifact's descriptive normal interval.
    e3intervals=[]
    for scenario,values in sorted(per_scenario.items()):
        for i,name in enumerate(['Mechanical','Scheduling','Combined','Interaction']):
            deltas=[v[i] for v in values]
            center=statistics.mean(deltas)
            half=2.045*statistics.stdev(deltas)/math.sqrt(len(deltas))
            e3intervals.append([scenario,name,len(deltas),f'{center:.4f}',f'{center-half:.4f}',f'{center+half:.4f}'])
    table(build/'e3_intervals.tex',['Scenario','Contrast','$n$','Mean','Lower 95\\%','Upper 95\\%'],e3intervals)
    table(build / 'e3_runs.tex', ['Scenario', 'Seed', 'Original/FIFO', 'Original/dispatch', 'Revised/FIFO', 'Revised/dispatch'],
          [[s, seed] + [f'{quartets[s,seed][k]:g}' for k in ['M00','M01','M10','M11']] for s,seed in sorted(quartets)], True)
    e2b = [json.loads(line) for line in (e / 'data/v6_1/e2b/trials_confirmatory.jsonl').read_text().splitlines()]
    assert len(e2b) == 200 and len({(r['seed'],r['variant']) for r in e2b}) == 100
    gantry = []
    for law in ['baseline', 'optimised']:
        rows = [r for r in e2b if r['law'] == law]
        count = sum(r['success'] for r in rows)
        check(f'V6.1-E2b-{law}-success', count, 'Abstract; gantry results', [f"{r['seed']}:{r['variant']}:{law}" for r in rows])
        times = {r['time_s'] for r in rows}
        assert len(times) == 1
        # Primary registry stores a three-decimal display value; the separate
        # film metric retains the full observed floating-point duration.
        check(f'V6.1-E2b-{law}-planned-clock', round(next(iter(times)), 3), 'Gantry results (three-decimal display)', [law])
        check(f'V6.1-E2b-film.e2b.clock_s.{law}', next(iter(times)), 'Gantry full-precision source check', [law])
        gantry.append([law, count, len(rows), f'{next(iter(times)):.3f}', sum(r['mode'] == 'tray_not_seated' for r in rows), f'{statistics.mean(r["work_J"] for r in rows):.3f}'])
        prefix='GantryBaseline' if law=='baseline' else 'GantryOptimised'
        bind(prefix+'Pass',count)
        bind(prefix+'Time',next(iter(times)),3)
        bind(prefix+'Precise',next(iter(times)),9)
    feasibility=load(e/'e2b/V6_1_E2B_FEASIBILITY.json')
    gantry_primary=feasibility['paired_trials']['primary']
    # Recalculate the paired seed-cluster interval from observations, keeping
    # both variants together. No simulator or provider session is rerun.
    import numpy as np
    by_pair=collections.defaultdict(dict)
    for r in e2b:
        key=(r['seed'],r['variant'])
        assert r['law'] not in by_pair[key]
        by_pair[key][r['law']]=r
    assert all(set(v)=={'baseline','optimised'} for v in by_pair.values())
    by_seed=collections.defaultdict(list)
    for (seed,variant),pair in sorted(by_pair.items()):
        by_seed[seed].append(int(pair['optimised']['success'])-int(pair['baseline']['success']))
    assert len(by_seed)==50 and all(len(v)==2 for v in by_seed.values())
    differences=np.array([by_seed[seed] for seed in sorted(by_seed)],dtype=float)
    indices=np.random.default_rng(gantry_primary['boot_rng_seed']).integers(0,len(differences),size=(gantry_primary['n_boot'],len(differences)))
    ci=np.percentile(differences[indices].mean(axis=(1,2)),[2.5,97.5]).tolist()
    assert math.isclose(float(differences.mean()),gantry_primary['estimate'],abs_tol=1e-12)
    assert np.allclose(ci,gantry_primary['ci95_cluster_bootstrap_by_seed'],rtol=0,atol=1e-12)
    bind('GantryDifference',gantry_primary['estimate'],2)
    bind('GantryLow',gantry_primary['ci95_cluster_bootstrap_by_seed'][0],2)
    bind('GantryHigh',gantry_primary['ci95_cluster_bootstrap_by_seed'][1],2)
    formulation=load(e/'data/v6_1/e2b/formulation/formulation.json')
    bind('GantryContinuous',formulation['exact_continuous']['T_s'],9)
    continuous_sum=0.0
    for group in formulation['exact_continuous']['groups']:
        assert group['exact'] is True
        length,cap=group['L_m'],group['V_m_s']
        positive,negative=group['A_plus_m_s2'],group['A_minus_m_s2']
        peak=min(cap,math.sqrt(2*length/(1/positive+1/negative)))
        cruise=length-peak*peak*(1/positive+1/negative)/2
        duration=peak/positive+peak/negative+max(0.0,cruise)/peak
        assert math.isclose(duration,group['T'],rel_tol=0,abs_tol=1e-9)
        continuous_sum+=duration
    assert math.isclose(continuous_sum,formulation['exact_continuous']['T_s'],rel_tol=0,abs_tol=1e-9)
    diagnostic_rows=[]
    for law in ['baseline','optimised']:
        arm=feasibility['paired_trials']['arms'][law]
        for successful in [False,True]:
            rows=[r for r in e2b if r['law']==law and (not successful or r['success'])]
            saturation=sum(any(v>0 for v in r['meas']['contact']['saturation_fraction']) for r in rows)
            receipt_saturation=arm['successful_trials_only']['contact_trials_with_any_saturation'] if successful else arm['phases']['contact']['trials_with_any_saturation']
            assert saturation==receipt_saturation
            slip=max(r['meas']['contact']['tray_slip_in_fork_mm_max'] for r in rows)
            tracking=max(max(r['meas']['contact']['track_err_mm_max']) for r in rows)
            diagnostic_rows.append([law,'Successful' if successful else 'All',len(rows),saturation,f'{tracking:.4f}',f'{slip:.4f}'])
    table(build/'gantry_diagnostics.tex',['Profile','Population','$n$','Saturation','Max track mm','Max slip mm'],diagnostic_rows)
    refinement=[]
    for r in feasibility['formulation']['refinement_table']:
        refinement.append([r['n_per_group'],r['solver'],'Inaccurate group' if 'inaccurate' in r['status'] else 'Optimal',f'{r["T_s"]:.9f}',f'{r["max_violation"]:.3g}'])
    table(build/'gantry_refinement.tex',['Grid/group','Solver','Status','Time s','Raw max violation'],refinement)
    plant=formulation['plant']
    plant_rows=[]
    for i,axis in enumerate(['Y','X','Z']):
        plant_rows.append([axis,f'{plant["c1_m_per_rad"][i]:.8f}',f'{plant["M_shaft_kgm2"][i][i]:.7g}',f'{plant["D_shaft_nms"][i][i]:.7g}',f'{plant["fc_shaft_nm"][i]:.7g}',f'{plant["tau_lim_nm"][i]:.5f}',f'{plant["v_top_mm_s"][i]:.4f}'])
    table(build/'gantry_plant.tex',['Axis','$c_1$ m/rad','$M$ kg m$^2$','$D$ N m s','Friction N m','Limit N m','Top mm/s'],plant_rows)
    path_rows=[]
    for i,(name,length) in enumerate(zip(formulation['segments'],formulation['segment_length_mm'])):
        path_rows.append([i+1,name.replace('_',' ').replace('->',' to '),f'{length:.2f}','Yes' if i in formulation['carried_segments'] else 'No'])
    table(build/'gantry_path.tex',['Segment','Endpoints','Length mm','Tray carried'],path_rows)
    (build/'numbers.tex').write_text('\n'.join(r'\newcommand{' + chr(92)+name+'}{'+value+'}' for name,value in sorted(macros.items()))+'\n')
    for source in [p/'main.tex',p/'supplement.tex']:
        if source.exists():
            text=source.read_text()
            referenced=set(re.findall(r'\\((?:EOne|ETwo|EThree|Gantry|Tokens)[A-Za-z0-9]+)',text))
            assert referenced <= set(macros), (source.name,referenced-set(macros))
            required={'EOneCadPass','EOneAggregatePass','EOneStructuredPass','EOnePrimary','ETwoReferenceOriginal','ETwoReferenceRevised','GantryBaselinePass','GantryOptimisedPass','GantryOptimisedTime','GantryContinuous'}
            if source.name=='main.tex':
                required|={'EThreeMechanical','EThreeScheduling','EThreeCombined','EThreeInteraction','EOnePrimaryRounded'}
            assert required <= referenced, (source.name,'missing numerical bindings',required-referenced)
    table(build / 'gantry.tex', ['Profile', 'Pass', '$n$', 'Planned s', 'Unseated', 'Work J'], gantry)
    table(build / 'gantry_pairs.tex', ['Seed', 'Variant', 'Profile', 'Success', 'Failure mode', 'Planned s'],
          [[r['seed'],r['variant'],r['law'],int(r['success']),str(r['mode']).replace('_',' '),f'{r["time_s"]:.3f}'] for r in e2b], True)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10, 'svg.hashsalt':'forge-paper', 'axes.spines.top':False, 'axes.spines.right':False})
    fig, axes = plt.subplots(1, 2, figsize=(8, 2.8), constrained_layout=True)
    for i, family in enumerate(['arm6','arm6_cell','gantry_v4b']):
        values = [sum(r['heldout_selected_final']=='PASS' for r in e1 if r['family']==family and r['arm']==arm) for arm in arms]
        axes[0].bar([x+(i-1)*0.24 for x in range(3)], values, width=0.24, color=['#287e86','#de8c43','#92989b'][i], label=['Arm sorting','Arm seating','Gantry timing'][i])
    axes[0].set(xticks=range(3),xticklabels=['CAD','Aggregate','Structured'],ylabel='Held-out passes (6 episodes/family)',ylim=(0,7.3))
    axes[0].legend(fontsize=8,frameon=False,loc='upper center',bbox_to_anchor=(.5,1.32),ncol=3)
    axes[1].bar(range(3),repairs,color=['#92989b','#287e86','#de8c43'])
    axes[1].set(xticks=range(3),xticklabels=['CAD','Aggregate','Structured'],ylabel='First-fail to selected-pass episodes',ylim=(0,6))
    fig.savefig(figures/'repair_outcomes.pdf',metadata={'Author':'Suneet Dungrani','CreationDate':None})
    fig.savefig(figures/'repair_outcomes.svg',metadata={'Creator':'Suneet Dungrani','Date':None})
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7,2.2),constrained_layout=True)
    left = [0,0]
    for category,label,color in [((True,True),'Both pass','#287e86'),((False,True),'Revised only','#de8c43'),((True,False),'Original only','#9b709e'),((False,False),'Both fail','#92989b')]:
        values=[counts[category] for counts in paired_counts]
        ax.barh(range(2),values,left=left,label=label,color=color)
        for i,v in enumerate(values):
            if v >= 3:
                ax.text(left[i]+v/2,i,str(v),ha='center',va='center',fontsize=9)
        left=[x+y for x,y in zip(left,values)]
    assert left == [100,100]
    ax.set(yticks=range(2),yticklabels=['Reference approach','Fast approach'],xlabel='Paired held-out disturbance draws',xlim=(0,100))
    ax.legend(ncol=4,fontsize=8,frameon=False,loc='upper center',bbox_to_anchor=(.5,1.35))
    fig.savefig(figures/'paired_placements.pdf',metadata={'Author':'Suneet Dungrani','CreationDate':None})
    fig.savefig(figures/'paired_placements.svg',metadata={'Creator':'Suneet Dungrani','Date':None})
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7,3.0),constrained_layout=True)
    for i,(s,values) in enumerate(sorted(per_scenario.items())):
        center=[statistics.mean(v[j] for v in values) for j in range(4)]
        half=[2.045*statistics.stdev(v[j] for v in values)/math.sqrt(len(values)) for j in range(4)]
        ax.errorbar([j+(i-2.5)*.1 for j in range(4)],center,yerr=half,fmt='o',markersize=3,capsize=2,label=s)
    ax.axhline(0,color='gray',linewidth=.7)
    ax.set(xticks=range(4),xticklabels=['Mechanical','Scheduling','Combined','Interaction'],ylabel='Good units per 1200 s run')
    ax.legend(ncol=6,fontsize=8,frameon=False,loc='upper center',bbox_to_anchor=(.5,1.18))
    fig.savefig(figures/'production_contrasts.pdf',metadata={'Author':'Suneet Dungrani','CreationDate':None})
    fig.savefig(figures/'production_contrasts.svg',metadata={'Creator':'Suneet Dungrani','Date':None})
    plt.close(fig)
    def derived(source, key, value, location, rows, unit, denominator, scope):
        """Map additional checked metrics without inventing canonical registry IDs."""
        relative = str(source.relative_to(a))
        record = {field: '' for field in used[0]}
        record.update(manuscript_location=location, claim_id='derived:' + relative + '#' + key,
            source_path=relative, source_sha256=sha(source), row_ids=rows,
            value=json.dumps(value,sort_keys=True), manuscript_value=json.dumps(value,sort_keys=True),
            sign_transform='identity', unit=unit, manuscript_unit=unit,
            source_denominator=denominator, manuscript_denominator=denominator,
            cohort='See exact source key and rows', evidence_class='PAPER_DERIVED_FROM_HASH_VERIFIED_SOURCE',
            estimand=key, analysis_unit=scope, scope_tags=json.dumps([scope]),
            limitation_tags=json.dumps(['No new canonical registry ID; exact source-key mapping']))
        used.append(record)

    for key in ['primary_A2_minus_A1','secondary_A2_minus_A0','secondary_A1_minus_A0']:
        derived(e/'data/v6_1/e1/summary.json', key, e1summary[key],
            'Main/supplement repair estimates and intervals', '|'.join(tasks),
            'success fraction difference', 9, 'Task-cluster uncertainty; two templates')
    derived(e/'e2b/V6_1_E2B_FEASIBILITY.json', 'paired_trials.primary', gantry_primary,
        'Main/supplement gantry difference and seed-cluster interval', 'seeds9100-9149;variantsA,B',
        'success fraction difference', 50, 'Paired simulation; 100 pairs clustered by 50 seeds')
    for key in ['exact_continuous','plant','segments','segment_length_mm','carried_segments']:
        derived(e/'data/v6_1/e2b/formulation/formulation.json', key, formulation[key],
            'Gantry model, path and analytic continuous-model check', key,
            'Source fields carry units', '', 'Fixed modeled path; no hardware certificate')
    derived(e/'e2b/V6_1_E2B_FEASIBILITY.json', 'formulation.refinement_table',
        feasibility['formulation']['refinement_table'], 'Supplement solver-refinement table',
        'all supplied refinement rows', 'Seconds and raw constraint scales', '',
        'Includes inaccurate statuses; requested tolerance is not achieved accuracy')
    derived(e/'data/v6_1/e2b/trials_confirmatory.jsonl', 'derived_tracking_saturation_slip',
        diagnostic_rows, 'Supplement tracking/saturation/slip table', 'all200 rows;successful subsets explicit',
        'trials;mm', 200, 'Paired simulated trials; conditional successful subset')
    derived(a/'data/operation_trials.csv', 'derived_joint_holdout_counts',
        [{str(k):v for k,v in counts.items()} for counts in paired_counts],
        'Main paired-placement figure and caption', 'holdout;reference,historical_fast;seeds630100-630199',
        'paired draws', 100, 'Previously collected; two profiles kept separate')
    derived(a/'data/production_runs.csv', 'derived_scenario_paired_t_intervals', e3intervals,
        'Main production figure; supplement conditional interval table',
        'D1-D6;seeds0-29;M00,M01,M10,M11', 'good units per1200s run', 30,
        't975(29)=2.045; conditional on fixed envelope and scenario')
    with (p.parent / 'PAPER_CLAIM_EVIDENCE_MAP.csv').open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(used[0]),lineterminator='\n');writer.writeheader();writer.writerows(used)
    for path in figures.glob('*.svg'):
        path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    generated = sorted(str(path.relative_to(p)) for directory in [build,figures]
                       for path in directory.iterdir() if path.is_file())
    print(json.dumps({'claim_map_rows':len(used),'generated_outputs':generated}))


if __name__ == '__main__':
    main()
