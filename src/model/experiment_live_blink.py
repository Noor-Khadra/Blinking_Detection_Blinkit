"""Compare Rana CNN training and alert policies on saved recordings.

Run with the Rana training environment. Writes reports/live_blink_experiments.
The deployed model and source recordings are read only.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfilt
import torch

os.environ.setdefault('MPLCONFIGDIR', '/tmp/blinkit-live-experiment-mpl')
import train_rana_new as base
from train_rana_cnn_xgb import normalize, predict_cnn, train_cnn

sys.path.insert(0, str(base.ROOT / 'src/live_blink'))
from engine import PeakGate, Predictor, StreamDetector

OUT = base.ROOT / 'reports/live_blink_experiments'
SESSIONS = {'rana_01_clean': 'rana_new', 'rana_02': 'rana_new', 'noor4-clean': 'noor4_new'}
CONFIG = {'width': 16, 'dropout': .2, 'lr': .001}
THRESHOLDS = sorted(set([.8, *np.round(np.arange(.35, .951, .025), 3)]))
POLICIES = [(3, 3, .35, 1.), (3, 3, .35, .10), (3, 3, .35, .05),
            (3, 2, .35, 1.), (2, 2, .25, 1.), (2, 2, .25, .10),
            (3, 3, .45, 1.)]
GRID = np.linspace(0, 1, 192)


def waves(signal, left, right):
    return np.asarray([np.interp(GRID, np.linspace(0, 1, r-l), signal[l:r])
                       for l, r in zip(left, right)], dtype=np.float32)


def distance(times, reference):
    reference = np.sort(reference)
    if not len(reference):
        return np.full(len(times), np.inf)
    j = np.searchsorted(reference, times)
    return np.minimum(abs(times-reference[np.maximum(0, j-1)]),
                      abs(times-reference[np.minimum(len(reference)-1, j)]))


def load_stream(session, family, predictor):
    folder = base.ROOT / 'data/model' / family
    with np.load(folder / session / 'aligned_eeg.npz') as eeg:
        raw = eeg['adc']
        timeline = eeg['session_elapsed_s']
        index = eeg['sample_index']
    source = pd.read_csv(base.ROOT / 'data/raw/datasets' / session / 'eeg.csv',
                         usecols=['sample_index', 'sequence', 'adc'])
    assert np.array_equal(source.adc, raw) and np.array_equal(source.sample_index, index)
    assert np.all(np.diff(index) == 1)
    assert np.all(np.diff(source.sequence.to_numpy()) % 16 == 1)
    detector = StreamDetector(predictor)
    recorded_scores, alerts = [], []
    for rows in np.array_split(source[['sequence', 'adc']].to_numpy(), 120):
        _, scores, emitted = detector.push(rows)
        recorded_scores.extend(scores)
        alerts.extend(emitted)
    score = pd.DataFrame(recorded_scores)
    endpoint = np.arange(3499, len(raw), 100)
    center = endpoint - 749.5
    np.testing.assert_allclose(score.center_s, center / 1000)
    score['stream_s'] = score.center_s
    score['center_s'] = np.interp(center, np.arange(len(raw)), timeline)
    score['minute'] = (score.center_s // 60).astype(int)
    low_filter = butter(3, [.5, 12], fs=1000, btype='bandpass', output='sos')
    broad_filter = butter(3, [.5, 30], fs=1000, btype='bandpass', output='sos')
    inputs = {name: waves(sosfilt(filt, raw), endpoint-1499, endpoint+1)
              for name, filt in [('low', low_filter), ('broad', broad_filter)]}
    batched = np.concatenate([predictor.score_waveforms(b) for b in np.array_split(inputs['low'], 24)])
    np.testing.assert_allclose(batched, score.score, atol=2e-5, rtol=0)
    recorded_alerts = np.asarray([a['peak_s'] for a in alerts])
    exact = select_alerts(score, score.score.to_numpy(), .8, POLICIES[0])
    np.testing.assert_array_equal(score.stream_s.to_numpy()[exact], recorded_alerts)
    events = pd.read_csv(folder / session / 'aligned_events.csv')
    prepared = pd.read_csv(folder / 'windows.csv')
    prepared = prepared[prepared.session == session]
    score['scorable'] = False
    for minute, block in score.groupby('minute'):
        previous = prepared.loc[prepared.minute == minute, 'center_s'].to_numpy()
        score.loc[block.index, 'scorable'] = distance(block.center_s.to_numpy(), previous) <= .051
    score['closure'] = False
    for row in events[events.event_type == 'sustained_closure'].itertuples():
        score.loc[score.center_s.between(row.onset_session_s, row.offset_session_s), 'closure'] = True
    print(session, len(score), 'live scores', len(alerts), 'actual alerts, batch parity passed', flush=True)
    # Match the old independent minute filters for any new training inputs.
    broad_prepared = None
    if family == 'rana_new':
        chunks = []
        for minute, part in prepared.groupby('minute', sort=False):
            a, b = np.searchsorted(timeline, [minute*60, (minute+1)*60])
            chunks.append(waves(sosfilt(broad_filter, raw[a:b]),
                                part.start_sample.to_numpy()-index[a],
                                part.end_sample.to_numpy()-index[a]+1))
        broad_prepared = np.concatenate(chunks)
    return score, inputs, events, broad_prepared


def select_alerts(score, probability, threshold, policy):
    back, forward, refractory, rail = policy
    if policy == POLICIES[0]:
        gate = PeakGate(threshold)
        return np.asarray([i-forward for i, (t, p, flat) in enumerate(
            zip(score.stream_s, probability, score.flat)) if gate.push(t, p, not flat)], dtype=int)
    middle = np.arange(back, len(score)-forward)
    p = probability[middle]
    keep = ((p >= threshold) & ~score.flat.to_numpy()[middle]
            & (score.rail_fraction.to_numpy()[middle] <= rail))
    for lag in range(1, back+1):
        keep &= p > probability[middle-lag]
    for lag in range(1, forward+1):
        keep &= p >= probability[middle+lag]
    last = -1e9
    result = []
    times = score.stream_s.to_numpy()
    for i in middle[keep]:
        if times[i]-last >= refractory:
            result.append(i)
            last = times[i]
    return np.asarray(result, dtype=int)


def event_pairs(alerts, truth):
    """Chronological one-to-one matching within 250 ms, maximizing the match count."""
    i = j = 0
    pairs = []
    while i < len(alerts) and j < len(truth):
        delta = alerts[i] - truth[j]
        if delta < -.25:
            i += 1
        elif delta > .25:
            j += 1
        else:
            pairs.append((i, j))
            i += 1
            j += 1
    return pairs


def evaluate(streams, probabilities, threshold, policy, split, sessions, audit=None,
             reference_offset_s=0.):
    tp = fp = fn = scored = closure_alerts = outside_alerts = blocked = 0
    for session in sessions:
        score, _, events, _ = streams[session]
        chosen = select_alerts(score, probabilities[session], threshold, policy)
        relevant = score.minute.map(base.SPLITS).eq(split).to_numpy() if split != 'all' else np.ones(len(score), bool)
        scorable = relevant & score.scorable.to_numpy()
        blocked += np.sum(scorable & (score.flat.to_numpy() | (score.rail_fraction.to_numpy() > policy[3])))
        scored += int(scorable.sum())
        closure_alerts += np.sum(relevant[chosen] & score.closure.to_numpy()[chosen])
        outside_alerts += np.sum(relevant[chosen] & ~scorable[chosen])
        refs = events[events.event_type.eq('blink') & events.usable_auto_label].peak_session_s.to_numpy()
        for minute in np.unique(score.minute.to_numpy()[scorable]):
            mask = scorable & score.minute.eq(minute).to_numpy()
            eligible = score.center_s.to_numpy()[mask]
            labels = np.sort(refs[(refs // 60) == minute])
            labels = labels[distance(labels, eligible) <= .15]
            hits = chosen[mask[chosen]]
            detected = score.center_s.to_numpy()[hits]
            pairs = event_pairs(detected, labels + reference_offset_s)
            tp += len(pairs)
            fp += len(detected)-len(pairs)
            fn += len(labels)-len(pairs)
            if audit is not None:
                used_p, used_r = {i for i, _ in pairs}, {j for _, j in pairs}
                audit.extend({'session': session, 'time_s': float(t), 'kind': 'unmatched_alert',
                              'score': probabilities[session][k], 'rail_fraction': score.rail_fraction.iloc[k]}
                             for i, (t, k) in enumerate(zip(detected, hits)) if i not in used_p)
                for j, t in enumerate(labels):
                    if j in used_r:
                        continue
                    near = np.flatnonzero(mask & (abs(score.center_s.to_numpy() - t) <= .25))
                    peak = near[np.argmax(probabilities[session][near])]
                    audit.append({'session': session, 'time_s': float(t), 'kind': 'unmatched_auto_label',
                                  'score': probabilities[session][peak],
                                  'rail_fraction': score.rail_fraction.iloc[peak],
                                  'below_threshold': bool(probabilities[session][peak] < threshold)})
    precision, recall = tp/max(1, tp+fp), tp/max(1, tp+fn)
    return dict(tp=tp, fp=fp, fn=fn, precision=precision, recall=recall,
                f1=2*tp/max(1, 2*tp+fp+fn), false_per_min=fp/max(scored/600, 1e-9),
                measured_minutes=scored/600, closure_alerts=int(closure_alerts),
                outside_scored_alerts=int(outside_alerts), blocked_fraction=float(blocked/max(1, scored)),
                delay_s=.7495+policy[1]*.1)


def fit_variants(streams, predictor):
    d = pd.read_csv(base.OUT / 'windows.csv')
    with np.load(base.OUT / 'windows.npz') as archive:
        low = archive['waveforms_128hz']
    assert tuple(d.session.drop_duplicates()) == base.SESSIONS
    broad = np.concatenate([streams[s][3] for s in base.SESSIONS])
    assert len(low) == len(broad) == len(d)
    target = d.label.to_numpy()
    train = d.split.eq('train').to_numpy()
    valid = np.flatnonzero(d.split.eq('validation') & d.classification_eligible)
    regular = np.flatnonzero(d.selected_for_fit)
    positive = regular[target[regular] == 1]
    negatives = regular[target[regular] == 0]
    rng = np.random.default_rng(base.SEED)
    pool = np.flatnonzero(train & d.classification_eligible & d.label.eq(0))
    # Reweight high scoring training background. Eligible open-eye pseudo-labels
    # are retained because the video annotations are unreviewed.
    rank_score = np.concatenate([predictor.score_waveforms(x) for x in np.array_split(low[pool], 12)])
    ranked = pool[np.argsort(-rank_score)]
    chosen = []
    times = d.center_s.to_numpy()
    sessions = d.session.to_numpy()
    for i in ranked:
        if all(sessions[i] != sessions[j] or abs(times[i]-times[j]) >= .35 for j in chosen):
            chosen.append(i)
        if len(chosen) >= len(negatives)//3:
            break
    easy = np.setdiff1d(negatives, chosen)
    drop = rng.choice(easy, len(chosen), replace=False)
    hard = np.sort(np.r_[positive, np.setdiff1d(negatives, drop), chosen])
    assert len(hard) == len(regular)
    fits = {
        'hard_negatives': ('low', low, hard),
        'wider_band': ('broad', broad, regular),
        'low_clipping_training': ('low', low, regular[d.rail_fraction.to_numpy()[regular] <= .05]),
    }
    metadata = {'original_fit_windows': len(regular), 'mined_unique_windows': len(chosen),
                'mined_windows_original_score_ge_0_8': int(np.sum(rank_score >= .8))}
    output = {}
    for name, (band, wave, fit) in fits.items():
        assert train[fit].all()
        scale = float(wave[fit].std())
        normalized = normalize(wave, scale)
        predictions = {session: [] for session in SESSIONS}
        epochs = []
        for seed in (base.SEED, base.SEED+1):
            model, history, epoch = train_cnn(CONFIG, seed, normalized, target, fit, valid)
            torch.save({'config': CONFIG, 'scale': scale, 'band': band, 'best_epoch': epoch,
                        'state_dict': model.state_dict()}, OUT/f'{name}_{seed}.pt')
            base.save_json(OUT/f'{name}_{seed}_history.json', history)
            epochs.append(epoch)
            for session, (_, inputs, _, _) in streams.items():
                predictions[session].append(predict_cnn(model, normalize(inputs[band], scale)))
        output[name] = {session: np.mean(values, axis=0) for session, values in predictions.items()}
        metadata[name] = {'fit_windows': len(fit), 'positive_fit_windows': int(target[fit].sum()),
                          'best_epochs': epochs}
        print('Completed', name, metadata[name], flush=True)
    return output, metadata


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    predictor = Predictor()
    streams = {session: load_stream(session, family, predictor) for session, family in SESSIONS.items()}
    predictions = {'deployed': {s: streams[s][0].score.to_numpy() for s in SESSIONS}}
    alternatives, training = fit_variants(streams, predictor)
    predictions.update(alternatives)
    predictions['original_plus_hard'] = {s: (predictions['deployed'][s]+predictions['hard_negatives'][s])/2
                                         for s in SESSIONS}
    for s, (score, _, _, _) in streams.items():
        score.assign(**{name: p[s] for name, p in predictions.items()}).to_csv(OUT/f'{s}_scores.csv', index=False)
    rana = tuple(base.SESSIONS)
    baseline_val = evaluate(streams, predictions['deployed'], .8, POLICIES[0], 'validation', rana)
    budget = baseline_val['false_per_min']
    trials = []
    for name, probs in predictions.items():
        for policy in POLICIES:
            for threshold in THRESHOLDS:
                m = evaluate(streams, probs, threshold, policy, 'validation', rana)
                trials.append(dict(model=name, threshold=float(threshold), policy=policy, **m))
    frozen = {'deployed_as_is': dict(model='deployed', threshold=.8, policy=POLICIES[0], **baseline_val)}
    for name in predictions:
        candidates = [r for r in trials if r['model'] == name]
        frozen[f'{name}_max_f1'] = max(candidates, key=lambda r: (r['f1'], r['precision'], -r['delay_s']))
        within_budget = [r for r in candidates if r['false_per_min'] <= budget+1e-9]
        if within_budget:
            frozen[f'{name}_matched_budget'] = max(within_budget, key=lambda r: (r['recall'], r['precision'], -r['delay_s']))
    winners = {objective: max((r for k, r in frozen.items() if k.endswith(objective)), key=key)
               for objective, key in [('max_f1', lambda r: (r['f1'], r['precision'])),
                                      ('matched_budget', lambda r: (r['recall'], r['precision']))]}
    pd.DataFrame(trials).to_csv(OUT/'validation_trials.csv', index=False)
    base.save_json(OUT/'frozen_validation.json', {'training': training, 'false_alert_budget': budget,
                                                  'policies': frozen, 'winners': winners})
    comparison = []
    for name, row in frozen.items():
        args = (streams, predictions[row['model']], row['threshold'], row['policy'])
        for population, split, subset in [('Rana', 'test', rana), ('Noor4', 'all', ('noor4-clean',))]:
            metric = evaluate(*args, split, subset)
            comparison.append({'setting': name, 'population': population, 'validation_f1': row['f1'], **metric})
    pd.DataFrame(comparison).to_csv(OUT/'comparison.csv', index=False)
    print(pd.DataFrame(comparison)[['setting','population','tp','fp','fn','f1','false_per_min','delay_s']].to_string(index=False), flush=True)


def analyze_saved_scores():
    """Run two separate diagnostics after the main validation search."""
    streams = {}
    deployed = {}
    for session, family in SESSIONS.items():
        score = pd.read_csv(OUT/f'{session}_scores.csv', float_precision='round_trip')
        events = pd.read_csv(base.ROOT/'data/model'/family/session/'aligned_events.csv')
        streams[session] = (score, None, events, None)
        deployed[session] = score.score.to_numpy()
    noor = ('noor4-clean',)
    calibration = [dict(threshold=t, **evaluate(streams, deployed, t, POLICIES[0], 'validation', noor))
                   for t in THRESHOLDS]
    chosen = max(calibration, key=lambda x: (x['f1'], x['precision']))
    noor_check = {'noor_validation_threshold': chosen,
                  'noor_test_default': evaluate(streams, deployed, .8, POLICIES[0], 'test', noor),
                  'noor_test_calibrated': evaluate(streams, deployed, chosen['threshold'], POLICIES[0], 'test', noor)}
    rana = tuple(base.SESSIONS)
    audit = []
    evaluate(streams, deployed, .8, POLICIES[0], 'test', rana, audit)
    evaluate(streams, deployed, .8, POLICIES[0], 'all', noor, audit)
    pd.DataFrame(audit).sort_values(['session', 'time_s']).to_csv(OUT/'deployed_disagreements.csv', index=False)

    def timing(delta, split):
        return evaluate(streams, deployed, .8, POLICIES[0], split, rana, reference_offset_s=delta)

    offsets = [dict(offset_s=float(delta), **timing(delta, 'validation'))
               for delta in np.linspace(-.2, .2, 17)]
    best = max(offsets, key=lambda x: (x['f1'], -abs(x['offset_s'])))
    gates = []
    for limit in (1., .10, .05):
        policy = (3, 3, .35, limit)
        gates.append({'max_rail_fraction': limit,
                      'validation': evaluate(streams, deployed, .8, policy, 'validation', rana),
                      'test': evaluate(streams, deployed, .8, policy, 'test', rana)})
    base.save_json(OUT/'diagnostics.json', {'noor_threshold_calibration': noor_check,
                   'timing_validation_sweep': offsets, 'timing_validation_choice': best,
                   'timing_test_no_shift': timing(0, 'test'),
                   'timing_test_validation_shift': timing(best['offset_s'], 'test'),
                   'saturation_gate_only': gates})
    plot_comparison()


def plot_comparison():
    table = pd.read_csv(OUT/'comparison.csv')
    styles = [('deployed_as_is', 'Current live CNN'),
              ('deployed_max_f1', 'Current CNN, retuned alerts'),
              ('hard_negatives_max_f1', 'Hard-negative CNN'),
              ('wider_band_max_f1', 'Wider-band CNN'),
              ('low_clipping_training_max_f1', 'Lower-clipping training'),
              ('original_plus_hard_max_f1', 'Original + hard-negative CNN')]
    fig, axes = base.plt.subplots(1, 2, figsize=(10, 4.1), sharey=True)
    for ax, person in zip(axes, ('Rana', 'Noor4')):
        data = table[table.population.eq(person)].set_index('setting').loc[[s[0] for s in styles]]
        bars = ax.barh([s[1] for s in styles], 100*data.f1, color=['#1c668d']+['#7eaab6']*5)
        ax.bar_label(bars, labels=[f'{100*x:.1f}%' for x in data.f1], padding=3, fontsize=9)
        ax.set(xlim=(0, 100), xlabel='Blink event F1 (%)',
               title='Rana test blocks (reused)' if person == 'Rana' else 'Noor 4 transfer (full session)')
        ax.grid(axis='x', alpha=.25)
        ax.set_axisbelow(True)
    axes[0].invert_yaxis()
    fig.suptitle('Continuous EEG replay against automatic webcam labels', fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT/'comparison.png', dpi=150)
    base.plt.close(fig)


if __name__ == '__main__':
    if sys.argv[1:] == ['--analyze-saved-scores']:
        analyze_saved_scores()
    else:
        main()
        analyze_saved_scores()
