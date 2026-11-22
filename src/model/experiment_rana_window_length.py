"""Compare centered EEG window lengths with whole-session Rana holdouts."""

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfilt
import torch

import experiment_live_blink as live

base = live.base
OUT = base.ROOT/'reports/rana_window_length'
SESSIONS = tuple(base.SESSIONS)
WIDTHS = (750, 1000, 1500)
POLICIES = (live.POLICIES[0], (3, 2, .35, 1.))
THRESHOLDS = np.round(np.arange(.2, 1., .025), 3)
FALSE_ALERT_BUDGET = 5.0


def prepare(session, table):
    folder = base.OUT/session
    with np.load(folder/'aligned_eeg.npz') as eeg:
        adc, t = eeg['adc'], eeg['session_elapsed_s']
        assert np.all(np.diff(eeg['sample_index']) == 1)
    score = pd.read_csv(live.OUT/f'{session}_scores.csv', float_precision='round_trip')
    end = np.arange(3499, len(adc), 100)
    centers = np.interp(end-749.5, np.arange(len(t)), t)
    np.testing.assert_allclose(centers, score.center_s, rtol=0, atol=1e-10)
    filt = butter(3, [.5, 12], fs=1000, btype='bandpass', output='sos')
    replay_signal = sosfilt(filt, adc)
    train, replay, flat = {}, {}, {}
    for width in WIDTHS:
        parts = []
        for minute, block in table.groupby('minute', sort=False):
            lo, hi = np.searchsorted(t, [minute*60, (minute+1)*60])
            x = sosfilt(filt, adc[lo:hi])
            starts = np.searchsorted(t, block.center_s.to_numpy())-width//2-lo
            assert starts.min() >= 0 and starts.max()+width <= len(x)
            parts.append(live.waves(x, starts, starts+width))
        train[width] = np.concatenate(parts)
        left = end+1-750-width//2
        replay[width] = live.waves(replay_signal, left, left+width)
        flat[width] = np.lib.stride_tricks.sliding_window_view(adc, width)[left].std(axis=1) < 1.
    np.testing.assert_array_equal(flat[1500], score.flat)
    events = pd.read_csv(folder/'aligned_events.csv')
    return train, replay, score, events, flat


def score_for(data, width):
    duration = 1000 if width == '750+1000' else width
    return data[2].assign(flat=data[4][duration])


def evaluate(session, data, prob, threshold, gate, split, width, truth=None):
    stream = {session: (score_for(data, width), None, data[3] if truth is None else truth, None)}
    return live.evaluate(stream, {session: prob}, threshold,
                         gate, split, (session,))


def fit_fold(source, table, data):
    prepared = data[source][0]
    fit = np.flatnonzero(table.selected_for_fit)
    valid = np.flatnonzero(table.split.eq('validation') & table.classification_eligible)
    assert table.split.iloc[fit].eq('train').all()
    y = table.label.to_numpy()
    predictions = {}
    for width in WIDTHS:
        wave = prepared[width]
        scale = float(wave[fit].std())
        model, history, epoch = live.train_cnn(live.CONFIG, base.SEED, live.normalize(wave, scale),
                                               y, fit, valid, max_epochs=18)
        torch.save(dict(state_dict=model.state_dict(), scale=scale, window_samples=width,
                        config=live.CONFIG, history=history, best_epoch=epoch), OUT/f'{source}_{width}.pt')
        predictions[width] = {session: live.predict_cnn(model, live.normalize(record[1][width], scale))
                              for session, record in data.items()}
        print(source, width, 'ms, selected epoch', epoch, flush=True)
    return predictions


def summarize():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    table = pd.read_csv(base.OUT/'windows.csv')
    parts = {s: table[table.session.eq(s)].reset_index(drop=True) for s in SESSIONS}
    data = {s: prepare(s, parts[s]) for s in SESSIONS}
    comparisons, validation = [], []
    for source, target in (SESSIONS, SESSIONS[::-1]):
        predictions = fit_fold(source, parts[source], data)
        predictions['750+1000'] = {s: (predictions[750][s]+predictions[1000][s])/2 for s in SESSIONS}
        chosen = {}
        for width, pair in predictions.items():
            trials = []
            for gate in POLICIES:
                for t in map(float, THRESHOLDS):
                    m = evaluate(source, data[source], pair[source], t, gate, 'validation', width)
                    trials.append((t, gate, m))
                    validation.append(dict(source=source, width=width, threshold=t, peak_steps=gate[1], **m))
            for goal, eligible, key in [
                ('f1', trials, lambda r: (r[2]['f1'], r[2]['precision'], -r[1][1])),
                ('budget', [r for r in trials if r[2]['false_per_min'] <= FALSE_ALERT_BUDGET],
                 lambda r: (r[2]['recall'], r[2]['precision'], -r[1][1])),
            ]:
                if eligible:
                    t, gate, m = max(eligible, key=key)
                    chosen[f'{width}_{goal}'] = (width, t, gate, m)
        base.save_json(OUT/f'{source}_validation_choices.json',
                       dict(settings=chosen, false_alert_budget=FALSE_ALERT_BUDGET))
        for name, (width, t, gate, _) in chosen.items():
            result = evaluate(target, data[target], predictions[width][target], t, gate, 'all', width)
            duration = 1000 if width == '750+1000' else width
            result['delay_s'] = (duration-1)/2000+gate[1]*.1
            comparisons.append(dict(source=source, target=target, setting=name, scope='full',
                                    threshold=t, peak_steps=gate[1], **result))
        pd.DataFrame({'center_s': data[target][2].center_s,
                      **{str(width): pair[target] for width, pair in predictions.items()}}).to_csv(
            OUT/f'{target}_holdout_scores.csv', index=False)
    results = pd.DataFrame(comparisons)
    results.to_csv(OUT/'folds.csv', index=False)
    pd.DataFrame(validation).to_csv(OUT/'validation_sweep.csv', index=False)
    aggregate = results.groupby(['scope', 'setting']).agg(
        tp=('tp', 'sum'), fp=('fp', 'sum'), fn=('fn', 'sum'), minutes=('measured_minutes', 'sum'),
        min_delay_s=('delay_s', 'min'), max_delay_s=('delay_s', 'max')).reset_index()
    aggregate['precision'] = aggregate.tp/(aggregate.tp+aggregate.fp).clip(lower=1)
    aggregate['recall'] = aggregate.tp/(aggregate.tp+aggregate.fn).clip(lower=1)
    aggregate['f1'] = 2*aggregate.tp/(2*aggregate.tp+aggregate.fp+aggregate.fn).clip(lower=1)
    aggregate['false_per_min'] = aggregate.fp/aggregate.minutes
    aggregate.to_csv(OUT/'summary.csv', index=False)
    print(aggregate.to_string(index=False), flush=True)


if __name__ == '__main__':
    summarize()
