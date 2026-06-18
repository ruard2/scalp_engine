import sys
sys.path.insert(0, '.')
import live_engine as le
from backtest_v6 import load_10min_bars
from backtest_sweep import run, stats, patch_profiles

df = load_10min_bars()
cl = le.classify_bars(df)

print(f"{'CONFIG':<52} RESULT")
print('-' * 112)
for label, kw, patch in [
    ('SL 2.5x + EnvRange UIT + trail +2/+1', dict(sl_mult=2.5, ignore_env_range=True), dict(trail_trigger_add=2.0, trail_dist_add=1.0)),
    ('SL 3.0x + EnvRange UIT', dict(sl_mult=3.0, ignore_env_range=True), None),
    ('SL 3.0x + EnvRange UIT + trail +2/+1', dict(sl_mult=3.0, ignore_env_range=True), dict(trail_trigger_add=2.0, trail_dist_add=1.0)),
    ('SL 2.5x + EnvRange UIT + max2 cd6', dict(sl_mult=2.5, ignore_env_range=True, max_open=2, cooldown_bars=6), None),
]:
    orig = patch_profiles(**patch) if patch else None
    try:
        tr = run(cl, **kw)
    finally:
        if orig is not None:
            le.PROFILES = orig
    print(f'{label:<52} {stats(tr)}')
