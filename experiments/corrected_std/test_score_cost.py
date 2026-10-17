import tempfile
from pathlib import Path
from score_cost import structural_rows,summary
with tempfile.TemporaryDirectory() as d:
 p=Path(d);(p/'runtime_config.tsv').write_text('key\tvalue\nstructural_cost_scope\t1\n')
 (p/'ledger.tsv').write_text('batch\texecuted_raw\tpending_start\tpending_end\taccrual_raw\tsnapshot_scale\tconservation_error\n1\t0\t0\t10\t10\t100\t0\n2\t10\t10\t0\t0\t100\t0\n')
 assert structural_rows(p,2)==[(1,10,.1),(2,0,0)] and summary(p)['cost_mean']==.05
 # A partition-only scorer or executed-only mean cannot represent this trajectory.
 try:structural_rows(p,3)
 except ValueError:pass
 else:raise AssertionError('partial experiment accepted')
print('PASS structural scorer: deferred work booked once, settlement excluded from double charging, incomplete streams rejected')
