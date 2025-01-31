#!/bin/bash

OFILE="results.tsv"

source ../venv/bin/activate

echo -e 'Algorithm\tQuery\tDataset\tFinal perf\tFinal perf Std.Dev\tMax perf\tMax perf Std.Dev\tMax perf epoch\tExponentially Weighted Diff Average\tExponentially Weighted Diff Std.Dev\tOOV Ratio\tVector Size\tWindow Size\tA\tmin A\tCBOW\tSkip-Gram\tNegative Sampling\tNS Exponent\tMax N\tDown-sample Threshold' > $OFILE
python rank.py results | tail -n +2 | sort -n -t $'\t' -k 7,7r -k 6,6 >> $OFILE

deactivate
