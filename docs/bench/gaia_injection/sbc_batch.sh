#!/bin/zsh
# Fit each prior-drawn injection at turin's defaults from its candidate.
S=${1:?usage: sbc_batch.sh DIR (holding truth.csv and sbcNN.csv)}
R=$S/runs; mkdir -p $R
tail -n +2 $S/truth.csv | while IFS=, read name sid P t0 T14 dep rest; do
  [ -f $R/$name/done ] && continue
  start=$(date +%s)
  TURIN_CACHE_DIR=$S/cache PYTHONUNBUFFERED=1 turin --GaiaDR3-$sid --lcfile=$S/$name.csv --period=$P --t0=$t0 --T14=$T14 --depth=$dep --outdir=$R/$name > $R/$name.log 2>&1 < /dev/null
  code=$?
  echo "$name exit $code $(( $(date +%s) - start ))s" >> $R/batch.txt
  touch $R/$name/done
done
echo BATCH DONE >> $R/batch.txt
