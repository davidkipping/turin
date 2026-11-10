#!/bin/zsh
S=${1:?usage: inj_batch.sh WORKDIR (holding inj/ from inj_build.py)}
R=$S/inj/runs; mkdir -p $R
tail -n +2 $S/inj/truth.csv | while IFS=, read name sid P t0 T14 dep rest; do
  [ -f $R/$name/done ] && continue
  start=$(date +%s)
  TURIN_CACHE_DIR=$S/inj/cache PYTHONUNBUFFERED=1 turin --GaiaDR3-$sid --lcfile=$S/inj/$name.csv --period=$P --t0=$t0 --T14=$T14 --depth=$dep --outdir=$R/$name > $R/$name.log 2>&1 < /dev/null
  code=$?
  echo "$name exit $code $(( $(date +%s) - start ))s" >> $R/batch.txt
  touch $R/$name/done
done
echo BATCH DONE >> $R/batch.txt
