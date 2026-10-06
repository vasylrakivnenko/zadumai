#!/bin/sh
# One patch (git apply format, paths from the repo root) with everything the dev copy changes against the live tree.
cd /root/zadumai_nli_proto/contract_map/typed
LIVE=/root/projects/zadumai/legalbench_map
OUT=pt0_engine.patch
: > $OUT
for f in router/frames.py router/stages.py router/harness.py router/netreader.py router/admin.html router/ui.html \
         router/located.py router/doccompile.py ask_ui.py tests/test_located.py; do
  a=$LIVE/$f; [ -e "$a" ] || a=/dev/null
  diff -uN --label "a/legalbench_map/$f" --label "b/legalbench_map/$f" "$a" "dev/$f" >> $OUT
done
echo "$(grep -c '^+++' $OUT) files, $(grep -c '^[+-][^+-]' $OUT) changed lines -> $OUT"
cd /root/projects/zadumai && git apply --check /root/zadumai_nli_proto/contract_map/typed/$OUT && echo "applies cleanly to the live tree (checked, not applied)"
