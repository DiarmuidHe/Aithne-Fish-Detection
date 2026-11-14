#!/usr/bin/env bash
# Round 2 split mid-run (docs/smartbay-round-2.md): once each extra labeller's labelling has ended,
# copy its results (no videos or images) through this PC to the base instance, where
# run_round2.sh after-label (AITHNE_EXTRA_SHARDS) merges them with its own and trains. Run on the
# PC from the repository root, in Git Bash:
#   scripts/cloud_labelling/relay_shards.sh <base IP> <labeller 1 IP> [<labeller 2 IP> ...]
# Labeller i's results land on the base in round-2-shards/<i>, with .complete written last. Each
# archive is checked by SHA256 on the PC and again on the base, and kept on the PC in round-2/relay.
# Safe to rerun: a labeller whose .complete is already on the base is skipped. For a test before
# labelling starts, AITHNE_RELAY_LOG names another "labelling finished" log on the labellers and
# AITHNE_RELAY_TARGET another folder on the base.
set -uo pipefail
export MSYS_NO_PATHCONV=1

if (($# < 2)); then
  echo "usage: $0 <base IP> <labeller 1 IP> [<labeller 2 IP> ...]" >&2; exit 2
fi
BASE="$1"; shift
LABELLERS=("$@")
KEY="${AITHNE_KEY:-$HOME/.ssh/fish-sam3-key.pem}"
SSH=(ssh -o IdentitiesOnly=yes -o ConnectTimeout=20 -o ServerAliveInterval=30 -o BatchMode=yes -i "$KEY")
SCP=(scp -q -o IdentitiesOnly=yes -o ConnectTimeout=20 -o BatchMode=yes -i "$KEY")
LOCAL="data/outputs/smartbay-rounds/round-2/relay"
ROUNDS="aithne/data/outputs/smartbay-rounds"  # Under the instance user's home
POLL="${AITHNE_RELAY_POLL_SECONDS:-120}"
GIVE_UP_HOURS="${AITHNE_RELAY_HOURS:-6}"
DONE_LOG="${AITHNE_RELAY_LOG:-aithne/round2-label.log}"  # Under the labeller's home
TARGET="${AITHNE_RELAY_TARGET:-round-2-shards}"  # Under the base's smartbay-rounds
mkdir -p "$LOCAL"
log() { echo "$(date -u +%FT%TZ) $*"; }

relay() {  # index, IP: succeeds once that labeller's results are on the base
  local i="$1" ip="$2" sha local_sha
  if "${SSH[@]}" "ubuntu@$BASE" "test -f ~/$ROUNDS/$TARGET/$i/.complete"; then
    return 0
  fi
  if "${SSH[@]}" "ubuntu@$ip" "tmux has-session -t label 2>/dev/null"; then
    return 1  # Still labelling
  fi
  if ! "${SSH[@]}" "ubuntu@$ip" "grep -q 'labelling finished' ~/$DONE_LOG"; then
    log "labeller $i ($ip): no labelling session and no 'labelling finished' in its log (unreachable or not started)"
    return 1
  fi
  log "labeller $i ($ip): labelling ended; packing its results"
  sha=$("${SSH[@]}" "ubuntu@$ip" "cd ~/$ROUNDS && tar czf ~/shard-$i.tgz --exclude='*.mp4' --exclude='*.jpg' \
--exclude='*.png' --exclude='*.part' --exclude='*.tmp' --exclude='*.lock' --exclude='round-2/logs' round-2 \
&& sha256sum ~/shard-$i.tgz | cut -d' ' -f1") || { log "labeller $i: packing failed"; return 1; }
  "${SCP[@]}" "ubuntu@$ip:shard-$i.tgz" "$LOCAL/shard-$i.tgz" || { log "labeller $i: download failed"; return 1; }
  local_sha=$(sha256sum "$LOCAL/shard-$i.tgz" | cut -d' ' -f1)
  if [[ "$local_sha" != "$sha" ]]; then
    log "labeller $i: SHA256 differs after download ($local_sha, not $sha)"; return 1
  fi
  "${SCP[@]}" "$LOCAL/shard-$i.tgz" "ubuntu@$BASE:shard-$i.tgz" || { log "labeller $i: upload to the base failed"; return 1; }
  if ! "${SSH[@]}" "ubuntu@$BASE" "set -e; echo '$sha  shard-$i.tgz' | sha256sum --quiet -c -
d=~/$ROUNDS/$TARGET; rm -rf \"\$d/$i.part\"; mkdir -p \"\$d/$i.part\"
tar xzf shard-$i.tgz -C \"\$d/$i.part\" --strip-components=1; test -f \"\$d/$i.part/journal.json\"
rm -rf \"\$d/$i\"; mv -T \"\$d/$i.part\" \"\$d/$i\"; touch \"\$d/$i/.complete\""; then
    log "labeller $i: unpacking on the base failed"; return 1
  fi
  log "labeller $i: results on the base ($(du -h "$LOCAL/shard-$i.tgz" | cut -f1), SHA256 $sha)"
}

declare -A copied=()
deadline=$((SECONDS + GIVE_UP_HOURS * 3600))
log "relaying ${#LABELLERS[@]} labellers to $BASE every $POLL s, for up to $GIVE_UP_HOURS h"
while :; do
  pending=0
  for idx in "${!LABELLERS[@]}"; do
    i=$((idx + 1))
    [[ -n "${copied[$i]:-}" ]] && continue
    if relay "$i" "${LABELLERS[$idx]}"; then copied[$i]=1; else pending=$((pending + 1)); fi
  done
  if ((pending == 0)); then log "every labeller's results are on the base"; exit 0; fi
  if ((SECONDS > deadline)); then log "giving up after $GIVE_UP_HOURS h: $pending labellers not copied"; exit 1; fi
  sleep "$POLL"
done
