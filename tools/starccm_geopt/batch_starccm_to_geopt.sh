#!/usr/bin/env bash
set -euo pipefail

# Batch pipeline:
# 1) Run STAR-CCM+ macro for each .sim
# 2) Convert exported CSV to GeoPT npy triplets
#
# Example:
# bash tools/starccm_geopt/batch_starccm_to_geopt.sh \
#   /opt/Siemens/STAR-CCM+18.06.006/starccm+ \
#   /data/star_cases \
#   /data/geopt_npys \
#   /workspace/GeoPT/tools/starccm_geopt/ExportGeoPTFields.java

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 <starccm_bin> <sim_dir> <geopt_outdir> <macro_java> [n_parallel]"
  exit 1
fi

STARCCM_BIN="$1"
SIM_DIR="$2"
GEOPT_OUTDIR="$3"
MACRO_JAVA="$4"
N_PARALLEL="${5:-1}"

EXPORT_SUBDIR="geopt_exports"
CONVERTER="$(cd "$(dirname "$0")" && pwd)/starccm_csv_to_geopt.py"

mkdir -p "$GEOPT_OUTDIR"

run_one() {
  local sim_path="$1"
  local case_id="$2"

  local sim_name
  sim_name="$(basename "${sim_path%.sim}")"

  echo "[1/2] STAR-CCM export: ${sim_name}"
  "$STARCCM_BIN" -batch "$MACRO_JAVA" "$sim_path"

  local volume_csv="$(dirname "$sim_path")/${EXPORT_SUBDIR}/${sim_name}_volume.csv"
  local surface_csv="$(dirname "$sim_path")/${EXPORT_SUBDIR}/${sim_name}_surface.csv"

  if [[ ! -f "$volume_csv" || ! -f "$surface_csv" ]]; then
    echo "[ERROR] Missing exported CSV for ${sim_name}" >&2
    return 2
  fi

  echo "[2/2] Convert -> GeoPT npy: case_id=${case_id}"
  python "$CONVERTER" \
    --volume_csv "$volume_csv" \
    --surface_csv "$surface_csv" \
    --outdir "$GEOPT_OUTDIR" \
    --case_id "$case_id" \
    --compute_sdf
}

export -f run_one
export STARCCM_BIN MACRO_JAVA EXPORT_SUBDIR CONVERTER GEOPT_OUTDIR

mapfile -t sims < <(find "$SIM_DIR" -maxdepth 1 -name '*.sim' | sort)
if [[ ${#sims[@]} -eq 0 ]]; then
  echo "No .sim files found under $SIM_DIR"
  exit 1
fi

case_id=1
if [[ "$N_PARALLEL" -le 1 ]]; then
  for sim in "${sims[@]}"; do
    run_one "$sim" "$case_id"
    ((case_id++))
  done
else
  # Parallel mode: feed 'sim_path case_id' to xargs
  {
    for sim in "${sims[@]}"; do
      echo "$sim $case_id"
      ((case_id++))
    done
  } | xargs -n 2 -P "$N_PARALLEL" bash -lc 'run_one "$0" "$1"'
fi

echo "All cases finished. Output dir: $GEOPT_OUTDIR"
