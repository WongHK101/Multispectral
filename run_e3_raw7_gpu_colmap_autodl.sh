#!/usr/bin/env bash
set -u

REPO=/root/autodl-tmp/Multispectral
PY=/root/autodl-tmp/envs/spectralindexgs_bw/bin/python
DATA=/root/autodl-tmp/datasets/Multispectral
RUN_ROOT=/root/autodl-tmp/runs/paper_autodl_full_20260429/e3_raw7_gpu_colmap_$(date +%Y%m%d_%H%M%S)
COLMAP=/root/autodl-tmp/opt/colmap-cuda-3.9.1/bin/colmap
MINIMA=/root/autodl-tmp/src/MINIMA
GPU_ID=0
RGB_ITER=30000
BAND_ITER=60000
RAW_RES=8

mkdir -p "$RUN_ROOT" "$RUN_ROOT/logs" "$RUN_ROOT/status"
cat > "$RUN_ROOT/batch_config.json" <<JSON
{
  "method": "E3_independent_band_transfer",
  "batch": "raw7_gpu_colmap",
  "repo": "$REPO",
  "python": "$PY",
  "colmap": "$COLMAP",
  "minima": "$MINIMA",
  "gpu_id": $GPU_ID,
  "rgb_iter": $RGB_ITER,
  "band_iter": $BAND_ITER,
  "raw_res": $RAW_RES,
  "started_at": "$(date --iso-8601=seconds)"
}
JSON

SCENE_IDS=(raw_self raw001 raw002 raw003 raw004 raw005 raw006)
SCENE_ROOTS=(
  "$DATA/self_m3m"
  "$DATA/20240528/DJI_202405281154_001_reynoldsTR01crossrtk"
  "$DATA/20240528/DJI_202405281154_002_reynoldsAB02crossrtk"
  "$DATA/20240528/DJI_202405281220_003_reynoldsAb01crossrtk"
  "$DATA/20240528/DJI_202405281326_004_jprAb01crossrtk"
  "$DATA/20240528/DJI_202405281358_005_jprAr01crossrtk"
  "$DATA/20240528/DJI_202405281358_006_jprAb02crossrtk"
)

sample_gpu() {
  local out_csv="$1"
  echo "timestamp,index,memory.used,memory.total,utilization.gpu,power.draw" > "$out_csv"
  while true; do
    nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader,nounits >> "$out_csv" 2>/dev/null || true
    sleep 10
  done
}

run_with_monitor() {
  local scene_id="$1"
  local stage_name="$2"
  shift 2
  local scene_dir="$RUN_ROOT/$scene_id"
  mkdir -p "$scene_dir/logs"
  local log="$scene_dir/logs/${stage_name}.log"
  local trace="$scene_dir/logs/${stage_name}_gpu_trace.csv"
  echo "[$(date --iso-8601=seconds)] START $scene_id $stage_name" | tee -a "$RUN_ROOT/status/events.log"
  sample_gpu "$trace" &
  local mon_pid=$!
  set +e
  CUDA_VISIBLE_DEVICES=$GPU_ID PYTHONUNBUFFERED=1 SIGS_COLMAP_EXECUTABLE="$COLMAP" "$@" > "$log" 2>&1
  local rc=$?
  set -e
  kill "$mon_pid" 2>/dev/null || true
  wait "$mon_pid" 2>/dev/null || true
  echo "[$(date --iso-8601=seconds)] END $scene_id $stage_name rc=$rc" | tee -a "$RUN_ROOT/status/events.log"
  return "$rc"
}

cd "$REPO"

echo "Batch root: $RUN_ROOT" | tee "$RUN_ROOT/status/started.txt"
for i in "${!SCENE_IDS[@]}"; do
  scene_id="${SCENE_IDS[$i]}"
  raw_root="${SCENE_ROOTS[$i]}"
  scene_dir="$RUN_ROOT/$scene_id"
  prepared="$scene_dir/prepared"
  rectified="$scene_dir/rectified"
  out="$scene_dir/out"
  mkdir -p "$scene_dir" "$scene_dir/logs"
  echo "$raw_root" > "$scene_dir/raw_root.txt"

  if [ ! -d "$raw_root" ]; then
    echo "[$(date --iso-8601=seconds)] SKIP $scene_id missing raw root $raw_root" | tee -a "$RUN_ROOT/status/events.log"
    echo "missing_raw_root" > "$scene_dir/FAILED"
    continue
  fi

  run_with_monitor "$scene_id" pipeline \
    "$PY" run_spectralindexgs_pipeline.py \
      --raw_root "$raw_root" \
      --prepared_root "$prepared" \
      --rectified_root "$rectified" \
      --out_root "$out" \
      --minima_root "$MINIMA" \
      --minima_device cuda \
      --colmap_executable "$COLMAP" \
      --rgb_iter "$RGB_ITER" \
      --band_iter "$BAND_ITER" \
      --rgb_res "$RAW_RES" \
      --band_res "$RAW_RES" \
      --input_dynamic_range uint16 \
      --radiometric_mode exposure_normalized \
      --matching spatial \
      --matcher_args "--SpatialMatching.max_num_neighbors=80 --SpatialMatching.max_distance=500" \
      --auto_render
  rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "pipeline_rc=$rc" > "$scene_dir/FAILED"
    continue
  fi

  run_with_monitor "$scene_id" render_rgb \
    "$PY" render.py -m "$out/Model_RGB" -s "$prepared/RGB" -r "$RAW_RES" --iteration "$RGB_ITER" --skip_train
  rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "render_rgb_rc=$rc" > "$scene_dir/FAILED_RENDER_RGB"
  fi

  run_with_monitor "$scene_id" metrics \
    "$PY" metrics.py -m "$out/Model_RGB" "$out/Model_G" "$out/Model_R" "$out/Model_RE" "$out/Model_NIR" --mask_mode gt_nonzero
  rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "metrics_rc=$rc" > "$scene_dir/FAILED_METRICS"
  fi

  run_with_monitor "$scene_id" index_eval \
    "$PY" evaluate_spectral_indices.py \
      --g_model_dir "$out/Model_G" \
      --r_model_dir "$out/Model_R" \
      --re_model_dir "$out/Model_RE" \
      --nir_model_dir "$out/Model_NIR" \
      --iteration "$BAND_ITER" \
      --indices NDVI,GNDVI,NDRE \
      --out_json "$out/index_metrics_summary.json" \
      --mask_mode gt_nonzero_intersection
  rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "index_rc=$rc" > "$scene_dir/FAILED_INDEX"
  fi

  echo "done" > "$scene_dir/DONE"
done

date --iso-8601=seconds > "$RUN_ROOT/status/finished.txt"
echo "Batch finished: $RUN_ROOT" | tee -a "$RUN_ROOT/status/events.log"
