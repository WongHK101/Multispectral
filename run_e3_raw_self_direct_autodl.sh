#!/usr/bin/env bash
set -u
REPO=/root/autodl-tmp/Multispectral
PY=/root/autodl-tmp/envs/spectralindexgs_bw/bin/python
RUN=/root/autodl-tmp/runs/paper_autodl_full_20260429/e3_raw_self_gpu_colmap_direct_$(date +%Y%m%d_%H%M%S)
RAW=/root/autodl-tmp/datasets/Multispectral/self_m3m
COLMAP=/root/autodl-tmp/opt/colmap-cuda-3.9.1/bin/colmap
MINIMA=/root/autodl-tmp/src/MINIMA
mkdir -p "$RUN/logs" "$RUN/status"
echo "$RUN" > /root/autodl-tmp/runs/paper_autodl_full_20260429/latest_e3_raw_self_direct.txt
cat > "$RUN/config.json" <<JSON
{"method":"E3","scene":"raw_self","colmap":"$COLMAP","minima":"$MINIMA","rgb_iter":30000,"band_iter":60000,"res":8,"started_at":"$(date --iso-8601=seconds)"}
JSON
(
  echo "timestamp,index,memory.used,memory.total,utilization.gpu,power.draw"
  while [ ! -f "$RUN/status/DONE" ] && [ ! -f "$RUN/status/FAILED" ]; do
    nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader,nounits || true
    sleep 10
  done
) > "$RUN/logs/gpu_trace.csv" 2>/dev/null &
MON=$!
cd "$REPO"
echo "START pipeline $(date --iso-8601=seconds)" | tee "$RUN/status/events.log"
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 SIGS_COLMAP_EXECUTABLE="$COLMAP" "$PY" run_spectralindexgs_pipeline.py \
  --raw_root "$RAW" \
  --prepared_root "$RUN/prepared" \
  --rectified_root "$RUN/rectified" \
  --out_root "$RUN/out" \
  --minima_root "$MINIMA" \
  --minima_device cuda \
  --colmap_executable "$COLMAP" \
  --rgb_iter 30000 \
  --band_iter 60000 \
  --rgb_res 8 \
  --band_res 8 \
  --input_dynamic_range uint16 \
  --radiometric_mode exposure_normalized \
  --matching spatial \
  --matcher_args "--SpatialMatching.max_num_neighbors=80 --SpatialMatching.max_distance=500" \
  --auto_render > "$RUN/logs/pipeline.log" 2>&1
RC=$?
if [ "$RC" -ne 0 ]; then
  echo "pipeline_rc=$RC" > "$RUN/status/FAILED"
  kill "$MON" 2>/dev/null || true
  exit "$RC"
fi
echo "START render_rgb $(date --iso-8601=seconds)" | tee -a "$RUN/status/events.log"
CUDA_VISIBLE_DEVICES=0 "$PY" render.py -m "$RUN/out/Model_RGB" -s "$RUN/prepared/RGB" -r 8 --iteration 30000 --skip_train > "$RUN/logs/render_rgb.log" 2>&1 || echo render_rgb_failed > "$RUN/status/WARN_RENDER_RGB"
echo "START metrics $(date --iso-8601=seconds)" | tee -a "$RUN/status/events.log"
CUDA_VISIBLE_DEVICES=0 "$PY" metrics.py -m "$RUN/out/Model_RGB" "$RUN/out/Model_G" "$RUN/out/Model_R" "$RUN/out/Model_RE" "$RUN/out/Model_NIR" --mask_mode gt_nonzero > "$RUN/logs/metrics.log" 2>&1 || echo metrics_failed > "$RUN/status/WARN_METRICS"
echo "START index $(date --iso-8601=seconds)" | tee -a "$RUN/status/events.log"
CUDA_VISIBLE_DEVICES=0 "$PY" evaluate_spectral_indices.py \
  --g_model_dir "$RUN/out/Model_G" \
  --r_model_dir "$RUN/out/Model_R" \
  --re_model_dir "$RUN/out/Model_RE" \
  --nir_model_dir "$RUN/out/Model_NIR" \
  --iteration 60000 \
  --indices NDVI,GNDVI,NDRE \
  --out_json "$RUN/out/index_metrics_summary.json" \
  --mask_mode gt_nonzero_intersection > "$RUN/logs/index_eval.log" 2>&1 || echo index_failed > "$RUN/status/WARN_INDEX"
echo "DONE $(date --iso-8601=seconds)" | tee -a "$RUN/status/events.log"
touch "$RUN/status/DONE"
kill "$MON" 2>/dev/null || true
