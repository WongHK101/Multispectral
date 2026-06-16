#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/root/autodl-tmp/Multispectral}
PY=${PY:-/root/autodl-tmp/envs/spectralindexgs_bw/bin/python}
COLMAP=${COLMAP:-/root/autodl-tmp/opt/colmap-cuda-3.9.1/bin/colmap}
MINIMA=${MINIMA:-/root/autodl-tmp/src/MINIMA}
DATA=${DATA:-/root/autodl-tmp/datasets/UAV-MultiSpec3D/UAV-MultiSpec3D_Benchmark_16scenes_20260602}
WAIT_RUN=${WAIT_RUN:-/root/autodl-tmp/runs/uav_multispec3d_active17_registered100_umgs_i_20260602_120858}
RUN_ROOT=${RUN_ROOT:-/root/autodl-tmp/runs/uav_multispec3d_wogan_fix_after_batch_$(date +%Y%m%d_%H%M%S)}
GPU_ID=${GPU_ID:-0}
RGB_ITER=${RGB_ITER:-30000}
BAND_ITER=${BAND_ITER:-60000}
RAW_RES=${RAW_RES:-8}
MIN_FREE_GB=${MIN_FREE_GB:-200}

SCENES=(
  wogan_mandarin_01_20260525_1533
  wogan_mandarin_04_20260528_1558
)
BANDS=(G R RE NIR)

mkdir -p "$RUN_ROOT/status" "$RUN_ROOT/logs" "$RUN_ROOT/summary"

event() {
  echo "[$(date --iso-8601=seconds)] $*" | tee -a "$RUN_ROOT/status/events.log"
}

free_gb() {
  df -BG /root/autodl-tmp | awk 'NR==2 {gsub(/G/,"",$4); print $4}'
}

sample_gpu() {
  local out_csv="$1"
  echo "timestamp,index,memory.used,memory.total,utilization.gpu,power.draw" > "$out_csv"
  while true; do
    nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader,nounits >> "$out_csv" 2>/dev/null || true
    sleep 10
  done
}

run_stage() {
  local scene_id="$1"; local stage="$2"; shift 2
  local scene_dir="$RUN_ROOT/$scene_id"
  mkdir -p "$scene_dir/logs"
  local log="$scene_dir/logs/${stage}.log"
  local trace="$scene_dir/logs/${stage}_gpu_trace.csv"
  event "START $scene_id $stage"
  sample_gpu "$trace" &
  local mon=$!
  set +e
  CUDA_VISIBLE_DEVICES=$GPU_ID PYTHONUNBUFFERED=1 SIGS_COLMAP_EXECUTABLE="$COLMAP" "$@" > "$log" 2>&1
  local rc=$?
  set -e
  kill "$mon" 2>/dev/null || true
  wait "$mon" 2>/dev/null || true
  event "END $scene_id $stage rc=$rc"
  return "$rc"
}

clean_colmap_outputs() {
  local rgb_root="$1"
  rm -rf "$rgb_root/distorted" "$rgb_root/images" "$rgb_root/sparse"
}

write_registration_summary() {
  local scene_id="$1"; local rgb_root="$2"; local matching_used="$3"; local out_json="$4"
  "$PY" - "$scene_id" "$rgb_root/input" "$rgb_root/sparse/0/images.txt" "$matching_used" "$out_json" <<'PY'
import json, sys
from pathlib import Path

scene_id, input_dir, images_txt, matching_used, out_json = sys.argv[1:6]
input_dir = Path(input_dir)
images_txt = Path(images_txt)
out_json = Path(out_json)
expected = sorted({p.name for p in input_dir.glob("*.JPG")} | {p.name for p in input_dir.glob("*.jpg")})
registered = []
if images_txt.exists():
    lines = [l.strip() for l in images_txt.read_text(encoding="utf-8", errors="replace").splitlines() if l.strip() and not l.startswith("#")]
    for i in range(0, len(lines), 2):
        parts = lines[i].split()
        if len(parts) >= 10:
            registered.append(Path(parts[9]).name)
registered = sorted(set(registered))
missing = sorted(set(expected) - set(registered))
georeg_json = images_txt.parent / "georegistration_alignment_summary.json"
georeg = None
if georeg_json.exists():
    try:
        georeg = json.loads(georeg_json.read_text(encoding="utf-8"))
    except Exception as exc:
        georeg = {"parse_error": str(exc), "path": str(georeg_json)}
payload = {
    "scene_id": scene_id,
    "matching_used": matching_used,
    "input_rgb_count": len(expected),
    "registered_count": len(registered),
    "registration_rate": (len(registered) / len(expected)) if expected else None,
    "missing_count": len(missing),
    "missing_images": missing,
    "images_txt": str(images_txt),
    "georegistration_summary": georeg,
}
out_json.parent.mkdir(parents=True, exist_ok=True)
out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False))
PY
}

registration_is_complete() {
  local summary_json="$1"
  "$PY" - "$summary_json" <<'PY'
import json, sys
p=json.load(open(sys.argv[1], encoding="utf-8"))
ok = p.get("input_rgb_count") == p.get("registered_count") and p.get("input_rgb_count", 0) > 0
print("1" if ok else "0")
PY
}

write_registered_split() {
  local scene_id="$1"; local rgb_root="$2"; local out_json="$3"; local audit_json="$4"
  "$PY" - "$scene_id" "$rgb_root/sparse/0/images.txt" "$out_json" "$audit_json" <<'PY'
import json, sys, hashlib
from pathlib import Path

scene_id, images_txt, out_json, audit_json = sys.argv[1:5]
images_txt = Path(images_txt)
lines = [l.strip() for l in images_txt.read_text(encoding="utf-8", errors="replace").splitlines() if l.strip() and not l.startswith("#")]
names = []
for i in range(0, len(lines), 2):
    parts = lines[i].split()
    if len(parts) >= 10:
        names.append(Path(parts[9]).name)
names = sorted(set(names))
train, test = [], []
for i, name in enumerate(names):
    item = {"image_name": name, "registered_index": i}
    if i % 8 == 0:
        test.append(item)
    else:
        train.append(item)

def sha(items):
    return hashlib.sha256("\n".join([x["image_name"] for x in items]).encode("utf-8")).hexdigest()

payload = {
    "schema": "uav_multispec3d_registered_split_v1",
    "scene_id": scene_id,
    "split_name": "autodl_runtime_registered_llffhold8",
    "basis": "registered RGB camera names in final COLMAP sparse/0",
    "llffhold": 8,
    "test_rule": "sort registered RGB image names lexicographically; held-out test if zero-based sorted index % 8 == 0",
    "total_count": len(names),
    "train_count": len(train),
    "test_count": len(test),
    "train_sha256": sha(train),
    "test_sha256": sha(test),
    "train": train,
    "test": test,
}
Path(out_json).parent.mkdir(parents=True, exist_ok=True)
Path(out_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
Path(audit_json).write_text(json.dumps({
    "scene_id": scene_id,
    "images_txt": str(images_txt),
    "registered_count": len(names),
    "split_json": str(out_json),
}, indent=2), encoding="utf-8")
print(json.dumps({"scene_id": scene_id, "registered": len(names), "train": len(train), "test": len(test)}))
PY
}

install_split() {
  local split="$1"; shift
  "$PY" - "$split" "$@" <<'PY'
import json, shutil, sys, hashlib
from pathlib import Path

split = Path(sys.argv[1])
targets = [Path(x) for x in sys.argv[2:]]
p = json.loads(split.read_text(encoding="utf-8"))

def name(item):
    return Path(str(item.get("image_name") if isinstance(item, dict) else item).replace("\\", "/")).name

train = [name(x) for x in p.get("train", [])]
test = [name(x) for x in p.get("test", p.get("eval", []))]
if not train or not test:
    raise RuntimeError(f"empty train/test in {split}")
if set(train) & set(test):
    raise RuntimeError(f"train/test overlap in {split}")
for sparse in targets:
    sparse.mkdir(parents=True, exist_ok=True)
    (sparse / "train.txt").write_text("\n".join(train) + "\n", encoding="utf-8")
    (sparse / "test.txt").write_text("\n".join(test) + "\n", encoding="utf-8")
    shutil.copy2(split, sparse / "protocol_split.json")
    audit = {
        "installed_protocol_split": str(split),
        "installed_train_count": len(train),
        "installed_test_count": len(test),
        "target_sparse": str(sparse),
        "train_sha256": hashlib.sha256("\n".join(sorted(train)).encode()).hexdigest(),
        "test_sha256": hashlib.sha256("\n".join(sorted(test)).encode()).hexdigest(),
    }
    (sparse / "protocol_split_install_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
print(json.dumps({"split": str(split), "targets": len(targets), "train": len(train), "test": len(test)}))
PY
}

support_audit() {
  local scene_id="$1"; local out_root="$2"; local out_json="$3"
  "$PY" - "$scene_id" "$out_root" "$BAND_ITER" "$out_json" <<'PY'
import json, sys, numpy as np
from pathlib import Path
from plyfile import PlyData

scene, out_root, iteration, out_json = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4])
fields = ["x", "y", "z", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3", "opacity"]
rows = {}
ref = None
ref_band = None
for band in ["G", "R", "RE", "NIR"]:
    ply = out_root / f"Model_{band}" / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    row = {"exists": ply.exists(), "point_cloud": str(ply)}
    if ply.exists():
        v = PlyData.read(str(ply))["vertex"].data
        arr = np.stack([np.asarray(v[f], dtype=np.float64) for f in fields], axis=1)
        row["num_gaussians"] = int(arr.shape[0])
        if ref is None:
            ref = arr
            ref_band = band
            row["reference_band"] = True
        else:
            row[f"same_shape_as_{ref_band}"] = bool(arr.shape == ref.shape)
            if arr.shape == ref.shape:
                delta = np.abs(arr - ref)
                row[f"max_support_delta_vs_{ref_band}"] = float(delta.max()) if delta.size else 0.0
                row[f"mean_support_delta_vs_{ref_band}"] = float(delta.mean()) if delta.size else 0.0
    rows[band] = row
out_json.parent.mkdir(parents=True, exist_ok=True)
out_json.write_text(json.dumps({"scene": scene, "method": "UMGS-I", "iteration": iteration, "support_fields": fields, "bands": rows}, indent=2), encoding="utf-8")
print(json.dumps({"scene": scene, "audit": str(out_json)}))
PY
}

wait_for_current_batch() {
  event "WAIT current batch $WAIT_RUN"
  while true; do
    if pgrep -f "run_active17_registered100_umgs_i_autodl.sh" >/dev/null || pgrep -f "$WAIT_RUN" >/dev/null; then
      sleep 300
    else
      break
    fi
  done
  event "WAIT_DONE current batch finished"
}

run_colmap_scene() {
  local scene_id="$1"
  local raw_root="$DATA/scenes/$scene_id"
  local scene_dir="$RUN_ROOT/$scene_id"
  local prepared="$scene_dir/prepared"
  local out_placeholder="$scene_dir/out_placeholder_no_training"
  mkdir -p "$scene_dir/logs" "$scene_dir/summary"
  echo "$raw_root" > "$scene_dir/raw_root.txt"

  if [ ! -d "$raw_root" ]; then
    event "SKIP $scene_id missing raw root $raw_root"
    echo missing_raw_root > "$scene_dir/FAILED_RAW"
    return 1
  fi
  if ! run_stage "$scene_id" prepare_raw_step1 "$PY" run_spectralindexgs_pipeline.py \
      --raw_root "$raw_root" \
      --prepared_root "$prepared" \
      --rectified_root "$scene_dir/rectified_unused" \
      --out_root "$out_placeholder" \
      --minima_root "$MINIMA" \
      --minima_device cuda \
      --colmap_executable "$COLMAP" \
      --input_dynamic_range uint16 \
      --radiometric_mode exposure_normalized \
      --from_step 1 --to_step 1; then
    echo prepare_raw_failed > "$scene_dir/FAILED_PREPARE_RAW"
    return 1
  fi

  local rgb_root="$prepared/RGB"
  local matching_used="spatial"
  if ! run_stage "$scene_id" colmap_spatial_gpu "$PY" prepare_scene_colmap.py \
      -s "$rgb_root" \
      --colmap_executable "$COLMAP" \
      --exiftool_executable exiftool \
      --camera SIMPLE_RADIAL \
      --matching spatial \
      --matcher_args "--SpatialMatching.max_num_neighbors=80 --SpatialMatching.max_distance=500" \
      --sift_num_threads -1 \
      --sift_max_image_size 3200 \
      --sift_max_num_features 8192 \
      --sift_matching_max_num_matches 32768 \
      --sift_use_gpu 1 \
      --sift_matching_use_gpu 1 \
      --prior_position_std_m 1.0 \
      --georegistration_mode auto \
      --georegistration_backend custom_sim3; then
    echo colmap_spatial_failed_retry_exhaustive > "$scene_dir/WARN_SPATIAL_FAILED"
    clean_colmap_outputs "$rgb_root"
    matching_used="exhaustive"
    if ! run_stage "$scene_id" colmap_exhaustive_gpu "$PY" prepare_scene_colmap.py \
        -s "$rgb_root" \
        --colmap_executable "$COLMAP" \
        --exiftool_executable exiftool \
        --camera SIMPLE_RADIAL \
        --matching exhaustive \
        --sift_num_threads -1 \
        --sift_max_image_size 3200 \
        --sift_max_num_features 8192 \
        --sift_matching_max_num_matches 32768 \
        --sift_use_gpu 1 \
        --sift_matching_use_gpu 1 \
        --prior_position_std_m 1.0 \
        --georegistration_mode auto \
        --georegistration_backend custom_sim3; then
      echo colmap_failed > "$scene_dir/FAILED_COLMAP"
      return 1
    fi
  fi
  write_registration_summary "$scene_id" "$rgb_root" "$matching_used" "$scene_dir/summary/registration_summary.json" > "$scene_dir/logs/registration_summary.log" 2>&1 || true
  cp "$scene_dir/summary/registration_summary.json" "$RUN_ROOT/summary/${scene_id}_registration_summary.json" 2>/dev/null || true
}

train_scene_if_complete() {
  local scene_id="$1"
  local scene_dir="$RUN_ROOT/$scene_id"
  local summary="$scene_dir/summary/registration_summary.json"
  if [ "$(registration_is_complete "$summary")" != "1" ]; then
    event "SKIP_TRAIN $scene_id registration_not_100_percent"
    echo registration_not_100_percent > "$scene_dir/SKIP_TRAIN_REGISTRATION"
    return 0
  fi
  local raw_root="$DATA/scenes/$scene_id"
  local prepared="$scene_dir/prepared"
  local rectified="$scene_dir/rectified"
  local out="$scene_dir/out"
  local split="$scene_dir/summary/registered_llffhold8_split_v1.json"

  write_registered_split "$scene_id" "$prepared/RGB" "$split" "$scene_dir/summary/registered_split_audit.json" > "$scene_dir/logs/registered_split.log" 2>&1
  install_split "$split" "$prepared/RGB/sparse/0" > "$scene_dir/logs/install_split_rgb.log" 2>&1

  run_stage "$scene_id" train_rgb "$PY" run_spectralindexgs_pipeline.py \
      --raw_root "$raw_root" --prepared_root "$prepared" --rectified_root "$rectified" --out_root "$out" \
      --minima_root "$MINIMA" --minima_device cuda --colmap_executable "$COLMAP" \
      --rgb_iter "$RGB_ITER" --band_iter "$BAND_ITER" --rgb_res "$RAW_RES" --band_res "$RAW_RES" \
      --input_dynamic_range uint16 --radiometric_mode exposure_normalized --protocol_split "$split" \
      --from_step 2 --to_step 2 || { echo train_rgb_failed > "$scene_dir/FAILED_TRAIN_RGB"; return 1; }

  install_split "$split" "$prepared/RGB/sparse/0" > "$scene_dir/logs/install_split_after_rgb.log" 2>&1

  run_stage "$scene_id" train_bands_products_render "$PY" run_spectralindexgs_pipeline.py \
      --raw_root "$raw_root" --prepared_root "$prepared" --rectified_root "$rectified" --out_root "$out" \
      --minima_root "$MINIMA" --minima_device cuda --colmap_executable "$COLMAP" \
      --rgb_iter "$RGB_ITER" --band_iter "$BAND_ITER" --rgb_res "$RAW_RES" --band_res "$RAW_RES" \
      --input_dynamic_range uint16 --radiometric_mode exposure_normalized --from_step 3 --to_step 9 --auto_render \
      || { echo train_bands_or_render_failed > "$scene_dir/FAILED_TRAIN_BANDS_PRODUCTS_RENDER"; return 1; }

  run_stage "$scene_id" render_rgb "$PY" render.py -m "$out/Model_RGB" -s "$prepared/RGB" -r "$RAW_RES" --iteration "$RGB_ITER" --skip_train || echo render_rgb_failed > "$scene_dir/FAILED_RENDER_RGB"
  run_stage "$scene_id" metrics "$PY" metrics.py -m "$out/Model_RGB" "$out/Model_G" "$out/Model_R" "$out/Model_RE" "$out/Model_NIR" --mask_mode gt_nonzero || echo metrics_failed > "$scene_dir/FAILED_METRICS"
  run_stage "$scene_id" index_eval "$PY" evaluate_spectral_indices.py \
      --g_model_dir "$out/Model_G" --r_model_dir "$out/Model_R" --re_model_dir "$out/Model_RE" --nir_model_dir "$out/Model_NIR" \
      --iteration "$BAND_ITER" --indices NDVI,GNDVI,NDRE --out_json "$out/index_metrics_summary.json" --mask_mode gt_nonzero_intersection \
      || echo index_eval_failed > "$scene_dir/FAILED_INDEX_EVAL"
  support_audit "$scene_id" "$out" "$scene_dir/summary/support_audit.json" > "$scene_dir/logs/support_audit.log" 2>&1 || echo support_audit_failed > "$scene_dir/FAILED_SUPPORT_AUDIT"

  if compgen -G "$scene_dir/FAILED*" > /dev/null; then
    event "DONE_WITH_WARNINGS $scene_id"
  else
    echo done > "$scene_dir/DONE_TRAIN_EVAL"
    event "DONE $scene_id train_eval"
  fi
}

write_status_summary() {
  "$PY" - "$RUN_ROOT" <<'PY'
import json, sys
from pathlib import Path
root=Path(sys.argv[1])
rows=[]
for d in sorted([p for p in root.iterdir() if p.is_dir() and p.name not in {"logs","status","summary"}]):
    reg=d/"summary"/"registration_summary.json"
    reg_payload=json.loads(reg.read_text()) if reg.exists() else None
    rows.append({
        "scene_id": d.name,
        "registration_summary": reg_payload,
        "done_train_eval": (d/"DONE_TRAIN_EVAL").exists(),
        "skip_train_registration": (d/"SKIP_TRAIN_REGISTRATION").exists(),
        "failed_markers": sorted(p.name for p in d.glob("FAILED*")),
    })
summary={"run_root":str(root),"scene_count":len(rows),"done_train_eval_count":sum(r["done_train_eval"] for r in rows),"skipped_registration_count":sum(r["skip_train_registration"] for r in rows),"scenes":rows}
(root/"summary"/"wogan_fix_status_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
print(json.dumps(summary,ensure_ascii=False,indent=2))
PY
}

cd "$REPO"
cat > "$RUN_ROOT/batch_config.json" <<JSON
{
  "batch": "uav_multispec3d_wogan_fix_after_current_batch",
  "purpose": "Wait for current training batch, rerun fixed Wogan01/Wogan04 COLMAP, and train only if RGB registration is 100%",
  "wait_run": "$WAIT_RUN",
  "data_root": "$DATA",
  "scenes": ["${SCENES[0]}", "${SCENES[1]}"],
  "started_at": "$(date --iso-8601=seconds)"
}
JSON

event "RUN_ROOT $RUN_ROOT"
event "DATA $DATA"
event "SCENES ${SCENES[*]}"
wait_for_current_batch

for scene_id in "${SCENES[@]}"; do
  free="$(free_gb)"
  event "CHECK $scene_id free_gb=$free"
  if [ "$free" -lt "$MIN_FREE_GB" ]; then
    event "STOP before $scene_id free_gb=$free below MIN_FREE_GB=$MIN_FREE_GB"
    break
  fi
  run_colmap_scene "$scene_id" || true
  if [ -f "$RUN_ROOT/$scene_id/summary/registration_summary.json" ]; then
    train_scene_if_complete "$scene_id" || true
  fi
  write_status_summary || true
done

write_status_summary || true
date --iso-8601=seconds > "$RUN_ROOT/status/finished.txt"
event "ALL_DONE"
