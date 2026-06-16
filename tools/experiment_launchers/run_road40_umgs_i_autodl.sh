#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/root/autodl-tmp/Multispectral}
PY=${PY:-/root/autodl-tmp/envs/spectralindexgs_bw/bin/python}
COLMAP=${COLMAP:-/root/autodl-tmp/opt/colmap-cuda-3.9.1/bin/colmap}
MINIMA=${MINIMA:-/root/autodl-tmp/src/MINIMA}
DATA=${DATA:-/root/autodl-tmp/datasets/UAV-MultiSpec3D/UAV-MultiSpec3D_Benchmark_16scenes_20260602}
PRE_SCENE=${PRE_SCENE:-/root/autodl-tmp/runs/uav_multispec3d_road_altitude_colmap_after_wogan_20260603_040812/DJI_202606021648_001_road-40m}
SCENE_ID=${SCENE_ID:-road_01_20260602_1648_40m}
RUN_ROOT=${RUN_ROOT:-/root/autodl-tmp/runs/uav_multispec3d_road40_umgs_i_$(date +%Y%m%d_%H%M%S)}
GPU_ID=${GPU_ID:-0}
RGB_ITER=${RGB_ITER:-30000}
BAND_ITER=${BAND_ITER:-60000}
RAW_RES=${RAW_RES:-8}
MIN_FREE_GB=${MIN_FREE_GB:-200}
RESUME_AFTER_RGB=${RESUME_AFTER_RGB:-0}
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

write_registered_split() {
  local scene_id="$1"; local rgb_root="$2"; local out_json="$3"; local audit_json="$4"
  "$PY" - "$scene_id" "$rgb_root/sparse/0/images.txt" "$out_json" "$audit_json" <<'PY'
import json, sys, hashlib
from pathlib import Path

scene_id, images_txt, out_json, audit_json = sys.argv[1:5]
images_txt = Path(images_txt)
lines = [l.strip() for l in images_txt.read_text(encoding="utf-8", errors="replace").splitlines()
         if l.strip() and not l.startswith("#")]
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
    "note": "Road 40m run. Split generated from accepted 94/94 registered COLMAP sparse under unchanged benchmark protocol."
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
        "runtime_split_source_expected": "explicit_train_test_txt",
        "target_sparse": str(sparse),
        "train_sha256": hashlib.sha256("\n".join(sorted(train)).encode()).hexdigest(),
        "test_sha256": hashlib.sha256("\n".join(sorted(test)).encode()).hexdigest(),
    }
    (sparse / "protocol_split_install_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
print(json.dumps({"split": str(split), "targets": len(targets), "train": len(train), "test": len(test)}))
PY
}

fix_manifest_source_paths() {
  local prepared_root="$1"; local raw_root="$2"; local out_json="$3"
  "$PY" - "$prepared_root" "$raw_root" "$out_json" <<'PY'
import json, sys
from pathlib import Path

prepared = Path(sys.argv[1])
raw = Path(sys.argv[2])
out = Path(sys.argv[3])
fixed = {}
missing = {}
for sub in ["RGB", "G_raw", "R_raw", "RE_raw", "NIR_raw"]:
    manifest = prepared / sub / "spectral_manifest.json"
    if not manifest.exists():
        missing[sub] = ["missing_manifest"]
        continue
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["scene_root"] = str(raw)
    n_fixed = 0
    n_missing = 0
    examples = []
    for item in payload.get("images", []):
        source = Path(str(item.get("source_path", "")))
        if not source.name:
            n_missing += 1
            continue
        candidate = raw / source.name
        if candidate.exists():
            if str(candidate) != str(source):
                item["source_path"] = str(candidate)
                n_fixed += 1
        else:
            n_missing += 1
            if len(examples) < 5:
                examples.append(str(candidate))
    manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    fixed[sub] = {
        "manifest": str(manifest),
        "images": len(payload.get("images", [])),
        "source_paths_rewritten": n_fixed,
        "missing_sources": n_missing,
        "missing_examples": examples,
    }
out.parent.mkdir(parents=True, exist_ok=True)
summary = {
    "prepared_root": str(prepared),
    "raw_root": str(raw),
    "fixed": fixed,
    "missing": missing,
}
out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
bad = {k: v for k, v in fixed.items() if v.get("missing_sources")}
if bad:
    raise RuntimeError(f"Missing source paths after manifest rewrite: {bad}")
print(json.dumps(summary, ensure_ascii=False, indent=2))
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
out_json.write_text(json.dumps({
    "scene": scene,
    "method": "UMGS-I",
    "iteration": iteration,
    "support_fields": fields,
    "bands": rows,
}, indent=2), encoding="utf-8")
print(json.dumps({"scene": scene, "audit": str(out_json)}))
PY
}

write_final_summary() {
  "$PY" - "$RUN_ROOT" <<'PY'
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for d in sorted([p for p in root.iterdir() if p.is_dir() and p.name not in {"logs", "status", "summary"}]):
    failed = sorted(p.name for p in d.glob("FAILED*"))
    idx = d / "out" / "index_metrics_summary.json"
    rows.append({
        "scene_id": d.name,
        "done_train_eval": (d / "DONE_TRAIN_EVAL").exists(),
        "failed_markers": failed,
        "has_rgb_model": (d / "out" / "Model_RGB" / "point_cloud" / "iteration_30000" / "point_cloud.ply").exists(),
        "has_g_model": (d / "out" / "Model_G" / "point_cloud" / "iteration_60000" / "point_cloud.ply").exists(),
        "has_r_model": (d / "out" / "Model_R" / "point_cloud" / "iteration_60000" / "point_cloud.ply").exists(),
        "has_re_model": (d / "out" / "Model_RE" / "point_cloud" / "iteration_60000" / "point_cloud.ply").exists(),
        "has_nir_model": (d / "out" / "Model_NIR" / "point_cloud" / "iteration_60000" / "point_cloud.ply").exists(),
        "has_index_metrics": idx.exists(),
        "has_support_audit": (d / "summary" / "support_audit.json").exists(),
    })
summary = {
    "run_root": str(root),
    "scene_count": len(rows),
    "done_count": sum(r["done_train_eval"] for r in rows),
    "failed_count": sum(bool(r["failed_markers"]) for r in rows),
    "scenes": rows,
}
(root / "summary" / "train_eval_status_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
}

cd "$REPO"
cat > "$RUN_ROOT/batch_config.json" <<JSON
{
  "batch": "uav_multispec3d_road40_umgs_i",
  "purpose": "Full UMGS-I training/evaluation for the official Road 40m scene using accepted 94/94 COLMAP prepared output; no COLMAP rerun.",
  "repo": "$REPO",
  "python": "$PY",
  "colmap": "$COLMAP",
  "minima": "$MINIMA",
  "data_root": "$DATA",
  "scene_id": "$SCENE_ID",
  "pre_scene": "$PRE_SCENE",
  "gpu_id": $GPU_ID,
  "rgb_iter": $RGB_ITER,
  "band_iter": $BAND_ITER,
  "raw_res": $RAW_RES,
  "split_policy": "registered-camera LLFF hold-8 generated from the accepted 94/94 road 40m sparse/0 image names",
  "started_at": "$(date --iso-8601=seconds)"
}
JSON

event "RUN_ROOT $RUN_ROOT"
event "DATA $DATA"
event "SCENE $SCENE_ID"
event "PRE_SCENE $PRE_SCENE"

scene_dir="$RUN_ROOT/$SCENE_ID"
raw_root="$DATA/scenes/$SCENE_ID"
prepared="$scene_dir/prepared"
rectified="$scene_dir/rectified"
out="$scene_dir/out"
split="$scene_dir/summary/registered_llffhold8_split_v1.json"
mkdir -p "$scene_dir/logs" "$scene_dir/summary"
echo "$raw_root" > "$scene_dir/raw_root.txt"
echo "$PRE_SCENE" > "$scene_dir/preprocess_source.txt"

free="$(free_gb)"
event "CHECK $SCENE_ID free_gb=$free"
if [ "$free" -lt "$MIN_FREE_GB" ]; then
  event "STOP before $SCENE_ID free_gb=$free below MIN_FREE_GB=$MIN_FREE_GB"
  echo disk_free_below_threshold > "$scene_dir/FAILED_DISK"
  write_final_summary || true
  exit 1
fi
if [ ! -d "$raw_root" ] || [ ! -d "$PRE_SCENE/prepared/RGB/sparse/0" ] || [ ! -d "$PRE_SCENE/prepared/RGB/images" ]; then
  event "FAILED $SCENE_ID missing raw root or accepted prepared source"
  echo missing_inputs > "$scene_dir/FAILED_INPUTS"
  write_final_summary || true
  exit 1
fi

if [ ! -d "$prepared" ]; then
  cp -al "$PRE_SCENE/prepared" "$prepared"
fi

fix_manifest_source_paths "$prepared" "$raw_root" "$scene_dir/summary/prepared_manifest_source_path_fix.json" > "$scene_dir/logs/fix_manifest_source_paths.log" 2>&1

write_registered_split "$SCENE_ID" "$prepared/RGB" "$split" "$scene_dir/summary/registered_split_audit.json" > "$scene_dir/logs/registered_split.log" 2>&1
install_split "$split" "$prepared/RGB/sparse/0" > "$scene_dir/logs/install_split_rgb.log" 2>&1

rm -f "$scene_dir"/FAILED_TRAIN_BANDS_PRODUCTS_RENDER "$scene_dir"/FAILED_RENDER_RGB "$scene_dir"/FAILED_METRICS "$scene_dir"/FAILED_INDEX_EVAL "$scene_dir"/FAILED_SUPPORT_AUDIT

if [ "$RESUME_AFTER_RGB" = "1" ] && [ -f "$out/Model_RGB/point_cloud/iteration_${RGB_ITER}/point_cloud.ply" ]; then
  event "SKIP $SCENE_ID train_rgb existing Model_RGB iteration_${RGB_ITER}"
else
  if ! run_stage "$SCENE_ID" train_rgb \
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
    --protocol_split "$split" \
    --from_step 2 --to_step 2; then
    echo train_rgb_failed > "$scene_dir/FAILED_TRAIN_RGB"
    write_final_summary || true
    exit 1
  fi
fi

install_split "$split" "$prepared/RGB/sparse/0" > "$scene_dir/logs/install_split_after_rgb.log" 2>&1

if ! run_stage "$SCENE_ID" train_bands_products_render \
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
    --from_step 3 --to_step 9 \
    --auto_render; then
  echo train_bands_or_render_failed > "$scene_dir/FAILED_TRAIN_BANDS_PRODUCTS_RENDER"
  write_final_summary || true
  exit 1
fi

if ! run_stage "$SCENE_ID" render_rgb \
    "$PY" render.py -m "$out/Model_RGB" -s "$prepared/RGB" -r "$RAW_RES" --iteration "$RGB_ITER" --skip_train; then
  echo render_rgb_failed > "$scene_dir/FAILED_RENDER_RGB"
fi

if ! run_stage "$SCENE_ID" metrics \
    "$PY" metrics.py -m "$out/Model_RGB" "$out/Model_G" "$out/Model_R" "$out/Model_RE" "$out/Model_NIR" --mask_mode gt_nonzero; then
  echo metrics_failed > "$scene_dir/FAILED_METRICS"
fi

if ! run_stage "$SCENE_ID" index_eval \
    "$PY" evaluate_spectral_indices.py \
    --g_model_dir "$out/Model_G" \
    --r_model_dir "$out/Model_R" \
    --re_model_dir "$out/Model_RE" \
    --nir_model_dir "$out/Model_NIR" \
    --iteration "$BAND_ITER" \
    --indices NDVI,GNDVI,NDRE \
    --out_json "$out/index_metrics_summary.json" \
    --mask_mode gt_nonzero_intersection; then
  echo index_eval_failed > "$scene_dir/FAILED_INDEX_EVAL"
fi

if ! support_audit "$SCENE_ID" "$out" "$scene_dir/summary/support_audit.json" > "$scene_dir/logs/support_audit.log" 2>&1; then
  echo support_audit_failed > "$scene_dir/FAILED_SUPPORT_AUDIT"
fi

if compgen -G "$scene_dir/FAILED*" > /dev/null; then
  event "DONE_WITH_WARNINGS $SCENE_ID"
else
  echo done > "$scene_dir/DONE_TRAIN_EVAL"
  event "DONE $SCENE_ID train_eval"
fi
write_final_summary || true
touch "$RUN_ROOT/status/finished.txt"
event "ALL_DONE"
