#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/root/autodl-tmp/Multispectral}
PY=${PY:-/root/autodl-tmp/envs/spectralindexgs_bw/bin/python}
COLMAP=${COLMAP:-/root/autodl-tmp/opt/colmap-cuda-3.9.1/bin/colmap}
MINIMA=${MINIMA:-/root/autodl-tmp/src/MINIMA}
DATA=${DATA:-/root/autodl-tmp/datasets/UAV-MultiSpec3D/UAV-MultiSpec3D_Benchmark_16scenes_20260602}
PRE_A=${PRE_A:-/root/autodl-tmp/runs/uav_multispec3d_active17_remaining_colmap_20260602_042451}
PRE_B=${PRE_B:-/root/autodl-tmp/runs/uav_multispec3d_active17_wogan_resume_colmap_20260602_092926}
PRE_EUC=${PRE_EUC:-/root/autodl-tmp/runs/uav_multispec3d_pruned_eucalyptus_gpu_colmap_20260602_021411}
RUN_ROOT=${RUN_ROOT:-/root/autodl-tmp/runs/uav_multispec3d_benchmark16_missing_eucalyptus_umgs_i_$(date +%Y%m%d_%H%M%S)}
GPU_ID=${GPU_ID:-0}
RGB_ITER=${RGB_ITER:-30000}
BAND_ITER=${BAND_ITER:-60000}
RAW_RES=${RAW_RES:-8}
MIN_FREE_GB=${MIN_FREE_GB:-200}

SCENES=(
  eucalyptus_01_20260526_1053_pruned
  eucalyptus_02_20260526_1108_pruned
)
BANDS=(G R RE NIR)

mkdir -p "$RUN_ROOT/status" "$RUN_ROOT/logs" "$RUN_ROOT/summary"

cat > "$RUN_ROOT/batch_config.json" <<JSON
{
  "batch": "uav_multispec3d_benchmark16_missing_eucalyptus_umgs_i",
  "purpose": "Full UMGS-I training/evaluation for the two pruned eucalyptus benchmark16 scenes that already passed 100% RGB COLMAP registration; reuses prepared COLMAP products and does not rerun COLMAP",
  "repo": "$REPO",
  "python": "$PY",
  "colmap": "$COLMAP",
  "minima": "$MINIMA",
  "data_root": "$DATA",
  "preprocess_root_a": "$PRE_A",
  "preprocess_root_b": "$PRE_B",
  "preprocess_root_eucalyptus": "$PRE_EUC",
  "gpu_id": $GPU_ID,
  "rgb_iter": $RGB_ITER,
  "band_iter": $BAND_ITER,
  "raw_res": $RAW_RES,
  "scene_count": ${#SCENES[@]},
  "method": "UMGS-I support-locked independent band transfer",
  "split_policy": "registered-camera LLFF hold-8 generated from each scene's canonical sparse/0 image names",
  "started_at": "$(date --iso-8601=seconds)"
}
JSON

event() {
  echo "[$(date --iso-8601=seconds)] $*" | tee -a "$RUN_ROOT/status/events.log"
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

free_gb() {
  df -BG /root/autodl-tmp | awk 'NR==2 {gsub(/G/,"",$4); print $4}'
}

pre_scene_for() {
  local scene_id="$1"
  case "$scene_id" in
    eucalyptus_01_20260526_1053_pruned|eucalyptus_02_20260526_1108_pruned)
      echo "$PRE_EUC/$scene_id"
      ;;
    wogan_mandarin_03_20260528_1441|wogan_mandarin_04_20260528_1558|wogan_mandarin_05_20260528_1621)
      echo "$PRE_B/$scene_id"
      ;;
    *)
      echo "$PRE_A/$scene_id"
      ;;
  esac
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
    "note": "Generated from canonical sparse/0 after 100% registration check; matches the registered-camera LLFF hold-8 protocol used in prior AutoDL runs."
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

write_final_summary() {
  "$PY" - "$RUN_ROOT" <<'PY'
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for d in sorted([p for p in root.iterdir() if p.is_dir() and p.name not in {"logs", "status", "summary"}]):
    failed = sorted(p.name for p in d.glob("FAILED*"))
    done = (d / "DONE_TRAIN_EVAL").exists()
    idx = d / "out" / "index_metrics_summary.json"
    rows.append({
        "scene_id": d.name,
        "done_train_eval": done,
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
print(json.dumps({"done": summary["done_count"], "failed": summary["failed_count"], "scene_count": summary["scene_count"]}, indent=2))
PY
}

cd "$REPO"
event "RUN_ROOT $RUN_ROOT"
event "DATA $DATA"
event "SCENES ${SCENES[*]}"

for scene_id in "${SCENES[@]}"; do
  scene_dir="$RUN_ROOT/$scene_id"
  raw_root="$DATA/scenes/$scene_id"
  pre_scene="$(pre_scene_for "$scene_id")"
  prepared="$scene_dir/prepared"
  rectified="$scene_dir/rectified"
  out="$scene_dir/out"
  split="$scene_dir/summary/registered_llffhold8_split_v1.json"
  mkdir -p "$scene_dir/logs" "$scene_dir/summary"
  echo "$raw_root" > "$scene_dir/raw_root.txt"
  echo "$pre_scene" > "$scene_dir/preprocess_source.txt"

  free="$(free_gb)"
  event "CHECK $scene_id free_gb=$free"
  if [ "$free" -lt "$MIN_FREE_GB" ]; then
    event "STOP before $scene_id free_gb=$free below MIN_FREE_GB=$MIN_FREE_GB"
    echo disk_free_below_threshold > "$scene_dir/FAILED_DISK"
    break
  fi
  if [ ! -d "$raw_root" ] || [ ! -d "$pre_scene/prepared/RGB/sparse/0" ] || [ ! -d "$pre_scene/prepared/RGB/images" ]; then
    event "SKIP $scene_id missing raw or canonical prepared RGB from $pre_scene"
    echo missing_inputs > "$scene_dir/FAILED_INPUTS"
    continue
  fi

  if [ ! -d "$prepared" ]; then cp -al "$pre_scene/prepared" "$prepared"; fi
  write_registered_split "$scene_id" "$prepared/RGB" "$split" "$scene_dir/summary/registered_split_audit.json" > "$scene_dir/logs/registered_split.log" 2>&1 || {
    echo split_generation_failed > "$scene_dir/FAILED_SPLIT"
    write_final_summary || true
    continue
  }
  install_split "$split" "$prepared/RGB/sparse/0" > "$scene_dir/logs/install_split_rgb.log" 2>&1 || {
    echo split_install_failed > "$scene_dir/FAILED_SPLIT"
    write_final_summary || true
    continue
  }

  if ! run_stage "$scene_id" train_rgb \
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
    continue
  fi

  install_split "$split" "$prepared/RGB/sparse/0" > "$scene_dir/logs/install_split_after_rgb.log" 2>&1 || {
    echo split_reinstall_failed > "$scene_dir/FAILED_SPLIT_AFTER_RGB"
    write_final_summary || true
    continue
  }

  if ! run_stage "$scene_id" train_bands_products_render \
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
    continue
  fi

  if ! run_stage "$scene_id" render_rgb \
      "$PY" render.py -m "$out/Model_RGB" -s "$prepared/RGB" -r "$RAW_RES" --iteration "$RGB_ITER" --skip_train; then
    echo render_rgb_failed > "$scene_dir/FAILED_RENDER_RGB"
  fi

  if ! run_stage "$scene_id" metrics \
      "$PY" metrics.py -m "$out/Model_RGB" "$out/Model_G" "$out/Model_R" "$out/Model_RE" "$out/Model_NIR" --mask_mode gt_nonzero; then
    echo metrics_failed > "$scene_dir/FAILED_METRICS"
  fi

  if ! run_stage "$scene_id" index_eval \
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

  if ! support_audit "$scene_id" "$out" "$scene_dir/summary/support_audit.json" > "$scene_dir/logs/support_audit.log" 2>&1; then
    echo support_audit_failed > "$scene_dir/FAILED_SUPPORT_AUDIT"
  fi

  if compgen -G "$scene_dir/FAILED*" > /dev/null; then
    event "DONE_WITH_WARNINGS $scene_id"
  else
    echo done > "$scene_dir/DONE_TRAIN_EVAL"
    event "DONE $scene_id train_eval"
  fi
  write_final_summary || true
done

write_final_summary || true
date --iso-8601=seconds > "$RUN_ROOT/status/finished.txt"
event "ALL_DONE"
