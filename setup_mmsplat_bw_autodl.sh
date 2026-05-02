#!/usr/bin/env bash
set -u
LOG_ROOT=/root/autodl-tmp/runs/paper_autodl_full_20260429/mms_env_setup_$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOG_ROOT"
BASE_PY=/root/autodl-tmp/envs/spectralindexgs_bw/bin/python
ENV=/root/autodl-tmp/envs/mmsplat_bw
SRC=/root/autodl-tmp/src/MS-Splatting
{
  echo "started_at=$(date --iso-8601=seconds)"
  echo "base_python=$BASE_PY"
  echo "env=$ENV"
  echo "src=$SRC"
  "$BASE_PY" -V
  if [ ! -d "$ENV" ]; then
    "$BASE_PY" -m venv --system-site-packages "$ENV"
  fi
  "$ENV/bin/python" -m pip install -U pip setuptools wheel -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
  "$ENV/bin/python" -m pip install -e "$SRC" -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
  "$ENV/bin/python" - <<'PY'
import torch
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())
try:
    import nerfstudio
    print('nerfstudio import ok')
except Exception as exc:
    print('nerfstudio import failed', type(exc).__name__, exc)
try:
    import gsplat
    print('gsplat import ok', getattr(gsplat, '__version__', 'unknown'))
except Exception as exc:
    print('gsplat import failed', type(exc).__name__, exc)
try:
    import mmsplat
    print('mmsplat import ok')
except Exception as exc:
    print('mmsplat import failed', type(exc).__name__, exc)
PY
  echo "finished_at=$(date --iso-8601=seconds)"
  echo ok > "$LOG_ROOT/DONE"
} > "$LOG_ROOT/setup.log" 2>&1
rc=$?
if [ "$rc" -ne 0 ]; then
  echo "rc=$rc" > "$LOG_ROOT/FAILED"
fi
exit "$rc"
