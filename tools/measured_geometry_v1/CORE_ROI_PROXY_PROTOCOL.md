# Core Reference-Visible ROI Proxy v1

Protocol ID: `umgs_core_reference_first_hit_roi_proxy_v1`.
Introduced after the 2026-10-05 5K global-bbox failure; not a retrospective
claim that the original full-scene protocol passed. Old outputs stay immutable.

## Scope and References

Only Road/3K (12 held-out targets) and 5K (13), with five fixed method labels:
UMGS, JO, neural-color MS-Splatting, SIG mechanism, and the shared RGB anchor.
RGB/UMGS geometry is one verified physical model, not two trained assets.
Existing four proxy scenes keep their original full-frame domain. A mixed
six-scene primary mean is prohibited; a new two-core macro requires both scenes
and complete per-target values for the particular metric/mask being aggregated.

Use each scene's already frozen LiDAR XY ROI, control-only source Sim(3), and
EPSG:4545 to EPSG:32649 conversion. No new fit, ICP, vertical clipping, GCP
change, checkpoint change, training, or deletion of Gaussian primitives.
LiDAR scoring remains unchanged. The ROI transform locates the evaluation
region; it does NOT convert proxy RMSE from source-model units to metres.

## Reference Qualification

Bind original mesh/source/split/leakage audit/camera/reference hashes and replay
the original engineering qualification. In this NEW protocol only, the global
mesh/sparse bbox ratio is a range diagnostic, not an accuracy or eligibility
gate. Missing/nonfinite bbox evidence remains an error. Never regenerate
thresholds from 5K with `make_numeric_qualification_gates`.

All existing finite-geometry, topology and full-frame render gates remain.
Also explicitly check positive reference depths, the existing p98/p02 <= 200
rule, and actual full-frame usable-target fraction using the frozen threshold
(0.2), not the old summary field hardcoded at 0.05. Each frozen target must
have nonempty reference-visible ROI support. Report holes/subregions without
inventing a percentage threshold from observed results. Eligibility means
engineering proxy scorable ON THE REPORTED SUPPORT, not complete/accurate ROI
geometry. Surface agreement with independent LiDAR is diagnostic evidence,
not a reason to select a better reconstruction or discard difficult regions.

## Camera and Masks

Keep the full mesh's original first-hit render, including outside-ROI occluders.
Use the exact camera that generated the reference: Road's frozen canonical
fingerprint, or an explicitly bound COLMAP/reference camera for 5K. The MVS
pixel sample is `(x+0.5,y+0.5)`. Preserve stored float32 transforms by actual
matrix inversion; do not orthogonalize them or substitute a same-size camera.
Method export must match this reference camera/grid and source frame. An R8
707x512 native packet cannot be resized into a 1200x869 proxy packet.

Let V be finite, positive full-mesh reference first hits, and R be those hits
strictly inside the frozen XY ROI. Compute the old five-method masks inside R:
numeric-valid common primary C; opacity >= 0.5 intersection O; finite
nonnegative variance intersection Q. Never test a method's predicted world
point against the ROI to remove a high-error pixel. Reference ROI masks do not
depend on any method depth or score.

On each FINAL common mask, compute the existing reference high-gradient domain
once and use it for all methods. Do not threshold full-frame gradients first
and crop afterward. Existing six metric formulas, numeric rules and null
handling are unchanged. Equal-target scene aggregation requires every frozen
target for each metric; no successful-only primary mean.

## Reporting and Staging

Keep original `support_coverage = |mask|/(H*W)` and additionally report |mask|/|R|,
missing pixels, |R|/(H*W), |R|/|V|, and 4x4 reference-ROI subregion denominators
and common support. An unobserved reference subregion has null coverage, not
zero method accuracy. Reference eligibility and metric completeness are
separate states. A low method score never triggers tuning or reconstruction.

`proxy_roi.py` creates immutable reference-only masks and camera records.
`proxy_roi_score.py` accepts a hash-bound packet index, verifies real producer
associations and original M1/A gates, then invokes the existing frozen metric
implementation. The fresh 5K export-manifest interface is reserved for a
subsequent matching-camera GPU export; declaring the interface is not proof
that such packets already exist or were GPU-validated.

This phase is CPU preparation. Existing models and metrics stay unchanged;
fresh GPU export requires the user's resource authorization, not just reviewer
approval. No training is required by this repair.
