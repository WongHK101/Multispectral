# Preparation and result-accounting contract

This is orchestration support for the approved UMGS/TGRS scope, not a training
launcher or a replacement scoring protocol. There is no automatic GPU resume,
SSH connection, shutdown command, or external-review PASS in this module.

## Scope

- Core measured-geometry scenes: Road/3K and GCP 5K.
- Proxy extensions: Eucalyptus, Maize-02, Cassava and Papaya.
- Six-scene paired methods: UMGS, JO and neural-color MS-Splatting.
- Core-only additions: SIG mechanism and the RGB anchor.
- Ten new pipelines: UMGS-5K, JO-5K, SIG-3K/5K and neural-MS on all six.
- Twenty-two result rows, including audited reuse candidates. SIG is not a
  rename of JO; historical JO output cannot fill a neural-MS row.

The ledger retains missing/failed rows, with blank metrics rather than zero.
It does not mix two-scene and six-scene means. Full per-point/per-image outputs,
coverage, error distributions, transform provenance and band metrics remain
mandatory scoring artifacts; the ledger contains selected summary columns only.
LiDAR threshold names reflect the approved proposal, not evidence that its ROI,
CRS, reference and sampling are already bound. Runtime protocol identity must be
resolved before any real result is accepted.

## CPU preparation

```text
python -B -m tools.measured_geometry_v1.campaign prepare --campaign_id umgs_tgrs_20260930 --output_dir NEW_DIRECTORY
```

This writes an immutable preparation plan, empty receipt index, JSON summary
and UTF-8-BOM CSV table. The output directory must not exist. Repeating the same
campaign identity produces identical plan bytes. A preparation plan cannot be
edited into training approval: authorization is a separate evidence record.

Server downloads and compilation can be performed in project-owned isolated
environments, with low CPU/I/O priority and bounded workers. Do not modify old
dirty source trees, inherit another method's environment, install system-wide,
probe CUDA kernels, or run training while another project owns the GPU.
An environment import PASS is not a recipe or kernel-runtime PASS.

## GPU authorization and staged qualification

`gpu_start_decision` is a pure final-decision helper. Its caller must verify
the actual report files and their scene/stage/recipe bindings, not supply
unchecked PASS strings. It requires a matching plan hash, explicit user notice,
bound gate reports, and a fresh resource/ownership snapshot. Snapshots older
than 30 seconds, future timestamps, unknown ownership, foreign work or pending
transfers reject the decision. Recheck immediately before launching anything.

GPU qualification itself remains staged: approved CPU contract tests permit
only the specifically authorized GPU smoke; real packet/reference and camera
checks must pass before formal evaluation and dependent new training. A missing
real-export result cannot be replaced by a CPU header-only PASS. This module
does not bootstrap or bypass that supervised qualification workflow, and all
generated command bindings remain pending until its evidence exists.

## Result receipts

A `umgs_tgrs_result_index_v1` binds campaign ID, plan hash and one entry per
available result. Each entry contains `row_id`, evidence-root-relative `path`,
`size_bytes` and `sha256`. No duplicate or out-of-scope row is accepted.

A new receipt uses schema `umgs_tgrs_result_receipt_v2` and binds:

- `campaign_id`, `row_id`, `scene`, `method`, and `track`;
- `status`: COMPLETE, PARTIAL, FAILED or BLOCKED;
- `groups`: gcp/lidar/proxy/appearance/resources for the core; the last three
  for extensions. Group states are PASS/PARTIAL/FAILED/NOT_RUN;
- finite numeric `metrics`, using the summary names in `campaign.METRICS`;
- `artifacts`: path, size and SHA records, including failure evidence;
- `reason` for every non-complete result;
- an explicit `coverage_status`, not a conclusion inferred from low RMSE.
- `ranking` for each group: frozen population SHA, total, passed coverage gates,
  disposition and exclusion reason. GCP counts formal checkpoints passing the
  point-level coverage gate, not observations or merely finite residuals.

COMPLETE additionally requires all groups to pass, every applicable summary
metric, input/recipe/scoring-protocol/checkpoint hashes and an independently
generated audit included in the evidence. RGB anchors cannot publish spectral
SAM. Scientific incomplete coverage may be an audited valid method result, but
missing measurements or missing historic resource records remain PARTIAL.

Delivery COMPLETE does not imply COMPLETE_RANKED. A complete delivery with an
incomplete GCP population is INCOMPLETE_UNRANKED; its subset residuals remain
diagnostic. Missing historical resources can leave delivery PARTIAL while an
independently verified complete geometry track remains ranking eligible.
`macro_eligibility` requires every predetermined scene for that method/metric;
it never computes a surviving-scenes primary mean or mixes the two scene scopes.

The independent audit binds campaign ID, row ID and SHA-256 of the canonical
record `{identity, groups, metrics, coverage_status, ranking}`, including PARTIAL
delivery. Legacy v1 receipts retain their old audit format and are always
UNVERIFIED_UNRANKED until explicitly re-audited into v2. Canonical serialization
is `contracts.canonical_bytes`. The collector checks this binding; it does
not perform the independent recomputation or certify the audit author. The
supervised workflow must pin the receipt index and its provenance separately.

```text
python -B -m tools.measured_geometry_v1.campaign collect --plan PLAN.json --index VERIFIED_INDEX.json --evidence_root EVIDENCE_DIRECTORY --output_dir NEW_DIRECTORY
```

No threshold selection, model ranking, metric computation or paper table
replacement happens automatically. Failed/unrun groups cannot publish numbers.
External review remains NOT_ASSERTED_BY_COLLECTOR even when all rows complete.

## Safe closeout

The user authorized shutdown after the future experiment batch, including
REVIEW_FAILED or REVIEW_TOOL_ERROR. Neither failure becomes an audit PASS.
The current CPU preparation is explicitly outside this shutdown authorization.

`closeout_decision` requires the matching campaign/plan, separately captured
GPU notification and shutdown authorization, an explicit stage and an authorized
outcome. CPU_PREPARATION cannot close the server. GPU_QUALIFICATION can close
after BLOCKED/REVIEW_FAILED/REVIEW_TOOL_ERROR without pretending training started.
EXPERIMENT_BATCH requires a real post-notification experiment start, and can
close after completion or those failures. All noncomplete outcomes need a reason.
Both GPU stages require stopped owned children, flushed logs, hash-verified
artifact inventory and off-server evidence backup. Live foreign/unknown jobs,
active transfers or unverified process ownership always prevent shutdown.
The execution operator must recheck live ownership before the real command;
this module never runs that command, kills a process or modifies another project.
No final review success is promised merely because an entire batch was run.
