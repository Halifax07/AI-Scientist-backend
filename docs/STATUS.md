# Development status

Updated: 2026-08-10

## Completed

- [x] Normalized monorepo layout and Git repository.
- [x] Python 3.12 project environment and reproducible `uv.lock`.
- [x] Research domain models and JSON Research Ledger.
- [x] Autonomous workflow through human experiment approval and post-result stages.
- [x] Mock runtime that never fabricates verified evidence or metrics.
- [x] Qwen/AgentScope structured reasoning adapter.
- [x] Hypothesis Elo tournament and evolution request types.
- [x] Progressive experiment tree with information-gain-per-cost priority.
- [x] Auditable random/k-center support selection and manifest digest.
- [x] PatchCore, AnomalyDINO and SubspaceAD command adapters.
- [x] FastAPI endpoints and React research-workbench skeleton.
- [x] Official third-party source checkout and immutable version manifest.
- [x] Backend tests, lint, frontend type-check and production build.
- [x] Live arXiv/Crossref search with bibliographic-only verification status.
- [x] arXiv PDF extraction and page-anchored Qwen claim verification boundary.
- [x] MVTec manifest, mask audit, and train/test content-leak detection.
- [x] Cached DINOv2 train/good profiler and CUDA-aware vision environment.
- [x] Strict-K/pool-compression planning and detector-visible immutable views.
- [x] Single-category AnomalyDINO wrapper without editing pinned source.
- [x] Safe subprocess executor and PatchCore/AnomalyDINO/SubspaceAD parsers.
- [x] Paired bootstrap confidence intervals and paired sign permutation tests.
- [x] Evidence and run observability in the React workbench.
- [x] Bounded hypothesis-revision loop with parent/child provenance and history retention.
- [x] CUDA DINOv2-S feature extraction smoke test on the local RTX 4060.
- [x] End-to-end AnomalyDINO synthetic smoke run, including Windows Unicode paths and case-insensitive TIFF handling in the project-side wrapper.
- [x] Full MVTec AD audit: 15 categories, 6,612 files, 3,629 train images, 1,725 test images, 1,258 masks, no audit issues.
- [x] Persistent adaptive campaign/round/node/run model with information-gain-per-cost ordering.
- [x] Server-validated Qwen feedback actions and result-driven next-round construction.
- [x] Direction-B React experiment console with dataset audit, loop rail, round summaries, efficiency, execution controls, and ledger trace.
- [x] Three-round, 12-run real MVTec campaign on the local RTX 4060.
- [x] Two live Qwen replanning decisions over real results: bottle → cable/capsule → transistor plus validator-filled carpet boundary test.
- [x] Formal paired bootstrap/sign-permutation analysis over six registered pairs.
- [x] Human-guidance gates before every real run and before each evidence-driven research-cycle restart.
- [x] Qwen guidance interpretation with server-side queued-Run validation and immutable Research Ledger records.
- [x] Campaign history rollover so a second research cycle preserves all earlier runs and negative findings.
- [x] Physical frontend/backend separation: `frontend/` and `backend/`, with independent startup and build paths.

## Later scaling and research work

- [ ] Use the new human-guidance gate to decide whether to authorize research cycle 2 after the inconclusive primary finding.
- [ ] k-medoids and DPP production selectors.
- [ ] Docker/Celery multi-worker GPU scheduler (local executor is implemented).
- [ ] MLflow/PostgreSQL/MinIO persistence migration.
- [ ] Mixed-effect detector × strategy analysis (paired tests are implemented).
- [ ] Qwen-VL heatmap review.
- [ ] Competition PDF report generation.

## Known integration notes

- Google Co-Scientist does not provide public full source code; only its paper and public design are used.
- FastRef's official repository is empty at this sync point; its adapter remains disabled.
- PyTorch CUDA 12.8 is locked from the official PyTorch wheel index; DINO weights are cached only when profiling starts.
- MVTec AD is present at `data/mvtec_anomaly_detection`; its content digest is `ed7f275e1c0035f7ff40063bc7fd190ee742f106955c1280a561501d6dc7ebc6`.
- The configured AgentScope runtime is active. API secrets remain environment-only and are never written to project JSON or frontend code.
- Synthetic smoke run `run_ab29f3f191c2` completed on CUDA. Its deliberately easy synthetic metrics are integration diagnostics only and are excluded from scientific findings.

## Validation snapshot

- Backend: 26 tests passed.
- Static analysis: Ruff passed for `src`, `tests`, and `scripts`.
- Frontend: TypeScript and Vite production build passed.
- Dependency lock: 195 packages resolved with `uv lock --check`.
- Detector: real MVTec bottle runs completed audited-view loading, GPU feature extraction, result parsing, and immutable execution recording.
- No-dataset ideation UI and real Qwen acceptance flow reached hypothesis review without starting experiments.
- Real campaign `campaign_8b5dd742937d` / project `project_a9dde7c91426` completed 12 verified runs across bottle, cable, capsule, transistor and carpet.
- The campaign reached 6/6 registered pairs and stopped at the preregistered three-round boundary, while avoiding 388 of 400 single-detector candidate runs.
- The cumulative k-center minus random Image AUROC effect is `-0.0053774`; bootstrap 95% CI `[-0.0173770, 0.0022420]`; paired sign-permutation `p=0.75`.
- The primary claim is therefore `inconclusive`, not supported. All negative, zero and metric-conflict results remain in the ledger.
- Qwen feedback and the deterministic validator jointly changed actual execution: the validator rejected a duplicate bottle proposal and filled the last round with an admissible carpet/K=1 boundary cell.
- Visual browser QA confirmed the loop rail, round cards, Qwen rationale, run queue and ledger render without application console errors.

## Recommended next implementation order

1. Human decision: freeze the current three-round demo or authorize research cycle 2 with a narrowed detector-dependent hypothesis.
2. Add Qwen-VL heatmap review and mixed-effect analysis using the now-available cross-category failure cases.
3. Add durable multi-worker tracking and competition-report generation after the pilot schema is stable.
