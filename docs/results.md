# Kinesis experimental results

Ankit Prajapati | Simulated mobile-robot path tracking | 28 October 2026

## Scope

A bounded planar simulator compares Pure Pursuit with Stable-Baselines3 SAC. This is a simulation study with no physical robot or embedded deployment.

## Protocol

- Training source checkpoint: 0df854a866d30330ba32dad38c29c9d8410ea1cc.
- Python 3.12.11; CPU PyTorch 2.14.1; Stable-Baselines3 2.9.2; Gymnasium 1.4.0.
- Cloud hardware: 2 vCPUs, approximately 8 GiB RAM; AMD EPYC 7763 host CPU.
- 24 training, 12 validation and 24 test scenarios; route hashes check exact split overlap.
- Training/validation use straight, sine and arc routes. Double-sine geometry is held out.
- Four candidates per method; SAC pilots use 12,000 steps each, seed 101.
- Three final SAC models use 80,000 steps each, independent seeds 7, 17 and 27.
- Candidate counts and validation scenarios match; computational tuning costs do not.
- Selected Pure Pursuit: {'lookahead': 0.8, 'cruise_speed': 1.0}.
- Selected SAC: {'learning_rate': 0.0003, 'ent_coef': 'auto_0.1'}.
- All controllers share observations, action limits and route-specific time allowances.
- Ordered 0.5 m gates with 0.35 m tolerance prevent skipped routes from counting as success.

## Nominal held-out results

All models use the same 24 test scenarios. SAC values are mean +/- sample standard deviation across three training seeds, not confidence intervals.

| Metric | Pure Pursuit | SAC across three seeds |
| --- | ---: | ---: |
| Success (%) | 100.0 | 100.0 +/- 0.0 |
| Ordered gate completion (%) | 100.0 | 100.0 +/- 0.0 |
| Tracking RMSE (m) | 0.043 | 0.113 +/- 0.066 |
| Maximum error, episode mean (m) | 0.184 | 0.278 +/- 0.136 |
| Elapsed time, all attempts (s) | 12.65 | 12.24 +/- 1.02 |
| Linear command slew RMS (m/s^2) | 0.88 | 1.23 +/- 0.10 |
| Angular command slew RMS (rad/s^2) | 0.59 | 5.91 +/- 2.18 |
| Mean episode median inference (ms) | 0.032 | 0.390 +/- 0.058 |
| Mean episode p95 inference (ms) | 0.058 | 0.595 +/- 0.155 |

Errors and elapsed times include early failures. Low values on short failed attempts do not demonstrate better control.

| SAC model | Successes / 24 | RMSE (m) | Paired successes | PP time (s) | SAC time (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| SAC seed 7 | 24 | 0.188 | 24 | 12.65 | 13.33 |
| SAC seed 17 | 24 | 0.090 | 24 | 12.65 | 11.33 |
| SAC seed 27 | 24 | 0.062 | 24 | 12.65 | 12.05 |

## Stress tests

Nominally trained models face one disturbance at a time. Each condition uses the same routes and initial states for every controller.

| Condition | PP success (%) | SAC success (%) | PP RMSE (m) | SAC RMSE (m) |
| --- | ---: | ---: | ---: | ---: |
| nominal | 100.0 | 100.0 +/- 0.0 | 0.043 | 0.113 +/- 0.066 |
| noise | 100.0 | 100.0 +/- 0.0 | 0.046 | 0.115 +/- 0.069 |
| delay_200ms | 100.0 | 90.3 +/- 8.7 | 0.059 | 0.175 +/- 0.085 |
| slow_response | 100.0 | 100.0 +/- 0.0 | 0.054 | 0.149 +/- 0.120 |

Noise: 0.03 m Gaussian position noise per coordinate and 3 degree heading noise. Delay: two 0.1 s control steps with observed command queue. Slow response: both time constants multiplied by 2.5. Noise does not simulate full localization.

## Held-out route family

| Nominal route family | PP success (%) | SAC success (%) | PP RMSE (m) | SAC RMSE (m) |
| --- | ---: | ---: | ---: | ---: |
| straight | 100.0 | 100.0 +/- 0.0 | 0.046 | 0.110 +/- 0.068 |
| sine | 100.0 | 100.0 +/- 0.0 | 0.053 | 0.117 +/- 0.064 |
| arc | 100.0 | 100.0 +/- 0.0 | 0.036 | 0.112 +/- 0.068 |
| double_sine | 100.0 | 100.0 +/- 0.0 | 0.036 | 0.114 +/- 0.066 |

Each family has six test scenarios, a small sample.

## Failure records

| Controller | Condition | Failure reasons |
| --- | --- | --- |
| Pure Pursuit | nominal | none |
| SAC seed 7 | nominal | none |
| SAC seed 17 | nominal | none |
| SAC seed 27 | nominal | none |
| Pure Pursuit | noise | none |
| SAC seed 7 | noise | none |
| SAC seed 17 | noise | none |
| SAC seed 27 | noise | none |
| Pure Pursuit | delay_200ms | none |
| SAC seed 7 | delay_200ms | none |
| SAC seed 17 | delay_200ms | missed_gate: 4 |
| SAC seed 27 | delay_200ms | missed_gate: 2, time_limit: 1 |
| Pure Pursuit | slow_response | none |
| SAC seed 7 | slow_response | none |
| SAC seed 17 | slow_response | none |
| SAC seed 27 | slow_response | none |

## Resources and interpretation

Final training used 11.13 minutes summed over three learning calls. This excludes pilots, evaluation, setup and documentation. No GPU was used.

- Three seeds give limited evidence about variability. No significance or universal superiority claim is made.
- Reference previews and localization are simplified; the scenario distribution is finite.
- Stresses are tested separately. Combined disturbances were not evaluated.
- Episode latency percentiles are averaged, not pooled over all calls.
- Latency is CPU wall time around prediction, not an embedded deadline or worst-case guarantee.
- Wheel dynamics, slip, obstacles, collisions, physical sensing and safety validation are outside scope.
- Selection was frozen before testing; test outcomes do not feed tuning.

## Evidence

- artifacts/protocol.json and manifests/: budgets, versions, source checkpoint, cases and route hashes.
- baseline_selection.json and sac_selection.json: all validation candidates.
- training/seed_*/run.json and monitor.csv: each final training record.
- test_episodes.csv and test_results.json: all test attempts, including failures.
- figures/: fixed examples and training-return plots.
- docs/learning-report.md: concepts, code walkthrough and interview revision.

SAC: https://stable-baselines3.readthedocs.io/en/master/modules/sac.html
Pure Pursuit: https://publications.ri.cmu.edu/implementation-of-the-pure-pursuit-path-tracking-algorithm
