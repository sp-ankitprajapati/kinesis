# Kinesis

**Simulated mobile-robot path tracking with Pure Pursuit and Soft Actor-Critic.**

A portfolio and learning project by Ankit Prajapati. The project builds a small
validated 2D simulator, integrates a conventional controller and Stable-Baselines3
SAC, and evaluates both on held-out routes under shared constraints.

## Current status

The simulator, Pure Pursuit baseline, correctness tests, experiment pipeline and
learning report are implemented. SAC experiments and final conclusions are in
progress. No improvement over the conventional baseline is assumed.

## Start in a GitHub Codespace

The development container creates a Python 3.12 virtual environment and installs
the recorded dependencies, including CPU-only PyTorch.

If setting up an existing Linux x86_64 checkout manually:

~~~bash
bash .devcontainer/setup.sh
source .venv/bin/activate
python -m pytest
~~~

The first setup downloads PyTorch. A GPU is not required.

## Run experiments

~~~bash
# Four conventional configurations, selected on validation routes
python -m kinesis.experiments baseline

# Measure training speed before committing to a budget
python -m kinesis.experiments profile --steps 5000

# Independent reproduction; preserves the existing experiment artifacts
python -m kinesis.experiments pipeline \
  --pilot-steps 12000 --final-steps 80000 \
  --output-dir artifacts/reproduction
~~~

A full pipeline runs four SAC pilot configurations, selects on validation, trains
three new random seeds, freezes the controller selection and then evaluates test
scenarios. It can take substantially longer than the tests. Execution is CPU-only
with one PyTorch thread. A completed output directory is not overwritten.

## What is actually implemented here?

- Bounded linear-speed and turn-rate commands.
- Planar kinematics with first-order motion response and integration substeps.
- Local route projection and ordered waypoint gates.
- A Gymnasium environment with observed command-delay queues.
- Pure Pursuit and a wrapper around the library's trained SAC policy.
- Disjoint training/validation/test manifests and a held-out route family.
- Paired noise, delay and changed-response stress tests.
- Completion, tracking, command-slew and measured CPU inference metrics.

**SAC itself is provided by Stable-Baselines3/PyTorch.** It is not implemented from
scratch. This is not a physical robot, motorcycle model, ROS2 system, embedded
deployment or real-world safety validation.

## Read and learn

Start with [the learning report](docs/learning-report.md). It explains the minimum
concepts, equations, code organization, reward safeguards, experimental protocol
and interview questions.

Measurements are saved under artifacts/. Large intermediate traces and checkpoints
remain in the Codespace rather than normal Git history. Final results and their
limitations will be summarized here after evaluation.
