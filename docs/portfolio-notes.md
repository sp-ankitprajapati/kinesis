# Kinesis portfolio and interview notes

## Compact CV entry

**Kinesis - Simulated Mobile-Robot Path Tracking | Python, PyTorch, Gymnasium**

- Built a validated 2D tracking simulator and benchmarked Pure Pursuit against
  Stable-Baselines3 SAC using three training seeds, 24 held-out routes and 26 checks.
- Ran 384 nominal/stress evaluations; documented completion, tracking error,
  command smoothness and CPU inference cost, including failures under command delay.

These bullets describe the implemented study. They do not claim SAC was implemented
from scratch or that a physical robot, motorcycle or embedded system was deployed.
Use them once you can explain the code and measured findings.

## A 30-second project explanation

Kinesis compares conventional and learned path tracking for a simulated mobile robot.
I used a bounded planar motion model with command-response lag, then evaluated Pure
Pursuit and the Stable-Baselines3 implementation of SAC under the same observations,
action limits and time budgets. Three independently trained SAC models completed all
24 nominal test routes. Pure Pursuit had lower tracking error, smoother commands
and lower inference cost. A 200 ms delay exposed seven SAC failures across 72
attempts, which showed why nominal completion alone is not enough.

## A longer technical explanation

The robot state contains x, y, heading, linear speed and turn rate. Commands are
limited to 0-1.2 m/s and -2 to 2 rad/s, with first-order response. Five integration
substeps support a 0.1 s control update.

Both controllers receive the same 25-value observation, including local reference
previews, error, actual speeds, previous/pending commands, the next required gate
and remaining time. Ordered gates prevent skipping the route from counting as
successful completion.

Four configurations per method were compared on 12 validation cases. After
selection, SAC was retrained from scratch with seeds 7, 17 and 27 for 80,000 steps
each. All seeds were tested; a plot example was chosen on validation beforehand.

The test set contains 24 new scenarios, including six double-sine routes not used
for training or validation. Noise, delay and slower response were tested separately.
Every attempt, including failures, is retained in the data.

## Main measured findings

- Nominal completion: 24/24 for Pure Pursuit and for every SAC seed.
- Nominal mean episode tracking RMSE: 0.043 m for Pure Pursuit; 0.113 +/- 0.066 m
  across the three SAC seeds.
- Nominal angular-command slew RMS: 0.59 rad/s^2 for Pure Pursuit; 5.91 +/- 2.18
  rad/s^2 for SAC.
- Mean episode median CPU inference: 0.032 ms for Pure Pursuit; 0.390 +/- 0.058 ms
  for SAC. This is not an embedded timing guarantee.
- Delay completion: 24/24 for Pure Pursuit; 65/72 across the three SAC seeds.
- The conventional baseline is the stronger default for this simulator and budget.

The +/- values are sample standard deviations across three training seeds, not
confidence intervals. They should not be presented as a significance test.

## Questions to prepare for

1. Derive Pure Pursuit curvature from a target in robot coordinates.
2. Explain actuator lag and why integration substeps need convergence checks.
3. Describe what the actor, two critics, replay buffer and entropy coefficient do.
4. Explain why stopping or cutting a route cannot count as success here.
5. Explain what delay history must be observed and what was absent during training.
6. Distinguish validation selection from held-out testing.
7. Discuss why reward can improve while tracking error remains worse.
8. Explain why three seeds and six examples per route family are limited evidence.
9. State which parts were implemented here and which came from libraries.
10. Explain what would be required before any physical deployment.

## First extension to investigate

Training-time randomization of delays and response constants is a reasonable next
experiment. The existing queue observation is necessary but did not by itself teach
delay compensation: nominal training never showed a nonzero delay. Create a new
experiment directory and a new validation protocol before changing the model.
Preserve the current test results.
