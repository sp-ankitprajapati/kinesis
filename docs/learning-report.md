# Kinesis learning report

Author: Ankit Prajapati
Project: Deep reinforcement learning for simulated mobile-robot path tracking

## Status and evidence

- Implemented: Python development container, bounded planar motion, reference geometry,
  Gymnasium environment, Pure Pursuit controller, correctness checks and evaluation tools.
- Measured setup: Python 3.12.11; CPU-only PyTorch 2.14.1; Stable-Baselines3 2.9.2;
  Gymnasium 1.4.0. Package versions are recorded in the dependency files.
- Pending: training results, held-out evaluation and portfolio conclusions.
- No physical robot, motorcycle dynamics, embedded deployment, ROS2 integration or
  real-world safety validation is included in this version.

## 1. Development environment

The repository lives in a GitHub Codespace. VS Code is its editor; computation runs
in the cloud. A virtual environment separates project packages from system packages.
The development container specifies the base system; dependency files record Python
packages. These improve repeatability but do not guarantee identical numerical
results across hardware, operating systems or future package-index availability.

An inherited Yarn package source had a missing signing key. That source is unused
by Kinesis. The project Dockerfile disables it before installing container features;
package signature verification remains enabled. Git LFS is included because the
repository's existing Git hooks invoke it. SSH access supports the explicitly
authorized remote development workflow.

## 2. Minimum concepts

A robot state describes its current situation: x and y position, heading, linear
speed and angular speed. A command requests forward speed and turning speed.
A controller chooses the next command from observations. The simulator advances
the state by a short time interval and returns the next observation.

A reference route is a sequence of points with accumulated distance along it.
Cross-track error measures displacement sideways from the local route segment.
Heading error measures the angle between the robot's heading and route direction.

All distances use metres, times use seconds and angles use radians.
One full turn is 2*pi radians. Heading is wrapped into [-pi, pi).

## 3. Motion model

The state is (x, y, theta, v, omega). Its planar kinematics are:

    dx/dt = v*cos(theta)
    dy/dt = v*sin(theta)
    dtheta/dt = omega

Command limits are 0 <= v_command <= 1.2 m/s and
-2 <= omega_command <= 2 rad/s. Wheel speeds are not modelled independently.

The robot does not respond instantly:

    dv/dt = (v_command - v)/tau_v
    domega/dt = (omega_command - omega)/tau_omega

Nominal response constants are 0.25 s and 0.18 s. Control updates occur every
0.1 s. Each update uses five integration substeps. First-order speed response is
integrated analytically; the pose update uses a constant-twist arc with interval
mean speeds per substep. This is exact for constant twist and approximate for a
changing twist. A convergence test checks the latter approximation.

This abstraction omits wheel slip, actuator torque, friction, obstacles, collision
geometry, terrain, mass, battery effects and physical sensing/localization systems.
It is a useful control experiment, not a high-fidelity robot digital twin.

## 4. Code map

- kinesis/dynamics.py: state, command limits, angle wrapping and motion integration.
- kinesis/routes.py: local path projection and fixed scenario manifests.
- kinesis/env.py: observations, delay state, reward, success and failure.
- kinesis/controllers.py: Pure Pursuit and deterministic SAC inference wrapper.
- kinesis/evaluation.py: paired scenarios, metrics and exported figures.
- kinesis/experiments.py: validation-only selection, training and final evaluation.
- tests/: independent analytic checks and important environment/evaluation boundaries.

## 5. Shared observations and fair comparison

Both controllers receive exactly the same observation vector and issue the same
two normalized actions in [-1, 1]. The first maps to forward speed in [0, 1.2];
the second maps to turn rate in [-2, 2]. A first action value of -1 means stop.

The observation contains:
- Four reference points 0.4, 0.8, 1.4 and 2.2 m ahead, in the robot's body frame.
- Signed cross-track error and sine/cosine of heading error.
- Current linear and angular speed.
- Previous requested action.
- Two slots for pending delayed actions, with stop padding.
- Remaining route distance, time remaining and active delay length.
- The next required route gate in body coordinates and remaining gate fraction.

The sine/cosine representation avoids an artificial jump at the +/-pi boundary.
Scaling keeps typical values manageable. No running observation normalization or
test-derived scaling is fitted.

With a two-step command delay, the effect of an action depends on the commands
already waiting. Current pose and speed alone cannot describe that mechanism.
The observation includes the queue and delay length. Reference previews remain
finite: this is a local tracking task with a known route, not global navigation
with complete knowledge of arbitrary future geometry.

During noise stress tests only measured position and heading are perturbed.
Reference geometry remains known. This is simpler than a full localization system.

## 6. Pure Pursuit

Choose a reference point ahead. Express that target in the robot frame as (x, y).
The curvature of a circular arc toward it is:

    curvature = 2*y/(x*x + y*y)
    omega_command = v_command*curvature

The implementation reduces speed for poor alignment, turn-rate limits and approach
to the endpoint. Lookahead and cruise speed are selected on validation scenarios.
It consumes observations rather than accessing the environment's internal state.

## 7. Reinforcement learning and SAC

An agent repeatedly observes, acts and receives a reward. Its policy is the rule
that maps observations to actions. Training changes the policy using experience.

SAC learns a continuous-action policy, two value estimators and an entropy
coefficient. Entropy encourages exploration while reward encourages useful motion.
Experience is stored in a replay buffer and sampled for training updates.

Kinesis uses Stable-Baselines3's SAC implementation and PyTorch. It implements the
environment, controller integration, experiment pipeline and evaluation; the SAC
algorithm is not implemented from scratch. The policy uses two hidden layers of
64 units each, CPU execution and one PyTorch thread.

## 8. Reward and completion safeguards

Per-step reward:

    5*(signed route progress)
    - 0.8*(cross-track error)^2
    - 0.1*(1 - cos(heading error))
    - 0.02*sum((action - previous action)^2)
    - 0.01
    + 2*(new ordered gates passed)

Successful completion adds 10; failure subtracts 10.

Stopping receives a negative running reward. Signed progress prevents repeated
forward/backward motion from producing net progress reward. Local projection limits
route-index jumps at crossings.

Route gates occur every 0.5 m and must be reached in order within 0.35 m. Passing
more than 0.5 m beyond an unreached gate fails the episode. Success requires all
gates, progress within 0.15 m of the route end, and endpoint distance <= 0.25 m.
Thus merely reaching the endpoint after cutting the route does not count.

A 1.5 m local route-distance limit also ends an episode. These are simulator
evaluation rules, not physical safety guarantees. Gate tolerance allows some
corner-cutting within 0.35 m; tracking-error metrics quantify that remaining error.

## 9. Experimental protocol

Fixed manifests contain 24 training, 12 validation and 24 test scenarios. Route
coordinates have hashes and are checked for split overlap. All training and
validation routes are straight, sine or arc paths. Test scenarios include those
families with new parameters and initial conditions, plus held-out double-sine
geometry. Generalization within known families and to the held-out family must be
reported separately.

Each method has four candidate configurations and is evaluated on the same
12 validation scenarios. SAC candidates also consume a declared pilot training
budget; conventional tuning has a much lower computational cost. Equal candidate
counts do not imply equal optimization effort or a universally fair algorithm
comparison.

Final SAC training uses three independent random seeds with one selected
hyperparameter configuration. Each seed starts from a new model. All final seeds
are evaluated, not just the best. The representative trace model is selected on
validation before testing; figures use fixed route indices rather than choosing
the most flattering test examples.

Every controller receives the same route-specific time allowance:
route_length/0.5 + 5 seconds, rounded up to whole control steps.

Stress cases are nominal operation, 3 cm position and 3 degree heading observation
noise, a 200 ms command delay, and response time constants multiplied by 2.5.
Stresses are tested separately. They do not establish robustness to all disturbances.

## 10. Metrics and interpretation

- Success rate: fraction completing all ordered gates and endpoint conditions.
- Route completion: fraction of ordered gates passed, including failures.
- Tracking RMSE and maximum error: local Euclidean route distance, in metres.
- Elapsed time: includes complete attempts and early failures; interpret with success.
- Command smoothness: RMS finite-difference slew, separately for linear and angular commands.
- Inference latency: measured CPU wall time around controller prediction, excluding
  environment updates. This is not embedded worst-case execution time.

Tracking errors on early failures have fewer samples. A low error on a short failed
episode is not evidence of superior control. Always report completion alongside
error. Episode latency percentiles are averaged in summary tables; they are not
the pooled percentile of every inference call.

Three seeds provide a small estimate of training variability, not strong statistical
evidence. Scenarios are paired across controllers. Comparisons apply only to this
simulator, these budgets and this scenario distribution.

## 11. Interview revision

Q: Why implement a conventional baseline first?
A: It checks that the task is solvable and gives a useful reference for learned control.

Q: Why use SAC?
A: It handles bounded continuous commands and can reuse experience through a replay
buffer. That is a design choice, not proof it is optimal for this task.

Q: Why can reward alone be misleading?
A: A policy may exploit a reward definition. We independently check ordered-route
completion and physical tracking metrics, and retain failures.

Q: Why distinguish termination and truncation?
A: Success/off-route failure ends the task; a time limit cuts off an otherwise
continuing attempt. Gymnasium exposes both so training can handle bootstrapping correctly.

Q: Is this a real differential-drive robot model?
A: It is a planar differential-drive-style kinematic abstraction with speed-response
lag. It does not model the wheels, contacts or full mechanical dynamics.

Q: Does simulation performance demonstrate deployment readiness?
A: No. Deployment would require better models, sensing and hardware tests, safety
engineering and timing/resource validation.

## Sources

- Gymnasium environment API: https://gymnasium.farama.org/api/env/
- Stable-Baselines3 SAC: https://stable-baselines3.readthedocs.io/en/master/modules/sac.html
- Evaluation guidance: https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html
- PyTorch installation: https://pytorch.org/get-started/locally/
- Pure Pursuit derivation: R. Craig Coulter, Implementation of the Pure Pursuit Path
  Tracking Algorithm, CMU-RI-TR-92-01 (1992).

### First validation milestone

The 22 correctness checks passed. Before RL training, two of four Pure Pursuit
configurations completed all 12 validation scenarios; the two longer-lookahead
configurations completed 11 of 12. These are validation observations, not test
results. Candidate selection prioritizes success, then gate completion, with
tracking-error and a small elapsed-time penalty. The fixed score is
100*success_rate + 10*gate_completion - RMSE - 0.005*elapsed_seconds.

Pure Pursuit reference: https://publications.ri.cmu.edu/implementation-of-the-pure-pursuit-path-tracking-algorithm
