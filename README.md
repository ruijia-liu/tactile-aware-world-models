# Tactile-Aware World Models for Robotic Manipulation

Can fingertip sensing help a robot predict where a grasped object will move when its pose is temporarily hidden? This repository implements a paired comparison in FlexiTac. A simulated gripper lifts a plug, carries it to a target and holds it there. Two controllers receive the same movement commands and robot observations. One also reads the two tactile pads on the active gripper.

[Watch the native FlexiTac simulation recording](media/simulation.mp4)

## World model and controller

The world model is a two-hidden-layer MLP. At each 120 Hz step it combines plug position, gripper position and opening, the commanded end-effector position and opening, and relative positions. The tactile model adds eight pressure-map features. For each pad these are peak response, mean response and the two coordinates of the centre of pressure. Plug velocity is hidden.

The training target contains the next state change and next pad features. During an open-loop forecast each model feeds back its predicted state, and the tactile model feeds back its predicted pad features. Eight independently initialized members form each ensemble. Whole routes are assigned to training or checkpoint selection before windows are drawn. Checkpoints are selected by 20-step position error on the selection routes.

The receding-horizon controller scores up to 60 future steps per candidate and applies the first four steps before replanning. The cost combines distance to the target, excessive plug drift relative to the gripper, and a small closing penalty. The planner also computes disagreement across the eight predicted costs. The paired touch comparison uses mean cost for both policies. They share the same three opening targets (0.27, 0.35 and 0.45), rate limit and approach to transport. During carrying and the terminal hold they may select the tight or medium opening.

## Experimental design

| Stage | Routes | Executions per route | Purpose |
| --- | ---: | ---: | --- |
| Branch collection | 24 | 3 fixed grip actions | Fit on 16 routes and select checkpoints on 8 |
| Closed-loop development | 8 | 2 policies | Check each controller on fresh routes and freeze the policy |
| Held-out comparison | 100 | 2 policies | Measure forecast error and transport success |

The held-out assignment balances four conditions, with 25 independent routes in each. The plug pose is either visible or masked for steps 1–20 after lift begins. Contact placement is either nominal or perturbed by a signed lateral displacement of 2–4 mm. The perturbation and material labels are hidden from both policies. Each matched pair shares route geometry, material coefficients, perturbation and command prefix. The method order is randomized. The simulation logs physical state at the decision point so reset differences can be checked.

Two endpoints answer different questions.

- **Position prediction.** At lift step 20, replay each model against the same observed history and the same 20 commands actually applied on that trajectory. Measure the squared 3D distance from its predicted endpoint to the recorded plug position in mm². Average the two executions within a route before averaging across routes. This measures an action-conditioned forecast. The future commands are known in replay and future sensor readings are excluded.
- **Transport success.** Score the controller that actually acted. The plug must stay within 5 mm of the target and at least 15 mm above its starting height for 25 consecutive observations, spanning 0.20 s. The hold must finish within 3 s of lift onset and by the start of release. Compare successes per 100 executions for each method and the paired difference in percentage points.

The evaluator resamples complete paired routes within the four conditions to form 95% percentile intervals. It reports settings with and without occlusion and perturbation separately. Ensemble cost disagreement describes the model's forecasts. The bootstrap interval describes uncertainty in the estimated difference between methods. For scalar predictions with a positive ensemble standard deviation, the optional interval calibration module fits fixed-width and ensemble-scaled radii on a disjoint calibration set and measures coverage and interval loss on separate held-out routes. [Interval calibration](methods/calibration.py)

## Run the study

Use Python 3.11 or later for the protocol, training and evaluation commands. The simulator also requires a Linux NVIDIA RTX environment, Isaac Sim 6.0, Isaac Lab and a separate upstream FlexiTac checkout. The adapter targets FlexiTac revision `dc2e1c6607a1b1a0e4b196bb4e2d4ccdc1afdcb0`. Its simulator launcher must be able to import the upstream `isaaclab` package and scene dependencies. [FlexiTac source and installation instructions](https://github.com/binghao-huang/FlexiTac-IsaacSim-Simulation/tree/dc2e1c6607a1b1a0e4b196bb4e2d4ccdc1afdcb0)

Start in the repository root. Set the three paths below to your installation. The adapter uses the Isaac Sim 5.1 URDF importer once to convert the upstream scene assets and record their checksums. Isaac Sim 6.0 runs the experiment using those converted assets. The outer Python runs NumPy training and launches one fresh simulator process per route.

```bash
python -m pip install -r requirements.txt
export ISAAC_PYTHON=/path/to/isaac-sim/python.sh
export ISAAC51_PYTHON=/path/to/isaac-sim-5.1/python.sh
export FLEXITAC_REPO=/path/to/FlexiTac-IsaacSim-Simulation
python -m protocol.generate --output runs/protocol
"$ISAAC51_PYTHON" -m simulation.prepare_assets --repo "$FLEXITAC_REPO" --usd-manifest runs/usd-assets.json
```

Collect the three physical grip branches and fit the paired ensembles.

```bash
python -m protocol.collect_stage branches --isaac-python "$ISAAC_PYTHON" --repo "$FLEXITAC_REPO" --usd-manifest runs/usd-assets.json --protocol runs/protocol --output runs/branches
python -m methods.train fit runs/branches --protocol runs/protocol --design config/design.json --output runs/model
```

Evaluate the fitted controllers on the independent development routes, then freeze both weight sets and the controller configuration.

```bash
python -m protocol.collect_stage validation --isaac-python "$ISAAC_PYTHON" --repo "$FLEXITAC_REPO" --usd-manifest runs/usd-assets.json --protocol runs/protocol --output runs/validation --ensemble runs/model/ensemble.npz
python -m methods.train freeze runs/validation --protocol runs/protocol --design config/design.json --output runs/model
```

Once the development gate passes, collect and score the paired held-out cohort.

```bash
python -m protocol.collect_stage confirmation --isaac-python "$ISAAC_PYTHON" --repo "$FLEXITAC_REPO" --usd-manifest runs/usd-assets.json --protocol runs/protocol --output runs/confirmation --ensemble runs/model/ensemble.npz --freeze runs/model/freeze.json
python -m methods.evaluate --raw runs/confirmation --manifest runs/protocol/manifest.json --freeze runs/model/freeze.json --ensemble runs/model/ensemble.npz --design config/design.json --output runs/evaluation
```

For a first launch, append `--rows 0` to the branch collection command. Running the full command later verifies completed routes and continues with the remaining assignments. Each stage keeps its model inputs, material readback, actions, physics time, measurements and integrity metadata together under the chosen output directory.
