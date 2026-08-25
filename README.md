# External Python Trainer for Unity ML-Agents (DQN)

**Train Unity ML-Agents from your own Python script instead of the built-in
`mlagents-learn` CLI.** By connecting to the environment through the Unity Gym
wrapper, the training loop becomes ordinary Python code — so *any* RL algorithm
can be plugged in, not only the ones bundled with ML-Agents.

This repository demonstrates that pipeline with a from-scratch **Deep Q-Network
(DQN)** — an algorithm ML-Agents does *not* ship by default — implemented in
PyTorch with experience replay, a separate target network, ε-greedy exploration,
and export of trained policies to both PyTorch (`.pth`) and ONNX (`.onnx`) for
in-engine inference.

## Overview

The core contribution is the **external trainer**: `dqn_mlagents.py` builds,
steps and trains a Unity environment entirely from Python, replacing the
ML-Agents command-line workflow and opening the door to arbitrary RL algorithms.
As a case study it also examines how the trained DQN agent behaves in
**deterministic** vs. **stochastic** environments, and whether it converges to
the rational (reward-maximizing) policy. See [`REPORT.md`](REPORT.md) for the
full write-up **including result figures** from real training and inference runs.

Two families of environments are included:

- **RandomPath** — a level-based environment where, at each level, the agent
  picks one of three positions and a coin spawns according to a per-level
  probability distribution. Variants add stochastic action noise and a negative
  ("prison") coin that must be avoided.
- **A Coin in the Room** — a continuous navigation environment where the agent
  always moves forward and steers (±45°) using three custom SphereCast ray
  sensors that report wall/coin detections and distances (8 observations total).

## Features

- **External Python trainer** for Unity ML-Agents via the Gym wrapper — trains
  from a script, not the `mlagents-learn` CLI, so any RL algorithm can be used.
- DQN implemented in PyTorch: policy + target networks, replay memory, Huber
  loss, AdamW, soft target updates (τ).
- ε-greedy exploration with exponential decay.
- Custom Unity ML-Agents environments with configurable per-level probability
  distributions and optional action noise.
- Custom ray-sensor observations via Unity SphereCasting.
- Model export to `.pth` (PyTorch) and `.onnx` (for in-engine Unity inference).
- TensorBoard logging of reward-per-episode and training loss.

## Technologies

- Python 3.10.3, PyTorch, ONNX
- Unity ML-Agents (Unity package 2.0.1, `mlagents` Python 1.0.0)
- Unity 2022.3.12f1 (C#)
- TensorBoard, matplotlib

## Project Structure

```
dqn-unity-mlagents/
├── python-trainer/
│   ├── dqn_mlagents.py       # DQN trainer: training + inference, ONNX export, TensorBoard
│   └── requirementsDQN.txt   # Python dependencies
├── unity-agents/
│   ├── Agent412.cs           # RandomPath agent (discrete level choice)
│   └── Agent412p3.cs         # "Coin in the Room" agent (ray sensors, continuous nav)
├── models/                   # Small pre-trained policies (.pth / .onnx)
│   ├── RandomPathv0.{pth,onnx}
│   ├── RandomPathv1.{pth,onnx}
│   ├── RandomPathv2.{pth,onnx}
│   └── model2.pth
├── media/                    # Result figures used in REPORT.md
├── REPORT.md                 # Full project report (with training/inference results)
└── README.md
```

> **Note:** the compiled Unity standalone environment builds
> (`mlagentsProj.exe`, `UnityPlayer.dll`, `*_Data/`, etc.) are **not** included —
> they are large binaries and are listed in `.gitignore`. The C# agent scripts
> and the Python trainer are provided so the project can be reconstructed inside
> a Unity ML-Agents project.

## Installation

Requires **Python 3.10.3** (important for this ML-Agents version) and, on
Windows, the Visual C++ Build Tools (needed to build the `numpy` wheel).

```bash
python -m venv myVenv
# Windows:
myVenv\Scripts\activate
# macOS/Linux:
# source myVenv/bin/activate

python -m pip install --upgrade pip
pip install -r python-trainer/requirementsDQN.txt
```

For in-engine inference and rebuilding the environments you also need Unity
2022.3.12f1 with the ML-Agents package (2.0.1).

## Usage

The trainer expects a **built Unity environment folder** (passed via
`--environment`) in the working directory. See `REPORT.md` for how the
environments are structured.

**Train:**

```bash
python dqn_mlagents.py --environment "BuildFolderName" --num 4000
# add --n256 to use 256 neurons per layer (the "Coin in the Room" environment)
```

Training produces `model.pth` and `model.onnx`.

**Inference (PyTorch):**

```bash
python dqn_mlagents.py --environment "BuildFolderName" --modelname "RandomPathv1.pth" --inference
```

**Inference (Unity):** drag a `.onnx` model into the agent's Behavior Parameters
(Model) in the Unity Editor.

**View training curves:**

```bash
tensorboard --logdir="runs"
```

Full flag reference: `python dqn_mlagents.py -h`.

## Examples

The `models/` folder contains small trained policies you can load directly:

- **`RandomPathv0`** — deterministic RandomPath. Learned strategy: `bot, mid,
  mid, top`.
- **`RandomPathv1`** — stochastic RandomPath (per-level distributions). Learned
  strategy after 4000 episodes: `bot, top, mid, top` (optimal).
- **`RandomPathv2`** — stochastic actions **plus** a negative coin the agent must
  avoid. Converges to the optimal strategy; `REPORT.md` works through the
  expected-utility calculation for the level containing the negative coin.
- **`model2.pth`** — the "Coin in the Room" navigation agent (8 sensor
  observations, 256 neurons/layer, trained for 3000 episodes).

## Notes

- The environments are intentionally simple: training is CPU-bound and slow
  without a GPU, and the goal is to study convergence behaviour rather than to
  scale.
- Stochastic environments need substantially more episodes than deterministic
  ones to reliably reach the rational policy.
- The Python code uses Windows-style path separators in a couple of places
  (`os.getcwd() + '\\' + ...`); adjust for other operating systems if needed.
