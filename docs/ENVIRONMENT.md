# XLand environment and tasks

The adapter uses **XLand-MiniGrid 0.9.3**, running JAX exclusively on CPU. It does
not substitute a toy grid or download the upstream task benchmarks. Tasks are
small hand-authored rulesets built from the official goal/rule classes.

## Interface

```python
from ucl.tasks import load_tasks
from ucl.envs import make_env
manifest = load_tasks()
env = make_env(manifest["tasks"]["development-navigation"])
observation = env.reset(seed=1000)
transition = env.step(0)
frame_for_evaluator = env.render()
identity_for_recorder = env.get_replay_identity()
```

- `reset(seed)` returns a new `numpy.float32` vector.
- `step(action)` returns `Transition(observation, reward, terminated, truncated, info)`.
- `info` contains only public episode events: `success` and the step count.
- `observation_dim` and `action_count` declare the neural interface.
- `agent_observation()` returns a copy of the native visible crop for a text encoder.
- `get_replay_identity()` contains evaluator-only seed, JAX key, task/config hashes,
  package versions and configuration. Never pass it to a model.
- `render()` produces the official RGB visualization of the **full world**. This
  is privileged display/recording data, not an observation for the policy.
- An ended episode requires an explicit reset. No auto-reset hides terminal frames.

A new environment can implement `EnvironmentAdapter` and call
`register_environment(name, factory)` without changing a model or training loop.
No JAX import is required by the abstract interface, task loader or neural modules.

## Agent-visible data and action semantics

Native observations contain tile ID and color ID in the official egocentric
5-by-5 crop. The adapter divides the two channels by 12 and 11, then explicitly
flattens row-major into **50 float32 values**. It adds no rules, goal specification,
position, task identity or global map. The optional LLM receives a description of
this same crop, not the full rendered image or evaluator metadata.

The upstream crop is transparent to walls inside its viewing area. It does not
include the object carried in the agent's pocket. Both upstream conventions are
preserved, not silently replaced with standard MiniGrid assumptions.

| ID | Action |
|---|---|
| 0 | Move forward |
| 1 | Turn clockwise |
| 2 | Turn counterclockwise |
| 3 | Pick up object in front |
| 4 | Put down object in front |
| 5 | Toggle door in front |

The adapter rejects invalid action IDs before JAX (whose low-level switch can
clamp indices). Upstream LAST with discount zero is goal termination. LAST with
discount one is time-limit truncation. Goal reward is the actual simulator reward,
`1 - 0.9 * elapsed_steps / max_steps`; other steps give zero. No reward shaping or
oracle actions are present in the adapter.

## Small task manifest

`configs/tasks.json` defines 19 task identities: 6 development, 6 validation and
7 held-out. All use the registered R1 layout with explicit 5-by-5 dimensions,
5-by-5 view and a 32-step episode cap. This is a configured small variant of the
official environment, not a claim to reproduce its large benchmark distribution.

| Family | Official mechanic | What varies |
|---|---|---|
| navigation | AgentOnTileGoal | Map layout and goal color |
| pickup | AgentHoldGoal | Target/distractor objects |
| hold_transform | AgentHoldRule | Object transforms when picked up |
| adjacency_transform | TileNearRule | Objects combine when placed next to each other |
| changed_rule | AgentHoldRule | Same visible initial object set, different trigger |
| composition | AgentHoldRule then TileNearRule | Two different interactions must compose |
| directional_adjacency | TileNearUpRule | Reserved directional interaction, held-out only |

Changed dynamics means changing a supported production rule; movement physics is
not changed. Composition requires pickup, placement and final pickup. It is not
just two instantaneous rules chained by one action.

Initial objects and agent positions are sampled on every seeded reset. Training
on one map is not treated as transferable mechanic learning. Native observation
omits the hidden objective/rules, so an untrained agent is not told the solution.
The reward and subsequent visible consequences provide experimental feedback.

## Splits and curricula

- Task SHA256 includes adapter, environment, dimensions, view, episode limit, goal,
  rules and initial tiles. Relabeling a task cannot bypass the duplicate check.
- Ruleset SHA256 includes goal, rules and initial tiles; rulesets cannot cross splits.
- Split seed lists and evaluation seeds are disjoint. All architectures consume the
  same task IDs and seed lists for evaluation.
- Training uses the task's explicit finite seed pool and cycles within that pool.
  It does not increment seeds without limit or add the model-initialization seed
  to map seeds. Long runs therefore cannot drift into reserved evaluation seeds.
  Reusing a seed repeats that map; more training steps do not create more unique
  maps. Report both interactions and the number of distinct training maps.
- Model initialization seeds are separate from map seeds. Comparisons preserve
  the same environment experience schedule when changing neural components.
- Task manifests fail validation if held-out or validation stages request training.
- Palette changes in validation/held-out are **color/ruleset generalization**, not
  cross-domain transfer. The reserved directional mechanic is a different test.
- The loader provides configs; the runner must enforce split separation for any
  command that directly requests training rather than using a curriculum.

`return_to_previous`: train A (navigation), B (pickup), C (transformation), D
(adjacency), then revisit A without updates. Evaluate A on the same reserved maps
before/after for retention and on further fresh maps for map generalization.

`progressive`: adds changed production rules, composition and explicitly
non-training validation/held-out stages. `learning_to_learn`: specifies prior
families, a target family and matched target exposure for an experienced and a
fresh model. A single smoke curve cannot establish meta-learning. Multiple seeds,
matched capacity/exposure, learning curves and an informative task difficulty are
needed for scientific interpretation.

## Reexecution versus replay

Seed plus full environment/ruleset configuration reproduces a trajectory when
versions and action sequence match. JAX key data and PRNG implementation are saved.
Resume requires a stable environment identity: task/ruleset hashes, observation
encoding, action mapping, JAX and XLand versions, and PRNG implementation. Runtime
seed/key fields are reconstructed separately from the saved training seed pool
and action history. A changed hidden goal or rule must reject resume even when
its visible crop happens to match the old checkpoint. The training seed pool is
part of resume state; continuation stays within that same declared pool.

A **recorded replay** instead displays saved frames and actions without running
JAX or model inference again. The adapter supplies rendering and identity; the
experiment recorder stores those frames. **Resume** additionally restores model,
optimizer and training RNG state; it is not the same operation as replay.

## Evidence and limits

Tests cover the real reset/step, exact seeded reexecution, all action IDs, proper
termination/truncation, actual goal reward, real production-rule composition,
rendering, copy isolation, hidden-state exclusion, split validation and custom
environment registration. Goal-reaching fixture/oracle logic exists only in tests
and is never provided to agents. These checks validate integration; they do not
claim an agent learned the tasks or a scientific advantage.

Sources inspected before implementation:

- https://github.com/dunnolab/xland-minigrid
- https://github.com/dunnolab/xland-minigrid/blob/main/src/xminigrid/environment.py
- https://github.com/dunnolab/xland-minigrid/blob/main/src/xminigrid/core/observation.py
- https://github.com/dunnolab/xland-minigrid/blob/main/src/xminigrid/core/actions.py
- https://github.com/dunnolab/xland-minigrid/blob/main/src/xminigrid/core/goals.py
- https://github.com/dunnolab/xland-minigrid/blob/main/src/xminigrid/core/rules.py
- https://docs.jax.dev/en/latest/installation.html

Upstream main inspected at `0d306d1414dcb6e451358d94e7d9a91879462ee1`.
The PyPI 0.9.3 wheel SHA256 is
`564823b12e09fb8eac9acb6c1fdc5ab678112a1ae45848ff9911c449b8c27389`.
