# RIM component: scope and implementation

This is a **small RIM-inspired component**, not a reproduction of the paper's experiments or its published checkpoints. The old laboratory and its trained RIMs are not used or changed.

Reference study completed before implementation:
- [Goyal et al., Recurrent Independent Mechanisms, sections 2.2–2.4 and C.1](https://arxiv.org/html/1909.10893v6).
- [Official implementation, frozen revision aa61773a949c1e3f499d4c05dd77fcd2adbffa19](https://github.com/anirudh9119/RIMs/tree/aa61773a949c1e3f499d4c05dd77fcd2adbffa19), especially `event_based/blocks_core.py` and `blocks_atari/a2c_ppo_acktr/attention.py`.

The reference mechanism selects recurrent modules by input attention, maintains independent recurrent dynamics, preserves inactive states, and lets active modules read other modules through attention. Our implementation uses a zero null input and chooses the modules with the lowest attention on that null input. Each selected module updates its own GRU state. A residual attention update communicates information to active modules; inactive states remain identical.

## Deliberate simplifications and deviations

- A single attention head, rather than the reference communication module's four heads.
- The encoded vector is one real input token plus a zero token; there is no spatial per-object attention in this initial adapter.
- Independent PyTorch GRUCell objects rather than a masked block implementation of LSTM/GRU.
- Per-module query projections have biases, allowing distinct initial routing with zero recurrent states. Input key/value projections have no biases, so the null remains zero.
- No attention dropout, gated-tanh communication or custom sparse-gradient operator from the reference code. Ordinary autograd flows through selected input processing, communication and the inactive-state identity path. Top-k selection itself is discrete.
- Inactive recurrent cells are skipped for inactive batch rows. Attention still computes scores for all modules; sparse activation does not imply constant compute.
- Explicit mean pooling returns a fixed output dimension independent of module count. This makes expansion possible without replacing the downstream policy head. It can lose module-specific information and changes output scaling when modules are added.

These are implementation choices for the laboratory, not claims of novelty or improved memory. Smoke training checks gradients and integration, not retention or superiority.

## Expansion

`expandable_rim` appends independently initialized mechanisms at an explicit boundary. Existing parameter objects and their optimizer moments remain in place. New parameters enter a new optimizer group inheriting the current settings of the first existing mechanism's group (including its current learning rate). Old state slices are preserved and new state slices begin at zero. There is no silent migration when a stale state is forwarded.

Both the component's module/parameter limits and the entire graph's trainable-parameter limit are checked before expansion. Frozen model parameters are reported separately and do not count against the trainable budget; the runner must also enforce memory limits. Expansion changes model behavior and incurs extra attention cost. Maximum module count bounds it; no claim of unbounded efficient growth is made.

Checkpoints must save current graph configuration, strict model state, optimizer parameter-group **names**, optimizer state and growth-policy state. Reconstruct the topology and named optimizer groups before loading their states. Recurrent state is neither a parameter nor external memory.

`none`, `fixed_schedule` and `plateau` policies are controls. Plateau uses insufficient improvement between the first and second halves of a recent training-return window, with cooldown and a maximum number of growth events. No held-out scores guide growth. This heuristic can expand because of noise or task difficulty and is not evidence that capacity was needed.

## Contract and extension

Components accept finite float32 `[batch, input_dim]` tensors and explicit state dictionaries. They return `(features, next_state, instrumentation)`. Input/output widths, state shapes, device support, trainable/frozen parameter names, checkpoint version and structural operations are declared by `contract()`.

To register a component, subclass `ucl.components.Component` and call `register_component('my_component', MyComponent)` before constructing `GraphPolicy`. The constructor accepts `input_dim` plus configured parameters. The experiment runner remains unchanged. Configuration declares branches explicitly; `concat` and `sum` are the only initial merging operations. Use an explicit MLP adapter for incompatible widths.

Automated tests cover swapping components, custom registration and gradients, branching, shape/state rejection, episode resets, inactive RIM identity and gradients, communication, parameter/moment preservation during growth, strict expanded restoration, continued training and frozen tensors.
