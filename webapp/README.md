# STATRL Interactive Playground

Interactive web interface for experimenting with the environments and learning algorithms provided by [STATRl](https://github.com/StatisticalRL/statrl).

The playground is designed to make the statistical reinforcement-learning components of STATRL directly explorable from a browser, without requiring the user to write an experiment script.

The central idea is simple:

> **An interaction history belongs to an experiment, not to a particular learner.**

A user can therefore start interacting with an environment manually, ask one learner for recommendations, switch to another learner, return to manual control, and continue the same experiment. Whenever a learner is attached to an existing experiment, it is updated from the history that has actually occurred.

The playground also provides a second, more exploratory mode in which the user can act as the environment: after an algorithm recommends an action, the user specifies the resulting reward and next state.

---

# 1. Objectives

The playground is intended to provide four complementary capabilities.

## 1.1 Explore STATRL settings

The user first navigates the STATRL settings hierarchy.

For example:

```text
STATRL
├── Bandits
│   ├── Stochastic
│   │   ├── Anytime
│   │   ├── Batch
│   │   ├── Knownhorizon
│   │   └── Kernel
│   └── Adversarial
│       └── Lipschitz
│
└── Markov Decision Processes
    ├── Discrete No Structure
    └── Gridworld
```

The hierarchy is discovered from the Python package structure rather than hard-coded into the HTML page.

Settings containing an `envs/environments.yaml` registry are selectable.

---

## 1.2 Select and inspect an environment

After selecting a setting, the user is presented with the environments registered in:

```text
<setting>/envs/environments.yaml
```

For example:

```text
Setting:
    Bandits > Stochastic > Anytime

Environment:
    Bernoulli Hard
```

Environment definitions are loaded using STATRL's existing environment registry mechanism:

```python
from statrl.experiments.utils import load, make

envs = load("envs/environments.yaml")
env = make(envs["bernoulli_hard"])
```

The playground should **not duplicate the logic of `load()` or `make()`**.

The environment registry remains the authoritative source.

The environment panel should expose useful information such as:

* environment name;
* description;
* state space;
* action space;
* parameters;
* horizon, when relevant;
* number of states/actions;
* environment-specific metadata;
* visualization, when an appropriate renderer is available.

---

# 2. Two modes of interaction

The playground supports two conceptually different interaction modes.

## 2.1 Standard environment mode

In standard mode, STATRL controls the environment.

The interaction is:

```text
learner/manual controller
          │
          │ action
          ▼
       environment
          │
          │ reward, next state
          ▼
       experiment
```

For a bandit:

```text
action → reward
```

For an MDP:

```text
(state, action) → (next_state, reward)
```

The user may either:

* choose the action manually;
* ask a learner for its recommendation.

---

## 2.2 Custom environment mode

Custom mode allows the user to act as the environment.

The learner proposes an action, but the user determines what happens.

For example:

```text
Current state: 7

Learner recommendation:
    action = 2

User specifies:

    next state = 11
    reward     = -1
    terminal   = false
```

The resulting transition is then given to the learner exactly as if it had come from an ordinary STATRL environment.

This mode is intended for:

* teaching;
* interactive demonstrations;
* hypothetical scenarios;
* testing algorithms under manually specified observations;
* exploring robustness;
* reproducing individual trajectories;
* experimentation with environments that are difficult to encode programmatically.

The browser should never execute arbitrary Python code supplied by the user. Custom environments are represented through **user-provided transition outcomes**, not through uploaded or evaluated Python.

---

# 3. Shared interaction history

The most important design feature of the playground is the shared history.

Suppose the user performs:

```text
t = 0    action 1    reward 0.3
t = 1    action 2    reward 0.8
t = 2    action 1    reward 0.4
```

The experiment stores these transitions independently of whichever learner was active.

The user can then select another algorithm.

For example:

```text
                shared history
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
        IMED       UCB          TS
```

The newly selected learner is initialized and replayed over the existing history.

Consequently, switching algorithms does **not** reset the experiment.

---

# 4. Example interaction

A typical session might look like:

```text
1. Select:

   Bandits
      └── Stochastic
             └── Anytime

2. Select:

   Bernoulli Hard

3. Inspect environment parameters.

4. Start manually.

5. Choose:

   action = 2

6. Environment returns:

   reward = 1

7. Ask IMED for a recommendation.

8. Execute the recommended action.

9. Switch to Thompson Sampling.

10. Thompson Sampling is initialized from the complete history.

11. Ask Thompson Sampling for another action.

12. Return to manual control.

13. Continue the same experiment.
```

The experiment is therefore not tied to a single learner.

---

# 5. Architecture

The webapp is deliberately separated into several layers.

```text
                         Browser
                            │
                            ▼
                     ┌────────────┐
                     │ index.html │
                     └─────┬──────┘
                           │ HTTP/JSON
                           ▼
                     ┌────────────┐
                     │   app.py   │
                     └─────┬──────┘
                           │
             ┌─────────────┼─────────────┐
             ▼             ▼             ▼
        Catalog       Session/Core   Custom Environment
             │             │
             ▼             │
      STATRL package       │
             │             │
             ▼             ▼
       environments      learners
```

---

# 6. File organization

The intended structure is:

```text
src/
└── statrl/
    │
    ├── settings/
    │   ├── bandits/
    │   ├── markovdecisionprocess/
    │   └── ...
    │
    └── webapp/
        ├── __init__.py
        ├── __main__.py
        ├── app.py
        ├── catalog.py
        ├── core.py
        ├── custom_environment.py
        │
        ├── adapters/
        │   ├── __init__.py
        │   ├── bandit.py
        │   ├── mdp.py
        │   └── gridworld.py
        │
        └── static/
            └── index.html
```

Not all adapters need to exist immediately.

The architecture is deliberately designed so that new settings can be added progressively.

---

# 7. `catalog.py`

`catalog.py` is responsible for discovering the STATRL taxonomy and available resources.

Its responsibilities are:

1. discover settings;
2. construct the hierarchical setting tree;
3. identify selectable settings;
4. locate `envs/environments.yaml`;
5. load environment registries;
6. discover available learners.

The setting hierarchy should be obtained from the installed Python package structure, rather than duplicated manually in the web application.

For example:

```python
import statrl.settings
import pkgutil
```

can be used to inspect the package hierarchy.

A setting becomes selectable when its package contains the expected environment registry:

```text
envs/
└── environments.yaml
```

This distinction is useful:

```text
bandits
└── stochastic
    └── anytime       ← selectable
```

rather than treating every package as an independent experiment setting.

---

# 8. Environment loading

Environment construction must use the existing STATRL utilities:

```python
from statrl.experiments.utils import load, make
```

The registry is loaded:

```python
envs = load("envs/environments.yaml")
```

and an environment is instantiated with:

```python
env = make(envs["bernoulli_hard"])
```

The implementation of `load()` records the directory containing the YAML file as `_base_dir`.

`make()` subsequently uses:

```text
entrypoint
kwargs
_base_dir
```

to load the environment implementation.

The webapp should therefore **not reproduce this logic**.

This ensures that environments used in the playground are constructed in exactly the same way as environments used by ordinary STATRL experiments.

---

# 9. `core.py`

`core.py` contains the setting-independent experiment/session logic.

A session conceptually contains:

```python
Session(
    environment,
    history,
    current_state,
    learner,
    ...
)
```

The history should be represented explicitly.

For an MDP, for example:

```python
Transition(
    state=s,
    action=a,
    reward=r,
    next_state=s_next,
    done=done,
    info=info,
)
```

For a bandit, the transition can be simpler:

```python
BanditTransition(
    action=a,
    reward=r,
)
```

The exact representation can be unified later if desirable, but the web layer should not depend on a particular learner implementation.

---

# 10. Learner switching

Learners are deliberately treated as replaceable controllers.

A learner is not the owner of the history.

When a new learner is selected:

```text
existing history
       │
       ▼
new learner
       │
       ├── update(t=0)
       ├── update(t=1)
       ├── update(t=2)
       └── ...
```

The learner therefore obtains estimates corresponding to the actual experiment history.

This approach has an additional advantage: it permits meaningful comparisons between algorithms on exactly the same trajectory.

---

# 11. Setting adapters

Different STATRL settings use different interaction protocols.

The webapp should therefore use adapters.

For example:

```text
adapters/
├── bandit.py
├── mdp.py
└── gridworld.py
```

A bandit adapter translates the generic session operations into the bandit learner protocol.

Conceptually:

```python
action = learner.select_arm()

learner.update(
    action,
    reward,
)
```

An MDP adapter may instead use:

```python
action = learner.select_action(state)

learner.update(
    state,
    action,
    reward,
    next_state,
)
```

The browser does not need to know these details.

---

# 12. Custom environment controller

Custom interaction should be isolated in:

```text
custom_environment.py
```

The controller receives an action and a user-specified outcome.

For example:

```python
controller.step(
    action=2,
    next_state=7,
    reward=-1.0,
    done=False,
)
```

The controller produces the same transition object that a genuine environment would have produced.

This means that learners cannot distinguish between:

```text
real environment
```

and

```text
human-provided transition
```

once the transition has entered the experiment history.

---

# 13. Browser ergonomics

The interface should remain deliberately simple.

The recommended layout is:

```text
┌──────────────────────────────────────────────────────────────┐
│                       STATRL Playground                      │
├──────────────────────┬───────────────────────────────────────┤
│ SETTING              │ ENVIRONMENT                           │
│                      │                                       │
│ 🔎 Search            │ Environment description               │
│                      │ Parameters                             │
│ ▾ Bandits            │ State/action information              │
│   ▾ Stochastic       │                                       │
│     Anytime          │                                       │
│     Batch            │                                       │
│     Kernel           │                                       │
│                      │                                       │
├──────────────────────┴───────────────────────────────────────┤
│                     INTERACTION                              │
│                                                               │
│ State: 3                                                    │
│                                                               │
│ Controller: [ Manual ▼ ]                                     │
│                                                               │
│ Action: [ 2 ]       [Recommend] [Take step]                 │
│                                                               │
│ Reward: 0.8                                                  │
│                                                               │
├───────────────────────────────────────────────────────────────┤
│                     HISTORY                                  │
│                                                               │
│ t     state    action    reward    next state                │
│ 0       0        1        .3          2                     │
│ 1       2        2        .8          3                     │
│ ...                                                           │
└───────────────────────────────────────────────────────────────┘
```

The interface should favour immediate experimentation over configuration.

---

# 14. Setting navigation

The setting tree should support:

* expansion/collapse;
* search;
* highlighting the selected setting;
* automatic expansion of matching branches;
* clear indication of selectable and non-selectable nodes.

For example:

```text
Search: kernel

▾ Bandits
  ▾ Stochastic
      Kernel
```

The user should not need to know the Python import path.

The canonical identifier remains available internally:

```text
bandits.stochastic.kernel
```

---

# 15. Environment selection

After selecting a setting, the environment selector should be populated from its registry.

For example:

```text
Environment

[ Bernoulli Hard                 ▼ ]

Available environments:
    Bernoulli Easy
    Bernoulli Hard
    Gaussian Easy
    Gaussian Hard
```

The page should display the environment specification before starting an experiment.

This is particularly important for educational use: users should be able to understand the experiment before asking an algorithm to act.

---

# 16. Controller selection

The interaction panel should expose:

```text
Controller

[ Manual ▼ ]

Manual
IMED
KL-UCB
Thompson Sampling
UCB
...
```

The precise list depends on the selected setting.

The learner list should therefore be discovered from the selected setting rather than globally hard-coded.

---

# 17. Recommended interaction workflow

A typical workflow should be:

```text
Select setting
       ↓
Select environment
       ↓
Inspect environment
       ↓
Start session
       ↓
Choose controller
       ↓
Recommend action
       ↓
Execute action
       ↓
Inspect result
       ↓
Repeat / switch controller
```

Switching controller must never silently reset the history.

If a reset is desired, it should be an explicit action:

```text
[ Reset experiment ]
```

---

# 18. History visualization

The history should remain visible throughout the experiment.

For bandits:

```text
Step    Action    Reward
 0        2        1
 1        0        0
 2        2        1
 3        1        0
```

For MDPs:

```text
Step    State    Action    Reward    Next state
 0        3        1        -1          7
 1        7        2         0          8
 2        8        0         1          4
```

This provides both an audit trail and an educational view of what the learner has actually observed.

---

# 19. Environment visualization

The playground should take advantage of the existing STATRL renderers.

For example, GridWorld can use the dedicated HTML renderer:

```text
GridWorldHTMLRenderer
```

rather than displaying only numerical states.

This suggests a future renderer interface for the playground:

```python
get_visualization(environment, session)
```

Possible visualizations include:

* GridWorld map;
* abstract MDP graph;
* bandit reward estimates;
* confidence intervals;
* action counts;
* empirical means;
* regret curves.

The visualization should be supplementary: the interaction history remains the authoritative record.

---

# 20. API design

The browser communicates with a lightweight Python HTTP server.

Conceptually:

```text
GET /api/settings/tree

GET /api/environments/<setting>

GET /api/learners/<setting>

POST /api/session

POST /api/session/<id>/step

POST /api/session/<id>/recommend

POST /api/session/<id>/learner

GET /api/session/<id>
```

The API should remain setting-independent wherever possible.

---

# 21. Session lifecycle

A session begins when the user starts interacting with an environment.

```text
POST /api/session
```

returns a session identifier.

The server then maintains:

```text
session_id
    │
    ├── environment
    ├── setting
    ├── history
    ├── current state
    └── active learner
```

A session can be reset explicitly.

A learner switch does **not** create a new session.

---

# 22. Security considerations

The custom-environment functionality must not allow arbitrary Python execution.

The browser should only submit structured data:

```json
{
    "action": 2,
    "next_state": 7,
    "reward": -1,
    "done": false
}
```

It must never submit Python code that the server evaluates.

Similarly, environment and learner construction should only use trusted definitions already present in the STATRL repository.

---

# 23. Extensibility

The architecture is designed so that adding a new STATRL setting requires little or no modification to the web application.

For a new setting:

```text
settings/
└── new_setting/
    ├── envs/
    │   └── environments.yaml
    ├── agents/
    ├── interaction.py
    └── ...
```

the catalogue can discover it automatically.

If its interaction protocol differs from existing adapters, only a new adapter should be required:

```text
webapp/
└── adapters/
    └── new_setting.py
```

The HTML interface should remain unchanged.

---

# 24. Current implementation status

The project should be developed incrementally.

### Phase 1 — infrastructure

* [x] HTTP server
* [x] HTML interface
* [x] setting discovery
* [x] hierarchical navigation
* [x] environment registry discovery
* [x] environment loading through STATRL utilities
* [x] basic session/history concept

### Phase 2 — bandit playground

* [ ] manual bandit interaction
* [ ] learner recommendations
* [ ] learner switching
* [ ] history replay
* [ ] bandit-specific diagnostics
* [ ] reward/action visualization

### Phase 3 — discrete MDPs

* [ ] MDP adapter
* [ ] state/action display
* [ ] manual action selection
* [ ] learner switching
* [ ] custom transition mode
* [ ] abstract MDP visualization

### Phase 4 — GridWorld

* [ ] GridWorld adapter
* [ ] integration with GridWorld HTML renderer
* [ ] interactive map
* [ ] state/action visualization
* [ ] custom transition mode

### Phase 5 — richer experimental functionality

Potential additions include:

* trajectory export;
* experiment replay;
* comparison of several learners;
* synchronized learner comparison;
* empirical regret visualization;
* confidence intervals;
* action-count plots;
* learner internal statistics;
* reproducible seeds;
* downloadable experiment logs.

---

# 25. Design principles

The implementation should follow a few principles.

### STATRL remains authoritative

The webapp should reuse:

```python
load()
make()
```

and existing environment/agent APIs rather than reimplementing them.

### The browser is a client

The HTML/JavaScript layer should display state and send user commands. It should not contain the statistical learning logic.

### History is independent

The experiment history belongs to the session, not to a learner.

### Learners are replaceable

Any compatible learner can be attached to an existing history.

### Settings are discovered

The webapp should not contain a manually maintained list of STATRL settings.

### Custom interaction is explicit

The user can specify transition outcomes without executing arbitrary code.

### Visualization is modular

Environment-specific visualizations should be implemented independently of the session engine.

---

# 26. Long-term vision

The final playground should provide a bridge between the mathematical/statistical algorithms implemented in STATRL and interactive experimentation.

A user should be able to go from:

```text
STATRL setting
      ↓
environment
      ↓
algorithm
      ↓
trajectory
      ↓
learner estimates
      ↓
visualization
```

without writing an experiment script.

At the same time, the playground should remain faithful to the underlying STATRL APIs so that an experiment performed interactively can later be reproduced programmatically.

The intended relationship is therefore:

```text
                  STATRL
                    │
       ┌────────────┴────────────┐
       │                         │
Programmatic experiments     Playground
       │                         │
       └────────────┬────────────┘
                    │
             same environments
             same learners
             same protocols
```

The playground is not a separate experimentation framework. It is an interactive front-end to the existing STATRL experimentation infrastructure.
