# Settings Repository Guidelines

Each setting is represented by a folder.

A setting folder either

- contains sub-settings, or
- defines a complete setting.

A complete setting must have the following structure:

```
setting_name/
├── agent.py          # Defines the Agent class.
├── environment.py    # Defines the Environment class.
│                     # Must implement: step(), reset(), render().
├── interaction.py    # Defines the interaction protocol.
│                     # Must implement: run(), renderrun().
├── _test.py          # Contains test_render(), test_run(),
│                     # test_load(), test_massive().
├── agents/
│   ├── _Oracle.py    # Oracle agent following the optimal policy.
│   └── _Random.py    # Uniform random policy.
├── envs/
│   └── environments.yaml
├── renderers/
└── wrappers/
```

The `envs/environments.yaml` file defines registered environments.

Example:

```yaml
bernoulli_simple:
  entrypoint: "parametric:BernoulliBandit"
  kwargs:
    means: [0.2, 0.8, 0.7, 0.5]

random:
  entrypoint: "parametric:RandomBernoulliBandit"
  kwargs:
    Delta: 0.2
    K: 5
```

Each environment specification must have the form

```yaml
environment_name:
  entrypoint: "<module>:<Class>"
  kwargs:
    ...
```

where

- `entrypoint` specifies the Python class to instantiate,
- `kwargs` contains the keyword arguments passed to the constructor.