# Structuration Guidelines for the Settings

Each setting folder should either contain sub-settings folder or define a setting.
To define a setting, the structure of the folder must contain at least:

[Name of the setting]

- agent.py:  "Specifies the class Agent"
- environment.py: "Specifies the class Environment
                  implementing methods .step, .reset, .render"
- interaction.py: "Specifies the Interaction (see src.experiments.onerun.Interaction)
                  implementing methods  .run and .renderrun"
- _test.py: "Specifies test functions test_render, test_run, test_massive ""

#### agents: "Folder containing the agent class definitions"
- _Oracle.py : "the Oracle agent which plays according to the optimal policy."
- _Random.py : "the Random agent which plays according to the uniformly random policy."

#### envs: "Folder containiing the environment class definitions"

#### renderers: "Folder containing the renderer class definitions"

#### wrappers: "Folder containing the wrappers from and to other settings (both from env and agent)"