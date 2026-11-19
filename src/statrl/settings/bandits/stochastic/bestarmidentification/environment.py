



"""Environment for the BAIsetting.

Best arm identification setting  changes what the agent may do, not what the environment
is, so this module re-exports :class:`~statrl.settings.bandits.stochastic.anytime.environment.StochasticBanditEnv`
unchanged. Any environment built for the anytime setting therefore works here
as is.
"""

from statrl.settings.bandits.stochastic.anytime.environment import StochasticBanditEnv as StochasticBanditEnv