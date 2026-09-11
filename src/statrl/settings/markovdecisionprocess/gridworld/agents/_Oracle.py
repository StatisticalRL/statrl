import numpy as np


from statrl.settings.markovdecisionprocess.discrete_nostructure.agents._Oracle import Opti_controller as MDPOracle

class Opti_77_4room:
    """Hand-coded oracle for the 7x7 four-room gridworld.

    Encodes the optimal action of each cell as a 7x7 table.

    Parameters
    ----------
    env : object
        The gridworld instance. Must expose a ``mapping`` from environment
        states to grid cells.

    Notes
    -----
    Unlike the other oracles this class does not inherit from
    :class:`~statrl.settings.markovdecisionprocess.discrete_nostructure.agent.MDPAgent`,
    yet its constructor calls ``super().__init__(nS, nA, name)``, which
    resolves to :class:`object` and raises :exc:`TypeError`. The class is
    therefore not currently constructible; the same applies to
    :class:`Opti_911_2room`.
    """

    def __init__(self, env):
        self.env = env
        self.name="Opti_77_4room"
        super(Opti_77_4room, self).__init__(env.nS, env.nA, self.name)
        pol = (
            [[0, 0, 0, 0, 0, 0, 0],
             [0, 1, 3, 3, 1, 1, 0],
             [0, 1, 2, 0, 1, 2, 0],
             [0, 1, 0, 0, 1, 0, 0],
             [0, 3, 3, 3, 3, 1, 0],
             [0, 3, 0, 0, 3, 1, 0],
             [0, 0, 0, 0, 0, 0, 0]]
        )
        self.policy = np.zeros(49)
        for x in range(7):
            for y in range(7):
                self.policy[x * 7 + y] = pol[x][y]
        self.mapping = env.mapping


    def reset(self, inistate):
        """Start a new run. The policy is fixed, so nothing is cleared.

        Parameters
        ----------
        inistate : int
            Initial state; ignored.
        """
        ()

    def play(self, state):
        """Look up the hand-coded action for this state's grid cell.

        Parameters
        ----------
        state : int
            Current state, mapped through ``env.mapping`` to a grid cell.

        Returns
        -------
        float
            The tabulated action. Note this is a float, since the policy table
            is a :class:`~numpy.ndarray` of floats.
        """
        s = self.mapping[state]
        return self.policy[s]

    def update(self, state, action, reward, observation):
        """Ignore the transition (the oracle has nothing to learn).

        Parameters
        ----------
        state : int
            State the action was taken in.
        action : int
            Action taken.
        reward : float
            Reward observed.
        observation : int
            State reached.
        """
        ()


class Opti_911_2room:
    """Hand-coded oracle for the 9x11 two-room gridworld.

    Encodes the optimal action of each cell as a 9x11 table.

    Parameters
    ----------
    env : object
        The gridworld instance. Must expose a ``mapping`` from environment
        states to grid cells.

    Notes
    -----
    Not constructible as written; see :class:`Opti_77_4room`.
    """

    def __init__(self, env):
        self.env = env
        self.name="Opti_911_2room"
        super(Opti_911_2room, self).__init__(env.nS, env.nA, self.name)
        pol = (
             [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 3, 3, 3, 1, 1, 1, 2, 2, 2, 0],
             [0, 3, 3, 3, 1, 1, 1, 2, 2, 2, 0],
             [0, 3, 3, 3, 3, 1, 2, 2, 2, 2, 0],
             [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
             [0, 3, 3, 3, 3, 3, 3, 3, 3, 1, 0],
             [0, 3, 3, 3, 3, 3, 3, 3, 1, 1, 0],
             [0, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
        )
        self.policy = np.zeros(9 * 11)
        for x in range(9):
            for y in range(11):
                self.policy[x * 11 + y] = pol[x][y]
        self.mapping = env.mapping


    def reset(self, inistate):
        """Start a new run. The policy is fixed, so nothing is cleared.

        Parameters
        ----------
        inistate : int
            Initial state; ignored.
        """
        ()

    def play(self, state):
        """Look up the hand-coded action for this state's grid cell.

        Parameters
        ----------
        state : int
            Current state, mapped through ``env.mapping`` to a grid cell.

        Returns
        -------
        float
            The tabulated action. Note this is a float, since the policy table
            is a :class:`~numpy.ndarray` of floats.
        """
        s = self.mapping[state]
        return self.policy[s]

    def update(self, state, action, reward, observation):
        """Ignore the transition (the oracle has nothing to learn).

        Parameters
        ----------
        state : int
            State the action was taken in.
        action : int
            Action taken.
        reward : float
            Reward observed.
        observation : int
            State reached.
        """
        ()
