class Agent:
    def play(self, observation):
        """
        Returns an action x_t in continuous action space.
        """
        raise NotImplementedError

    def update(self, action, reward, observation=None):
        """
        Optional update step after observing reward.
        """
        pass