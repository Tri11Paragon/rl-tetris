import numpy as np

class TrainingSupervisor:
    """Manages the learning process and coordination of agents."""

    def __init__(self, initial_learn_rate = 1e-4, *optimizers):
        self.learn_rate = initial_learn_rate
        self.optimizers = optimizers
        pass

    def epoch(self):
        """Called after the agent performs one epoch of training."""
        pass

    def iterate(self):
        """Called after the agent performs a whole batch of epochs, usually denoting that the current training has finished.
        This function is used to adjust the learning rate"""
        pass