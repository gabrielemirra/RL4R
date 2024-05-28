import numpy as np
from state import State
from reward_system.reward_system import RewardSystem
from action import Action

class environment:
    def __init__(self, state, reward_system):
        self.state = None
        self.reward_sytem = None
    
    def step(self, action: Action):
        if action.placement_flag:
            self.state.update(action)

        return self.reward_sytem.reward(self.state)
