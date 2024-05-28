import numpy as np


class Action:
    def __init__(self, position, element, place):
        self.cursor_position = position
        self.element = element
        self.placement_flag = place