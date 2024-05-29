import numpy as np
from stock import Stock
from action import Action
from rasterizer import Rasterizer
import cv2


class State:
    def __init__(self, design_domain, support_domain, stock):
        self.support_domain = support_domain
        self.design_domain = design_domain
        self.vector_rep = []
        self.stock = stock
        self.cursor = np.array([0, 0])
        self.grid_size = 32
        self.rasterizer = Rasterizer(self.support_domain, self.design_domain)

    def get_observation(self):
        return self.rasterizer.canvas

    def update(self, action: Action):
        if not action.placement_flag:
            self.cursor = np.array(
                [
                    action.cursor_position // self.grid_size,
                    action.cursor_position % self.grid_size,
                ]
            )
            return

        element = self.stock.fetch_element(action.element)
        if element == 0:
            return

        new_element = self._calc_placement(element, action.cursor_position)
        self._add_element(new_element)  # Add element to state

    # Helper Functions
    # These functions need to know about the element, this calls for a Element class :D
    def _add_element(self, element):
        self.vector_rep.append(element)  # Add to list
        self.rasterizer.draw_element(element)  # Update rasterizer
        self.cursor = element[1]  # Update cursor

    def _calc_placement(self, element, position):
        direction = position - self.cursor
        unit_direction = direction / np.linalg.norm(direction)
        scaled_vector = unit_direction * element
        end_point = (self.cursor + scaled_vector).astype(np.uint8)

        return (self.cursor, end_point)


if __name__ == "__main__":
    Stock.initiate_stock(5)
    stock = Stock()

    canvas_size = 64

    support_domain = np.eye(canvas_size, canvas_size, -5)
    
    gauss = (cv2.getGaussianKernel(canvas_size, 5) * 256).astype(np.uint8)
    design_domain = np.outer(gauss, gauss)

    print(design_domain)

    state = State(design_domain, support_domain, stock)
    action = Action([5, 5], 4, True)

    state.update(action)

    img = state.get_observation()
    cv2.imshow("IMAGE", cv2.resize(img, (800, 800)))
    cv2.waitKey(0)
