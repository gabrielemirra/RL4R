import numpy as np
from stock import Stock
from action import Action
import cv2


class State:
    def __init__(self, design_domain, support_domain, stock):
        self.design_domain = design_domain
        self.support_domain = support_domain
        self.vector_rep = []
        self.stock = stock
        self.cursor = np.array([0, 0])
        self.grid_size = 32

    def get_observation(self):
        img = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        for element in self.vector_rep:
            print(element)
            img = cv2.line(img, (element[0]), (element[1]), color=128)

        return img

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

        direction = action.cursor_position - self.cursor
        unit_direction = direction / np.linalg.norm(direction)
        scaled_vector = unit_direction * element
        end_point = (self.cursor + scaled_vector).astype(np.uint8)

        self.vector_rep.append((self.cursor, end_point))
        self.cursor = end_point


if __name__ == "__main__":
    Stock.initiate_stock(5)
    stock = Stock()

    canvas_size = 32

    design_domain = np.diag((canvas_size, canvas_size))
    support_domain = np.eye(canvas_size)

    state = State(design_domain, support_domain, stock)
    action = Action([5, 5], 4, True)
    state.update(action)
    img = state.get_observation()
    cv2.imshow("IMAGE", img)
    cv2.waitKey(0)
