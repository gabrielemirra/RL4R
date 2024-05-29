import numpy as np
import cv2


class Rasterizer:
    def __init__(self, supports_domain, design_domain, canvas_size=64):
        self.canvas_size = canvas_size
        self.elements_channel = np.zeros((self.canvas_size, self.canvas_size))
        self.supports_channel = None
        self.design_channel = None

        self.draw_supports_channel(supports_domain)
        self.draw_design_channel(design_domain)

    def draw_supports_channel(self, supports_domain):
        # TODO: Convert domain to np.array
        self.supports_channel = supports_domain

    def draw_design_channel(self, design_domain):
        # TODO: Convert domain to np.array
        self.design_channel = design_domain

    # This function needs to know how to draw an element, this calls for a Element class :D
    def draw_element(self, element):
        self.elements_channel = cv2.line(
            self.elements_channel, element[0], element[1], color=200
        )

    @property
    def canvas(self):
        return np.stack(
            [
                self.elements_channel,
                self.design_channel,
                self.supports_channel,
            ],
            axis=-1,
        )


if __name__ == "__main__":
    supports_domain = np.eye(64)
    design_domain = np.eye(64)

    rasterizer = Rasterizer(supports_domain, design_domain)
    rasterizer.draw_element([(10, 5), (3, 14)])

    img = rasterizer.canvas
    cv2.imshow("CANVAS", cv2.resize(img, (800, 800)))
    cv2.waitKey(0)
