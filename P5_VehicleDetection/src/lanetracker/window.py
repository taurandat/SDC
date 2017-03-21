import numpy as np


class Window(object):
    """
    Represents a scanning window used to detect points likely to represent lane edge lines.
    """

    def __init__(self, y1, y2, x, m=100, tolerance=50):
        """
        Initialises a window object.
        """
        self.x = x
        self.mean_x = x
        self.y1 = y1
        self.y2 = y2
        self.m = m
        self.tolerance = tolerance

    def pixels_in(self, nonzero, x=None):
        """
        Returns indices of the pixels in `nonzero` that are located within this window.
        """
        if x is not None:
            self.x = x
        win_indices = (
            (nonzero[0] >= self.y1) & (nonzero[0] < self.y2) &
            (nonzero[1] >= self.x - self.m) & (nonzero[1] < self.x + self.m)
        ).nonzero()[0]
        if len(win_indices) > self.tolerance:
            self.mean_x = np.int(np.mean(nonzero[1][win_indices]))
        else:
            self.mean_x = self.x

        return win_indices

    def coordinates(self):
        """
        Returns coordinates of the bounding rect.
        """
        return ((self.x - self.m, self.y1), (self.x + self.m, self.y2))
