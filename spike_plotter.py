import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

class SpikePlotter():

    def __init__(self, callback_func, title='Event Camera Output', width=None, height=None) -> None:
        self.fig = plt.figure()
        self.ax = p3.Axes3D(self.fig)

        # Setting the axes properties
        self.ax.set_xlim3d([0.0, 1.0])
        self.ax.set_xlabel('time')
        self.ax.set_ylim3d([0.0, width])
        self.ax.set_ylabel('X')
        self.ax.set_zlim3d([0.0, height])
        self.ax.set_zlabel('Y')
        self.ax.set_title(title)
        self.func = callback_func
        self.animation = animation.FuncAnimation(self.fig, self.func)
        plt.show()