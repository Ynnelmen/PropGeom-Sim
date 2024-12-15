import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline

from Acoustic_Solver import AcousticObserver

class ObserverManager:
    def __init__(self, observers=None):
        self.observers = observers or []

    @classmethod
    def from_positions(cls, positions):
        return cls(observers=[AcousticObserver(pos) for pos in positions])

    @classmethod
    def from_iso3744(cls):
        positions = [
            [0.336, -2.016, 0.462],
            [1.638, -1.260, 0.420],
            [1.638, 1.155, 0.651],
            [0.336, 1.890, 0.861],
            [-1.743, 0.672, 0.945],
            [-1.743, -0.840, 0.798],
            [-0.546, -1.365, 1.491],
            [1.554, -0.147, 1.407],
            [-0.546, 1.050, 1.743],
            [0.210, -0.210, 2.079],
        ]
        return cls.from_positions(positions)

    @classmethod
    def from_fibonacci(cls, n_points, radius):
        indices = np.arange(0, n_points, dtype=float) + 0.5
        phi = np.arccos(1 - indices / n_points)  # Polar angle (latitude)
        theta = np.pi * (1 + 5**0.5) * indices  # Azimuthal angle (longitude)

        positions = [
            [np.sin(phi[i]) * np.cos(theta[i]) * radius,
             np.sin(phi[i]) * np.sin(theta[i]) * radius,
             np.cos(phi[i]) * radius]
            for i in range(n_points) if np.cos(phi[i]) >= 0
        ]

        return cls.from_positions(positions)

    def __getitem__(self, index):
        return self.observers[index]

    def __iter__(self):
        return iter(self.observers)

    def __len__(self):
        return len(self.observers)

    def plot_observer_positions(self):
        """
        Plots observer positions in 3D space.
        """
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

        for observer in self.observers:
            pos = observer()
            ax.scatter(pos[0], pos[1], pos[2], color='r', s=50)
            ax.text(pos[0], pos[1], pos[2], f'{observer}', size=10, zorder=1)

        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')
        plt.show()


