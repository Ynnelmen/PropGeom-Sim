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
    def from_iso3744(cls, radius=None):
        default_radius = np.linalg.norm([0.336, -2.016, 0.462])
        if radius is None:
            radius = default_radius

        iso_data = np.array([
            [0.336, -2.016, 0.462],
            [1.638, -1.260, 0.420],
            [1.638, 1.155, 0.651],
            [0.336, 1.890, 0.861],
            [-1.743, 0.672, 0.945],
            [-1.743, -0.840, 0.798],
            [-0.546, -1.365, 1.491],
            [1.554, -0.147, 1.407],
            [-0.546, 1.050, 1.743],
            [0.210, -0.210, 2.079]
        ]) * (radius / default_radius)
        return cls.from_positions(iso_data)

    @classmethod
    def from_iso3745(cls, radius=None):
        default_radius = np.linalg.norm([-1.887, 0, 0.844444444-0.75])
        if radius is None:
            radius = default_radius
        iso_data = np.array([
            [-1.887, 0, 0.844444444-0.75],
            [0.933111111, -1.616888889, 1.033333333-0.75],
            [0.914222222, 1.584777778, 1.222222222-0.75],
            [-0.884, 1.531888889, 1.411111111-0.75],
            [-0.844333333, -1.460111111, 1.6-0.75],
            [1.577222222, 0, 1.788888889-0.75],
            [0.717777778, 1.242888889, 1.977777778-0.75],
            [-1.248555556, 0, 2.166666667-0.75],
            [0.496777778, -0.861333333, 2.355555556-0.75],
            [0.589333333, 0, 2.544444444-0.75]
        ]) * (radius / default_radius)
        # rotate 90 degrees around z-axis:
        rotation_matrix = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        iso_data = np.dot(iso_data, rotation_matrix)
        return cls.from_positions(iso_data)

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

    def add_observers_to_ax(self, ax):
        for num, observer in enumerate(self.observers):
            pos = observer()
            ax.scatter(pos[0], pos[1], pos[2], color='r', s=50)
            ax.text(pos[0], pos[1], pos[2], '%d' % int(num+1), size=10, zorder=1)

    def plot_observer_positions(self):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        self.add_observers_to_ax(ax)

        # Set labels and limits
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim([-2.5, 2.5])
        ax.set_ylim([-2.5, 2.5])
        ax.set_zlim([0, 2.5])

        # Plot the hemisphere
        radius = np.linalg.norm([self.observers[0]()])  # assumption: all observers are at the same distance form the origin!!
        u = np.linspace(0, np.pi / 2, 100)  # Limit to hemisphere by using pi/2 for u
        v = np.linspace(0, 2 * np.pi, 100)

        x = np.outer(np.sin(u), np.cos(v)) * radius
        y = np.outer(np.sin(u), np.sin(v)) * radius
        z = np.outer(np.cos(u), np.ones_like(v)) * radius

        ax.plot_surface(x, y, z, color='c', alpha=0.3, rstride=5, cstride=5)

        # plot the semi-circle in the y-z plane
        u = np.linspace(0, np.pi, 100)
        v = np.linspace(0, np.pi, 100)

        ax.plot(np.zeros(100), np.cos(u)*radius, np.sin(v)*radius, color='blue', alpha=0.5, label="Blade Plane", linestyle='dashed')
        ax.text(0, -radius, 0, 'Blade Plane', color='blue', fontsize=12)


        # Plot coordinate system axes with arrows
        ax.quiver(0, 0, 0, 1, 0, 0, color='r', arrow_length_ratio=0.1, label='X-axis')
        ax.quiver(0, 0, 0, 0, 1, 0, color='g', arrow_length_ratio=0.1, label='Y-axis')
        ax.quiver(0, 0, 0, 0, 0, 1, color='b', arrow_length_ratio=0.1, label='Z-axis')

        # Add labels for the arrows at their ends
        ax.text(1, 0, 0, 'X', color='r', fontsize=12)
        ax.text(0, 1, 0, 'Y', color='g', fontsize=12)
        ax.text(0, 0, 1, 'Z', color='b', fontsize=12)

        plt.title('Receiver Positions')
        plt.show()

if __name__ == "__main__":
    observers = ObserverManager.from_iso3745()
    observers.plot_observer_positions()