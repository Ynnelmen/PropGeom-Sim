import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline


class CompactSourceElement:
    """COMPACT SOURCE ELEMENT FUNCTIONALITY"""
    def __init__(self, rho, a_inf, dr, area, y0d, y1d, y2d, y3d, f0d, f1d, tau):
        self.rho = rho
        self.a_inf = a_inf
        self.dr = dr
        self.area = area
        self.y0d = y0d
        self.y1d = y1d
        self.y2d = y2d
        self.y3d = y3d
        self.f0d = f0d
        self.f1d = f1d
        self.tau = tau

    @classmethod
    def from_params(cls, rho, a_inf, r, blade_angle, dr, area, dT, dR, dQ, tau):
        y0dot = np.array([0, r * np.cos(blade_angle), r * np.sin(blade_angle)])
        y1dot = np.zeros(3)
        y2dot = np.zeros(3)
        y3dot = np.zeros(3)
        f0dot = np.array([dT, dR * np.cos(blade_angle) - dQ * np.sin(blade_angle), dR * np.sin(blade_angle) + dQ * np.cos(blade_angle)])
        f1dot = np.zeros(3)
        return cls(rho, a_inf, dr, area, y0dot, y1dot, y2dot, y3dot, f0dot, f1dot, tau)

    def transform(self, trans):
        y0d, y1d, y2d, y3d = trans(self.tau, self.y0d, self.y1d, self.y2d, self.y3d, linear_only=False)
        f0d, f1d = trans(self.tau, self.f0d, self.f1d, linear_only=True)
        return CompactSourceElement(self.rho, self.a_inf, self.dr, self.area, y0d, y1d, y2d, y3d, f0d, f1d, self.tau)

    def time_to_observer(self, observer):
        r = np.linalg.norm(observer() - self.y0d)
        t = self.tau + r / self.a_inf
        return t

    def coordinate_transform(self, omega, v_inf):
        # define rotational and translational transformations
        y0d_nu = self.y0d
        angle = omega * self.tau
        x = v_inf * self.tau

        R0d = omega**0 * np.array([[1, 0, 0],
                                   [0, np.cos(angle), -np.sin(angle)],
                                   [0, np.sin(angle), np.cos(angle)]])
        T0d = np.array([x, 0, 0])

        R1d = omega**1 * np.array([[0, 0, 0],
                                   [0, -np.sin(angle), -np.cos(angle)],
                                   [0, np.cos(angle), -np.sin(angle)]])
        T1d = np.array([v_inf, 0, 0])

        R2d = omega**2 * np.array([[0, 0, 0],
                                   [0, -np.cos(angle), np.sin(angle)],
                                   [0, -np.sin(angle), -np.cos(angle)]])
        T2d = np.zeros(3)

        R3d = omega**3 * np.array([[0, 0, 0],
                                   [0, np.sin(angle), np.cos(angle)],
                                   [0, -np.cos(angle), np.sin(angle)]])
        T3d = np.zeros(3)

        # Apply transformations to quantities
        self.y0d = R0d @ y0d_nu + T0d
        self.y1d = R1d @ y0d_nu + T1d
        self.y2d = R2d @ y0d_nu + T2d
        self.y3d = R3d @ y0d_nu + T3d

        self.f0d = R0d @ self.f0d
        self.f1d = R1d @ self.f1d

        return self

    def __repr__(self):
        return (f"CompactSourceElement(rho={self.rho}, a_inf={self.a_inf}, dr={self.dr}, area={self.area}, "
                f"y0d={self.y0d}, y1d={self.y1d}, y2d={self.y2d}, y3d={self.y3d}, "
                f"f0d={self.f0d}, f1d={self.f1d}, tau={self.tau})")


# class CSEs:
#     def __init__(self, n_source_times, n_sections, n_blades):
#         self.cse = np.empty((n_source_times, n_sections, n_blades), dtype=object)
#
#         for i in range(n_source_times):
#             for j in range(n_sections):
#                 for k in range(n_blades):
#                     self.cse[i, j, k] = CompactSourceElement.from_params(
#                         rho, a_inf, radial_section[j], blade_angles[k], dr[j], airfoil_area[j], -dT[j], dR[j], dQ[j],
#                         src_times[i]
#                     )


"""DEFINE CLASS ACOUSTIC RECEIVER"""
class acousticReceiver:
    p_ref = 2 * 10 ** (-5)  # Reference pressure according to CFD simulation

    # Init method reads raw data of a specific receiver and calculates SPL spectrum
    def __init__(self, observer, data = None):
        assert isinstance(observer, AcousticObserver), "Observer must be an instance of AcousticObserver"
        self.observer = observer
        if data is None:
            self.rawData = pd.DataFrame(np.array((observer.pressure_history.t,
                                                  observer.pressure_history.p_m +
                                                  observer.pressure_history.p_d)).T,
                                        columns=['Time', 'Pressure'])
        else:
            self.rawData = data
        self.rawData['Pressure'] = self.rawData['Pressure'] - self.rawData['Pressure'].mean() #subtract mean pressure value here
        self._fourierCharacteristics()

    # Method returns fourier characteristcs of pressure signal
    def _fourierCharacteristics(self):
        # Compute Fourier quantities
        self.dt = self.rawData.iloc[1, 0] - self.rawData.iloc[0, 0]
        self.n_points = len(self.rawData.iloc[:, 0])
        self.frequency = np.fft.rfftfreq(self.n_points, self.dt)
        self.fft_pressure = np.fft.rfft(self.rawData.iloc[:, 1])

        # Compute magnitude and phase of each complex number
        self.fft_pressure_amplitude = np.absolute(self.fft_pressure) / self.n_points
        self.fft_pressure_phase = np.arctan2(self.fft_pressure.imag, self.fft_pressure.real)

    def timeData(self):
        return self.rawData

    # Method returns SPL spectrum
    def SPL_Spectrum(self):
        # Z weigthed SPL
        self.SPL = 20 * np.log10(self.fft_pressure_amplitude / acousticReceiver.p_ref)
        # Write data in dataframe
        d = {'Frequency': self.frequency, 'SPL': self.SPL}
        SPL_spectrum = pd.DataFrame(data=d)
        return SPL_spectrum

    # Method returns A-weigthed SPL spectrum
    def SPLA_Spectrum(self):
        SPL_spectrum = self.SPL_Spectrum()
        # A weigthed SPL
        R_a = lambda f: (12194 ** 2 * f ** 4) / (
                    (f ** 2 + 20.6 ** 2) * np.sqrt((f ** 2 + 107.7 ** 2) * (f ** 2 + 737.9 ** 2)) * (
                        f ** 2 + 12194 ** 2))
        A_weight = lambda f: 20 * np.log10(R_a(f)) - 20 * np.log10(R_a(1000))
        weight = np.array([A_weight(f) for f in self.frequency])
        self.SPL_A = SPL_spectrum['SPL'] + weight
        d = {'Frequency': self.frequency, 'SPL': self.SPL_A}
        SPL_spectrum = pd.DataFrame(data=d)
        return SPL_spectrum

    # Method returns overall SPL value
    def OSPL(self):
        self.SPL_Spectrum()
        pressure_amplitude = 10 ** (1 / 20 * self.SPL) * acousticReceiver.p_ref
        p_rms = np.sqrt(pressure_amplitude[0] ** 2 + 2 * np.sum(pressure_amplitude[1:] ** 2))
        OSPL = 20 * np.log10(p_rms / acousticReceiver.p_ref)
        return OSPL

    # Method returns overall A-weighted SPL value -> Note that fluent calculates this value by only using the 5? highest values of the pressure amplitude vector!
    def OASPL(self):
        self.SPLA_Spectrum()
        pressure_amplitude = 10 ** (1 / 20 * self.SPL_A) * acousticReceiver.p_ref
        p_rms = np.sqrt(pressure_amplitude[0] ** 2 + 2 * np.sum(pressure_amplitude[1:] ** 2))
        OASPL = 20 * np.log10(p_rms / acousticReceiver.p_ref)
        return OASPL

    # Method returns overall SPL value calculated in time domain
    def OSPL_TimeDomain(self):
        time_vector = self.rawData.iloc[:, 0]
        t_start = time_vector.iloc[0]
        t_end = time_vector.iloc[-1]
        pressure_timeDomain = self.rawData.iloc[:, 1]
        T = t_end - t_start
        InterObject = InterpolatedUnivariateSpline(time_vector, pressure_timeDomain ** 2, k=3)
        p_rms_timeDomain = np.sqrt((1 / T) * InterObject.integral(t_start, t_end))
        OSPL_timeDomain = 20 * np.log10(p_rms_timeDomain / acousticReceiver.p_ref)
        return OSPL_timeDomain


    def A_weigthed_Pressure(self):
        # reconstruct pressure signal with weighted amplitude
        phase = self.fft_pressure_phase
        amplitude = 10 ** (1 / 20 * self.SPL_A) * acousticReceiver.p_ref * self.n_points
        pressure_fft_reconstructed = amplitude * np.exp(1j * phase)
        pressure_reconstructed = np.fft.irfft(pressure_fft_reconstructed)
        return pressure_reconstructed

"""ACOUSTIC OBSERVER FUNCTIONALITY"""
class AcousticObserver:
    def __init__(self, position_vector):
        self.position_vector = np.array(position_vector)
        self.pressure_history = None  #placeholder for data computed by the acoustic solver
    def __call__(self):
        return self.position_vector

class ObserverManager:
    def __init__(self, observer_positions=None, type="iso3745", number_of_observers=24, radius = 2.1):
        """ observer_positions: [[x1, y1, z1], [x2, y2, z2], ...] or accepted types: "iso",... """
        if isinstance(observer_positions, list) or isinstance(observer_positions, np.ndarray):
            if len(observer_positions[0]) == 3:
                self.observers = [AcousticObserver(pos) for pos in observer_positions]
        elif type.lower() == "iso":
            self.observers = self.get_iso3745_observers()
        elif type.lower() == "iso3744":
            self.observers = self.get_iso3744_observers()
        elif type.lower() == "iso3745":
            self.observers = self.get_iso3745_observers()
        elif type.lower() == "even":
            self.observers = self.spherical_grid_sampled_observers(number_of_observers, radius)
        elif type.lower() == "fibonacci":
            self.observers = self.fibonacci_sampled_observers(number_of_observers, radius)
        else:
            raise ValueError("Invalid observer positions")

    # helper functions such that the class can be used as a list
    def __getitem__(self, index):
        return self.observers[index]

    def __iter__(self):
        return iter(self.observers)

    def __len__(self):
        return len(self.observers)


    def fibonacci_sampled_observers(self, n_points, radius):
        indices = np.arange(0, n_points*2, dtype=float) + 0.5

        phi = np.arccos(1 - indices / n_points)  # Polar angle (latitude)
        theta = np.pi * (1 + 5 ** 0.5) * indices  # Azimuthal angle (longitude)

        x = np.sin(phi) * np.cos(theta) * radius
        y = np.sin(phi) * np.sin(theta) * radius
        z = np.cos(phi) * radius  # For the hemisphere, z is always positive

        observers = [AcousticObserver([x, y, z]) for x, y, z in zip(x, y, z) if z>=0]
        return observers

    def spherical_grid_sampled_observers(self, n_observers, radius):
        n_theta = int(np.sqrt(n_observers))
        n_phi = int(n_observers / n_theta)
        if n_theta * n_phi != n_observers:
            print(f"Grid_dimensions (theta x phi): {n_theta} x {n_phi} = {n_theta * n_phi}")

        theta = np.linspace(0, 2 * np.pi, n_theta)
        phi = np.linspace(0, np.pi / 2, n_phi)  # Hemisphere: 0 to pi/2

        theta, phi = np.meshgrid(theta, phi)

        x = np.sin(phi) * np.cos(theta) * radius
        y = np.sin(phi) * np.sin(theta) * radius
        z = np.cos(phi) * radius
        observers = []
        for i in range(10):
            for j in range(10):
                observers.append(AcousticObserver([x[i][j], y[i][j], z[i][j]]))
        return observers

    def inplane_observers(self, n_observers, radius):
        theta = np.linspace(0, 2 * np.pi, n_observers)
        phi = [np.pi / 4]

        #xy-plane
        a = np.cos(theta) * radius
        b = np.sin(theta) * radius

        # todo


    def get_iso3744_observers(self):
        # NR, x, y, z [m]
        iso_data = np.array([
            [1, 0.336, -2.016, 0.462],
            [2, 1.638, -1.260, 0.420],
            [3, 1.638, 1.155, 0.651],
            [4, 0.336, 1.890, 0.861],
            [5, -1.743, 0.672, 0.945],
            [6, -1.743, -0.840, 0.798],
            [7, -0.546, -1.365, 1.491],
            [8, 1.554, -0.147, 1.407],
            [9, -0.546, 1.050, 1.743],
            [10, 0.210, -0.210, 2.079]
        ])
        observers = [AcousticObserver([x, y, z]) for _, x, y, z in iso_data]
        return observers

    def get_iso3745_observers(self):
        # NR, x, y, z [m]
        iso_data = np.array([
            [0, -1.887, 0, 0.844444444-0.75],
            [1, 0.933111111, -1.616888889, 1.033333333-0.75],
            [2, 0.914222222, 1.584777778, 1.222222222-0.75],
            [3, -0.884, 1.531888889, 1.411111111-0.75],
            [4, -0.844333333, -1.460111111, 1.6-0.75],
            [5, 1.577222222, 0, 1.788888889-0.75],
            [6, 0.717777778, 1.242888889, 1.977777778-0.75],
            [7, -1.248555556, 0, 2.166666667-0.75],
            [8, 0.496777778, -0.861333333, 2.355555556-0.75],
            [9, 0.589333333, 0, 2.544444444-0.75]
        ])

    def add_observers_to_ax(self, ax):
        for num, observer in enumerate(self.observers):
            pos = observer()
            ax.scatter(pos[0], pos[1], pos[2], color='r', s=50)
            ax.text(pos[0], pos[1], pos[2], '%d' % int(num), size=10, zorder=1)


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
        radius = np.linalg.norm([self.observers[0]()])
        u = np.linspace(0, np.pi / 2, 100)  # Limit to hemisphere by using pi/2 for u
        v = np.linspace(0, 2 * np.pi, 100)

        x = np.outer(np.sin(u), np.cos(v)) * radius
        y = np.outer(np.sin(u), np.sin(v)) * radius
        z = np.outer(np.cos(u), np.ones_like(v)) * radius

        ax.plot_surface(x, y, z, color='c', alpha=0.3, rstride=5, cstride=5)

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


"""COMBINE ACOUSTIC PRESSURE"""
class F1APressureTimeHistory:
    def __init__(self, t_common, p_m, p_d):
        self.t = t_common
        self.p_m = p_m
        self.p_d = p_d


def common_obs_time(f1a_output_array, time_range, n_common_time_steps):
    shape = f1a_output_array.shape
    t_obs = np.array([f1a_output_array[i, j, k].t for i in range(shape[0]) for j in range(shape[1]) for k in
                      range(shape[2])]).reshape(shape)
    p_m = np.array([f1a_output_array[i, j, k].p_m for i in range(shape[0]) for j in range(shape[1]) for k in
                    range(shape[2])]).reshape(shape)
    p_d = np.array([f1a_output_array[i, j, k].p_d for i in range(shape[0]) for j in range(shape[1]) for k in
                    range(shape[2])]).reshape(shape)

    n_source_times = shape[0]
    n_sections = shape[1]
    n_blades = shape[2]

    time_matrix = np.empty((n_source_times, n_sections * n_blades))  # receiver times for each source element in columns
    pressure_matrix_m = np.empty(
        (n_source_times, n_sections * n_blades))  # monopole pressure for each source element in columns
    pressure_matrix_d = np.empty(
        (n_source_times, n_sections * n_blades))  # dipole pressure for each source element in columns
    for j in range(n_sections):
        for k in range(n_blades):
            time_matrix[:, 2 * j + k] = t_obs[:, j, k]
            pressure_matrix_m[:, 2 * j + k] = p_m[:, j, k]
            pressure_matrix_d[:, 2 * j + k] = p_d[:, j, k]

    # common starting time (max time for first recevier time of each source element)
    t_common_start = np.max(time_matrix[0, :])
    dt = time_range / n_common_time_steps
    # common time vector for all sources
    t_common = t_common_start + np.arange(n_common_time_steps) * dt
    return t_common, time_matrix, pressure_matrix_m, pressure_matrix_d


def combine_pressure_history(time_matrix, pressure_matrix_m, pressure_matrix_d, t_common):
    p_m_interp = np.zeros_like(t_common)
    p_d_interp = np.zeros_like(t_common)

    n_sources_tot = time_matrix.shape[1]
    for source in range(0, n_sources_tot):
        p_m_interp += np.interp(t_common, time_matrix[:, source], pressure_matrix_m[:, source])
        p_d_interp += np.interp(t_common, time_matrix[:, source], pressure_matrix_d[:, source])

    return F1APressureTimeHistory(t_common, p_m_interp, p_d_interp)

"""COMPACT F1A CALCULATION"""
class F1AOutput:
    def __init__(self, t, p_m, p_d):
        self.t = t
        self.p_m = p_m
        self.p_d = p_d

def f1a(compact_elements, observer, observer_time):
    observer_position = observer()

    #0th order derivatives
    r_vec_0d = observer_position - compact_elements.y0d
    r0d = np.linalg.norm(r_vec_0d)
    r_hat_0d = r_vec_0d / r0d
    v_vec_0d = compact_elements.y1d
    M_vec_0d = v_vec_0d / compact_elements.a_inf
    M0d = np.linalg.norm(v_vec_0d) / compact_elements.a_inf
    Mr_0d = np.dot(M_vec_0d, r_hat_0d)
    R_m1m2_0d = lambda m1, m2: r0d**(-m1) * (1 - Mr_0d)**(-m2)

    #1st order derivatives
    # r_vec_1d = -compact_elements.y1d
    # r1d = -np.dot(r_hat_0d, v_vec_0d)
    v_vec_1d = compact_elements.y2d
    M1d = 1/compact_elements.a_inf * np.dot(v_vec_0d, v_vec_1d) / (np.linalg.norm(v_vec_0d))
    r_hat_1d = -compact_elements.a_inf/r0d * (M_vec_0d - Mr_0d*r_hat_0d)
    Mr_1d = 1/compact_elements.a_inf * np.dot(v_vec_1d, r_hat_0d) + compact_elements.a_inf/r0d * (Mr_0d**2 - M0d**2)
    R_m1m2_1d = lambda m1, m2: (1/compact_elements.a_inf * np.dot(v_vec_1d, r_hat_0d) * m2 * R_m1m2_0d(m1, m2+1) +
                               compact_elements.a_inf * m2 * (Mr_0d - M0d**2) * R_m1m2_0d(m1+1, m2+1) +
                               compact_elements.a_inf * (m1 - m2) * Mr_0d * R_m1m2_0d(m1+1, m2))
    # R_m1m2_1d = lambda m1, m2: (1/compact_elements.a_inf * np.dot(v_vec_1d, r_hat_0d) * m2 * R_m1m2_0d(m1, m2+1) +
    #                            compact_elements.a_inf * (m1) * Mr_0d * R_m1m2_0d(m1+1, m2)+
    #                            -m2 * compact_elements.a_inf * R_m1m2_0d(m1+1,m2+1)*(M0d**2-Mr_0d**2))


    #2nd order derivatives
    v_vec_2d = compact_elements.y3d
    R_11_2d = (1/compact_elements.a_inf * (np.dot(v_vec_2d, r_hat_0d) * R_m1m2_0d(1,2) + np.dot(v_vec_1d, r_hat_1d) * R_m1m2_0d(1,2) + np.dot(v_vec_1d,r_hat_0d) * R_m1m2_1d(1,2)) +
              compact_elements.a_inf * (Mr_1d * R_m1m2_0d(2,2) + Mr_0d * R_m1m2_1d(2,2,) - 2*M0d*M1d*R_m1m2_0d(2,2) - M0d**2*R_m1m2_1d(2,2)))

    #Monopole coefficient
    C1A = R_m1m2_0d(0,2) * R_11_2d + R_m1m2_0d(0,1) * R_m1m2_1d(0,1) * R_m1m2_1d(1,1)

    # Dipole coefficients
    D1A = R_m1m2_0d(0,1)*R_m1m2_0d(1,1)*r_hat_0d
    E1A = R_m1m2_0d(0,1) * (R_m1m2_1d(1,1) * r_hat_0d + R_m1m2_0d(1,1) * r_hat_1d) + compact_elements.a_inf * R_m1m2_0d(2,1) * r_hat_0d

    # Monopole acoustic pressure
    p_m = compact_elements.rho / (4.0 * np.pi) * compact_elements.area * C1A * compact_elements.dr

    # Dipole acoustic pressure
    p_d = 1/(compact_elements.a_inf*4*np.pi) * (np.dot(compact_elements.f1d, D1A) * compact_elements.dr + np.dot(compact_elements.f0d, E1A) * compact_elements.dr)

    return F1AOutput(observer_time, p_m, p_d)