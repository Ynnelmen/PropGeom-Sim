import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

"""COMPACT F1A OBJECT"""
class F1AOutput:
    def __init__(self, t, p_m, p_d):
        self.t = t
        self.p_m = p_m
        self.p_d = p_d

"""COMPACT SOURCE ELEMENT FUNCTIONALITY"""
class CompactSourceElement:
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
    def from_params(cls, rho, a_inf, r, blade_angle, COM_x, COM_y, dr, area, dT, dR, dQ, tau):
        y0dot = np.array([COM_y, r * np.cos(blade_angle), r * np.sin(blade_angle) + COM_x])
        y1dot = np.zeros(3)
        y2dot = np.zeros(3)
        y3dot = np.zeros(3)
        f0dot = np.array([dT, dR * np.cos(blade_angle) - dQ * np.sin(blade_angle), dR * np.sin(blade_angle) + dQ * np.cos(blade_angle)])
        f1dot = np.zeros(3)
        return cls(rho, a_inf, dr, area, y0dot, y1dot, y2dot, y3dot, f0dot, f1dot, tau)

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

    
    def time_to_observer(self, observer):
        r = np.linalg.norm(observer() - self.y0d)
        t = self.tau + r / self.a_inf
        return t
    
    def f1a_calculation(self, compact_elements, observer, observer_time):
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
        v_vec_1d = compact_elements.y2d
        M1d = 1/compact_elements.a_inf * np.dot(v_vec_0d, v_vec_1d) / (np.linalg.norm(v_vec_0d))
        r_hat_1d = -compact_elements.a_inf/r0d * (M_vec_0d - Mr_0d*r_hat_0d)
        Mr_1d = 1/compact_elements.a_inf * np.dot(v_vec_1d, r_hat_0d) + compact_elements.a_inf/r0d * (Mr_0d**2 - M0d**2)
        R_m1m2_1d = lambda m1, m2: (1/compact_elements.a_inf * np.dot(v_vec_1d, r_hat_0d) * m2 * R_m1m2_0d(m1, m2+1) +
                                    compact_elements.a_inf * (m1) * Mr_0d * R_m1m2_0d(m1+1, m2)+
                                    -m2 * compact_elements.a_inf * R_m1m2_0d(m1+1,m2+1)*(M0d**2-Mr_0d**2))

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

    def __repr__(self):
        return (f"CompactSourceElement(rho={self.rho}, a_inf={self.a_inf}, dr={self.dr}, area={self.area}, "
                f"y0d={self.y0d}, y1d={self.y1d}, y2d={self.y2d}, y3d={self.y3d}, "
                f"f0d={self.f0d}, f1d={self.f1d}, tau={self.tau})")

"""COMBINE ACOUSTIC PRESSURE"""
class AcousticObserver:
    p_ref = 2 * 10 ** (-5)
    def __init__(self, position_vector):
        self.position_vector = np.array(position_vector)
        self.t = None
        self.p_m = None
        self.p_d = None
        self.p_tot = None
        self.frequency = None
        self.fft_pressure_amplitue = None

    def __call__(self):
        return self.position_vector

    def __calculate_common_obs_time(self, f1a_output, time_range, n_common_time_steps):
        shape = f1a_output.shape
        t_obs = np.array([f1a_output[i, j, k].t for i in range(shape[0]) for j in range(shape[1]) for k in
                            range(shape[2])]).reshape(shape)
        p_m = np.array([f1a_output[i, j, k].p_m for i in range(shape[0]) for j in range(shape[1]) for k in
                        range(shape[2])]).reshape(shape)
        p_d = np.array([f1a_output[i, j, k].p_d for i in range(shape[0]) for j in range(shape[1]) for k in
                        range(shape[2])]).reshape(shape)

        n_source_times = shape[0]
        n_sections = shape[1]
        n_blades = shape[2]

        time_array = np.empty((n_source_times, n_sections * n_blades))  # receiver times for each source element in columns
        pressure_array_m = np.empty(
            (n_source_times, n_sections * n_blades))  # monopole pressure for each source element in columns
        pressure_array_d = np.empty(
            (n_source_times, n_sections * n_blades))  # dipole pressure for each source element in columns
        for j in range(n_sections):
            for k in range(n_blades):
                time_array[:, 2 * j + k] = t_obs[:, j, k]
                pressure_array_m[:, 2 * j + k] = p_m[:, j, k]
                pressure_array_d[:, 2 * j + k] = p_d[:, j, k]

        t_common_start = np.max(time_array[0, :])
        dt = time_range / n_common_time_steps
        t_common = t_common_start + np.arange(n_common_time_steps) * dt
        return t_common, time_array, pressure_array_m, pressure_array_d
    
    def combine_source_elements(self, f1a_output, time_range, n_common_time_steps):

        t_common, time_arr, pressure_arr_m, pressure_arr_d = self.__calculate_common_obs_time(f1a_output, time_range, n_common_time_steps)

        p_m_interp = np.zeros_like(t_common)
        p_d_interp = np.zeros_like(t_common)
        n_sources_tot = time_arr.shape[1]
        for source in range(0, n_sources_tot):
            spl_1 = InterpolatedUnivariateSpline(time_arr[:, source], pressure_arr_m[:, source], k=3)
            p_m_interp += spl_1(t_common)
            spl_2 = InterpolatedUnivariateSpline(time_arr[:, source], pressure_arr_d[:, source], k=3)
            p_d_interp += spl_2(t_common)

        self.t = t_common
        self.p_m = p_m_interp
        self.p_d = p_d_interp
        p_tot = p_m_interp+p_d_interp
        self.p_tot = p_tot - np.mean(p_tot)
        #####
        self._compute_pressure_amplitude()
        self._compute_SPL_Spectrum()
        self._compute_SPLA_Spectrum()
        self._compute_OSPL()
        self._compute_OASPL()


    def _compute_pressure_amplitude(self):
        dt = self.t[1] - self.t[0]
        n_points = len(self.t)
        self.frequency = np.fft.rfftfreq(n_points, dt)
        self.fft_pressure = np.fft.rfft(self.p_tot)
        self.fft_pressure_amplitude = np.absolute(self.fft_pressure * np.sqrt(2)) / n_points # TODO check sqrt!!

    def _compute_SPL_Spectrum(self):
        if self.fft_pressure_amplitude is None:
            self._compute_pressure_amplitude()
        self.SPL = 20 * np.log10(self.fft_pressure_amplitude / AcousticObserver.p_ref)

    def _compute_SPLA_Spectrum(self):
        self._compute_SPL_Spectrum()
        R_a = lambda f: (12194 ** 2 * f ** 4) / (
                    (f ** 2 + 20.6 ** 2) * np.sqrt((f ** 2 + 107.7 ** 2) * (f ** 2 + 737.9 ** 2)) * (
                        f ** 2 + 12194 ** 2))
        A_weight = lambda f: 20 * np.log10(R_a(f)) - 20 * np.log10(R_a(1000))
        weight = np.array([A_weight(f) for f in self.frequency])
        self.SPL_A = self.SPL + weight

    def _compute_OSPL(self):
        self._compute_SPL_Spectrum()
        pressure_amplitude = 10 ** (1 / 20 * self.SPL) * AcousticObserver.p_ref
        p_rms = np.sqrt(pressure_amplitude[0] ** 2 + 2 * np.sum(pressure_amplitude[1:] ** 2))
        self.OSPL = 20 * np.log10(p_rms / AcousticObserver.p_ref)

    def _compute_OASPL(self):
        self._compute_SPLA_Spectrum()
        pressure_amplitude = 10 ** (1 / 20 * self.SPL_A) * AcousticObserver.p_ref
        p_rms = np.sqrt(pressure_amplitude[0] ** 2 + 2 * np.sum(pressure_amplitude[1:] ** 2))
        self.OASPL = 20 * np.log10(p_rms / AcousticObserver.p_ref)

