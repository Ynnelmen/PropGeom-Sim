import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.interpolate import griddata
from APC_Reader import APC_Reader
from BEMT_Blade import BEMT_Blade
from BEMT_Solver import PropellerAnalysis, PropellerParameters
from Acoustic_Solver import CompactSourceElement, f1a, common_obs_time, combine_pressure_history, acousticReceiver, ObserverManager
import warnings
warnings.filterwarnings("ignore")

class Job:
    def __init__(self, name="default", description="None", propeller_name="10x7E", interpolation_points=200, blade=None, observer_manager=None, RPM=5000):
        self.name = name
        self.description = description
        self.propeller_name = propeller_name.upper()  # e.g. "10x7E"
        self.interpolation_points = interpolation_points

        self.apc_reader = APC_Reader(os.getcwd() + fr"\APC Propeller Geometry Data\{propeller_name}-PERF.PE0")
        if blade is None:
            blade = BEMT_Blade(self.apc_reader, interpolation_points)
        self.propeller_geometry = blade.export_geometry_for_analysis()

        # Geometrical Propeller Parameters
        self.prop_radius = int(self.propeller_name.split("X")[0]) / 2 * 0.0254
        self.hub_radius = 0.4 * 0.0254 # todo adapt to propeller name
        self.n_blades = 2  # todo: read from APC file
        self.blade_angles = np.linspace(0, 2 * np.pi, self.n_blades, endpoint=False)

        # Operating conditions
        self.RPM = RPM
        self.v_inf = 0
        self.omega = 2 * np.pi * self.RPM / 60  # Angular velocity in rad/s

        # Analysis parameters
        self.period = 1 / self.RPM * 60 # time to complete one revolution
        self.n_periods = 4  # number of revolutions to be evaluated
        self.n_source_times = 1000  # number of source times to be evaluated
        self.source_times = np.arange(0, self.n_source_times)*(self.period*self.n_periods/(self.n_source_times-1))

        # Fluid parameters
        self.rho = 1.225
        self.mu = 1.81e-5
        self.a_inf = 343

        # Results
        self.total_thrust = None
        self.total_torque = None
        self.Cp = None
        self.Ct = None
        self.receivers = []

        if observer_manager is None:
            self.observer_manager = ObserverManager(type="fibonacci", number_of_observers=100)
        else:
            self.observer_manager = observer_manager

        self.propeller_params = PropellerParameters(
            prop_radius=self.prop_radius,
            hub_radius=self.hub_radius,
            n_blades=self.n_blades,
            RPM=self.RPM,
            rho=self.rho,
            a_inf=self.a_inf,
            mu=self.mu,
            v_inf=self.v_inf
        )

    def run_BEMT(self):
        print(f"Running BEMT for propeller {self.propeller_name}...")
        # Create the analysis object
        self.analysis = PropellerAnalysis(
            propeller_geometry=self.propeller_geometry,
            propeller_params=self.propeller_params
        )

        # Run BEMT
        n_jobs = 12
        self.analysis.run_BEMT(n_jobs=n_jobs)

        # Compute total thrust, torque, CT and CP
        self.total_thrust, self.total_torque, self.Ct, self.Cp = self.analysis.compute_total_forces()
        print(f"Total thrust: {self.total_thrust} N")

    def run_acoustic_analysis(self):
        if self.total_thrust is None:
            self.run_BEMT()
        print("Running acoustic analysis...")
        self.compact_source_elements = np.empty((self.n_source_times, len(self.propeller_geometry['r']), self.n_blades), dtype=object)
        self.observer_time = np.empty((self.n_source_times, len(self.propeller_geometry['r']), self.n_blades, len(self.observer_manager)), dtype=object)
        self.f1a_output = np.empty((self.n_source_times, len(self.propeller_geometry['r']), self.n_blades, len(self.observer_manager)), dtype=object)

        self.dTdr = self.analysis.solution_data['dT'] / self.propeller_geometry['dr'] / self.n_blades
        self.dQdr = self.analysis.solution_data['dQ'] / self.propeller_geometry['dr'] / self.propeller_geometry['r'] / self.n_blades
        self.dR = np.zeros(len(self.dTdr))

        for i in range(self.n_source_times):
            for j in range(len(self.propeller_geometry['r'])):
                for k in range(self.n_blades):
                    # [rho, a_inf, r, blade_angle, dr, area, dT, dR, dQ, tau]
                    element = CompactSourceElement.from_params(
                        self.rho, self.a_inf, self.propeller_geometry['r'][j], self.blade_angles[k],
                        self.propeller_geometry['dr'][j],
                        self.propeller_geometry['airfoil'][j].calculate_cross_section_area()*0.0254**2,
                        -self.dTdr[j], self.dR[j], self.dQdr[j], self.source_times[i]
                    )
                    self.compact_source_elements[i, j, k] = element.coordinate_transform(omega=self.omega, v_inf=self.v_inf)

                    for o_nr, observer in enumerate(self.observer_manager):
                        self.observer_time[i, j, k, o_nr] = element.time_to_observer(observer)
                        self.f1a_output[i, j, k, o_nr] = f1a(self.compact_source_elements[i, j, k], observer,
                                                       self.observer_time[i, j, k, o_nr])

        obs_time_range = self.n_periods * self.period / self.n_blades

        for o_nr, observer in enumerate(self.observer_manager):
            self.t_common, self.time_matrix, self.pressure_matrix_m, self.pressure_matrix_d = common_obs_time(self.f1a_output[:, :, :, o_nr],
                                                                                                              obs_time_range,2000)
            observer.pressure_history = combine_pressure_history(self.time_matrix, self.pressure_matrix_m, self.pressure_matrix_d, self.t_common)
            receiver = acousticReceiver(observer=observer)
            self.receivers.append(receiver)

    def plot_pressure_history_single(self, observer_nr=0):
        fig, axs = plt.subplots(1, 1)
        fig.set_figheight(7)
        fig.set_figwidth(12)
        fontsize = 10

        observer = self.observer_manager[observer_nr]

        axs.plot(observer.pressure_history.t, observer.pressure_history.p_m, marker='.', label='monopole pressure')
        axs.plot(observer.pressure_history.t, observer.pressure_history.p_d, marker='.', label='dipole pressure')
        axs.plot(observer.pressure_history.t, observer.pressure_history.p_m + observer.pressure_history.p_d, marker='.', label='total pressure')
        axs.set_xlabel('time [s]', fontsize=fontsize)
        axs.set_ylabel('acoustic pressure [Pa]', fontsize=fontsize)
        axs.grid(True)
        axs.legend()
        plt.show()

    def plot_pressure_history_all_observers(self):
        fig, axs = plt.subplots(len(self.observer_manager), 1, sharex=True, sharey=True)
        fig.set_figheight(7 * len(self.observer_manager))
        fig.set_figwidth(12)
        fontsize = 10

        # Get min and max values for uniform y-axis limits
        all_pressures = [
            observer.pressure_history.p_m + observer.pressure_history.p_d
            for observer in self.observer_manager
        ]
        all_pressures_flat = [item for sublist in all_pressures for item in sublist]
        y_min, y_max = min(all_pressures_flat), max(all_pressures_flat)

        for o_nr, observer in enumerate(self.observer_manager):
            axs[o_nr].plot(observer.pressure_history.t, observer.pressure_history.p_m, marker='.',
                           label='monopole pressure')
            axs[o_nr].plot(observer.pressure_history.t, observer.pressure_history.p_d, marker='.',
                           label='dipole pressure')
            axs[o_nr].plot(observer.pressure_history.t,
                           observer.pressure_history.p_m + observer.pressure_history.p_d, marker='.',
                           label='total pressure')
            axs[o_nr].grid(True)

            # Add numbering to the subplot
            axs[o_nr].annotate(f'Receiver {o_nr + 1}', xy=(0.03, 0.6), xycoords='axes fraction', fontsize=fontsize)

        # Set uniform y-axis range for all subplots
        axs[0].set_ylim([y_min, y_max])

        axs[-1].legend()
        axs[-1].set_xlabel('time [s]', fontsize=fontsize)
        axs[0].set_ylabel('Pressure [Pa]', fontsize=fontsize)  # Single y-axis label

        plt.show()

    def show_observer_positions(self):
        self.observer_manager.plot_observer_positions()

    def OSPL_analysis_single(self, observer_nr=0):
        # observer = self.observer_manager[observer_nr]
        self.receiver = acousticReceiver(observer=self.observer_manager[observer_nr])

        fig, axs = plt.subplots(1, 2)
        fig.set_figheight(5)
        fig.set_figwidth(12)
        fontsize = 10
        labelsize = 10

        axs[0].plot(self.receiver.timeData()['Time'], self.receiver.timeData()['Pressure'][0:], marker='.',
                    label=f"OSPL: {np.round(self.receiver.OSPL_TimeDomain(), 2)} dB")
        axs[1].semilogx(self.receiver.SPL_Spectrum()['Frequency'], self.receiver.SPL_Spectrum()['SPL'], marker='.',
                        label=f"OSPL: {np.round(self.receiver.OSPL(), 2)} dB")

        axs[0].set_xlabel(r'Time [s]', fontsize=fontsize)
        axs[0].set_ylabel(r'Acoustic Pressure [Pa]', fontsize=fontsize)
        axs[0].tick_params(axis='both', labelsize=labelsize)
        axs[0].grid('on')
        axs[0].legend(loc='upper right', bbox_to_anchor=(1, 1), fontsize=labelsize)

        axs[1].set_xlabel(r'Frequency [Hz]', fontsize=fontsize)
        axs[1].set_ylabel(r'SPL [dB]', fontsize=fontsize)
        axs[1].tick_params(axis='both', labelsize=labelsize)
        axs[1].grid('on')
        axs[1].legend(loc='upper right', bbox_to_anchor=(1, 1), fontsize=labelsize)
        plt.show()

    def OSPL_analysis_all_observers(self):
        fig, axs = plt.subplots(1, 2)
        fig.set_figheight(10)
        fig.set_figwidth(12)
        fontsize = 10
        labelsize = 10

        for obs_nr, observer in enumerate(self.observer_manager):
            receiver = self.receivers[obs_nr]
            axs[0].plot(receiver.timeData()['Time'], receiver.timeData()['Pressure'][0:], marker='.',
                                label=f"OSPL: {np.round(receiver.OSPL_TimeDomain(), 2)} dB")
            axs[1].semilogx(receiver.SPL_Spectrum()['Frequency'], receiver.SPL_Spectrum()['SPL'], marker='.',
                                    label=f"OSPL Receiver {obs_nr}: {np.round(receiver.OSPL(), 2)} dB")

            axs[0].set_xlabel(r'Time [s]', fontsize=fontsize)
            axs[0].set_ylabel(r'Acoustic Pressure [Pa] \n  ' if obs_nr == 0 else '[Pa]', fontsize=fontsize)
            axs[0].tick_params(axis='both', labelsize=labelsize)
            axs[0].grid('on')
            axs[0].legend(loc='upper right', bbox_to_anchor=(1, 1), fontsize=labelsize)

            axs[1].set_xlabel(r'Frequency [Hz]', fontsize=fontsize)
            axs[1].set_ylabel(r'SPL [dB]', fontsize=fontsize)
            axs[1].tick_params(axis='both', labelsize=labelsize)
            axs[1].grid('on')
            axs[1].legend(loc='upper right', bbox_to_anchor=(1,1), fontsize=labelsize)
            axs[1].set_ylim([-100, 80])

        plt.show()

    def plot_OSPL_surface(self):
        resolution = 100
        import numpy as np
        x = [x()[0] for x in self.observer_manager]
        y = [x()[1] for x in self.observer_manager]
        z = [z()[2] for z in self.observer_manager]
        radius = np.linalg.norm(np.stack([x, y, z]), axis=0).mean()
        OSPL = [receiver.OSPL() for receiver in self.receivers]

        # Convert the lists into NumPy arrays
        x = np.array(x)
        y = np.array(y)
        z = np.array(z)
        OSPL = np.array(OSPL)

        # Create a 2D grid in the x-y plane for interpolation
        grid_x, grid_y = np.mgrid[-radius:radius:200j, -radius:radius:200j]

        # Interpolate the OSPL values onto the grid using the known 3D coordinates
        grid_OSPL = griddata((x, y), OSPL, (grid_x, grid_y), method='cubic')
        norm_OSPL = (grid_OSPL - np.nanmin(grid_OSPL)) / (np.nanmax(grid_OSPL) - np.nanmin(grid_OSPL))

        # Now, to avoid issues with invalid sqrt values for a spherical surface, we mask out non-real regions
        grid_z = np.sqrt(np.clip(radius**2 - grid_x**2 - grid_y**2, 0, None))

        # set OSPL to 0, where grid_z = 0
        # grid_OSPL = np.where(grid_z == 0, np.nan, grid_OSPL)

        # Plotting the data
        fig = plt.figure(figsize=(14, 6))

        # Subplot 1: Plot the first surface (e.g., Cylinder or Sphere)
        ax = fig.add_subplot(111, projection='3d')
        surface1 = ax.plot_surface(grid_x, grid_y, grid_z, facecolors=plt.cm.viridis(norm_OSPL),
                                    rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_zlabel('z [m]')
        mappable = plt.cm.ScalarMappable(cmap='viridis',
                                         norm=plt.Normalize(vmin=np.nanmin(grid_OSPL), vmax=np.nanmax(grid_OSPL)))
        mappable.set_array(grid_OSPL)
        plt.colorbar(mappable, ax=ax, label='OSPL [dB]')

        #plot observers
        for num, observer in enumerate(self.observer_manager):
            pos = observer()
            ax.scatter(pos[0], pos[1], pos[2], color='r', s=50)

            ax.text(pos[0], pos[1], pos[2], '%d' % int(num), size=10, zorder=1)

        plt.suptitle("3D Sound Pressure Level Representation")
        plt.show()


OM = ObserverManager(type="iso", number_of_observers=24)
self = Job(observer_manager=OM, description="None", propeller_name="10x7E")
self.run_BEMT()
# save thrust to file
# np.savetxt('data.txt', [self.total_thrust, self.Cp, self.Ct], delimiter=',')

self.run_acoustic_analysis()

# # self.plot_pressure_history_all_observers()
# self.show_observer_positions()
# # self.OSPL_analysis(1)
# self.OSPL_analysis_all_observers()
# self.plot_OSPL_surface()