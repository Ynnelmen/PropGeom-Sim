#General packages
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import griddata
import warnings
warnings.filterwarnings("ignore")

#Geometry creation
from APC_Reader import APC_Reader
from BEMT_Blade import BEMT_Blade

#BEMT Analysis
from BEMT_Solver import PropellerParameters, PropellerAnalysis

#Acoustic Analysis
from Acoustic_Solver import CompactSourceElement
from Observer_Manager import ObserverManager

class Job:
    interpolation_points = 200
    def __init__(self, propeller_name="10x7E", RPM=5000, v_inf=0, revolutions=6, observer_manager=None):
        self.propeller_name = propeller_name.upper()
        self._base_dir = os.path.dirname(os.path.abspath(__file__))

        # --- GEOMETRY ---
        # Geometrical propeller parameters and export for analysis
        self.apc_reader = APC_Reader(self._base_dir + fr"\APC Propeller Geometry Data\{propeller_name}-PERF.PE0")
        self.blade = BEMT_Blade(self.apc_reader, self.interpolation_points)
        self.bemt_input = self.blade.export_geometry_for_analysis()

        # Global Propeller Parameters
        self.prop_radius = self.bemt_input['tip_radius']
        self.hub_radius = self.bemt_input['hub_radius']
        self.n_blades = self.bemt_input['n_blades']
        self.revolutions = revolutions

        # Fluid Parameters
        self.rho = 1.225
        self.mu = 1.81e-5
        self.a_inf = 343

        # Operating conditions
        self.RPM = int(RPM)
        self.v_inf = v_inf
        self.omega = 2 * np.pi * self.RPM / 60  # Angular velocity in rad/s

        # Results
        self.total_thrust = None
        self.total_torque = None
        self.Cp = None
        self.Ct = None
        self.observer_list = []

        # Observer preparation
        if observer_manager is None:
            r_observer = [[0, 1.8, 0],
                          [1.8, 0, 0],
                          [0, 0, 1.8]]
            self.observer_manager = ObserverManager().from_positions(r_observer)
        else:
            self.observer_manager = observer_manager

    def run_BEMT(self):
        print(f"Running BEMT for propeller {self.propeller_name} at {self.RPM} RPM and {self.v_inf} m/s...")

        #Create geometry object for BEMT
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

        # Create the analysis object
        self.bemt_analysis = PropellerAnalysis(
            propeller_geometry=self.bemt_input,
            propeller_params=self.propeller_params
        )

        # Run BEMT
        n_jobs = 8
        self.bemt_analysis.run_BEMT(n_jobs=n_jobs)

        # Compute total thrust, torque, CT and CP
        self.total_thrust, self.total_torque, self.Ct, self.Cp = self.bemt_analysis.compute_total_forces()

    def run_acoustic_analysis(self):
        if self.total_thrust is None:
            self.run_BEMT()

        #Temporal discretization in observer coordinate system
        self.revolutions = self.revolutions
        duration = self.revolutions * (2*np.pi/self.omega) 
        blade_passing_period = duration / self.revolutions / self.n_blades
        observer_time_range = self.revolutions*blade_passing_period
        num_obs_times = 50*self.revolutions

        #Temporal discretization in source coordinate system
        n_source_times = 2*num_obs_times
        dt = duration/(n_source_times-1)
        src_times = np.arange(0,n_source_times)*dt

        #Prepare data for acoustic analysis
        radial_section = self.bemt_input['r']
        n_sections = len(radial_section)
        dr = self.bemt_input['dr']
        blade_angles = 2*np.pi/self.n_blades * np.arange(0, self.n_blades)
        airfoil_area = []
        COM_x = []
        COM_y = []
        for i in range(len(self.bemt_input['airfoil'])):
            #airfoil_area.append(BEMT_input['airfoil'][i]._APC_cross_section_area * 0.0254**2)
            airfoil_area.append(self.bemt_input['airfoil'][i].calculate_cross_section_area() * self.bemt_input['chord'][i]**2)
            COM_x.append(self.bemt_input['COM_shift'][i][0])
            COM_y.append(self.bemt_input['COM_shift'][i][1])
        dT = self.bemt_analysis.solution_data['dT'] / self.bemt_input['dr'] / self.n_blades
        dQ = self.bemt_analysis.solution_data['dQ'] / self.bemt_input['dr'] / self.bemt_input['r'] / self.n_blades
        dR = np.zeros_like(dT)

        for observer in self.observer_manager:

            #Construct compact source elements, apply coordinate transformation from \nu to y-frame, compute observer times for all compact elements and perform f1a calculation
            compact_source_elements = np.empty((n_source_times, n_sections, self.n_blades), dtype=object)
            observer_time = np.empty((n_source_times, n_sections, self.n_blades), dtype=float)
            F1A_output = np.empty((n_source_times, n_sections, self.n_blades), dtype=object)

            for i in range(n_source_times):
                for j in range(n_sections):
                    for k in range(self.n_blades):
                        #compact source element
                        element = CompactSourceElement.from_params(
                            self.rho, self.a_inf, radial_section[j], blade_angles[k], COM_x[j], COM_y[j], dr[j], airfoil_area[j] , -dT[j] , dR[j], dQ[j], src_times[i]
                        )
                        compact_source_elements[i,j,k] = element.coordinate_transform(omega=self.omega, v_inf=self.v_inf)

                        #observer time
                        obs_time = element.time_to_observer(observer)
                        observer_time[i,j,k] = obs_time

                        #f1a calculation
                        f1a_output = element.f1a_calculation(element, observer, obs_time)
                        F1A_output[i,j,k] = f1a_output

            #combine the source elements at the observer position
            observer.combine_source_elements(f1a_output=F1A_output, time_range=observer_time_range, n_common_time_steps=num_obs_times)


"""
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

"""
j = Job()
print(j._base_dir)
if __name__ == "__main__":
    OM = ObserverManager().from_iso3745()
    self = Job(observer_manager=OM, propeller_name="10x7E")
    self.run_BEMT()
    # save thrust to file
    # np.savetxt('data.txt', [self.total_thrust, self.Cp, self.Ct], delimiter=',')

    self.run_acoustic_analysis()

    # # self.plot_pressure_history_all_observers()
    # self.show_observer_positions()
    # # self.OSPL_analysis(1)
    # self.OSPL_analysis_all_observers()
    # self.plot_OSPL_surface()