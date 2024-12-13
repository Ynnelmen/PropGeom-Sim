
import numpy as np

from Airfoil_Section import Airfoil_Section
from ocp_vscode import show_object
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import cadquery as cq
import copy


class Blade():
    def __init__(self, APCReader, hub, interpolation_points, linear_interpolation=True, thickness_variation=None, section_adaptation=[0.4, 1, 1.7, 0.9], E63_correction=1):
        self.coordinate_rotation_matrix = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
        self.inverse_coordinate_rotation_matrix = np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]])

        self.blade_solid = None

        self.hub_ellipses_revert_distance = -0.1  # has to be negative!

        self.hub = hub
        self.APCReader = APCReader
        self.interpolation_points = interpolation_points
        self.linear_interpolation = linear_interpolation
        self.thickness_variation = thickness_variation
        if self.thickness_variation is None:
            self.thickness_variation = [0] * len(self.APCReader.thickness_ratio)
        assert len(self.thickness_variation) == len(self.APCReader.thickness_ratio)

        if section_adaptation is not None:
            if not (len(section_adaptation) == 3 or len(section_adaptation) == 4):
                if not len(section_adaptation) == 0:
                    print("section_adaptation has to be a list of 3 or 4 values: [start, end, factor, (mid)]")
                section_adaptation = None
        self.section_adaptation = section_adaptation
        self.E63_correction = E63_correction

        self.get_reader_data()
        self.build_airfoil_sections()

    def get_reader_data(self):
        self.airfoil1 = self.APCReader.airfoil1
        self.airfoil2 = self.APCReader.airfoil2
        self.transition = self.APCReader.transition
        self.thickness_ratio = self.APCReader.thickness_ratio
        self.max_thicknesses = self.APCReader.geometry_data['MAX-THICK'].to_numpy()
        self.chord_length = self.APCReader.chord_length
        self.twist_angle = self.APCReader.twist_angle
        self.radial_position = self.APCReader.radial_position
        self.x_trans = self.APCReader.x_trans
        self.y_trans = self.APCReader.y_trans
        self.z_trans = self.APCReader.z_trans

        self.xa_trans = self.APCReader.xa_trans
        self.ya_trans = self.APCReader.ya_trans
        self.za_trans = self.APCReader.za_trans


    def build_airfoil_sections(self):
        self.airfoil_sections = []
        self.airfoils_pre_rotation = []
        self.calculated_thicknesses = []
        self.adapted_thicknesses = []
        for i in range(len(self.chord_length)):
            airfoil = Airfoil_Section(airfoil_type1=self.airfoil1[i], airfoil_type2=self.airfoil2[i],
                                      transition=self.transition[i], thickness_ratio=self.thickness_ratio[i],
                                      n=self.interpolation_points, thickness_mode="vertically", E63_correction=self.E63_correction, center=True)

            if self.section_adaptation is not None:
                if len(self.section_adaptation) == 3:
                    airfoil.scale_section_vertically(self.section_adaptation[0], self.section_adaptation[1], factor=self.section_adaptation[2])
                else:
                    airfoil.scale_section_vertically(self.section_adaptation[0], self.section_adaptation[1], factor= self.section_adaptation[2], mid_section=self.section_adaptation[3])

            if i == 0:
                self.airfoil = airfoil ## For testing

            airfoil.scale(self.chord_length[i])
            airfoil.translate([self.xa_trans[i], self.ya_trans[i]])

            # calculated_airfoil_thickness = airfoil.get_max_thickness()
            # self.calculated_thicknesses.append(calculated_airfoil_thickness)
            # airfoil.scale_across_chamber(self.max_thicknesses[i] / calculated_airfoil_thickness)
            # adapted_thickness = airfoil.get_max_thickness()
            # self.adapted_thicknesses.append(adapted_thickness)

            # airfoil.increase_thickness_across_chamber(self.thickness_variation[i])

            self.airfoils_pre_rotation.append(copy.deepcopy(airfoil))
            airfoil.rotate(-self.twist_angle[i])
            self.airfoil_sections.append(airfoil)

        # self.thickness_comparison = pd.DataFrame([self.calculated_thicknesses, self.max_thicknesses, self.adapted_thicknesses], index=["Calculated", "APCReader", "Adapted"])
        # print(self.thickness_comparison.T)

        ## move trailing edge of last airfoil to trailing edge of second last airfoil
        shift_x = self.airfoil_sections[-2].X[-1] - self.airfoil_sections[-1].X[-1]
        # shift_y = airfoil_sections[-2].Y[-1] - airfoil_sections[-1].Y[-1]
        self.airfoil_sections[-1].translate([shift_x*.85, 0]) #TODO: 0.9 and 0.99 are arbitrary values

        # Extrapolate y position of last airfoil such that lofting leads to a continuous body
        f_y = interp1d(self.radial_position[:-1], [a.Y[-1] for a in self.airfoil_sections[:-1]], kind='cubic', fill_value='extrapolate')
        y_soll = f_y(self.radial_position.iloc[-1])
        y_ist = self.airfoil_sections[-1].Y[-1]
        shift_y = y_soll - y_ist
        self.airfoil_sections[-1].translate([0, shift_y])

    def create_blade(self, export=False, show=False):
        ## Create hub ellipses for transition to Blade
        self.hub_ellipses = []
        self.hub_wires = []
        for x_dist in np.linspace(self.hub_ellipses_revert_distance, 0, 4):
            el = cq.Edge.makeSpline([cq.Vector(p) for p in zip(self.hub.ellipse_cord_X + x_dist, self.hub.ellipse_cord_Y, self.hub.ellipse_cord_Z)])
            # show_object(el)
            self.hub_ellipses.append(el)
            self.hub_wires.append(cq.Wire.assembleEdges([el]))

        self.spline_wire_list = []
        for i in self.hub_wires:
            self.spline_wire_list.append(i)

        ## add airfoil sections to spline_wire_list
        for i, rad_pos in enumerate(self.radial_position):
            self.airfoil_matrix = np.array([self.airfoil_sections[i].X, self.airfoil_sections[i].Y, -rad_pos*np.ones(len(self.airfoil_sections[i].X))]).T
            self.X, self.Y, self.Z = np.matmul(self.airfoil_matrix, self.inverse_coordinate_rotation_matrix).T

            spline_edge = cq.Edge.makeSpline([cq.Vector(p) for p in zip(self.X, self.Y, self.Z)])
            # show_object(spline_edge)
            self.spline_wire_list.append(cq.Wire.assembleEdges([spline_edge]))

        self.blade_solid = cq.Workplane().add(self.spline_wire_list).toPending().loft(ruled =self.linear_interpolation)
        self.blade_solid = self.blade_solid.faces("<X").workplane(invert=False).circle(2).extrude(self.hub_ellipses_revert_distance, combine="cut")
        print("### Blade created ###")

        if export:
            cq.exporters.export(self.blade_solid, 'blade.step')

        if show:
            show_object(self.blade_solid)
            pass

        return self.blade_solid

    def export_geometry_for_analysis(self):
        # airfoil_coordinates = [[airfoil.X, airfoil.Y] for airfoil in self.airfoil_sections]
        distance_to_preceeding_airfoil = [self.radial_position[i] - self.radial_position[i-1] for i in range(1, len(self.radial_position))] + [0]
        self.export_data = {
            "r": [],  # radial positions
            "dr": [],  # distances to preceding airfoil
            "chord": [],  # chord lengths
            "twist": [],  # twist angles
            "airfoil": []  # airfoil objects
        }
        for i, airfoil in enumerate(self.airfoils_pre_rotation[:-1]):
            self.export_data["r"].append(self.radial_position[i] * 0.0254)
            self.export_data["dr"].append(distance_to_preceeding_airfoil[i] * 0.0254)
            self.export_data["chord"].append(self.chord_length[i] * 0.0254)
            self.export_data["twist"].append(self.twist_angle[i])
            self.export_data["airfoil"].append(airfoil)
        self.export_data["#blades"] = self.APCReader.blades
        self.export_data["hub_radius"] = self.radial_position[-1]
        return self.export_data

    def export_geometry_for_BEMT_analysis(self):
        """ deprecated: Reverse engineering of Marcos version"""
        # airfoil_coordinates = [[airfoil.X, airfoil.Y] for airfoil in self.airfoil_sections]
        distance_to_preceeding_airfoil = [self.radial_position[i] - self.radial_position[i-1] for i in range(1, len(self.radial_position))] + [0]
        self.export_data = {}
        for i, airfoil in enumerate(self.airfoil_sections[:-1]):
            key = f"Airfoil section {i}"
            self.export_data[key] = [self.radial_position[i]*0.0254, distance_to_preceeding_airfoil[i]*0.0254,
                                     self.chord_length*0.0254, self.twist_angle[i], airfoil.X, airfoil.Y]
        return self.export_data

    def show_blade(self):
        if self.blade_solid is None:
            self.create_blade()
        show_object(self.blade_solid)

    def APC_comparisons(self):
        transition_start = np.where(self.APCReader.transition.to_numpy() != 0)[0][0]
        transition_end = np.where(self.APCReader.transition.to_numpy() == 1)[0][0]
        transition_mid = np.where(self.APCReader.transition.to_numpy() >= 0.5)[0][0]

        drawn_cross_section_area = [airfoil.calculate_cross_section_area() for airfoil in self.airfoils_pre_rotation]
        # cord_lengths = [airfoil.get_chord_length() for airfoil in self.airfoils_pre_rotation]
        # thicknesses = [airfoil.get_max_thickness_perpendicular_to_camber() for airfoil in self.airfoils_pre_rotation]
        # Z_high_drawn = [airfoil.get_highest_point() for airfoil in self.airfoil_sections]

        area_error = (drawn_cross_section_area/self.APCReader.cross_section_area - 1) * 100 #%
        area_error_after_transition = np.mean(area_error[int((transition_mid+transition_end*2)/3):-2])
        area_error_before_transition = np.mean(area_error[:int((transition_start*2+transition_mid)/3)])
        return area_error_before_transition, area_error_after_transition

    def comparisons_plot(self, show = True, save = False):
        transition_start = np.where(self.APCReader.transition.to_numpy() != 0)[0][0]
        transition_end = np.where(self.APCReader.transition.to_numpy() == 1)[0][0]
        transition_mid = np.where(self.APCReader.transition.to_numpy() >= 0.5)[0][0]

        drawn_cross_section_area = [airfoil.calculate_cross_section_area() for airfoil in self.airfoils_pre_rotation]
        cord_lengths = [airfoil.get_chord_length() for airfoil in self.airfoils_pre_rotation]
        thicknesses = [airfoil.get_max_thickness_perpendicular_to_camber() for airfoil in self.airfoils_pre_rotation]
        Z_high_drawn = [airfoil.get_highest_point() for airfoil in self.airfoil_sections]

        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        plt.grid()
        plt.text(x=0, y=0.52, s=f'{blade.APCReader.propeller_name}, SA: {self.section_adaptation}, E63A: {self.E63_correction}', fontsize=15)

        # First and second plots combined with secondary y-axis
        ax1 = axs[0, 0]
        ax2 = ax1.twinx()
        ax1.plot(self.APCReader.cross_section_area, label="APC")
        ax1.plot(drawn_cross_section_area, label="inspire", linestyle="--")
        ax1.plot(drawn_cross_section_area - self.APCReader.cross_section_area, label="Difference")
        ax1.set_ylabel("Cross Section Area [IN]")
        ax1.set_xlabel("Radial Position")
        ax1.set_title("Cross Section Area Comparison")
        ax1.legend(loc="center left")
        ax1.vlines(transition_start, ymin=min(drawn_cross_section_area), ymax=max(drawn_cross_section_area), color='black', linestyle='--')
        ax1.text(transition_start - 1.2, min(drawn_cross_section_area), self.airfoil1[0], rotation=90)
        ax1.vlines(transition_mid, ymin=min(drawn_cross_section_area), ymax=max(drawn_cross_section_area), color='gray', linestyle='--')
        ax1.text(transition_mid - 1.2, min(drawn_cross_section_area), "50%", rotation=90)
        ax1.vlines(transition_end, ymin=min(drawn_cross_section_area), ymax=max(drawn_cross_section_area), color='black', linestyle='--')
        ax1.text(transition_end + 0.3, min(drawn_cross_section_area), self.airfoil2[0], rotation=90)

        area_error = (drawn_cross_section_area[:-1] / self.APCReader.cross_section_area[:-1] - 1) * 100
        ax2.plot(area_error, color='tab:red',
                 label="Difference [%]")
        ax2.set_ylabel("Difference [%]", color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')
        ax2.legend(loc="upper right")

        # Second subplot
        axs[0, 1].plot(self.APCReader.geometry_data['CHORD'].to_numpy(), label="APC")
        axs[0, 1].plot(cord_lengths, label="inspire", linestyle="--")
        axs[0, 1].plot(self.APCReader.thickness_ratio, label="Thickness Ratio", linestyle="--")
        axs[0, 1].legend()
        axs[0, 1].set_ylabel("Chord Length [IN]")
        axs[0, 1].set_xlabel("Radial Position")
        axs[0, 1].set_title("Chord Length Comparison")

        # Third subplot
        ax3 = axs[1, 0].twinx()
        axs[1, 0].plot(self.APCReader.geometry_data['MAX-THICK'].to_numpy()[:-1], label="APC")
        axs[1, 0].plot(thicknesses[:-1], label="inspire", linestyle="--")
        axs[1, 0].legend()
        axs[1, 0].set_ylabel("Max Thickness [IN]")
        axs[1, 0].set_xlabel("Radial Position")
        axs[1, 0].set_title("Max Thickness Comparison")

        ax3.plot((np.array(thicknesses[:-1]) / self.APCReader.geometry_data['MAX-THICK'].to_numpy()[:-1] - 1) * 100, color='tab:red',
                    label="Difference [%]", linestyle="--")
        ax3.set_ylabel("Difference [%]", color='tab:red')
        ax3.tick_params(axis='y', labelcolor='tab:red')
        ax3.legend(loc="center right")

        # Fourth subplot
        ax4 = axs[1, 1].twinx()
        axs[1, 1].plot(self.APCReader.geometry_data['ZHIGH'].to_numpy(), label="APC")
        axs[1, 1].plot(Z_high_drawn, label="inspire", linestyle="--")
        axs[1, 1].plot(self.APCReader.geometry_data['CGZ'].to_numpy(), label="CGY_APC")
        axs[1, 1].legend()
        axs[1, 1].set_ylabel("Z High [IN]")
        axs[1, 1].set_xlabel("Radial Position")
        axs[1, 1].set_title("Z High Comparison")
        axs[1, 1].vlines(transition_start, ymin=min(Z_high_drawn), ymax=max(Z_high_drawn), color='black', linestyle='--')
        axs[1, 1].text(transition_start - 1.2, min(Z_high_drawn), self.airfoil1[0], rotation=90)
        axs[1, 1].vlines(transition_mid, ymin=min(Z_high_drawn), ymax=max(Z_high_drawn), color='gray', linestyle='--')
        axs[1, 1].text(transition_mid - 1.2, min(Z_high_drawn), "50%", rotation=90)
        axs[1, 1].vlines(transition_end, ymin=min(Z_high_drawn), ymax=max(Z_high_drawn), color='black', linestyle='--')
        axs[1, 1].text(transition_end + 0.3, min(Z_high_drawn), self.airfoil2[0], rotation=90)

        diff = (np.array(Z_high_drawn) / self.APCReader.geometry_data['ZHIGH'].to_numpy() - 1) * 100
        diff = [d if abs > 0.01 or abs < -0.01 else 0 for d, abs in zip(diff, self.APCReader.geometry_data['ZHIGH'].to_numpy())]
        ax4.plot(diff, color='tab:red',
                    label="Difference [%]", linestyle="--")
        ax4.set_ylabel("Difference [%]", color='tab:red')
        ax4.tick_params(axis='y', labelcolor='tab:red')
        ax4.legend(loc="center right")

        plt.tight_layout()

        if save:
            plt.savefig(r"C:\Users\RhinerLenny\OneDrive - inspire AG\BAZL\EvaluationPICs\Comparison_" + self.APCReader.propeller_name + ".png")
        if show:
            plt.show()
        else:
            plt.close()

        area_error_after_transition = np.mean(area_error[int((transition_mid+transition_end)/2):-2])
        area_error_before_transition = np.mean(area_error[:int((transition_start+transition_mid)/2)])
        return area_error_before_transition, area_error_after_transition




if __name__ == "__main__":
    from APC_Reader import APC_Reader
    from Hub import Hub
    import os

    interpolation_points = 100
    hub = Hub(interpolation_points * 2 - 1, 0.65 / 2, 0.15, 0.36)
    apcreader = APC_Reader(os.getcwd() + r"\APC Propeller Geometry Data\10x8E-PERF.PE0")
    blade = Blade(apcreader, hub, interpolation_points, linear_interpolation=False, E63_correction=1, section_adaptation=None)
    # blade.create_blade(show=True)
    # blade.export_geometry_for_analysis()
    blade.comparisons_plot()
    # self = blade.airfoil_sections[1]
    # blade.airfoil_sections[5].plot()
    # blade.airfoil_sections[30].plot()
    # blade.airfoils_pre_rotation[1].get_max_thickness_perpendicular_to_camber(plot=True)

    # plt.figure()
    # plt.plot(self.X, self.Y)
    # plt.plot(self.x_camber, self.y_camber)
    #
    # # equal aspect ratio
    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.show()
    self = blade

