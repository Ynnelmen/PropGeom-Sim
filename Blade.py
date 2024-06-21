# import cadquery as cq
import numpy as np
import pandas as pd

from Airfoil_Section import Airfoil_Section
from ocp_vscode import show_object
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


class Blade():
    def __init__(self, APCReader, hub, interpolation_points, linear_interpolation=True, thickness_variation=None):
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

        self.get_reader_data()
        self.build_airfoil_sections()

    def get_reader_data(self):
        self.airfoil1 = self.APCReader.airfoil1
        self.airfoil2 = self.APCReader.airfoil2
        self.transition = self.APCReader.transition
        self.thickness_ratio = self.APCReader.thickness_ratio
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
        for i in range(len(self.chord_length)):
            airfoil = Airfoil_Section(airfoil_type1=self.airfoil1[i], airfoil_type2=self.airfoil2[i],
                                      transition=self.transition[i], thickness_ratio=self.thickness_ratio[i],
                                      n=self.interpolation_points)

            if i == 0:
                self.airfoil = airfoil ## For testing

            airfoil.scale(self.chord_length[i])
            airfoil.translate([self.xa_trans[i], self.ya_trans[i]])
            airfoil.increase_thickness_across_chamber(self.thickness_variation[i])
            airfoil.rotate(-self.twist_angle[i])
            self.airfoil_sections.append(airfoil)

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
            show_object(el)
            self.hub_ellipses.append(el)
            self.hub_wires.append(cq.Wire.assembleEdges([el]))

        self.spline_wire_list = []
        for i in self.hub_wires:
            self.spline_wire_list.append(i)

        ## add airfoil sections to spline_wire_list
        for i, rad_pos in enumerate(self.radial_position):
            self.airfoil_matrix = np.array([airfoil_sections[i].X, airfoil_sections[i].Y, -rad_pos*np.ones(len(airfoil_sections[i].X))]).T
            self.X, self.Y, self.Z = np.matmul(self.airfoil_matrix, self.inverse_coordinate_rotation_matrix).T

            spline_edge = cq.Edge.makeSpline([cq.Vector(p) for p in zip(self.X, self.Y, self.Z)])
            show_object(spline_edge)
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
        airfoil_coordinates = [[airfoil.X, airfoil.Y] for airfoil in self.airfoil_sections]
        distance_to_preceeding_airfoil =  [self.radial_position[i] - self.radial_position[i-1] for i in range(1, len(self.radial_position))] + [0]
        self.export_data = {}
        for i, airfoil in enumerate(self.airfoil_sections):
            self.export_data[self.radial_position[i]] = [distance_to_preceeding_airfoil[i], self.twist_angle[i], airfoil.X, airfoil.Y]
        return self.export_data

    def show_blade(self):
        if self.blade_solid is None:
            self.create_blade()
        show_object(self.blade_solid)


if __name__ == "__main__":
    from APCReader import APCReader
    from Hub import Hub
    import os

    interpolation_points = 100
    hub = Hub(interpolation_points * 2 - 1, 0.65 / 2, 0.15, 0.36)
    apcreader = APCReader(os.getcwd() + r"\APC Propeller Geometry Data\10x7E-PERF.PE0")
    blade = Blade(apcreader, hub, interpolation_points, linear_interpolation=True)
    self = blade
    self.export_geometry_for_analysis()

