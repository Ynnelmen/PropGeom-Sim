import cadquery as cq
import numpy as np
from Airfoil_Section import Airfoil_Section
from ocp_vscode import show_object


class Blade():
    def __init__(self, APCReader, hub, interpolation_points, linear_interpolation=True):
        self.coordinate_rotation_matrix = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
        self.inverse_coordinate_rotation_matrix = np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]])
        
        self.blade_solid = None

        self.hub_ellipses_revert_distance = -0.1  # negative!

        self.hub = hub
        self.APCReader = APCReader
        self.interpolation_points = interpolation_points
        self.linear_interpolation = linear_interpolation
        self.get_reader_data()

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
        

    def create_blade(self, export=False, show=False):
        # hub_ellipse_spline0 = cq.Edge.makeSpline(
        #     [cq.Vector(p) for p in zip(self.hub.ellipse_cord_X-, self.hub.ellipse_cord_Y, self.hub.ellipse_cord_Z)])
        # hub_ellipse_spline = cq.Edge.makeSpline([cq.Vector(p) for p in zip(self.hub.ellipse_cord_X, self.hub.ellipse_cord_Y, self.hub.ellipse_cord_Z)])
        # hub_ellipse_spline2 = cq.Edge.makeSpline([cq.Vector(p) for p in
        #                                           zip(self.hub.ellipse_cord_X+0.1, self.hub.ellipse_cord_Y,
        #                                               self.hub.ellipse_cord_Z)])
        self.hub_ellipses = []
        self.hub_wires = []
        for x_dist in np.linspace(self.hub_ellipses_revert_distance, 0, 4):
            el = cq.Edge.makeSpline([cq.Vector(p) for p in zip(self.hub.ellipse_cord_X + x_dist, self.hub.ellipse_cord_Y, self.hub.ellipse_cord_Z)])
            # show_object(el)
            self.hub_ellipses.append(el)
            self.hub_wires.append(cq.Wire.assembleEdges([el]))

        # self.t = cq.Workplane().add(self.hub_wires).toPending().loft(ruled=self.linear_interpolation)
        # show_object(self.t)

        airfoil_sections = []
        for i in range(len(self.chord_length)):
            airfoil = Airfoil_Section(airfoil_type1=self.airfoil1[i], airfoil_type2=self.airfoil2[i],
                                      transition=self.transition[i], thickness_ratio=self.thickness_ratio[i],
                                      n=self.interpolation_points)
            # airfoil.plot_airfoil(show=False, save=True, filename="airfoil_" + str(i) + ".png")

            airfoil.scale(self.chord_length[i])
            airfoil.rotate(-self.twist_angle[i])
            airfoil.translate([self.xa_trans[i], self.ya_trans[i]])
            airfoil_sections.append(airfoil)

            # airfoil.plot_airfoil(show=False, save=True, filename="airfoil_scaled_" + str(i) + ".png")  ## For testing
        self.airfoil = airfoil ## For testing

        ## move trailing edge of last airfoil to trailing edge of second last airfoil
        shift_x = airfoil_sections[-2].X[-1] - airfoil_sections[-1].X[-1]
        shift_y = airfoil_sections[-2].Y[-1] - airfoil_sections[-1].Y[-1]
        airfoil_sections[-1].translate([shift_x*0.9, shift_y*0.99]) #TODO: 0.9 and 0.99 are arbitrary values
        # sharpen last airfoil to a point
        # airfoil_sections[-1].X = np.ones(len(airfoil_sections[-1].X)) * shift_x*0.9
        # airfoil_sections[-1].Y = np.ones(len(airfoil_sections[-1].Y)) * shift_y*0.99

        # remove first airfoil from list, such that it is not duplicated with the transition part
        airfoil_sections = airfoil_sections[1:]

        self.spline_wire_list = []
        for i in self.hub_wires:
            self.spline_wire_list.append(i)

        # 3d plot
        # import matplotlib.pyplot as plt
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')

        for i, rad_pos in enumerate(self.radial_position[:-1]):
            self.airfoil_matrix = np.array([airfoil_sections[i].X, airfoil_sections[i].Y, -rad_pos*np.ones(len(airfoil_sections[i].X))]).T
            self.X, self.Y, self.Z = np.matmul(self.airfoil_matrix, self.inverse_coordinate_rotation_matrix).T
            # print(self.X, self.Y, self.Z)
            # print("Shape: ", self.X.shape, self.Y.shape, self.Z.shape)

            # ax.plot(self.X, self.Y, self.Z)

            spline_edge = cq.Edge.makeSpline([cq.Vector(p) for p in zip(self.X, self.Y, self.Z)])
            # show_object(spline_edge)
            self.spline_wire_list.append(cq.Wire.assembleEdges([spline_edge]))

        # plt.show()

        # self.blade_solid = cq.Solid.makeLoft(self.spline_wire_list, self.linear_interpolation)
        self.blade_solid = cq.Workplane().add(self.spline_wire_list).toPending().loft(ruled =self.linear_interpolation)
        self.blade_solid = self.blade_solid.faces("<X").workplane(invert=False).circle(2).extrude(self.hub_ellipses_revert_distance, combine="cut")
        print("### Blade created ###")

        

        if export:
            cq.exporters.export(self.blade_solid, 'blade.step')

        if show:
            show_object(self.blade_solid)
            pass

        return self.blade_solid

    def show_blade(self):
        if self.blade_solid is None:
            self.create_blade()
        show_object(self.blade_solid)
