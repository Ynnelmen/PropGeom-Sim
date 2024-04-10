import cadquery as cq
import numpy as np
from Airfoil_Section import Airfoil_Section
# from ocp_vscode import *
# from build123d import *
from ocp_vscode import *

class Propeller():
    def __init__(self, Blade, Hub, linear_interpolation=True, ccw=True):
        self.blade = Blade
        self.hub = Hub
        
        self.linear_interpolation = linear_interpolation
        self.counterclockwise_rotation = ccw

        self.create_propeller()

    def create_propeller(self):
        self.create_transition()
        self.create_2nd_blade()
        self.merge_parts()

    def create_transition(self):
        hub_ellipse_spline = cq.Edge.makeSpline([cq.Vector(p) for p in zip(self.hub.ellipse_cord_X, self.hub.ellipse_cord_Y, self.hub.ellipse_cord_Z)])
        hub_edge = cq.Wire.assembleEdges([hub_ellipse_spline])
        hub_ellipse_spline2 = cq.Edge.makeSpline([cq.Vector(p) for p in zip(self.hub.ellipse_cord_X+0.01, self.hub.ellipse_cord_Y, self.hub.ellipse_cord_Z)])
        hub_edge2 = cq.Wire.assembleEdges([hub_ellipse_spline2])
        blade_edge1 = self.blade.spline_wire_list[0] # transtion part overlaps with airfoil for spline interpolation
        blade_edge2 = self.blade.spline_wire_list[1]
        blade_edge3 = self.blade.spline_wire_list[2]
        # show_object(hub_edge)
        # show_object(blade_edge)
        self.transition_part = cq.Solid.makeLoft([hub_edge, hub_edge2, blade_edge1, blade_edge2, blade_edge3], self.linear_interpolation)
        print("### Transition part created ###")
        show_object(self.transition_part)
        self.complete_blade = self.blade.blade_solid.union(self.transition_part)
        show_object(self.complete_blade)
        return self.transition_part
    
    def create_2nd_blade(self):
        # self.blade2 = self.blade.blade_solid.rotate((0,0,0), (0,0,1), 180)
        # self.transition2 = self.transition_part.rotate((0,0,0), (0,0,1), 180)
        self.blade2 = self.complete_blade.rotate((0,0,0), (0,0,1), 180)
        # show_object(self.transition2)
        show_object(self.blade2)
        print("### 2nd Blade created ###")

    
    def merge_parts(self):
        # self.part = cq.Compound.makeCompound([self.blade.blade_solid, self.hub.part, self.transition_part])
        # self.part1 = self.hub.part.union(self.transition_part)
        # show_object(self.part1)
        # print("### Hub and Transition merged ###")
        # self.part2 = self.part1.union(self.blade.blade_solid)
        # show_object(self.part2)
        # print("### Blade merged ###")
        # self.part3 = self.part2.union(self.transition2)
        # show_object(self.part3)
        # print("### Transition2 merged ###")
        # self.part4 = self.part3.union(self.blade2)
        # show_object(self.part4)
        # print("### Blade2 merged ###")
        # print("### Parts merged ###")
        # self.part = self.part.workplane("XY", origin=(0,0,-1)).hole(self.hub.inner_radius*2)  # remake hole
        # self.part = self.part4.faces("<Z").workplane().hole(self.hub.inner_radius*2)  # remake hole
        # try:
        #     self.part = self.part.workplane((0,0,1), origin=(0,0,-1)).hole(self.hub.inner_radius*2)  # remake hole
        # except:
        #     pass
        self.part = self.complete_blade.union(self.blade2).union(self.hub.part)
        self.part = self.part.faces("<Z").workplane().hole(self.hub.inner_radius*2)  # remake hole

        if not self.counterclockwise_rotation:
            self.part = self.part.mirror("XZ")

        print("### Propeller created ###")

        return self.part
    
