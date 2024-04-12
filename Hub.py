import cadquery as cq
import numpy as np
from Airfoil_Section import Airfoil_Section
# from ocp_vscode import *
# from build123d import *

class Hub():
    def __init__(self, interpolation_points, outer_radius, inner_radius, thickness):
        self.outer_radius = outer_radius
        self.inner_radius = inner_radius
        self.thickness = thickness
        z_offset = 0 #-thickness*0.6
        y_offset = 0

        x_offset_ellipse = 0 #outer_radius*0.2
        z_offset_ellipse = -thickness*0.05
        y_scaler_ellipse = 0.99
        z_scaler_ellipse = 0.98
        #Define Wires
        outer_circle = cq.Wire.makeCircle(radius=self.outer_radius, center=cq.Vector(0,y_offset,-self.thickness/2), normal=cq.Vector(0,0,1))
        inner_circle = cq.Wire.makeCircle(radius=self.inner_radius, center=cq.Vector(0,y_offset,-self.thickness/2), normal=cq.Vector(0,0,1))
        #Create Solid Object from Wires
        base_solid = cq.Solid.extrudeLinear(outer_circle, [inner_circle], cq.Vector(0,0,self.thickness))

        part = cq.Workplane(inPlane='XY', origin=((0,0,-self.thickness/2+z_offset)))
        # part = cq.Workplane(inPlane='XY', origin=((0,0,-thickness)))
        part = part.circle(self.outer_radius)
        # part = part.circle(self.inner_radius)
        part = part.extrude(self.thickness)

        # part = part.copyWorkplane(cq.Workplane("YZ", origin=(outer_radius+0.00001,0,0)))
        # part = part.ellipse(outer_radius*0.9, thickness*0.49)
        # part = part.extrude(until='next', combine=True)

        self.ellipse_cord_X = np.ones(interpolation_points)*self.outer_radius*0 + x_offset_ellipse
        self.ellipse_cord_Z = np.sin(np.linspace(0,2*np.pi,interpolation_points))*thickness*0.5*z_scaler_ellipse + z_offset + z_offset_ellipse
        self.ellipse_cord_Y = -(np.cos(np.linspace(0,2*np.pi,interpolation_points))*outer_radius*y_scaler_ellipse + y_offset)
        

        # show(part)
        self.part = part