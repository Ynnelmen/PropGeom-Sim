import numpy as np

class Hub():
    def __init__(self, interpolation_points, outer_radius, inner_radius, thickness):
        self.outer_radius = outer_radius
        self.inner_radius = inner_radius
        self.thickness = thickness
        self.z_offset = -1.1 /25.4
        self.y_offset = 0
        x_offset_ellipse = 0 #outer_radius*0.2
        z_offset_ellipse = -thickness*0.12
        y_scaler_ellipse = 0.99
        z_scaler_ellipse = 0.95

        self.ellipse_cord_X = np.ones(interpolation_points)*self.outer_radius*0 + x_offset_ellipse
        self.ellipse_cord_Z = np.sin(np.linspace(0,2*np.pi,interpolation_points))*thickness*0.5*z_scaler_ellipse + self.z_offset + z_offset_ellipse
        self.ellipse_cord_Y = -(np.cos(np.linspace(0,2*np.pi,interpolation_points))*outer_radius*y_scaler_ellipse + self.y_offset)

    def create_hub_outline(self):
        # The created hub is thrice as thick and will be refined in Propeller.cleanup()
        import cadquery as cq # import now to allow for testing without cadquery

        part = cq.Workplane(inPlane='XY', origin=((0,0,-self.thickness+self.z_offset)))
        part = part.circle(self.outer_radius)
        part = part.extrude(self.thickness*3)

        # show_object(part)
        self.part = part


    def create_hub_geometry(self):
        # for evaluation of the hub only. Not used in the final propeller
        import cadquery as cq
        outer_circle = cq.Wire.makeCircle(radius=self.outer_radius, center=cq.Vector(0,self.y_offset,-self.thickness/2), normal=cq.Vector(0,0,1))
        inner_circle = cq.Wire.makeCircle(radius=self.inner_radius, center=cq.Vector(0,self.y_offset,-self.thickness/2), normal=cq.Vector(0,0,1))
        #Create Solid Object from Wires
        base_solid = cq.Solid.extrudeLinear(outer_circle, [inner_circle], cq.Vector(0,0,self.thickness))
        part = cq.Workplane(inPlane='XY', origin=((0,0,-self.thickness/2+self.z_offset)))
        # show_object(part)
        self.part = part