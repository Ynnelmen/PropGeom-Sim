import cadquery as cq
from ocp_vscode import show_object

class Propeller():
    def __init__(self, Blade1, Hub, Blade2=None, linear_interpolation=True, ccw=True, attachment_points=False):
        self.blade1 = Blade1
        self.blade2 = Blade2
        self.hub = Hub

        if self.blade2 is None:
            self.create_2nd_blade() # duplicate blade1, rotate 180 degrees
        else:
            self.blade2 = self.blade2.rotate((0,0,0), (0,0,1), 180)
        
        self.linear_interpolation = linear_interpolation
        self.counterclockwise_rotation = ccw
        self.attachment_points = attachment_points

        self.create_propeller()

        # self.cleanup()

    def create_propeller(self):
        self.merge_parts()
    
    def create_2nd_blade(self):
        self.blade2 = self.blade1.rotate((0,0,0), (0,0,1), 180)
        print("### 2nd Blade created ###")

    
    def merge_parts(self):
        self.part = self.blade1.union(self.blade2).union(self.hub.part)
        # show_object(self.part)

        print("### Propeller created ###")
        return self.part

    def cleanup(self):
        self.part = self.part.faces(">Z").workplane(centerOption="CenterOfMass").hole(self.hub.inner_radius*2)  # remake hole
        self.part = self.part.faces("<Z").workplane(invert=True).circle(self.hub.outer_radius).extrude(self.hub.thickness/2, combine="cut") # remove "debris" above hub
        self.part = self.part.faces(">Z").workplane(invert=True).circle(self.hub.outer_radius).extrude(self.hub.thickness/2, combine="cut") # remove "debris" below hub


        if self.attachment_points:
            ap_exzenter_distance = 15/2/25.4  #radius in inch
            pin_radius = 3/2/25.4
            pin_length = self.hub.thickness+1  # throughhole

            self.part = self.part.faces("<Z").workplane(invert=True).center(ap_exzenter_distance, 0).circle(pin_radius).extrude(
                pin_length, combine="cut")
            self.part = self.part.faces("<Z").workplane(invert=True).center(-2*ap_exzenter_distance, 0).circle(pin_radius).extrude(
                pin_length, combine="cut")


        if not self.counterclockwise_rotation:
            self.part = self.part.mirror("XZ")


        print("### Cleanup complete ###")

        return self.part


    
