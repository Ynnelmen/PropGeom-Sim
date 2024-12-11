from Airfoil_Section import Airfoil_Section

class BEMT_Blade:
    def __init__(self, apc_reader, interpolation_points):
        self.APC_Reader = apc_reader
        self.interpolation_points = interpolation_points

        self.get_reader_data()
        self.build_airfoil_sections()

    def get_reader_data(self): # giving more immediate access to the data
        self.airfoil1 = self.APC_Reader.airfoil1
        self.airfoil2 = self.APC_Reader.airfoil2
        self.transition = self.APC_Reader.transition
        self.thickness_ratio = self.APC_Reader.thickness_ratio
        self.max_thicknesses = self.APC_Reader.geometry_data['MAX-THICK'].to_numpy()
        self.chord_length = self.APC_Reader.chord_length
        self.twist_angle = self.APC_Reader.twist_angle
        self.radial_position = self.APC_Reader.radial_position

        self.xa_trans = self.APC_Reader.xa_trans
        self.ya_trans = self.APC_Reader.ya_trans

    def build_airfoil_sections(self):
        self.airfoil_sections = []

        for i in range(len(self.chord_length)):
            airfoil = Airfoil_Section(airfoil_type1=self.airfoil1[i], airfoil_type2=self.airfoil2[i],
                                      transition=self.transition[i], thickness_ratio=self.thickness_ratio[i],
                                      n=self.interpolation_points, thickness_mode="vertically", center=False)
            airfoil._APC_cross_section_area = self.APC_Reader.cross_section_area[i]
            self.airfoil_sections.append(airfoil)

    def get_hub_radius_from_radius_and_pitch(self, radius, pitch):
        # Information derived from https://www.apcprop.com/product/ pages
        if radius < 5:
            raise ValueError('Tip radius must be greater than 5 inches') # no propeller with radius less than 5 inches available
        elif radius < 8:
            if pitch <= radius:
                hub_outer_diameter = 0.5
            else:
                hub_outer_diameter = 0.65
        elif radius < 15:
            if pitch <= radius:
                hub_outer_diameter = 0.8
            else:
                hub_outer_diameter = 0.8
        elif radius < 18:
            hub_outer_diameter = 1.0
        elif radius < 21:
            hub_outer_diameter = 1.25
        return hub_outer_diameter * 0.0254 ## converted to meters

    def export_geometry_for_analysis(self):
        distance_to_preceeding_airfoil = [self.radial_position[i] - self.radial_position[i-1] for i in range(1, len(self.radial_position))] + [0]
        self.export_data = {
            "r": [],  # radial positions
            "dr": [],  # distances to preceding airfoil
            "chord": [],  # chord lengths
            "twist": [],  # twist angles
            "airfoil": []  # airfoil objects
        }
        #asdf
        for i, airfoil in enumerate(self.airfoil_sections[:-1]):
            self.export_data["r"].append(self.radial_position[i] * 0.0254) # converted to meters
            self.export_data["dr"].append(distance_to_preceeding_airfoil[i] * 0.0254)
            self.export_data["chord"].append(self.chord_length[i] * 0.0254)
            self.export_data["twist"].append(self.twist_angle[i])
            self.export_data["airfoil"].append(airfoil)
        self.export_data["n_blades"] = self.APC_Reader.blades
        self.export_data["tip_radius"] = self.APC_Reader.radius * 0.0254
        self.export_data["hub_radius"] = self.get_hub_radius_from_radius_and_pitch(self.APC_Reader.radius, self.APC_Reader.pitch)
        return self.export_data



# import numpy as np
# Propeller_Measurement = {
#     'propeller_name': '10x7E',
#     'rpm': 3005,
#     'time': np.array([], dtype='datetime64'),
#     'temperature': np.array([]),
#     'atmospheric_pressure': np.array([]),
#     "thrust": {
#         'time': np.array([]),
#         'thrust': np.array([])
#     },
#     "sound": {
#         'time': np.array([]),
#         "channels": {
#             "1": np.array([]),
#             "2": np.array([]),
#             "3": np.array([]),
#             "4": np.array([])
#         },
#         "channel_positions": { #[x, y, z], [meter]
#             "1": [0, 0, 0],
#             "2": [0, 0, 1],
#             "3": [0, 1, 1],
#             "4": [1, 0, 1]
#         }
#     }
# }

