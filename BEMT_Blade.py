import numpy as np
import pandas as pd
from Airfoil_Section import Airfoil_Section

class BEMT_Blade:
    def __init__(self, APCReader, interpolation_points):
        self.APCReader = APCReader
        self.interpolation_points = interpolation_points

        self.get_reader_data()
        self.build_airfoil_sections()

    def get_reader_data(self): # giving more immediate access to the data
        self.airfoil1 = self.APCReader.airfoil1
        self.airfoil2 = self.APCReader.airfoil2
        self.transition = self.APCReader.transition
        self.thickness_ratio = self.APCReader.thickness_ratio
        self.max_thicknesses = self.APCReader.geometry_data['MAX-THICK'].to_numpy()
        self.chord_length = self.APCReader.chord_length
        self.twist_angle = self.APCReader.twist_angle
        self.radial_position = self.APCReader.radial_position

        self.xa_trans = self.APCReader.xa_trans
        self.ya_trans = self.APCReader.ya_trans

    def build_airfoil_sections(self):
        self.airfoil_sections = []

        for i in range(len(self.chord_length)):
            airfoil = Airfoil_Section(airfoil_type1=self.airfoil1[i], airfoil_type2=self.airfoil2[i],
                                      transition=self.transition[i], thickness_ratio=self.thickness_ratio[i],
                                      n=self.interpolation_points, thickness_mode="vertically", center=False)
            self.airfoil_sections.append(airfoil)

    def export_geometry_for_analysis(self):
        distance_to_preceeding_airfoil = [self.radial_position[i] - self.radial_position[i-1] for i in range(1, len(self.radial_position))] + [0]
        self.export_data = {
            "r": [],  # radial positions
            "dr": [],  # distances to preceding airfoil
            "chord": [],  # chord lengths
            "twist": [],  # twist angles
            "airfoil": []  # airfoil objects
        }
        for i, airfoil in enumerate(self.airfoil_sections[:-1]):
            self.export_data["r"].append(self.radial_position[i] * 0.0254) # converted to meters
            self.export_data["dr"].append(distance_to_preceeding_airfoil[i] * 0.0254)
            self.export_data["chord"].append(self.chord_length[i] * 0.0254)
            self.export_data["twist"].append(self.twist_angle[i])
            self.export_data["airfoil"].append(airfoil)
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