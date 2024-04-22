import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class APCReader():
    def __init__(self, filename):
        self.coordinate_rotation_matrix = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])

        self.geometry_data = self.read_geom_data(filename)
        # self.geometry_data = self.geometry_data[:-1]  ## TODO  (Skipping size 0 airfoil)
        self.geometry_data.loc[self.geometry_data.index[-1], "CHORD"] = 0.1   # (Arbitrary value for last airfoil) - kept this way to avoid step export error
        self.interpret_geom_data()


    def read_geom_data(self, filename):
        # Read geometry data from APC
        geometry_data = []
        airfoil_section_info = {}
        blades = 0
        parse_table = False

        # read and parse file
        with open(filename, "r") as f:
            # for index, line in enumerate(f):
            #     match index:
            #         case 0:
            #             prop_label = line[0:5]
            #         case 25:
            #             prop_attributes = line.split()
            #         case 26:
            #             prop_attributes_units = line.split()
            #         case 67:
            #             prop_radius = float(line[10:14])
            #         case 68:
            #             prop_hub_tra = float(line[10:14])
            #         case 69:
            #             prop_nblades = int(line[10:12])
            #         case 102:
            #             prop_airfoil_type_rad_1 = float(line[12:16])
            #             prop_airfoil_type_1 = "".join(line[18:25].split())
            #         case 103:
            #             prop_airfoil_type_rad_2 = float(line[12:16])
            #             prop_airfoil_type_2 = "".join(line[18:25].split())
            #
            #     if index > 27 and index < 65:
            #         line_float = []
            #         for item in line.split():
            #             line_float.append(float(item))
            #         geometry_data.append(line_float)
            for line in f:
                # Detect start of the airfoil summary data table
                if line.strip().startswith('STATION'):
                    parse_table = True
                    hold = 2  # start of the table is 2 lines after the header
                    columns = line.strip().split()
                    continue

                # Detect end of the table/data block
                if line.strip() == "" and parse_table:
                    if hold > 0:
                        hold = hold - 1
                        continue
                    parse_table = False
                    continue

                # Parse table data
                if parse_table:
                    if hold > 0:
                        hold = hold - 1
                        continue
                    else:
                        geometry_data.append(line.strip().split())

                # Extract airfoil sections information
                if line.strip().startswith('AIRFOIL1:'):
                    transition_start, airfoil1 = line.strip().split(':')[1].split("(")[0].split(",")
                    airfoil1 = airfoil1.strip().strip('\t')
                    transition_start = float(transition_start)
                if line.strip().startswith('AIRFOIL2:'):
                    transition_end, airfoil2 = line.strip().split(':')[1].split("(")[0].split(",")
                    airfoil2 = airfoil2.strip().strip('\t')
                    transition_end = float(transition_end)

                # Extract number of blades
                if line.strip().startswith('BLADES:'):
                    blades = int(line.strip().split()[1])

        def transition_state(rad, rad_1, rad_2):
            if rad <= rad_1:
                return 0
            elif rad >= rad_2:
                return 1
            else:
                return (rad - rad_1) / (rad_2 - rad_1)

        # construct data frame
        prop_geometry_data = pd.DataFrame(geometry_data, columns=columns, dtype=float)
        prop_geometry_data['AIRFOILNAME1'] = [airfoil1] * len(prop_geometry_data)
        prop_geometry_data['AIRFOILNAME2'] = [airfoil2] * len(prop_geometry_data)
        prop_geometry_data['TRANSITION'] = [transition_state(rad, transition_start, transition_end) for
                                        rad in prop_geometry_data['STATION']]
        return prop_geometry_data

    def interpret_geom_data(self):
        self.airfoil1 = self.geometry_data['AIRFOILNAME1']
        self.airfoil2 = self.geometry_data['AIRFOILNAME2']
        self.transition = self.geometry_data['TRANSITION']
        self.thickness_ratio = self.geometry_data['THICKNESS']
        self.chord_length = self.geometry_data['CHORD']
        self.twist_angle = self.geometry_data['TWIST']
        self.radial_position = self.geometry_data['STATION']
        self.y_trans = self.geometry_data['CGY'].to_numpy()
        self.z_trans = self.geometry_data['CGZ'].to_numpy()
        self.x_trans = np.zeros(len(self.y_trans))  # no translation in radial direction
        self.blade_trans = np.array([self.x_trans, self.y_trans, self.z_trans]).T

        # # Manipulation of last airfoil
        # self.chord_length[-1] = 0.1
        # self.

        self.airfoil_trans = np.dot(self.blade_trans, self.coordinate_rotation_matrix)
        self.xa_trans = self.airfoil_trans[:, 0]
        self.ya_trans = self.airfoil_trans[:, 1]
        self.za_trans = self.airfoil_trans[:, 2]

        ## test shift = 0
        # self.xa_trans = self.xa_trans - self.xa_trans
        # self.ya_trans = self.xa_trans - self.xa_trans
        # self.za_trans = self.xa_trans - self.xa_trans
        return 0
    
    def plot_trans(self):
        filename1 = r"C:\Users\RhinerLenny\OneDrive - inspire AG\BAZL\10x6E-PERF.PE0"
        filename2 = r"C:\Users\RhinerLenny\OneDrive - inspire AG\BAZL\10x6-PERF.PE0"
        df1 = self.read_geom_data(filename1)
        df2 = self.read_geom_data(filename2)

        fig, ax = plt.subplots(2, 1, figsize=(12, 6))

        ax[0].plot(df1['STATION'], df1['CGY'], label='10x6E-PERF.PE0')
        ax[0].plot(df2['STATION'], df2['CGY'], label='10x6-PERF.PE0')
        ax[0].set_xlabel('Radial position')
        ax[0].set_ylabel('Y Translation')
        ax[0].legend()
        ax[0].grid()

        ax[1].plot(df1['STATION'], df1['CGZ'], label='10x6E-PERF.PE0')
        ax[1].plot(df2['STATION'], df2['CGZ'], label='10x6-PERF.PE0')
        ax[1].set_xlabel('Radial position')
        ax[1].set_ylabel('Z Translation')
        ax[1].legend()
        ax[1].grid()

        # plt.plot(df1['CGY'], label='10x6E')
        # plt.plot(df2['CGY'], label='10x6')
        # plt.xlabel('Radial position')
        # plt.ylabel('Y Translation')
        # plt.legend()

        plt.show()