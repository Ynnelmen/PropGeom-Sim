import numpy as np
import os
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')  # Switch to TkAgg backend, do this before importing pyplot
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
# from XFoil import XFoil
import neuralfoil as nf
# import aerosandbox as asb

# Airfoil construction class
class Airfoil_Section():
    def __init__(self, airfoil_type1, airfoil_type2, transition, thickness_ratio, n, center = True):
        self.n = n
        self.thickness_ratio = thickness_ratio
        self.transition = transition
        # print(transition)
        self.center = center # Whether to center the airfoil in its centroid (Flächenmittelpunkt)

        self.remove_trailing_double = 1  # 0 = No, 1 = Yes

        self.airfoil_type1 = airfoil_type1.upper()  # Airfoil is always uppercase
        self.airfoil_type2 = airfoil_type2.upper()

        self.X = None
        self.Y = None

        self.alpha_variation = np.linspace(-20, 20, 41)

        self.initialize()

    def initialize(self):
        if self.transition == 0:
            self.X, self.Y = self.draw_airfoil(self.airfoil_type1)
        elif self.transition == 1:
            self.X, self.Y = self.draw_airfoil(self.airfoil_type2)
        elif self.transition > 0 and self.transition < 1:
            self.X1, self.Y1 = self.draw_airfoil(self.airfoil_type1)
            self.X2, self.Y2 = self.draw_airfoil(self.airfoil_type2)
            self.X = self.X1 * (1 - self.transition) + self.X2 * self.transition
            self.Y = self.Y1 * (1 - self.transition) + self.Y2 * self.transition
        else:
            raise ValueError(f"Transition value must be between 0 and 1, but is {self.transition}")

    def draw_airfoil(self, airfoil_type):
        # decides which function to use based on airfoil type. Centers airfoil
        if airfoil_type.startswith("NACA"):
            self.X, self.Y = self.naca_airfoil(airfoil_type)
        else:
            match airfoil_type:
                case "E63":
                    self.X, self.Y = self.e63_airfoil()
                case "APC12":
                    self.X, self.Y = self.naca_airfoil("NACA 4412")
                case "CLARK-Y":
                    self.X, self.Y = self.clarky_airfoil()
                case _:
                    raise ValueError(f"Invalid airfoil type: {airfoil_type}. Airfoil type might not be supported yet.")
        # print('Airfoil output:', self.X, self.Y)
        # print("Shapes X, Y:", self.X.shape, self.Y.shape, self.X.argmin())
        if self.center:
            self.center_airfoil()
        return self.X, self.Y


    ########### Airfoil type functions ###########
    def naca_airfoil(self, NACA_number):
        self.NACA_number = NACA_number

        m = float(self.NACA_number[5]) / 100.0
        p = float(self.NACA_number[6]) / 10.0
        t = float(self.NACA_number[7:]) / 100.0
        x = np.linspace(0, 1, self.n)

        a0 = 0.2969
        a1 = -0.1260
        a2 = -0.3516
        a3 = 0.2843
        # a4 = -0.1036
        a4 = -0.1015

        # Thickness function
        yt_func = lambda x: 5 * self.thickness_ratio * (a0 * np.sqrt(x) +
        # yt_func = lambda x: 5 * t * (a0 * np.sqrt(x) +
                                                   a1 * x +
                                                   a2 * x ** 2 +
                                                   a3 * x ** 3 +
                                                   a4 * x ** 4)

        # Definition of camber line and upper/lower airfoil coordinates
        if p == 0:
            x_upper = x
            y_upper = yt_func(x)

            x_lower = x
            y_lower = -y_upper

            x_camber = x
            y_camber = np.zeros(len(x_camber))
        else:
            yc_func = lambda x: (m / p ** 2) * (2 * p * x - x ** 2) if (x < p) else (m / (1 - p) ** 2) * (
                        (1 - 2 * p) + 2 * p * x - x ** 2)
            dycdx_func = lambda x: (2 * m / p ** 2) * (p - x) if (x < p) else (2 * m / (1 - p) ** 2) * (p - x)
            theta_func = lambda x: np.arctan(x)

            x_upper = []
            y_upper = []
            x_lower = []
            y_lower = []
            y_camber = []
            x_camber = x
            for val in x:
                x_upper.append(val - yt_func(val) * np.sin(theta_func(dycdx_func(val))))
                y_upper.append(yc_func(val) + yt_func(val) * np.cos(theta_func(dycdx_func(val))))
                x_lower.append(val + yt_func(val) * np.sin(theta_func(dycdx_func(val))))
                y_lower.append(yc_func(val) - yt_func(val) * np.cos(theta_func(dycdx_func(val))))
                y_camber.append(yc_func(val))

            x_upper[-1] = 1
            x_lower[-1] = 1
            y_upper[-1] = 0
            y_lower[-1] = 0
            y_upper[0] = 0
            y_lower[0] = 0

        self.x_camber = x_camber
        self.y_camber = np.asarray(y_camber)
        self.x_chord = self.x_camber
        self.y_chord = np.zeros(len(self.x_camber))
        self.X = np.concatenate((x_upper[::-1], x_lower[self.remove_trailing_double:]))
        self.Y = np.concatenate((y_upper[::-1], y_lower[self.remove_trailing_double:]))
        # print("NACA airfoil output shapes x, y, x_camber, y_camber:", self.X.shape, self.Y.shape, self.x_camber.shape, self.y_camber.shape)

        return self.X, self.Y

    def naca_airfoil2(self, naca):
        m = int(naca[0]) / 100.0  # Maximum camber
        p = int(naca[1]) / 10.0  # Position of maximum camber
        t = int(naca[2:]) / 100.0  # Maximum thickness

        # Define the chord line from 0 to 1
        x = np.linspace(0, 1, self.n, dtype='float64')

        # Calculate the camber line
        yc = np.where(x < p, m * (x / np.power(p, 2)) * (2 * p - x),
                      m * ((1 - x) / np.power(1 - p, 2)) * (1 + x - 2 * p))

        # Calculate the thickness distribution
        yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * np.power(x, 2)
                      + 0.2843 * np.power(x, 3) - 0.1015 * np.power(x, 4))

        # Calculate the angle of the camber line
        dyc_dx = np.where(x < p, 2 * m / np.power(p, 2) * (p - x),
                          2 * m / np.power(1 - p, 2) * (p - x))
        theta = np.arctan(dyc_dx)

        # Upper and lower surface coordinates
        xu = x - yt * np.sin(theta)
        yu = yc + yt * np.cos(theta)
        xl = x + yt * np.sin(theta)
        yl = yc - yt * np.cos(theta)

        # Combine upper and lower coordinates
        x_coords = np.concatenate((xu[::-1], xl[1:]))
        y_coords = np.concatenate((yu[::-1], yl[1:]))
        return x_coords, y_coords

    def e63_airfoil(self):
        self.max_thickness = 0.0425
        file_loc = os.getcwd() + "\\airfoil_data\\e63_selig.txt"
        self.UIUC_selig_format_reader(file_loc)

        self.scale_across_chamber(self.thickness_ratio / self.max_thickness)
        # print("scaled xy:", self.X, self.Y, self.X.argmin())
        return self.X, self.Y

    def clarky_airfoil(self):
        self.max_thickness = 0.117
        file_loc = os.getcwd() + "\\airfoil_data\\clark-y_selig.txt"
        self.UIUC_selig_format_reader(file_loc)
        self.scale_across_chamber(self.thickness_ratio / self.max_thickness)
        return self.X, self.Y


    ########### Airfoil drawing helper functions ###########
    def UIUC_selig_format_reader(self, filename):
        data = pd.read_csv(filename, sep="\s+", skiprows=1, engine='python')
        data.columns = ["X", "Y"]
        data = data.astype(float)

        self.X, self.Y, self.x_camber, self.y_camber = self.interpolate_airfoil(data.to_numpy())

        self.x_chord = np.array([0, 1])
        self.y_chord = np.array([0, 0])

        # print("UIUC_selig_format_reader output shapes x, y, x_camber, y_camber:", self.X.shape, self.Y.shape, self.x_camber.shape, self.y_camber.shape)
        # print(self.X, self.Y)

        return self.X, self.Y, self.x_camber, self.y_camber

    def separate_airfoil_data(self, data):
        ''' data = np.array of x, y coordinates'''
        # this function separates the points in data into equally sized upper and lower arrays
        upper = data[:data[:,0].argmin()+1]
        lower = data[data[:,0].argmin():]
        # print(data.shape, upper.shape, lower.shape, data[:,0].argmin())
        # print("Data shape:", data.shape)
        # print(upper, lower)
        # print("Upper shape:", upper.shape, "Lower shape:", lower.shape)
        return upper, lower

    def interpolate_airfoil(self, xy):
        ''' xy = np.array '''
        upper, lower = self.separate_airfoil_data(xy)
        upper[0] = [0, 0]
        lower[0] = [0, 0]
        upper[-1] = [1, 0]
        lower[-1] = [1, 0]

        f_upper = interp1d(upper[:,0], upper[:,1], kind='linear', fill_value='extrapolate')
        f_lower = interp1d(lower[:,0], lower[:,1], kind='linear', fill_value='extrapolate')

        x_new = np.linspace(0, 1, self.n, dtype='float64')
        y_new_upper = f_upper(x_new)
        y_new_lower = f_lower(x_new)

        # connect the edges
        y_new_upper[0] = 0.0
        y_new_lower[0] = 0.0
        y_new_upper[-1] = 0.0
        y_new_lower[-1] = 0.0

        y_camber = (y_new_upper + y_new_lower) / 2

        # Combine upper and lower coordinates
        x_coords = np.concatenate([x_new[::-1], x_new[self.remove_trailing_double:]])
        y_coords = np.concatenate([y_new_upper[::-1], y_new_lower[self.remove_trailing_double:]])

        # print("interpolation output shapes x, y, x_new, y_chamber:", x_coords.shape, y_coords.shape, x_new.shape, y_chamber.shape)
        # print("interpolation output x, y, x_new, y_chamber:", x_coords, y_coords)

        return x_coords, y_coords, x_new, y_camber

    ########### Airfoil transformation functions ###########
    # Private method to move airfoil origin to mid chord
    def __moveToMidChord(self):
        self.translate([-self.getChordMidPoint()[0], -self.getChordMidPoint()[1]])

    # Private method to move airfoil origin to mid camber
    def __moveToMidCamber(self):
        self.translate([-self.getCamberMidPoint()[0], -self.getCamberMidPoint()[1]])

    # method to translate the airfoil coordinates
    def translate(self, pos_vector):
        self.X = self.X + pos_vector[0]
        self.Y = self.Y + pos_vector[1]
        self.x_camber = self.x_camber + pos_vector[0]
        self.y_camber = self.y_camber + pos_vector[1]
        self.x_chord = self.x_chord + pos_vector[0]
        self.y_chord = self.y_chord + pos_vector[1]

    # method to scale the airfoil coordinates
    def scale(self, factor):
        self.X = self.X * factor
        self.Y = self.Y * factor
        self.x_camber = self.x_camber * factor
        self.y_camber = self.y_camber * factor
        self.x_chord = self.x_chord * factor
        self.y_chord = self.y_chord * factor

    def scale_vertically(self, factor):
        self.Y = self.Y * factor
        self.y_camber = self.y_camber * factor
        self.y_chord = self.y_chord * factor

    def scale_across_chamber(self, factor):
        yu_dist_from_chamber = self.Y[:self.n] - self.y_camber
        yl_dist_from_chamber = self.Y[self.n-1:] - self.y_camber
        y_upper_new = self.y_camber + yu_dist_from_chamber * factor
        y_lower_new = self.y_camber + yl_dist_from_chamber * factor
        # print("y_upper_new:", y_upper_new, "y_lower_new:", y_lower_new)
        # print(self.X)

        # print("yu_dist_from_chamber:", yu_dist_from_chamber)
        # print("upper:", upper[:, 1])

        self.Y = np.concatenate([y_upper_new, y_lower_new[1:]])
        self.y_camber = self.y_camber * factor

    def increase_thickness_across_chamber(self, increase_mm):
        # increases the thickness of the airfoil by a flat mm value across the chamber
        max_dist_from_chamber_mm = (self.y_camber - self.Y[self.n-1:]).max() * 25.4 # inches to mm
        print(max_dist_from_chamber_mm)
        factor = 1 + increase_mm / max_dist_from_chamber_mm
        self.scale_across_chamber(factor)


    # method to rotate the airfoil coordinates (in plane)
    def rotate(self, angle):
        ang = angle
        coordinates = np.vstack((self.X, self.Y))
        coordinates_camber = np.vstack((self.x_camber, self.y_camber))
        coordinates_chord = np.vstack((self.x_chord, self.y_chord))
        rotation_matrix = np.array(([]))
        ang = np.pi / 180 * ang

        rotation_matrix = np.array([[np.cos(ang), -np.sin(ang)],
                                    [np.sin(ang), np.cos(ang)]])
        self.X = np.matmul(rotation_matrix, coordinates)[0, :]
        self.Y = np.matmul(rotation_matrix, coordinates)[1, :]
        self.x_camber = np.matmul(rotation_matrix, coordinates_camber)[0, :]
        self.y_camber = np.matmul(rotation_matrix, coordinates_camber)[1, :]
        self.x_chord = np.matmul(rotation_matrix, coordinates_chord)[0, :]
        self.y_chord = np.matmul(rotation_matrix, coordinates_chord)[1, :]

    def getChordMidPoint(self):
        return [self.x_chord[int(self.n / 2)], self.y_chord[int(self.n / 2)]]

    def getCamberMidPoint(self):
        return [self.x_camber[int(self.n / 2)], self.y_camber[int(self.n / 2)]]
    
    def center_airfoil(self):
        # Calculate the centroid
        x = self.X
        y = self.Y
        # print("X, Y shapes:", x.shape, y.shape)
        A = 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])
        Cx = (1 / (6 * A)) * np.sum((x[:-1] + x[1:]) * (x[:-1] * y[1:] - x[1:] * y[:-1]))
        Cy = (1 / (6 * A)) * np.sum((y[:-1] + y[1:]) * (x[:-1] * y[1:] - x[1:] * y[:-1]))

        self.translate([-Cx, -Cy])
        return self.X, self.Y

    ########### Airfoil visualization functions ###########
    def plot_airfoil(self, show=True, save=False, filename="airfoil.png", chord=True, camber=True, interpolated_only=False):
        if not interpolated_only:
            try:
                plt.plot(self.X1, self.Y1, label = self.airfoil_type1)
            except:
                pass
            try:
                plt.plot(self.X2, self.Y2, label = self.airfoil_type2)
            except:
                pass
        try:
            plt.plot(self.X, self.Y, label = "Interpolated")
        except:
            pass
        if camber:
            plt.plot(self.x_camber, self.y_camber, label = "Camber")
        if chord:
            plt.plot(self.x_chord, self.y_chord, label = "Chord")
        plt.axis('equal')
        plt.title(f"Airfoil: {self.airfoil_type1} to {self.airfoil_type2} with transition {self.transition}")
        plt.legend()
        plt.xlabel("X [in]")
        plt.ylabel("Y [in]")
        scope = 0.55
        if self.center:
            plt.xlim([-scope, scope])
        else:
            plt.xlim([-0.05, 2*scope-0.05])
        plt.ylim([-scope, scope])
        plt.grid()
        if show:
            plt.show()
        if save:
            save_folder = os.getcwd() + "\\airfoil_imgs"
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            plt.savefig(save_folder + "\\" + filename)
            plt.close()

    ### Airfoil analysis functions ###
    def get_aero(self, alpha_variation=None, Re=1e6, mach=0.2, n_crit=9, model_size="xxxlarge"):
        if alpha_variation is None:
            alpha_variation = self.alpha_variation
        # self.af = asb.Airfoil("NACA4412")
        # self.nf = self.af.get_aero_from_neuralfoil(alpha=self.alpha_variation, Re=Re, mach=mach, n_crit=n_crit, model_size=model_size)
        self.aero = nf.get_aero_from_coordinates(
            coordinates=np.array([self.X, self.Y]).T,
            alpha=alpha_variation,
            Re=Re,
            # mach=mach,
            # n_crit=n_crit,
            model_size=model_size
        )

        return self.aero

    def plot_aero(self, Re):
        # plt.title(f"Aerodynamic Analysis of {self.airfoil_type1} to {self.airfoil_type2} with transition {self.transition} with Re={Re}")
        plt.plot(self.alpha_variation, self.aero["CL"], label=f"CL, Re={Re:.0g}")
        plt.plot(self.alpha_variation, self.aero["CD"], label=f"CD, Re={Re:.0g}")
        plt.ylabel('CL [-]')
        plt.xlabel('Alpha [deg]')
        plt.legend()
        # plt.show()


if __name__ == "__main__":
    self = Airfoil_Section("NACA 4412", "E63", transition=1, thickness_ratio=0.0425, n=150, center=True)
    # self.scale(0.877)

    # self.plot_airfoil(chord=False, camber=False, interpolated_only=True, show=False)
    # self.increase_thickness_across_chamber(0.05)
    # self.plot_airfoil(chord=False, camber=True, interpolated_only=True)

    self.alpha_variation = np.linspace(-9, 9, 21)

    for Re in [1e5, 5e5, 1e6]:
        self.get_aero(Re=Re, mach=0, n_crit=9, model_size="xxxlarge")
        self.plot_aero(Re)

    from XFoil import XFoil
    xf = XFoil()
