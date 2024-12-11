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

def xspace(start, stop, num=None):
    if num is None:
        num = 100
    num20 = int(num * 0.2)
    m1 = int((stop-start) * 0.1)  # 10% of the range   ###
    m2 = int((stop-start) * 0.3)  # 30% of the range
    m3 = int((stop-start) * 0.4)  # 40% of the range
    s1 = np.linspace(start, m1, num20)  # 0-10% at 20% resolution
    s2 = np.linspace(m1, m2, num20)  # 10-30% at 20% resolution
    s3 = np.linspace(m2, m3, num20)  # 30-70% at 20% resolution
    s4 = np.linspace(stop-m2-m1, stop-m1, num20)  # 70-90% at 20% resolution
    s5 = np.linspace(stop-m1, stop, num20)  # 90-100% at 20% resolution
    return np.concatenate([s1, s2, s3, s4, s5])

def tanhspace(start, stop, num=None, tahnhlimit=None):
    if num is None:
        num = 100
    else:
        num = int(num)
        assert num > 2, "Number of points must be larger than 2"

    if tahnhlimit is None:
        tanhlimit = np.pi * 0.67  ## how strong the low and top point concentration is: 1*np.pi = complete tahnh, ~0.001 = linear
    else:
        assert tahnhlimit < np.pi, "tahnhlimit must be smaller than pi"

    space = np.tanh(np.linspace(-tanhlimit, tanhlimit, num, dtype='float64'))
    #resize space between 0 and 1
    space = (space - space.min()) / (space.max() - space.min())
    return space * (stop-start) + start


# Airfoil construction class
class Airfoil_Section():
    def __init__(self, airfoil_type1, airfoil_type2, transition, thickness_ratio, n, thickness_mode="perpendicular_to_chamber", center = True, use_cosine_spacing = True, E63_correction=1):
        self.n = n
        self.thickness_ratio = thickness_ratio  # Thickness ratio to chord length
        self.transition = transition
        # print(transition)
        self.center = center # Whether to center the airfoil in its centroid (Flächenmittelpunkt)
        self.use_cosine_spacing = use_cosine_spacing
        self.thickness_mode = thickness_mode  # "perpendicular_to_chamber" or "vertically"

        self.remove_trailing_double = 1  # 0 = No, 1 = Yes

        self.COM = [0, 0]
        self.shifts = []
        self.rotations = []
        self.resizes = []

        self.airfoil_type1 = airfoil_type1.upper()  # Airfoil is always uppercase
        self.airfoil_type2 = airfoil_type2.upper()


        if self.airfoil_type1 == "E63":
            self.E63_content = 1 - self.transition
        elif self.airfoil_type2 == "E63":
            self.E63_content = self.transition
        else:
            self.E63_content = 0
        self.E63_correction = E63_correction * self.E63_content + (1-self.E63_content)
        assert self.E63_correction > 0, "E63 correction must be larger than 0"
        # print(self.E63_correction, self.E63_content, self.transition)

        self.X = None
        self.Y = None
        self.x_chord = np.array([0, 1])
        self.y_chord = np.array([0, 0])

        self.alpha_variation = np.linspace(-20, 20, 41)

        self.initialize()

    def initialize(self):
        if self.transition == 0.0:
            self.X0, self.Y0 = self.draw_airfoil(self.airfoil_type1)
            self.X, self.Y = self.scale_across_chamber(self.thickness_ratio / self.get_max_thickness_vertically())
        elif self.transition == 1.0:
            self.X, self.Y = self.draw_airfoil(self.airfoil_type2)
            self.X, self.Y = self.scale_across_chamber(self.thickness_ratio / self.get_max_thickness_vertically())
        elif self.transition > 0 and self.transition < 1:
            self.X1, self.Y1 = self.draw_airfoil(self.airfoil_type1)
            self.X1, self.Y1 = self.scale_across_chamber(self.thickness_ratio / self.get_max_thickness_vertically())
            self.X2, self.Y2 = self.draw_airfoil(self.airfoil_type2)
            self.X2, self.Y2 = self.scale_across_chamber(self.thickness_ratio / self.get_max_thickness_vertically())
            self.X = self.X1 #* (1 - self.transition) + self.X2 * self.transition
            self.Y = self.Y1 * (1 - self.transition) + self.Y2 * self.transition
            self.recalculate_camber()
        else:
            raise ValueError(f"Transition value must be between 0 and 1, but is {self.transition}")

        self.scale_across_chamber(self.thickness_ratio / self.get_max_thickness_vertically()* self.E63_correction)  ## [WIKIPEDIA] The thickness ratio is the maximum vertical thickness divided by the chord length.
        if self.center:
            self.center_airfoil()
        else:
            self.COM = self.getCOM()
            # pass

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
                case "SQUARE":
                    self.X, self.Y = self.squarefoil()
                case _:
                    raise ValueError(f"Invalid airfoil type: {airfoil_type}. Airfoil type might not be supported yet.")

        # self.scale_across_chamber(self.thickness_ratio / self.get_max_thickness_perpendicular_to_camber())
        # self.scale_across_chamber(self.thickness_ratio / self.get_max_thickness_vertically())
        # average_thickness = (self.get_max_thickness_vertically() + self.get_max_thickness_perpendicular_to_camber()) / 2
        # self.scale_across_chamber(self.thickness_ratio / average_thickness)
        self.COM = self.getCOM()
        return self.X, self.Y


    ########### Airfoil type functions ###########
    def naca_airfoil(self, NACA_number):
        self.NACA_number = NACA_number

        m = float(self.NACA_number[5]) / 100.0
        p = float(self.NACA_number[6]) / 10.0
        t = float(self.NACA_number[7:]) / 100.0
        if self.use_cosine_spacing:
            x = tanhspace(0, 1, self.n)
        else:
            x = np.linspace(0, 1, self.n, dtype='float64')

        a0 = 0.2969
        a1 = -0.1260
        a2 = -0.3516
        a3 = 0.2843
        # a4 = -0.1036
        a4 = -0.1015

        # Thickness function
        yt_func = lambda x: 5 * t * (a0 * np.sqrt(x) +
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
        self.X = np.concatenate((xu[::-1], xl[1:]))
        self.Y = np.concatenate((yu[::-1], yl[1:]))
        # resample the airfoil such that X-Coordinates agree with the camber line
        # x_new = np.linspace(0, 1, self.n, dtype='float64')
        # yu_new = np.interp(x_new, x, yu)
        # yl_new = np.interp(x_new, x, yl)
        # self.X = np.concatenate([x_new[::-1], x_new[1:]])
        # self.Y = np.concatenate([yu_new[::-1], yl_new[1:]])
        return self.X, self.Y

    def e63_airfoil(self):
        self.max_thickness = 0.0425
        file_loc = os.getcwd() + "\\airfoil_data\\e63_selig.txt"
        self.UIUC_selig_format_reader(file_loc)
        return self.X, self.Y

    def clarky_airfoil(self):
        self.max_thickness = 0.117
        file_loc = os.getcwd() + "\\airfoil_data\\clark-y_selig.txt"
        self.UIUC_selig_format_reader(file_loc)
        return self.X, self.Y

    def squarefoil(self):
        xl = np.linspace(0, 1, self.n)
        yl = np.zeros(self.n)
        xu = np.linspace(0, 1, self.n)
        yu = np.array([0])
        yu = np.append(yu, np.ones(self.n-2))
        yu = np.append(yu, [0])
        self.X = np.concatenate((xl[::-1], xu[1:]))
        self.Y = np.concatenate((yl[::-1], yu[1:]))
        self.x_camber = np.linspace(0, 1, self.n)
        self.y_camber = np.array([0])
        self.y_camber = np.append(self.y_camber, np.ones(self.n-2)/2)
        self.y_camber = np.append(self.y_camber, [0])
        return self.X, self.Y


    ########### Airfoil drawing helper functions ###########
    def UIUC_selig_format_reader(self, filename):
        data = pd.read_csv(filename, sep="\s+", skiprows=1, engine='python')
        data.columns = ["X", "Y"]
        data = data.astype(float)

        self.X, self.Y, self.x_camber, self.y_camber = self.interpolate_airfoil(data.to_numpy())

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

        if self.use_cosine_spacing:
            x_new = tanhspace(0, 1, self.n)
        else:
            x_new = np.linspace(0, 1, self.n, dtype='float64')
        y_new_upper = f_upper(x_new)
        y_new_lower = f_lower(x_new)

        # force connected edges
        y_new_upper[0] = 0.0
        y_new_lower[0] = 0.0
        y_new_upper[-1] = 0.0
        y_new_lower[-1] = 0.0

        y_camber = (y_new_upper + y_new_lower) / 2

        # Combine upper and lower coordinates
        self.X = np.concatenate([x_new[::-1], x_new[self.remove_trailing_double:]])
        y_coords = np.concatenate([y_new_upper[::-1], y_new_lower[self.remove_trailing_double:]])

        # print("interpolation output shapes x, y, x_new, y_chamber:", self.X.shape, y_coords.shape, x_new.shape, y_chamber.shape)
        # print("interpolation output x, y, x_new, y_chamber:", self.X, y_coords)

        return self.X, y_coords, x_new, y_camber

    ########### Airfoil transformation functions ###########
    # Private method to move airfoil origin to mid chord
    def __moveToMidChord(self):
        self.translate([-self.getChordMidPoint()[0], -self.getChordMidPoint()[1]])

    # Private method to move airfoil origin to mid camber
    def __moveToMidCamber(self):
        self.translate([-self.getCamberMidPoint()[0], -self.getCamberMidPoint()[1]])

    def translate(self, pos_vector):
        self.X = self.X + pos_vector[0]
        self.Y = self.Y + pos_vector[1]
        self.x_camber = self.x_camber + pos_vector[0]
        self.y_camber = self.y_camber + pos_vector[1]
        self.x_chord = self.x_chord + pos_vector[0]
        self.y_chord = self.y_chord + pos_vector[1]
        self.shifts.append([pos_vector, self.COM])
        self.COM = [self.COM[0] + pos_vector[0], self.COM[1] + pos_vector[1]]


    def scale(self, factor):
        self.X = self.X * factor
        self.Y = self.Y * factor
        self.x_camber = self.x_camber * factor
        self.y_camber = self.y_camber * factor
        self.x_chord = self.x_chord * factor
        self.y_chord = self.y_chord * factor
        self.resizes.append(factor)

    def scale_vertically(self, factor):
        self.Y = self.Y * factor
        self.y_camber = self.y_camber * factor
        self.y_chord = self.y_chord * factor

    def scale_across_chamber(self, factor):
        yu_dist_from_chamber = self.Y[:self.n] - self.y_camber
        yl_dist_from_chamber = self.Y[self.n-1:] - self.y_camber  #negative values
        y_upper_new = self.y_camber + (yu_dist_from_chamber * factor)
        y_lower_new = self.y_camber + (yl_dist_from_chamber * factor)
        self.Y = np.concatenate([y_upper_new, y_lower_new[1:]])
        return self.X, self.Y

    def increase_thickness_across_chamber(self, increase_mm):
        # increases the thickness of the airfoil by a flat mm value across the chamber
        max_dist_from_chamber_mm = (self.y_camber - self.Y[self.n-1:]).max() * 25.4 # inches to mm
        print(max_dist_from_chamber_mm)
        factor = 1 + increase_mm / max_dist_from_chamber_mm
        self.scale_across_chamber(factor)

    def scale_section_vertically(self, section_start = 0.3, section_end = 1, mid_section=None, factor = 2, mode="linear", plot=False):
        if plot:
            pre_cs = self.calculate_cross_section_area()
            plt.figure()
            plt.plot(self.X, self.Y, label="Pre-scaled")
            plt.axis('equal')
            plt.title(f"Airfoil section scaling from {section_start} to {section_end} with factor {factor}. Airfoil: {self.transition*100}% {self.airfoil_type1} + {100-self.transition*100}% {self.airfoil_type2}")
            plt.xlabel("X [in]")
            plt.ylabel("Y [in]")

        # sections in % of chord length
        if mid_section is None:
            mid_section = (section_end - section_start) / 2 + section_start  # %

        section_start_index = int(section_start * self.n)
        section_end_index = int(section_end * self.n)
        mid_section_index = int(mid_section * self.n)

        if mode == "linear":
            self.factors = np.ones(self.n)
            linear_up = np.linspace(1, factor, mid_section_index - section_start_index + 1)
            linear_down = np.linspace(factor, 1, section_end_index - mid_section_index)
            self.factors[section_start_index:mid_section_index+1] = linear_up
            self.factors[mid_section_index:section_end_index] = linear_down

            yu_dist_from_chamber = self.Y[:self.n] - self.y_camber
            yl_dist_from_chamber = self.Y[self.n - 1:] - self.y_camber  # negative values
            y_upper_new = self.y_camber + (yu_dist_from_chamber * self.factors[::-1])
            y_lower_new = self.y_camber + (yl_dist_from_chamber * self.factors)
            self.Y = np.concatenate([y_upper_new, y_lower_new[1:]])

        if plot:
            post_cs = self.calculate_cross_section_area()
            plt.plot(self.X, self.Y, label="Post-scaled")
            plt.text(0.1, 0.1, f"Crosssection_increase: {post_cs/pre_cs*100-100:2f}%", transform=plt.gca().transAxes)
            plt.legend()
            plt.show()

    # method to rotate the airfoil coordinates (in plane)
    def rotate(self, angle):
        distance_COM_to_origin = self.getCOM()
        self.translate([-distance_COM_to_origin[0], -distance_COM_to_origin[1]]) # rotate around origin

        self.rotations.append([angle, self.getCOM()])
        coordinates = np.vstack((self.X, self.Y))
        coordinates_camber = np.vstack((self.x_camber, self.y_camber))
        coordinates_chord = np.vstack((self.x_chord, self.y_chord))
        ang = np.pi / 180 * angle

        rotation_matrix = np.array([[np.cos(ang), -np.sin(ang)],
                                    [np.sin(ang), np.cos(ang)]])
        self.X = np.matmul(rotation_matrix, coordinates)[0, :]
        self.Y = np.matmul(rotation_matrix, coordinates)[1, :]
        self.x_camber = np.matmul(rotation_matrix, coordinates_camber)[0, :]
        self.y_camber = np.matmul(rotation_matrix, coordinates_camber)[1, :]
        self.x_chord = np.matmul(rotation_matrix, coordinates_chord)[0, :]
        self.y_chord = np.matmul(rotation_matrix, coordinates_chord)[1, :]

        self.translate(distance_COM_to_origin)

    def getChordMidPoint(self):
        return [self.x_chord[int(self.n / 2)], self.y_chord[int(self.n / 2)]]

    def getCamberMidPoint(self):
        return [self.x_camber[int(self.n / 2)], self.y_camber[int(self.n / 2)]]

    def recalculate_camber(self):
        self.x_camber = self.X[:self.n]
        self.y_camber = (self.Y[:self.n] + self.Y[:self.n - 2:-1]) / 2
        return self.x_camber, self.y_camber

    def getCOM(self):
        x = self.X
        y = self.Y
        A = np.abs(0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))
        Cx = (1 / (6 * A)) * np.sum((x[:-1] + x[1:]) * (x[:-1] * y[1:] - x[1:] * y[:-1]))
        Cy = (1 / (6 * A)) * np.sum((y[:-1] + y[1:]) * (x[:-1] * y[1:] - x[1:] * y[:-1]))
        self.COM = [Cx, Cy]
        return self.COM

    def center_airfoil(self):
        Cx, Cy = self.getCOM()
        self.translate([-Cx, -Cy])
        return self.X, self.Y

    def calculate_cross_section_area(self):
        self.A = np.abs(0.5 * (np.sum(self.X[:-1] * self.Y[1:] - self.X[1:] * self.Y[:-1]) + self.X[-1] * self.Y[0] - self.X[0] * self.Y[-1]))
        return self.A

    def get_chord_length(self):
        return np.sqrt((self.x_chord[-1] - self.x_chord[0])**2 + (self.y_chord[-1] - self.y_chord[0])**2)

    def get_highest_point(self):
        return self.Y.max()


    def get_max_thickness(self):
        match self.thickness_mode:
            case "perpendicular_to_chamber":
                return self.get_max_thickness_perpendicular_to_camber()
            case "vertically":
                return self.get_max_thickness_vertically()
            case _:
                raise ValueError(f"Invalid thickness mode: {self.thickness_mode}. This thickness mode might not be supported yet.")

    def get_max_thickness_vertically(self):
        max_thickness = (self.Y[:self.n-1] - self.Y[:self.n-1:-1]).max()
        return max_thickness

    def get_max_thickness_perpendicular_to_camber(self, plot=False):
        from scipy.interpolate import interp1d
        from scipy.optimize import fsolve
        # Separate the points into upper and lower surfaces
        data = pd.DataFrame({'x': self.X, 'y': self.Y})
        mid_index = len(self.X) // 2
        upper_surface = data.iloc[:mid_index]
        lower_surface = data.iloc[mid_index:]

        # Interpolate the upper and lower surfaces
        upper_interp = interp1d(upper_surface['x'], upper_surface['y'], kind='cubic', fill_value="extrapolate")
        lower_interp = interp1d(lower_surface['x'], lower_surface['y'], kind='cubic', fill_value="extrapolate")

        # Calculate the camber line
        upper_y_new = upper_interp(self.x_camber[1:-1])
        lower_y_new = lower_interp(self.x_camber[1:-1])

        # Calculate the slope of the camber line
        self.recalculate_camber()
        self.camber_slope = np.gradient(self.y_camber, self.x_camber)

        # Function to find the intersection of the normal line with the surface
        def find_intersection(x, y_cam, slope, interp):
            normal_slope = -1 / slope

            def equations(p):
                x_i, y_i = p
                return (y_i - y_cam - normal_slope * (x_i - x), y_i - interp(x_i))

            x_i, y_i = fsolve(equations, (x, interp(x)))
            return x_i, y_i

        # Calculate the thickness at each x-coordinate
        thickness = []
        x_upper_intersections = []
        x_lower_intersections = []
        y_upper_intersections = []
        y_lower_intersections = []
        for x, y, m in zip(self.x_camber, self.y_camber, self.camber_slope):
            if m == 0.0:
                x_upper_intersect = x
                x_lower_intersect = x
                y_upper_intersect = upper_interp(x)
                y_lower_intersect = lower_interp(x)
                upper_distance = np.abs(upper_interp(x) - y)
                lower_distance = np.abs(lower_interp(x) - y)
            else:
                x_upper_intersect, y_upper_intersect = find_intersection(x, y, m, upper_interp)
                x_lower_intersect, y_lower_intersect = find_intersection(x, y, m, lower_interp)
                upper_distance = np.sqrt((x_upper_intersect - x) ** 2 + (y_upper_intersect - y) ** 2)
                lower_distance = np.sqrt((x_lower_intersect - x) ** 2 + (y_lower_intersect - y) ** 2)
            thickness.append(upper_distance + lower_distance)
            x_upper_intersections.append(x_upper_intersect)
            x_lower_intersections.append(x_lower_intersect)
            y_upper_intersections.append(y_upper_intersect)
            y_lower_intersections.append(y_lower_intersect)
    
        # Find the maximum thickness
        self.max_thickness = np.max(thickness)
        self.max_thickness_location = self.x_camber[np.argmax(thickness)]  # Location of maximum thickness (x-coordinate on the camber line)

        if plot:
            # Output the results
            print(f'Max thickness {self.max_thickness*100:.2f}% at {self.max_thickness_location*100:.2f}% chord.')
            print(f"Max camber {np.max(self.y_camber)*100:.2f}% at {self.x_camber[np.argmax(self.y_camber)]*100:.2f}% chord")

            # Plot the airfoil and the thickness distribution
            plt.figure(figsize=(10, 5))
            plt.plot(self.X, self.Y, label='Airfoil')
            plt.plot(self.x_camber[1:-1], upper_y_new, label='Upper Surface', linestyle='--')
            plt.plot(self.x_camber[1:-1], lower_y_new, label='Lower Surface', linestyle='--')
            plt.plot(self.x_camber, self.y_camber, label='Camber Line', linestyle='-.', color='orange')
            plt.scatter(self.max_thickness_location, self.y_camber[np.argmax(thickness)], color='red',
                        label=f'Max Thickness: {self.max_thickness:.4f}')
            for i,_ in enumerate(x_upper_intersections):
                if i == np.argmax(thickness):
                    plt.plot([x_upper_intersections[i], x_lower_intersections[i]], [y_upper_intersections[i], y_lower_intersections[i]], color = 'black', linewidth=2)
                else:
                    plt.plot([x_upper_intersections[i], x_lower_intersections[i]],
                             [y_upper_intersections[i], y_lower_intersections[i]])
            plt.legend()
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title('Airfoil and Thickness Distribution')
            plt.grid(True)
            plt.gca().set_aspect('equal', adjustable='box')
            plt.show()
        return self.max_thickness

    ########### Airfoil visualization functions ###########
    def plot(self, show=True, save=False, filename="airfoil.png", chord=True, camber=True, interpolated_only=False):
        plt.figure()
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
        # try:
        #     plt.scatter(self.unshifted_COM[0], self.unshifted_COM[1], color='red', label='Unshifted Center of Mass')
        # except:
        #     pass
        if camber:
            plt.plot(self.x_camber, self.y_camber, label = "Camber")
        if chord:
            plt.plot(self.x_chord, self.y_chord, label = "Chord")
        for shift in self.shifts:
            plt.scatter(shift[1][0], shift[1][1], color='red')
            plt.arrow(shift[1][0], shift[1][1], shift[0][0], shift[0][1], head_width=0.012, head_length=0.025, fc='red', ec='red', length_includes_head=True)
        for rotation in self.rotations:
            arc = matplotlib.patches.Arc(rotation[1], 0.1, 0.1, angle =0, theta1=rotation[0], theta2=0, color='green', linewidth=3)
            plt.gca().add_patch(arc)

        plt.axis('equal')
        plt.title(f"Airfoil: {self.airfoil_type1} to {self.airfoil_type2} with transition {self.transition}")
        plt.legend()
        plt.xlabel("X [in]")
        plt.ylabel("Y [in]")
        # scope = 0.55
        # if self.center:
        #     plt.xlim([-scope, scope])
        # else:
        #     plt.xlim([-0.05, 2*scope-0.05])
        # plt.ylim([-scope, scope])
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
    self = Airfoil_Section("NACA 4412", "E63", transition=1, thickness_ratio=0.1658, n=100, center=True)
    self.scale(0.8776)
    self.plot()
    print(self.calculate_cross_section_area())
    self.get_max_thickness_perpendicular_to_camber()
    self.get_max_thickness_vertically() # slightly thicker

    # self.scale_section_vertically(0.3, 1, mid_section=0.9, factor= 1.5, plot=True)
    self.plot()
    print(self.calculate_cross_section_area())


    # self.plot_airfoil(chord=False, camber=False, interpolated_only=True, show=False)
    # self.increase_thickness_across_chamber(0.05)
    # self.plot_airfoil(chord=False, camber=True, interpolated_only=True)

    # self.alpha_variation = np.linspace(-9, 9, 21)
    #
    # for Re in [1e5, 5e5, 1e6]:
    #     self.get_aero(Re=Re, mach=0, n_crit=9, model_size="xxxlarge")
    #     self.plot_aero(Re)

    # from XFoil import XFoil
    # xf = XFoil()
