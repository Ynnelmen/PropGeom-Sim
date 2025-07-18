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
    def __init__(self, coordinates=None, airfoil_name="Custom", thickness_ratio=None, n=100, 
                 thickness_mode="vertically", center=False, use_cosine_spacing=True):
        """
        Initialize airfoil section from coordinates.
        
        Parameters:
        -----------
        coordinates : array-like, shape (n_points, 2)
            Array of [x, y] coordinates defining the airfoil shape.
            Points should be ordered starting from trailing edge, going around upper surface,
            then lower surface back to trailing edge.
        airfoil_name : str
            Name/identifier for the airfoil
        thickness_ratio : float, optional
            Target thickness ratio. If provided, airfoil will be scaled to this thickness.
        n : int
            Number of points for interpolation
        thickness_mode : str
            "perpendicular_to_chamber" or "vertically"
        center : bool
            Whether to center the airfoil at its centroid
        use_cosine_spacing : bool
            Whether to use cosine spacing for interpolation
        """
        self.n = n
        self.thickness_ratio = thickness_ratio
        self.airfoil_name = airfoil_name
        self._APC_cross_section_area = None
        self.center = center
        self.use_cosine_spacing = use_cosine_spacing
        self.thickness_mode = thickness_mode

        self.remove_trailing_double = 1  # 0 = No, 1 = Yes

        self.COM = [0, 0]
        self.shifts = []  # To keep track of the alterations to the airfoil
        self.rotations = []
        self.resizes = []

        self.X = None
        self.Y = None
        self.x_chord = np.array([0, 1])
        self.y_chord = np.array([0, 0])

        self.alpha_variation = 0

        # Initialize from coordinates or generate default airfoil
        if coordinates is not None:
            self.initialize_from_coordinates(coordinates)
        else:
            # Default to NACA 4412 if no coordinates provided
            self.initialize_default_airfoil()

    def initialize_from_coordinates(self, coordinates):
        """Initialize airfoil from provided coordinates."""
        coordinates = np.array(coordinates)
        if coordinates.shape[1] != 2:
            raise ValueError("Coordinates must have shape (n_points, 2)")
        
        # Process the coordinates
        # self.X, self.Y, self.x_camber, self.y_camber = self.interpolate_airfoil(coordinates)
        self.X, self.Y = coordinates[:, 0], coordinates[:, 1]
        
        # Scale to target thickness if specified
        if self.thickness_ratio is not None:
            current_thickness = self.get_max_thickness_vertically()
            self.scale_across_chamber(self.thickness_ratio / current_thickness)
        
        # Center airfoil if requested
        if self.center:
            self.center_airfoil()
        else:
            self.COM = self.getCOM()

    def initialize_default_airfoil(self):
        """Initialize with default NACA 4412 airfoil if no coordinates provided."""
        print("No coordinates provided. Initializing with NACA 4412 airfoil.")
        self.X, self.Y = self.naca_airfoil("NACA 4412")
        
        if self.thickness_ratio is not None:
            current_thickness = self.get_max_thickness_vertically()
            self.scale_across_chamber(self.thickness_ratio / current_thickness)
        
        if self.center:
            self.center_airfoil()
        else:
            self.COM = self.getCOM()

    def naca_airfoil(self, NACA_number):
        """Generate NACA airfoil coordinates."""
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
        a4 = -0.1015

        # Thickness function
        yt_func = lambda x: 5 * t * (a0 * np.sqrt(x) +
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

        return self.X, self.Y

    def separate_airfoil_data(self, data):
        """Separate airfoil data into upper and lower surfaces."""
        # Find the point with minimum x-coordinate (usually leading edge)
        min_x_idx = data[:, 0].argmin()
        
        # Split at the leading edge
        upper = data[:min_x_idx+1]
        lower = data[min_x_idx:]
        
        return upper, lower

    def interpolate_airfoil(self, xy):
        """Interpolate airfoil coordinates to desired number of points."""
        xy = np.array(xy)
        upper, lower = self.separate_airfoil_data(xy)
        
        # Ensure trailing edge points are at (0,0) and (1,0)
        if len(upper) > 0:
            upper[0] = [0, 0] if upper[0, 0] < 0.1 else upper[0]
            upper[-1] = [1, 0] if upper[-1, 0] > 0.9 else upper[-1]
        
        if len(lower) > 0:
            lower[0] = [0, 0] if lower[0, 0] < 0.1 else lower[0]
            lower[-1] = [1, 0] if lower[-1, 0] > 0.9 else lower[-1]

        # Create interpolation functions
        f_upper = interp1d(upper[:, 0], upper[:, 1], kind='linear', fill_value='extrapolate')
        f_lower = interp1d(lower[:, 0], lower[:, 1], kind='linear', fill_value='extrapolate')

        # Generate new x coordinates
        if self.use_cosine_spacing:
            x_new = tanhspace(0, 1, self.n)
        else:
            x_new = np.linspace(0, 1, self.n, dtype='float64')
        
        y_new_upper = f_upper(x_new)
        y_new_lower = f_lower(x_new)

        # Force connected edges
        y_new_upper[0] = 0.0
        y_new_lower[0] = 0.0
        y_new_upper[-1] = 0.0
        y_new_lower[-1] = 0.0

        y_camber = (y_new_upper + y_new_lower) / 2

        # Combine upper and lower coordinates
        X = np.concatenate([x_new[::-1], x_new[self.remove_trailing_double:]])
        Y = np.concatenate([y_new_upper[::-1], y_new_lower[self.remove_trailing_double:]])

        return X, Y, x_new, y_camber

    ########### Airfoil transformation functions ###########
    def translate(self, pos_vector):
        """Translate airfoil by given vector."""
        self.X = self.X + pos_vector[0]
        self.Y = self.Y + pos_vector[1]
        self.x_camber = self.x_camber + pos_vector[0]
        self.y_camber = self.y_camber + pos_vector[1]
        self.x_chord = self.x_chord + pos_vector[0]
        self.y_chord = self.y_chord + pos_vector[1]
        self.shifts.append([pos_vector, self.COM])
        self.COM = [self.COM[0] + pos_vector[0], self.COM[1] + pos_vector[1]]

    def scale(self, factor):
        """Scale airfoil uniformly."""
        self.X = self.X * factor
        self.Y = self.Y * factor
        self.x_camber = self.x_camber * factor
        self.y_camber = self.y_camber * factor
        self.x_chord = self.x_chord * factor
        self.y_chord = self.y_chord * factor
        self.resizes.append(factor)

    def scale_vertically(self, factor):
        """Scale airfoil vertically only."""
        self.Y = self.Y * factor
        self.y_camber = self.y_camber * factor
        self.y_chord = self.y_chord * factor

    def scale_across_chamber(self, factor):
        """Scale thickness perpendicular to camber line."""
        yu_dist_from_chamber = self.Y[:self.n] - self.y_camber
        yl_dist_from_chamber = self.Y[self.n-1:] - self.y_camber
        y_upper_new = self.y_camber + (yu_dist_from_chamber * factor)
        y_lower_new = self.y_camber + (yl_dist_from_chamber * factor)
        self.Y = np.concatenate([y_upper_new, y_lower_new[1:]])
        return self.X, self.Y

    def rotate(self, angle):
        """Rotate airfoil by given angle in degrees."""
        distance_COM_to_origin = self.getCOM()
        self.translate([-distance_COM_to_origin[0], -distance_COM_to_origin[1]])

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

    def recalculate_camber(self):
        """Recalculate camber line from current airfoil coordinates."""
        self.x_camber = self.X[:self.n]
        self.y_camber = (self.Y[:self.n] + self.Y[:self.n - 2:-1]) / 2
        return self.x_camber, self.y_camber

    def getCOM(self):
        """Calculate center of mass (centroid) of airfoil."""
        x = self.X
        y = self.Y
        A = np.abs(0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))
        Cx = (1 / (6 * A)) * np.sum((x[:-1] + x[1:]) * (x[:-1] * y[1:] - x[1:] * y[:-1]))
        Cy = (1 / (6 * A)) * np.sum((y[:-1] + y[1:]) * (x[:-1] * y[1:] - x[1:] * y[:-1]))
        self.COM = [Cx, Cy]
        return self.COM

    def center_airfoil(self):
        """Center airfoil at its centroid."""
        Cx, Cy = self.getCOM()
        self.translate([-Cx, -Cy])
        return self.X, self.Y

    def calculate_cross_section_area(self, chord_length=1):
        """Calculate cross-sectional area of airfoil."""
        scalar = chord_length / self.get_chord_length()
        self.A = np.abs(0.5 * (np.sum(self.X[:-1] * self.Y[1:] - self.X[1:] * self.Y[:-1]) + 
                              self.X[-1] * self.Y[0] - self.X[0] * self.Y[-1])) * scalar
        return self.A

    def get_chord_length(self):
        """Get chord length of airfoil."""
        return np.sqrt((self.x_chord[-1] - self.x_chord[0])**2 + (self.y_chord[-1] - self.y_chord[0])**2)

    def get_max_thickness_vertically(self):
        """Get maximum vertical thickness."""
        max_thickness = (self.Y[:self.n-1] - self.Y[:self.n-1:-1]).max()
        return max_thickness

    def get_max_thickness_perpendicular_to_camber(self, plot=False):
        """Get maximum thickness perpendicular to camber line."""
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
        self.max_thickness_location = self.x_camber[np.argmax(thickness)]

        if plot:
            print(f'Max thickness {self.max_thickness*100:.2f}% at {self.max_thickness_location*100:.2f}% chord.')
            print(f"Max camber {np.max(self.y_camber)*100:.2f}% at {self.x_camber[np.argmax(self.y_camber)]*100:.2f}% chord")

            plt.figure(figsize=(10, 5))
            plt.plot(self.X, self.Y, label='Airfoil')
            plt.plot(self.x_camber[1:-1], upper_y_new, label='Upper Surface', linestyle='--')
            plt.plot(self.x_camber[1:-1], lower_y_new, label='Lower Surface', linestyle='--')
            plt.plot(self.x_camber, self.y_camber, label='Camber Line', linestyle='-.', color='orange')
            plt.scatter(self.max_thickness_location, self.y_camber[np.argmax(thickness)], color='red',
                        label=f'Max Thickness: {self.max_thickness:.4f}')
            for i, _ in enumerate(x_upper_intersections):
                if i == np.argmax(thickness):
                    plt.plot([x_upper_intersections[i], x_lower_intersections[i]], 
                            [y_upper_intersections[i], y_lower_intersections[i]], color='black', linewidth=2)
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

    def get_max_thickness(self):
        """Get maximum thickness based on specified mode."""
        if self.thickness_mode == "perpendicular_to_chamber":
            return self.get_max_thickness_perpendicular_to_camber()
        elif self.thickness_mode == "vertically":
            return self.get_max_thickness_vertically()
        else:
            raise ValueError(f"Invalid thickness mode: {self.thickness_mode}")

    ########### Airfoil visualization functions ###########
    def plot(self, show=True, save=False, filename="airfoil.pdf", chord=False, camber=True):
        """Plot the airfoil."""
        fig, ax = plt.subplots()

        ax.plot(self.X, self.Y, label=f"Airfoil: {self.airfoil_name}")

        if camber:
            ax.plot(self.x_camber, self.y_camber, label="Camber", linestyle='--')
        if chord:
            ax.plot(self.x_chord, self.y_chord, label="Chord", linestyle=':')

        # Show transformations
        for shift in self.shifts:
            ax.scatter(shift[1][0], shift[1][1], color='red')
            ax.arrow(shift[1][0], shift[1][1], shift[0][0], shift[0][1],
                    head_width=0.012, head_length=0.025, fc='red', ec='red', length_includes_head=True)

        for rotation in self.rotations:
            arc = matplotlib.patches.Arc(rotation[1], 0.1, 0.1, angle=0, 
                                       theta1=rotation[0], theta2=0, color='green', linewidth=3)
            ax.add_patch(arc)

        ax.set_aspect('equal')
        ax.set_title(f"Airfoil: {self.airfoil_name}")
        ax.set_xlabel("X [-]")
        ax.set_ylabel("Y [-]")
        ax.grid(linestyle='dotted')
        ax.legend()

        if show:
            plt.show()
        if save:
            save_folder = os.path.join(os.getcwd(), "airfoil_imgs")
            os.makedirs(save_folder, exist_ok=True)
            fig.savefig(f'{save_folder}/{filename}', dpi=1200)
            plt.close(fig)

    ### Airfoil analysis functions ###
    def get_aero(self, alpha_variation=None, Re=1e6, mach=0.2, n_crit=9, model_size="xxxlarge"):
        """Get aerodynamic properties using NeuralFoil."""
        if alpha_variation is None:
            alpha_variation = self.alpha_variation
            
        self.aero = nf.get_aero_from_coordinates(
            coordinates=np.array([self.X, self.Y]).T,
            alpha=alpha_variation,
            Re=Re,
            model_size=model_size
        )
        return self.aero

    def plot_aero(self, Re):
        """Plot aerodynamic characteristics."""
        plt.plot(self.alpha_variation, self.aero["CL"], label=f"CL, Re={Re:.0g}")
        plt.plot(self.alpha_variation, self.aero["CD"], label=f"CD, Re={Re:.0g}")
        plt.ylabel('CL/CD [-]')
        plt.xlabel('Alpha [deg]')
        plt.legend()

    @classmethod
    def from_file(cls, filename, **kwargs):
        """Create airfoil from coordinate file."""
        if filename.endswith('.txt'):
            # Assume Selig format
            data = pd.read_csv(filename, sep="\s+", skiprows=1, engine='python')
            data.columns = ["X", "Y"]
            coordinates = data.astype(float).values
        else:
            # Assume CSV format
            coordinates = np.loadtxt(filename, delimiter=',')
        
        return cls(coordinates=coordinates, **kwargs)

    @classmethod
    def from_naca(cls, naca_designation, **kwargs):
        """Create airfoil from NACA designation."""
        # Create temporary instance to generate NACA coordinates
        temp = cls(coordinates=None, **kwargs)
        temp.X, temp.Y = temp.naca_airfoil(naca_designation)
        
        # Get coordinates and create new instance
        coordinates = np.column_stack([temp.X, temp.Y])
        return cls(coordinates=coordinates, airfoil_name=naca_designation, **kwargs)


# Example usage
if __name__ == "__main__":
    # Example 1: Create from NACA designation
    airfoil1 = Airfoil_Section.from_naca("NACA 4412", thickness_ratio=0.12, n=100, center=True)
    airfoil1.plot()
    
    # Example 2: Create from custom coordinates
    # Generate some example coordinates (diamond shape)
    n_points = 50
    x_upper = np.linspace(0, 1, n_points//2)
    y_upper = 0.1 * np.sin(np.pi * x_upper)  # Simple curved upper surface
    x_lower = np.linspace(1, 0, n_points//2)
    y_lower = -0.05 * np.sin(np.pi * x_lower)  # Simple curved lower surface
    
    coordinates = np.column_stack([
        np.concatenate([x_upper, x_lower]),
        np.concatenate([y_upper, y_lower])
    ])
    
    airfoil2 = Airfoil_Section(coordinates=coordinates, airfoil_name="Custom Diamond", 
                              thickness_ratio=0.08, center=True)
    airfoil2.plot()
    
    # Example 3: Analyze airfoil
    alpha_range = np.linspace(-5, 15, 21)
    aero_data = airfoil1.get_aero(alpha_variation=alpha_range, Re=1e6)
    
    plt.figure()
    airfoil1.plot_aero(Re=1e6)
    plt.title(f"Aerodynamic Analysis: {airfoil1.airfoil_name}")
    plt.show()