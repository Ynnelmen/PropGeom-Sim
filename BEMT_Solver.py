"""IMPORT PACKAGES"""
import numpy as np
import scipy.optimize
import pandas as pd
import aerosandbox as asb
from joblib import Parallel, delayed
import warnings


# Suppress RuntimeWarnings for overflow encountered in exp
warnings.filterwarnings("ignore", category=RuntimeWarning)

class PropellerParameters:
    """DEFINE GLOBAL PROPELLER PARAMETERS"""
    def __init__(self, prop_radius, hub_radius, n_blades, RPM, v_inf, a_inf, rho, mu):
        self.prop_radius = prop_radius
        self.prop_diameter = 2 * prop_radius
        self.hub_radius = hub_radius
        self.n_blades = n_blades
        self.RPM = RPM
        self.omega = 2 * np.pi * RPM / 60

        self.v_inf = v_inf
        self.a_inf = a_inf
        self.rho = rho
        self.mu = mu

    def __repr__(self):
        return (f"PropellerParameters(prop_diameter={self.prop_diameter}, hub_radius={self.hub_radius}, "
                f"n_blades={self.n_blades}, RPM={self.RPM}, v_inf={self.v_inf}, rho={self.rho}, mu={self.mu})")

class SectionForces:
    """SOLVE BEMT FOR EACH SECTION"""
    def __init__(self, airfoil_coordinates, r, dr, chord, theta, propeller_params):
        self.airfoil_coordinates = airfoil_coordinates
        self.r = r
        self.dr = dr
        self.chord = chord
        self.theta = theta
        self.propeller_params = propeller_params
        self.Re = 1e5  #Initial guess for Reynolds number
        self.Ma = 0.05 #Initial guess for Mach number

    @property
    def sigma(self):
        return self.propeller_params.n_blades * self.chord / (2 * np.pi * self.r)

    def prandtl_loss(self, phi):
        def prandtl(d_r, r, phi):
            f = self.propeller_params.n_blades * d_r / (2 * r * np.sin(phi))
            return 1.0 if -f > 500 else 2 * np.arccos(min(1.0, np.exp(-f))) / np.pi

        if phi == 0:
            return 1.0
        Ftip = prandtl(self.propeller_params.prop_radius - self.r, self.r, phi)
        Fhub = prandtl(self.r - self.propeller_params.hub_radius, self.r, phi)
        return Ftip * Fhub

    def airfoil_coefficients(self, alpha, Re, Ma, model_size="xxxlarge"):
        airfoil = asb.Airfoil(coordinates=self.airfoil_coordinates)
        full_output = asb.Airfoil.get_aero_from_neuralfoil(airfoil, alpha=alpha, Re=Re, mach=Ma, model_size=model_size)
        return full_output["CL"].item(), full_output["CD"].item()

    def section_parameters(self, phi):
        alpha = np.degrees(self.theta - phi)
        c_l, c_d = self.airfoil_coefficients(alpha, self.Re, self.Ma)
        c_l_prime = c_l * np.cos(phi) - c_d * np.sin(phi)
        c_d_prime = c_l * np.sin(phi) + c_d * np.cos(phi)
        F = self.prandtl_loss(phi)
        a = 1 / ((4 * F * np.sin(phi)**2) / (self.sigma * c_l_prime) - 1)
        a_prime = 1 / ((4 * F * np.sin(phi) * np.cos(phi)) / (self.sigma * c_d_prime) + 1)
        v_a = (1 + a) * self.propeller_params.v_inf
        v_t = self.propeller_params.omega * self.r* (1 - a_prime)
        W = np.sqrt(v_a**2 + v_t**2)
        self.Re = self.propeller_params.rho * W * self.chord / self.propeller_params.mu
        self.Ma = W/self.propeller_params.a_inf
        return alpha, c_l, c_d, F, a, a_prime, W, c_l_prime, c_d_prime

    def residual_function(self, phi):
        _, _, _, _, a, a_prime, _, _, _ = self.section_parameters(phi)
        return np.sin(phi) / (1 + a) - self.propeller_params.v_inf / (self.propeller_params.omega * self.r) * (np.cos(phi) / (1 - a_prime))

    def solve(self):
        result = scipy.optimize.root_scalar(self.residual_function, method='brentq', xtol=1e-5, bracket=[np.radians(0.1), np.radians(89.9)])
        if not result.converged:
            raise RuntimeError("Root finding did not converge")

        phi = result.root
        alpha, cl, cd, F, a, a_prime, W, c_l_prime, c_d_prime = self.section_parameters(phi)
        dT = self.sigma * np.pi * self.propeller_params.rho * W**2 * c_l_prime * self.r * self.dr
        dQ = self.sigma * np.pi * self.propeller_params.rho * W**2 * c_d_prime * self.r**2 * self.dr
        return phi, dT, dQ, alpha, a, a_prime, c_l_prime, c_d_prime, F, W, self.Re, self.Ma

class PropellerAnalysis:

    def __init__(self, propeller_geometry, propeller_params):
        self.propeller_geometry = propeller_geometry
        self.propeller_params = propeller_params
        self.solution_data = pd.DataFrame(columns=[
            "radius", "chord", "twist", "phi", "alpha", "Cl", "Cd",
            "a", "a_prime", "dT", "dQ", "F", "W", "Re", "Ma"
        ])

    def process_section(self, r, dr, chord, theta_deg, airfoil):
        theta = np.radians(theta_deg)
        airfoil_coordinates = np.array([airfoil.X, airfoil.Y]).T

        section_force = SectionForces(
            airfoil_coordinates=airfoil_coordinates,
            r=r,
            dr=dr,
            chord=chord,
            theta=theta,
            propeller_params=self.propeller_params
        )

        try:
            phi, dT, dQ, alpha, a, a_prime, Cl, Cd, F, W, Re, Ma = section_force.solve()
            return [
                r, chord, np.degrees(theta), np.degrees(phi), alpha,
                Cl, Cd, a, a_prime, dT, dQ, F, W, Re, Ma
            ]
        except RuntimeError as e:
            print(f"Error in section {r}: {e}")
            return [np.nan]*14

    def run_BEMT(self, n_jobs):
        all_tasks = (
            delayed(self.process_section)(r, dr, chord, twist, airfoil) for r, dr, chord, twist, airfoil in zip(self.propeller_geometry['r'],
                                                                                                                self.propeller_geometry['dr'],
                                                                                                                self.propeller_geometry['chord'],
                                                                                                                self.propeller_geometry['twist'],
                                                                                                                self.propeller_geometry['airfoil'])
        )
        self.results = Parallel(n_jobs=n_jobs)(all_tasks)
        for r, chord, theta, phi, alpha, Cl, Cd, a, a_prime, dT, dQ, F, W, Re, Ma in self.results:
            self.solution_data = pd.concat([self.solution_data, pd.DataFrame([[r, chord, theta, phi, alpha, Cl, Cd, a, a_prime, dT, dQ, F, W, Re, Ma]], columns=self.solution_data.columns)], ignore_index=True)

    def compute_total_forces(self):
        total_thrust = self.solution_data['dT'].sum()
        total_torque = self.solution_data['dQ'].sum()
        Ct = total_thrust / (self.propeller_params.rho * (self.propeller_params.RPM / 60)**2 * (self.propeller_params.prop_diameter)**4)
        Cp = 2 * np.pi * total_torque / (self.propeller_params.rho * (self.propeller_params.RPM / 60)**2 * (self.propeller_params.prop_diameter)**5)
        return total_thrust, total_torque, Ct, Cp