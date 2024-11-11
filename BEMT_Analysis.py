import os
import numpy as np
from Blade import Blade
from APCReader import APCReader
from BEMT_Solver import PropellerAnalysis, SectionForces, PropellerParameters



"""IMPORT PROPELLER GEOMETRY DATA"""
interpolation_points = 200
apcreader_object = APCReader(os.getcwd() + r"\APC Propeller Geometry Data\10x7E-PERF.PE0")
propeller_geometry = Blade(apcreader_object, interpolation_points, linear_interpolation=True)

# # #[radius, dr, c, twist, airfoil_data_x, airfoil_data_y]
# # prop_geometry_BEMT = propeller_geometry.export_geometry_for_BEMT_analysis()
# key_to_remove = 'Airfoil Section 37'
# # del prop_geometry_BEMT[key_to_remove]
#
# #[radius, dr, c, twist, A, airfoil_data_x, airfoil_data_y]
# prop_geometry = propeller_geometry.export_geometry_for_analysis()
# del prop_geometry[key_to_remove]


"""DEFINE BEMT CASE"""
#Geometrical Propeller Parameters
prop_radius = propeller_geometry.prop_radius
hub_radius = 0.4*0.0254
n_blades = propeller_geometry.n_blades

#Fluid parameters
rho = 1.225
mu = 1.81e-5
a_inf = 343

"""RUN BEMT ANALYSIS"""
#Operating conditions
RPM = 2000
v_inf = 0

#Define propeller parameters
propeller_params = PropellerParameters(
    prop_radius=prop_radius,
    hub_radius=hub_radius,
    n_blades=n_blades,
    RPM=RPM,
    rho=rho,
    a_inf=a_inf,
    mu=mu,
    v_inf=v_inf
)

# Create the analysis object
analysis = PropellerAnalysis(
    propeller_geometry=prop_geometry_BEMT,
    propeller_params=propeller_params
)

#Run BEMT
n_jobs = 12
analysis.run_BEMT(n_jobs=n_jobs)

#Compute total thrust, torque, CT and CP
total_thrust, total_torque, Ct, Cp = analysis.compute_total_forces()


"""RUN BEMT RPM SWEEP"""
Ct_simulation = []
Cp_simulation = []
total_thrust_simulation = []
total_torque_simulation = []
RPM_vec = np.linspace(1000,8000,16)
J_vec = np.linspace(0, 0.75, 20)

#Operating conditions
RPM_vec = np.linspace(1000, 12000, 11)
#v_inf_vec = J_vec * (RPM/60) * 2*prop_radius
v_inf = 0

""" for v_inf in v_inf_vec:
    #Define propeller parameters
    propeller_params = PropellerParameters(
        prop_radius=prop_radius,
        hub_radius=hub_radius,
        n_blades=n_blades,
        RPM=RPM,
        rho=rho,
        a_inf=a_inf,
        mu=mu,
        v_inf=v_inf
    )

    # Create the analysis object
    analysis = PropellerAnalysis(
        propeller_geometry=prop_geometry_BEMT,
        propeller_params=propeller_params
    )

    #Run BEMT
    n_jobs = 16
    analysis.run_BEMT(n_jobs=n_jobs)

    #Compute total thrust, torque, CT and CP
    total_thrust, total_torque, Ct, Cp = analysis.compute_total_forces()
    Ct_simulation = np.append(Ct_simulation, Ct)
    Cp_simulation = np.append(Cp_simulation, Cp)
    total_thrust_simulation = np.append(total_thrust_simulation, total_thrust)
    total_torque_simulation = np.append(total_torque_simulation, total_torque)

    print(f"Simulation with {v_inf} m/s inflow velocity finished") """


for RPM in RPM_vec:
    #Define propeller parameters
    propeller_params = PropellerParameters(
        prop_radius=prop_radius,
        hub_radius=hub_radius,
        n_blades=n_blades,
        RPM=RPM,
        rho=rho,
        a_inf=a_inf,
        mu=mu,
        v_inf=v_inf
    )

    # Create the analysis object
    analysis = PropellerAnalysis(
        propeller_geometry=prop_geometry_BEMT,
        propeller_params=propeller_params
    )

    #Run BEMT
    n_jobs = 16
    analysis.run_BEMT(n_jobs=n_jobs)

    #Compute total thrust, torque, CT and CP
    total_thrust, total_torque, Ct, Cp = analysis.compute_total_forces()
    Ct_simulation = np.append(Ct_simulation, Ct)
    Cp_simulation = np.append(Cp_simulation, Cp)
    total_thrust_simulation = np.append(total_thrust_simulation, total_thrust)
    total_torque_simulation = np.append(total_torque_simulation, total_torque)

    print(f"Simulation with {RPM} RPM finished")