import numpy as np
import pandas as pd
import os
from Blade import Blade
from APCReader import APCReader
from Hub import Hub
from Propeller import Propeller

from ocp_vscode import show_object
from datetime import datetime
import cadquery as cq
import matplotlib
matplotlib.use('TkAgg') # Needed to show plots in a separate window
import matplotlib.pyplot as plt

hub_geometry_types = {
                "-PERF": [1/2, 0.15, 0.5],
                "E-PERF": [0.8/2, 0.15, 0.4],
                "MR-PERF": [0.65/2, 0.15, 0.35],
                "WE-PERF": [0.8/2, 0.15, 0.41],
                "SF-PERF": [0.5/2, 0.15, 0.3],
                "EP-PERF": [0.8/2, 0.15, 0.38],
                "C-PERF": [1/2, 5/16, 0.56],
                }

interpolation_points = 15 ## Number of Points to define each, the lower and upper side of the airfoil. Total number of points per airfoil is 2*interpolation_points
### this markably impacts filesize and to an extend processing speed. Quality (especially around leading edge) decreases at around 20 points
### Lofts are automatically splining to points, so the general shape is preserved even with fewer points, however then the shape begins to deviate from the with more points. If super low filesize is crucial, this can be reduced to 5-10 points
counterclockwise_rotation = True  ## if false, propeller is mirrored
linear_interpolation = False  ## Set to true for testing and faster processing. Products are not linearly interpolated
rotation_axis_is_X = True ## Default Axis of rotation is around Z. If true, propeller is rotated such that axis of rotation is around X
########################################################################################################################
# READ ME
# This script generates a propeller.step from an APC propeller geometry file as found here: https://www.apcprop.com/technical-information/file-downloads/.
# The propeller is generated in 3 parts: Hub, Blade and Propeller.
# (Visualization via ocp_Viewer (Visual Studio Code only) or CQ-Editor)
# The hub geometry is generated based on the propeller name. The hub geometry can also be defined manually if inferred incorrectly.
# The blade is generated based on the APC propeller geometry file. Supported airfoil types are NACA, E63 and CLARK-Y.
# The transition part between the hub and the blade is a loft (splined) between the first 2 airfoils and 2 ellipses at the hub center.
# The propeller is generated by combining the hub, blade and transition part.
# The propeller is exported as a .step (and .stl file in the "Generated Propeller Exports" folder).

# Usage instructions:
# Select the APC propeller geometry file and change the filename variable to the path of the file.
# Set the hub geometry type to the propeller name or define the hub geometry manually.
# Run the script.
# ########################################################################################################################
### CHANGE FILENAME HERE
filename = os.getcwd() + r"\APC Propeller Geometry Data\10x6-PERF.PE0"
### SET HUB GEOMETRY HERE (or leave as is to infer from propeller name)
infer_hub_geometry = True  # If true, hub geometry is inferred from the propeller name and overwrites the following values. If False, hub geometry has to be defined manually below
outer_radius = 0.65 / 2
inner_radius = 0.15
thickness = 0.35
########################################################################################################################

propeller_name = os.path.basename(filename).split(".")[0]
hubtype = ''.join([i for i in propeller_name.split("x")[1] if not i.isdigit()])

### Create Hub
if infer_hub_geometry:
    outer_radius, inner_radius, thickness = hub_geometry_types[hubtype]
hub = Hub(interpolation_points*2-1, outer_radius, inner_radius, thickness)
# show_object(hub.part)
print("### Hub created ###")

### Create Blade
apcreader = APCReader(filename)
blade = Blade(apcreader, hub, interpolation_points, linear_interpolation=linear_interpolation)
s = blade.create_blade(export=False)
# show_object(s)

### Create Propeller
propeller = Propeller(blade, hub, linear_interpolation=linear_interpolation,
                      ccw=counterclockwise_rotation)


if isinstance(propeller.part, cq.Workplane):
    propeller.part.objects[0] = propeller.part.objects[0].scale(25.4)  # Convert from inches to mm
else:
    propeller.part = propeller.part.scale(25.4)

if rotation_axis_is_X:
    propeller.part = propeller.part.rotate((0,0,0), (0,1,0), 90)

show_object(propeller.part)

save_name = os.getcwd() + f"\\Generated Propeller Exports\\{propeller_name}"
# cq.exporters.export(propeller.part, f"{save_name}.step")
propeller.part.objects[0].exportStep(f"{propeller_name}.step") #, precision_mode=-1, write_pcurves=False)
# cq.exporters.export(propeller.part, f"{save_name}.stl")
print("### Propeller exported ###")


### test export
# test_part = cq.importers.importStep("propeller.step")
# show_object(test_part)


# TODO:
# Add more airfoil types. Only NACA, E63 and CLARK-Y are implemented. However, most APC propellers use only these airfoils.
# RESOLVED: Hub dimensions have to be defined manually. Find possible connections to automate

# Known Deviations from Product/Manual
# last airfoil is shifted in X and Y to match trailing edge of second last airfoil (by 90%). This is an arbitrary shift. Find a better way to finish the blade at the outer radius
# RESOLVED (Probably): Transition part is simple loft between last airfoil and ellipse at hub center. However, products have better transition.
# Transition part is sometimes exceeding the upper hub surface, which is not observed on products. -> Ellipse is shifted in Z (-thickness*0.05) to avoid this.
# Transition includes a second hub_edge, which is a copy of the first hub_edge with a small offset in X for a more consistent loft. This has been implemented to more resemble to the product but is not part of the manual.
# Propeller has chamfers at the hub. This is not implemented

# Potential Issues:
# Loft shapes can sometimes (strongly) change after unions...?
# Export Filesize is large (~50-70MB). This is likely due to the high number of faces in the lofts. This can be reduced by reducing the number of interpolation points. However, this will also reduce the quality of the propeller.