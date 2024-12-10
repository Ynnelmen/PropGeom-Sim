import matplotlib.pyplot as plt
import pandas as pd
from APC_Reader import APC_Reader
from Blade import Blade
from Hub import Hub
import os
import numpy as np

interpolation_points = 100
hub = Hub(interpolation_points * 2 - 1, 0.65 / 2, 0.15, 0.36)

propeller_data_folder = os.getcwd() + r"\APC Propeller Geometry Data"
section_adaptation = np.array([0.4, 1, 1.5, 0.9])
E63_correction = 1

diffs = pd.DataFrame(columns=["propeller", "area_diff_early", "area_diff_late"])
blades = []
names = []
for file in os.listdir(propeller_data_folder):
    if file.endswith(".PE0") and "E-PERF" in file and int(file.split("x")[0]) <= 20 and not "W" in file:
        name = file.split("-")[0]

        apcreader = APC_Reader(os.getcwd() + r"\APC Propeller Geometry Data" + "\\" + file)
        blade = Blade(apcreader, hub, interpolation_points, linear_interpolation=False,
                      section_adaptation=section_adaptation, E63_correction=E63_correction)
        blade.comparisons_plot(show=False, save=True)

        area_diff_early, area_diff_late = blade.APC_comparisons()
        diffs = pd.concat([diffs, pd.DataFrame([[name, area_diff_early, area_diff_late]], columns=["propeller", "area_diff_early", "area_diff_late"])], ignore_index=True)

        blades.append(blade)
        names.append(name)

diffs["propeller"] = [f.split("-")[0] for f in diffs["propeller"]]
diffs["inch"] = [int(f.split("x")[0]) for f in diffs["propeller"]]
diffs = diffs.sort_values(by="inch")
average_diff_late = diffs["area_diff_late"].mean()
average_diff_early = diffs["area_diff_early"].mean()
print(average_diff_early,average_diff_late)

plt.figure(figsize=(16, 8))
plt.bar(diffs["propeller"], diffs["area_diff_late"], label = "83% Trans - End", alpha = 0.8)
plt.bar(diffs["propeller"], diffs["area_diff_early"], label = "start - 16% Trans", alpha = 0.8)
plt.xticks(rotation=90)
plt.ylabel("Area Difference [%]")
plt.legend()

