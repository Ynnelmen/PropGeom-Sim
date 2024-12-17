import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from BEMT_Acoustic_Job import Job
import pickle
from Data_Processor import Data_Processor
from UIUCProcessor import UIUCProcessor

## Notebook to compare Measurement results from different runs to BEMT method and UIUC data

### Load and Process Measurement Data
measurement_data_folder = r"D:\Propeller Measurement Files"
measurement_names = [n.name for n in os.scandir(measurement_data_folder) if "x" in n.name and n.name.endswith(".pkl") and "e_" in n.name]
unique_propeller_names = np.sort(np.array(list(set([n.split("_")[0] for n in measurement_names]))))
rpms = [n.split("RPM_")[-1].split('.')[0] for n in measurement_names]
unique_rpms = np.sort(np.array(list(set(rpms))))


mdp = Data_Processor(measurement_data_folder)  # Measurement Data Processor

measurement_results = pd.DataFrame(columns=["Propeller", "RPM", "Meas_Thrust", "Meas_CT"])
actual_rpms_per_prop = {}
for n, rpm in enumerate(unique_rpms):
    base_thrust_meas = [n.name for n in os.scandir(measurement_data_folder) if n.name.startswith("FORCE") and n.name.endswith(".pkl") and f"RPM_{rpm}" in n.name]
    base_thrust_path = os.path.join(measurement_data_folder, base_thrust_meas[0])
    mdp.thrust_sensor_offset = mdp.load_and_calculate_thrust_offset(base_thrust_path)

    for prop_name in unique_propeller_names:
        if n == 0:
            actual_rpms_per_prop[prop_name] = []
        file_path = [n.name for n in os.scandir(measurement_data_folder) if prop_name in n.name and f"RPM_{rpm}" in n.name and n.name.endswith(".pkl")]
        if not file_path:
            print(f"No measurement data found for {prop_name} at {rpm} RPM")
            continue
        for num, file in enumerate(file_path):
            meas_data = pickle.load(open(os.path.join(measurement_data_folder, file), "rb")) # TODO only loading the first measurement!

            thrust_data = meas_data['auxilliary_sensors']['thrust'] - mdp.thrust_sensor_offset

            temperature = np.array(meas_data['auxilliary_sensors']['temperature']) - 2
            humidity = 0.45
            pressure = np.array(meas_data['auxilliary_sensors']['air_pressure']) * 100
            actual_rpm = np.array(meas_data['auxilliary_sensors']['electric_rpm']).mean()
            actual_rpms_per_prop[prop_name].append(actual_rpm)
            ct = mdp.ct_calculation(prop_name, thrust_data, actual_rpm, temperature, pressure, humidity).mean()

            measurement_results = pd.concat([measurement_results, pd.DataFrame({"Propeller": prop_name, "RPM": actual_rpm, "meas_nr": num, "Meas_Thrust": thrust_data.mean(),
                                                                                "Meas_CT": ct},index=[0])])
measurement_results.reset_index(drop=True, inplace=True)
measurement_results.sort_values(by=["Propeller", "RPM", "meas_nr"], inplace=True)


### load and process UIUC data
uiuc_data_folder = r"D:\uiuc"
uiuc_data = {}

uiuc_proc = UIUCProcessor(uiuc_data_folder)
for prop_name in unique_propeller_names:
    try:
        file_name = uiuc_proc.find_uiuc_data(prop_name)
        df = pd.DataFrame(uiuc_proc.load_uiuc_data(file_name)['data'])
        # interpolate rpm values
        df_interpolated = pd.DataFrame({'rpm': actual_rpms_per_prop[prop_name]})
        df_interpolated['ct'] = np.interp(actual_rpms_per_prop[prop_name], df['rpm'], df['ct'])
        df_interpolated['cp'] = np.interp(actual_rpms_per_prop[prop_name], df['rpm'], df['cp'])
        df = pd.concat([df, df_interpolated])
        df.sort_values(by='rpm', inplace=True)
        df.reset_index(drop=True, inplace=True)

        uiuc_data[prop_name] = df
    except FileNotFoundError as e:
        print(e)


bemt_results = pd.DataFrame(columns=["Propeller", "RPM", "BEMT_Thrust", "BEMT_CT", "BEMT_CP"])
### create BEMT Data
for prop_name in unique_propeller_names:
    for n, rpm in enumerate(unique_rpms):
        if n == 0:
            bemt_results = pd.DataFrame(columns=["Propeller", "RPM", "BEMT_Thrust", "BEMT_CT", "BEMT_CP"])
        job = Job(prop_name, rpm)
        job.run_BEMT()
        bemt_results = pd.concat([bemt_results, pd.DataFrame({"Propeller": prop_name, "RPM": rpm, "BEMT_Thrust": job.total_thrust, "BEMT_CT": job.Ct, "BEMT_CT": job.Cp},
                                                             index=[0])])

bemt_results.reset_index(drop=True, inplace=True)

### Plotting CT
fig, ax = plt.subplots(1, len(unique_propeller_names), figsize=(10, 5), sharex=True, sharey=True)
ax[0].set_ylabel("CT [-]")
for num, prop_name in enumerate(unique_propeller_names):
    try:
        ax[num].plot(uiuc_data[prop_name]['rpm'], uiuc_data[prop_name]['ct'], label=f"UIUC")
    except:
        pass
    ax[num].plot(measurement_results[(measurement_results["Propeller"] == prop_name)]["RPM"], measurement_results[(measurement_results["Propeller"] == prop_name)]["Meas_CT"],
                 label=f"Meas")
    ax[num].plot(bemt_results[(bemt_results["Propeller"] == prop_name)]["RPM"], bemt_results[(bemt_results["Propeller"] == prop_name)]["BEMT_CT"], label=f"BEMT")
    ax[num].set_title(f"{prop_name}")
    ax[num].set_xlabel("RPM")
    ax[num].set_xticks([3000, 5000, 7000])
    ax[num].set_xlim(2500, 7250)
    ax[num].legend()


### Plotting CP
fig, ax = plt.subplots(1, len(unique_propeller_names), figsize=(10, 5), sharex=True, sharey=True)
ax[0].set_ylabel("CP [-]")
for num, prop_name in enumerate(unique_propeller_names):
    try:
        ax[num].plot(uiuc_data[prop_name]['rpm'], uiuc_data[prop_name]['cp'], label=f"UIUC")
    except:
        pass
    ax[num].plot(bemt_results[(bemt_results["Propeller"] == prop_name)]["RPM"], bemt_results[(bemt_results["Propeller"] == prop_name)]["BEMT_CP"], label=f"BEMT")
    ax[num].set_title(f"{prop_name}")
    ax[num].set_xlabel("RPM")
    ax[num].set_xticks([3000, 5000, 7000])
    ax[num].set_xlim(2500, 7250)
    ax[num].legend()