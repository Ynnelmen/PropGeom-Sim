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
unique_propeller_names = ["8x4e", "8x6e", "8x8e", "9x6e", "9x9e", "10x5e", "10x7e", "10x8e", "11x7e", "11x8e"]
unique_propeller_names = unique_propeller_names[::-1]
rpms = [int(n.split("RPM_")[-1].split('.')[0]) for n in measurement_names]
unique_rpms = np.sort(np.array(list(set(rpms))))


mdp = Data_Processor(measurement_data_folder)  # Measurement Data Processor

measurement_results = pd.DataFrame(columns=["Propeller", "RPM", "Meas_Thrust", "Meas_CT"])
measurement_results_avgd = pd.DataFrame(columns=["Propeller", "target_rpm", "ct_avgd", "ct_std"])
actual_rpms_per_prop = {}
all_cts = {}
all_thrusts = {}
for n, rpm in enumerate(unique_rpms):
    rpm = int(rpm)
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
        all_ct_per_prop = np.array([])
        all_thrusts_per_prop = np.array([])
        for num, file in enumerate(file_path):
            meas_data = pickle.load(open(os.path.join(measurement_data_folder, file), "rb")) # TODO only loading the first measurement!

            thrust_data = meas_data['auxilliary_sensors']['thrust'] - mdp.thrust_sensor_offset
            # plt.plot(thrust_data)
            # print(meas_data['measurement_initialization_time'])

            temperature = np.array(meas_data['auxilliary_sensors']['temperature']) - 2
            humidity = 0.45
            pressure = np.array(meas_data['auxilliary_sensors']['air_pressure']) * 100
            actual_rpm = np.array(meas_data['auxilliary_sensors']['electric_rpm'])
            actual_rpms_per_prop[prop_name].append(actual_rpm.mean())
            ct = mdp.ct_calculation(prop_name, thrust_data, actual_rpm, temperature, pressure, humidity)
            ct = np.array(ct)

            all_ct_per_prop = np.append(all_ct_per_prop, ct)
            all_cts[f"{prop_name} {rpm}"] = ct
            all_thrusts_per_prop = np.append(all_thrusts_per_prop, thrust_data)
            all_thrusts[f"{prop_name} {rpm}"] = thrust_data
            measurement_results = pd.concat([measurement_results, pd.DataFrame({"Propeller": prop_name, "RPM": actual_rpm.mean(),
                                                                                "target_rpm": rpm, "meas_nr": num, "Meas_Thrust": thrust_data.mean(),
                                                                                "Meas_CT": ct.mean()}, index=[0])])

        ct_avgd = all_ct_per_prop.mean()
        ct_avg_high = np.percentile(all_ct_per_prop, 65)
        ct_avg_low = np.percentile(all_ct_per_prop, 35)
        thrust_avgd = all_thrusts_per_prop.mean()
        thrust_avg_high = np.percentile(all_thrusts_per_prop, 65)
        thrust_avg_low = np.percentile(all_thrusts_per_prop, 35)
        measurement_results_avgd = pd.concat([measurement_results_avgd, pd.DataFrame({"Propeller": prop_name,
                                                                                      "target_rpm": rpm, "ct_avgd": ct_avgd,
                                                                                      "ct_high": ct_avg_high, "ct_low": ct_avg_low,
                                                                                      "thrust_avgd": thrust_avgd, "thrust_high": thrust_avg_high,
                                                                                      "thrust_low": thrust_avg_low}, index=[0])])

measurement_results.reset_index(drop=True, inplace=True)
measurement_results.sort_values(by=["Propeller", "RPM", "meas_nr"], inplace=True)
measurement_results_avgd.reset_index(drop=True, inplace=True)
measurement_results_avgd.sort_values(by=["Propeller", "target_rpm"], inplace=True)

### load and process UIUC data
uiuc_data_folder = r"D:\uiuc"
uiuc_data = {}

uiuc_proc = UIUCProcessor(uiuc_data_folder)
for prop_name in unique_propeller_names:
    try:
        file_name = uiuc_proc.find_uiuc_data(prop_name)
        df = pd.DataFrame(uiuc_proc.load_uiuc_data(file_name)['data'])
        # interpolate rpm values
        # df_interpolated = pd.DataFrame({'rpm': actual_rpms_per_prop[prop_name]})
        # df_interpolated['ct'] = np.interp(actual_rpms_per_prop[prop_name], df['rpm'], df['ct'])
        # df_interpolated['cp'] = np.interp(actual_rpms_per_prop[prop_name], df['rpm'], df['cp'])
        # df = pd.concat([df, df_interpolated])
        df.sort_values(by='rpm', inplace=True)
        df.reset_index(drop=True, inplace=True)

        uiuc_data[prop_name] = df
    except FileNotFoundError as e:
        print(e)

### create BEMT Data
import datetime
start = datetime.datetime.now()
bemt_results = pd.DataFrame(columns=["Propeller", "RPM", "BEMT_Thrust", "BEMT_CT", "BEMT_CP"])
for prop_name in unique_propeller_names:
    for n, rpm in enumerate(np.linspace(2500, 7500, (7500-2500)//250 + 1)):
        job = Job(prop_name, rpm)
        job.run_BEMT()
        bemt_results = pd.concat([bemt_results, pd.DataFrame({"Propeller": prop_name, "RPM": int(rpm), "BEMT_Thrust": job.total_thrust, "BEMT_CT": job.Ct, "BEMT_CP": job.Cp},
                                                             index=[0])])
bemt_results.reset_index(drop=True, inplace=True)
print(f"Time taken: {datetime.datetime.now() - start}")

### Plotting CT
colors = {"meas": "tab:blue", "bemt": "tab:orange", "uiuc": "tab:green"}
fig, ax = plt.subplots(1, len(unique_propeller_names), figsize=(16,9), sharex=True, sharey=True)
ax[0].set_ylabel("CT [-]")
for num, prop_name in enumerate(unique_propeller_names):

    # ax[num].plot(measurement_results[(measurement_results["Propeller"] == prop_name)]["RPM"], measurement_results[(measurement_results["Propeller"] == prop_name)]["Meas_CT"],
    #              label=f"Meas")
    ax[num].plot(measurement_results_avgd[(measurement_results_avgd["Propeller"] == prop_name)]["target_rpm"], measurement_results_avgd[(measurement_results_avgd["Propeller"] == prop_name)]["ct_avgd"],
                    label=f"Meas", color=colors["meas"], marker="o", markersize=3.5, linewidth=0.8)
    #use fill_between to show the standard deviation
    ax[num].fill_between(pd.to_numeric(measurement_results_avgd[(measurement_results_avgd["Propeller"] == prop_name)]["target_rpm"]),
                            measurement_results_avgd[(measurement_results_avgd["Propeller"] == prop_name)]["ct_low"],
                            measurement_results_avgd[(measurement_results_avgd["Propeller"] == prop_name)]["ct_high"],
                         alpha=0.4, color=colors["meas"])
    ax[num].plot(bemt_results[(bemt_results["Propeller"] == prop_name)]["RPM"], bemt_results[(bemt_results["Propeller"] == prop_name)]["BEMT_CT"], label=f"BEMT", color=colors["bemt"])
    try:
        ax[num].plot(uiuc_data[prop_name]['rpm'], uiuc_data[prop_name]['ct'], label=f"UIUC", color=colors["uiuc"], marker="o", markersize=3.5, linewidth=0.8)
    except:
        pass
    ax[num].set_title(f"{prop_name}")
    ax[num].set_xlabel("RPM")
    ax[num].set_xticks([3000, 5000, 7000])
    ax[num].set_xlim(2500, 7250)
ax[-1].legend()
plt.savefig(r"C:\Users\RhinerL\OneDrive - inspire AG\BAZL\CODE NEW\results\ct.png", dpi=500)
plt.close()

avg_uiuc_ct = [prop[(prop["rpm"]>3000) & (prop["rpm"]<7000)]["ct"].mean() for prop in uiuc_data.values()]
avg_meas_ct = [measurement_results_avgd[(measurement_results_avgd["Propeller"] == prop_name)]["ct_avgd"].mean() for prop_name in unique_propeller_names]
diff = np.array(avg_meas_ct).mean() / np.array(avg_uiuc_ct).mean()
# avg_bemt_ct = [bemt_results[(bemt_results["Propeller"] == prop_name)]["BEMT_CT"].mean() for prop_name in unique_propeller_names]

### Thrust Comparison
fig, ax = plt.subplots(1, len(unique_propeller_names), figsize=(16,9), sharex=True, sharey=True)
ax[0].set_ylabel("Thrust [N]")
for num, prop_name in enumerate(unique_propeller_names):
    ax[num].plot(measurement_results_avgd[(measurement_results_avgd["Propeller"] == prop_name)]["target_rpm"], measurement_results_avgd[(measurement_results_avgd["Propeller"] == prop_name)]["thrust_avgd"],
                    label=f"Meas", color=colors["meas"])
    ax[num].fill_between(pd.to_numeric(measurement_results_avgd[(measurement_results_avgd["Propeller"] == prop_name)]["target_rpm"]),
                            measurement_results_avgd[(measurement_results_avgd["Propeller"] == prop_name)]["thrust_low"],
                            measurement_results_avgd[(measurement_results_avgd["Propeller"] == prop_name)]["thrust_high"],
                            alpha=0.4, color=colors["meas"])
    ax[num].plot(bemt_results[(bemt_results["Propeller"] == prop_name)]["RPM"], bemt_results[(bemt_results["Propeller"] == prop_name)]["BEMT_Thrust"], label=f"BEMT", color=colors["bemt"])
    ax[num].set_title(f"{prop_name}")
    ax[num].set_xlabel("RPM")
    ax[num].set_xticks([3000, 5000, 7000])
    ax[num].set_xlim(2500, 7250)
ax[-1].legend()

plt.savefig(r"C:\Users\RhinerL\OneDrive - inspire AG\BAZL\CODE NEW\results\thrust.png", dpi=500)
plt.close()



### Plotting CP
fig, ax = plt.subplots(1, len(unique_propeller_names), figsize=(16,9), sharex=True, sharey=True)
ax[0].set_ylabel("CP [-]")
for num, prop_name in enumerate(unique_propeller_names):
    try:
        ax[num].plot(uiuc_data[prop_name]['rpm'], uiuc_data[prop_name]['cp'], label=f"UIUC", color=colors["uiuc"])
    except:
        pass
    ax[num].plot(bemt_results[(bemt_results["Propeller"] == prop_name)]["RPM"], bemt_results[(bemt_results["Propeller"] == prop_name)]["BEMT_CP"], label=f"BEMT", color=colors["bemt"])
    ax[num].set_title(f"{prop_name}")
    ax[num].set_xlabel("RPM")
    ax[num].set_xticks([3000, 5000, 7000])
    ax[num].set_xlim(2500, 7250)
ax[-1].legend()

plt.savefig(r"C:\Users\RhinerL\OneDrive - inspire AG\BAZL\CODE NEW\results\cp.png", dpi=500)
plt.close()

