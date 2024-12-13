from Data_Processor import *
from UIUC_Processor import *

if __name__ == "__main__":
    # Set the directory containing the pickle files
    data_directory = r"D:\\Propeller Measurement Files"
    uiuc_directory = r".\\uiuc"

    data_processor = Data_Processor(data_directory)
    uiuc_processor = UIUC_Processor(uiuc_directory)

    ##### PREPROCESSING OF THE EXPERIMENTAL DATA #####

    # Load and calculate thrust offset for the specific file
    thrust_file_path = "D:\Propeller Measurement Files\FORCE_2024-11-28-10-11-01_RPM_3000.pkl"
    data_processor.thrust_sensor_offset = data_processor.load_and_calculate_thrust_offset(thrust_file_path)
    #print(f"Thrust Sensor Offset: {signal_processor.thrust_sensor_offset}")

    ###################################################
    propeller_type = '10x7e'
    rpm = 3000

    pkl_data = data_processor.load_data_by_propeller_and_rpm(propeller_type, rpm)
    #print("Number of entries in the list:", len(pkl_data))
    filename = uiuc_processor.find_uiuc_data('10x7e')
    data = uiuc_processor.load_uiuc_data(filename)
    # Process all pickle files
    pkl_data = data_processor.apply_thrust_offset_to_data(pkl_data)
    #print(pkl_data["auxilliary_sensors"]["thrust_corrected"])
    #print(pkl_data["sound"])
    pkl_data = data_processor.dbfs_dbfsa_calc(pkl_data, True, 45, 1000)
    pkl_data = data_processor.apply_iso_correction(5000, pkl_data, True, 45, 10000)

    # Initialize data for mean and standard deviation
    channel_means = {}
    channel_stds = {}

    # Iterate over all entries
    for entry in pkl_data:
        channels = entry["sound"]["channels"]
        for channel_key, channel_data in channels.items():
            spl_value = channel_data["overall_spl_value"]
            if channel_key not in channel_means:
                channel_means[channel_key] = []
            channel_means[channel_key].append(spl_value)

    # Calculate mean and standard deviation for each channel
    for channel_key in channel_means:
        channel_mean = np.mean(channel_means[channel_key])
        channel_std = np.std(channel_means[channel_key])
        channel_means[channel_key] = channel_mean
        channel_stds[channel_key] = channel_std

    # Prepare data for plotting
    channels = list(channel_means.keys())
    means = [channel_means[channel] for channel in channels]
    std_devs = [channel_stds[channel] for channel in channels]

    # Create a bar plot with error bars
    plt.figure(figsize=(12, 6))
    plt.bar(channels, means, yerr=std_devs, color='skyblue', capsize=5)

    # Add labels and title
    plt.xlabel("Channel")
    plt.ylabel("Overall SPL Value (dB)")
    plt.title("Mean Overall SPL Values with Standard Deviation for All Channels")
    plt.ylim(0,90)
    # Show the plot
    plt.tight_layout()
    plt.show()

    # Load time and pressure data for all files with the same propeller type and RPM
    #propeller_type = "PropellerA"
    #rpm = 3000
    #time_pressure_data = signal_processor.load_time_and_pressure_by_propeller_and_rpm(propeller_type, rpm)
    #print(f"Loaded time and pressure data for propeller type {propeller_type} and RPM {rpm}:")
    #for data in time_pressure_data:
    #    print(f"File: {data['file_name']}, Channel: {data['channel_id']}")