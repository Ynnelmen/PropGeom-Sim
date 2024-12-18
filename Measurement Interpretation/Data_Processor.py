import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

class Data_Processor:
    """
    A class to process and evaluate signals in both the time and frequency domains, 
    with functionalities for plotting, calculating various metrics, 
    and saving plots.
    """

    def __init__(self, data_directory):
        """
        Initialize the Data_Processor.

        Args:
            data_directory (str): The directory containing the pkl files.
        """
        self.data_directory = data_directory
        self.thrust_sensor_offset = 0  # Default offset value

    def load_pkl_file(self, file_path):
        """
        Load and deserialize data from a pickle file.
        
        Args:
            file_path (str): Path to the pickle file.

        Returns:
            dict: The loaded pickle data.
        """
        with open(file_path, 'rb') as file:
            return pickle.load(file)

    def plot_time_signal(self, time, pressure, name, save=False, save_path="./"):
        """
        Plot and optionally save a time-domain signal.
        Args:
            time (list): Time vector.
            pressure (list): Pressure values.
            name (str): Title of the plot.
            save (bool): Whether to save the plot.
            save_path (str): Directory to save the plot.
        """
        plt.figure(figsize=(10, 6), dpi=1200)  # Set DPI during figure creation
        plt.plot(time, pressure, linestyle='-')
        plt.title(name, fontsize=16)
        plt.xlabel("Time (s)", fontsize=14)
        plt.ylabel("Pressure (Pa)", fontsize=14)
        plt.grid(True)
        if save:
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(os.path.join(save_path, f"{name}_time_plot.png"), dpi=300)
        plt.show()

    def std_dev_calc(self, spl_value_background_correction_vec):
        """
        Calculate the standard deviation for a given SPL value vector.
        Args:
            spl_value_background_correction_vec (list): Background noise-corrected SPL values.
        Returns:
            float: Standard deviation of the given values.
        """
        if len(spl_value_background_correction_vec) <= 1:
            return 0

        spl_mean = np.mean(spl_value_background_correction_vec)
        std_dev = np.sqrt(np.sum((spl_value_background_correction_vec - spl_mean) ** 2) / (len(spl_value_background_correction_vec) - 1))

        return std_dev

    def overall_std_dev_calc(self, std_dev):
        """
        Calculate the overall standard deviation for a microphone.
        Args:
            std_dev (float): Standard deviation of the microphone measurements.
        Returns:
            float: Overall standard deviation including compensation.
        """
        comp_std_dev = 0.5
        overall_std_dev = np.sqrt(std_dev**2 + comp_std_dev**2)
        return overall_std_dev

    def background_noise_correction_calc(self, spl_value_difference):
        """
        Calculate the background noise correction factor.
        Args:
            spl_value_difference (float): SPL value difference between noise and background.
        Returns:
            float: Background noise correction factor.
        """
        if spl_value_difference > 15:
            return 0
        elif spl_value_difference < 6:
            return 1.26
        K1 = -10 * np.log10(1 - 10**(-0.1 * spl_value_difference))
        return K1

    def spl_value_difference_calc(self, noise_spl_value, background_noise_spl_value):
        """
        Calculate SPL value difference between noise and background noise.
        Args:
            noise_spl_value (float): Noise SPL value.
            background_noise_spl_value (float): Background noise SPL value.
        Returns:
            float: SPL value difference.
        """
        return noise_spl_value - background_noise_spl_value

    def spl_value_background_correction_calc(self, noise_spl_value, K1):
        """
        Calculate background noise corrected SPL value.
        Args:
            noise_spl_value (float): Noise SPL value.
            K1 (float): Background noise correction factor.
        Returns:
            float: Corrected SPL value.
        """
        return noise_spl_value - K1

    def overall_spl_value_surface_calc(self, spl_value_background_correction_vec):
        """
        Calculate the overall SPL value of all microphones on the surface.
        Args:
            spl_value_background_correction_vec (list): Corrected SPL values.
        Returns:
            float: Overall SPL value.
        """
        overall_spl_value_surface = 10 * np.log10(np.mean(10**(0.1 * np.array(spl_value_background_correction_vec))))
        return overall_spl_value_surface

    def dbffta(self, dbfs, freq):
        """
        Perform A-weighting on frequency spectrum data.
        Args:
            dbfs (list): Amplitude values in dBFS.
            freq (list): Frequency vector.
        Returns:
            tuple: A-weighted frequencies and corresponding amplitudes.
        """
        np.seterr(divide='ignore')
        R_a = lambda f: (12194**2 * f**4) / ((f**2 + 20.6**2) * np.sqrt((f**2 + 107.7**2)*(f**2 + 737.9**2)) * (f**2 + 12194**2))
        A_weight = lambda f: 20 * np.log10(R_a(f)) - 20 * np.log10(R_a(1000))
        weight = np.array([A_weight(f) for f in freq])
        dbfs_A = dbfs + weight
        return freq, dbfs_A

    def blade_passing_frequency_calc(self, rpm, n_blades, j):
        """
        Calculate the blade passing frequency at the current RPM.
        Args:
            rpm (float): Rotations per minute.
            n_blades (int): Number of blades.
            j (int): The harmonic order.
        Returns:
            float: Blade passing frequency.
        """
        return rpm * n_blades / 60 * j

    def third_octave_calc(self, blade_passing_frequency):
        """
        Calculate the third octave bandwidth of a frequency.
        Args:
            blade_passing_frequency (float): Frequency to calculate for.
        Returns:
            tuple: Lower and upper bounds of the third octave.
        """
        f1 = blade_passing_frequency / (2**(1/6))
        f2 = blade_passing_frequency * (2**(1/6))
        return f1, f2

    def create_mask(self, dbfs, freq, f1, f2):
        """
        Create a mask for the frequency and amplitude values in a specific range.
        Args:
            dbfs (list): Amplitude values in dBFS.
            freq (list): Frequency values.
            f1 (float): Lower bound.
            f2 (float): Upper bound.
        Returns:
            tuple: Filtered frequency and amplitude values.
        """
        freq_mask_range = (freq >= f1) & (freq <= f2)
        freq_mask = freq[freq_mask_range]
        dbfs_mask = dbfs[freq_mask_range]
        return freq_mask, dbfs_mask

    def overall_spl_value_calc(self, dbfs):
        """
        Calculate the overall SPL value.
        Args:
            dbfs (list): Amplitude values in dBFS.
        Returns:
            float: Overall SPL value.
        """
        power_values = 10**(dbfs / 10)
        overall_spl_value = 10 * np.log10(np.sum(power_values)) if power_values.size > 0 else 0
        return overall_spl_value

    def dbfft(self, input_vec, fs, ref=1):
        """
        Calculate the frequency spectrum of a signal on a dB scale relative to a specified reference.

        Args:
            input_vec (list or np.array): The input time-domain signal.
            fs (float): Sampling frequency of the input signal in Hz.
            ref (float, optional): Reference value for the dB calculation. Default is 1.

        Returns:
            tuple: Frequency vector and the corresponding spectrum in dB.
                - freq_vec (np.array): Frequency vector.
                - spec_db (np.array): Spectrum magnitude in dB relative to the specified reference.
        """
        if len(input_vec) == 0:
            return None, None  # Return None if input vector is empty

        # Perform the Fast Fourier Transform (FFT) on the input signal
        spec = np.fft.rfft(input_vec)

        # Calculate the magnitude of the FFT and normalize it by the length of the input vector
        spec_mag = (np.abs(spec) * np.sqrt(2)) / len(input_vec)

        # Convert the magnitude to dB scale relative to the specified reference value
        spec_db = 20 * np.log10(spec_mag / ref)

        # Generate the frequency vector corresponding to the FFT result
        freq_vec = np.fft.rfftfreq(len(input_vec), d=1/fs)

        return freq_vec, spec_db

    def load_and_calculate_thrust_offset(self, file_path):
        """
        Load a specific pickle file and calculate the mean of the thrust data.

        Args:
            file_path (str): Path to the pickle file.

        Returns:
            float: Mean thrust value, named thrust_sensor_offset.
        """
        pkl_data = self.load_pkl_file(file_path)
        thrust_data = pkl_data['auxilliary_sensors']['thrust']
        thrust_sensor_offset = np.mean(thrust_data) if thrust_data else 0
        self.thrust_sensor_offset = thrust_sensor_offset
        return thrust_sensor_offset

    def subtract_thrust_offset(self, thrust_values):
        """
        Subtract the calculated thrust offset from each thrust value.

        Args:
            thrust_values (list or np.array): List of thrust values.

        Returns:
            np.array: Thrust values after offset subtraction.
        """
        return np.array(thrust_values) - self.thrust_sensor_offset

    def process_pkl_files(self, thrust_offset_file = "./output/FORCE_2024-11-28-10-11-01_SoundMeasurement.pkl"):
        """
        Process all pickle files in the data directory, except for the pickle files containing the force offset measurements.
        Per default: FORCE_2024-11-28-10-11-01_SoundMeasurement.pkl file.
        """
        for file_name in os.listdir(self.data_directory):
            if file_name.endswith('.pkl') and file_name != thrust_offset_file:
                file_path = os.path.join(self.data_directory, file_name)
                print(f"Processing file: {file_path}")

                # Load the data
                pkl_data = self.load_pkl_file(file_path)

                # Subtract thrust offset from thrust data
                if 'auxilliary_sensors' in pkl_data and 'thrust' in pkl_data['auxilliary_sensors']:
                    pkl_data['auxilliary_sensors']['thrust'] = self.subtract_thrust_offset(pkl_data['auxilliary_sensors']['thrust'])

                # Process all channels in the file
                self.process_all_channels(pkl_data)

    def process_all_channels(self, pkl_data):
        """
        Process all channels in the given pkl_data.

        Args:
            pkl_data (dict): The loaded pickle data.
        """
        for channel_id, channel_data in pkl_data['sound']['channels'].items():
            time = channel_data['time']
            pressure = channel_data['pressure']

            #if time is not None and pressure is not None:
            #    self.plot_time_signal(time, pressure, f"Channel {channel_id} Time-Domain Signal")
            #else:
            #    print(f"No data found for Channel {channel_id}.")

    def load_time_and_pressure_by_propeller_and_rpm(self, propeller_type, rpm):
        """
        Load the time and pressure data of all pickle files for the same propeller type and RPM.

        Args:
            propeller_type (str): The type of the propeller.
            rpm (int): The target RPM value.

        Returns:
            list: A list of dictionaries containing grouped channel data for each matching file.
        """
        rpm_data_set = []

        for file_name in os.listdir(self.data_directory):
            if file_name.endswith('.pkl') and propeller_type in file_name and f"RPM_{rpm}" in file_name:
                file_path = os.path.join(self.data_directory, file_name)
                print(f"Loading time and pressure from file: {file_path}")

                # Load the data
                pkl_data = self.load_pkl_file(file_path)

                # Group channels into a single entry
                channels = []
                for channel_id, channel_data in pkl_data['sound']['channels'].items():
                    time = channel_data['time']
                    pressure = channel_data['pressure']
                    sampling_frequency = channel_data['sampling_frequency']

                    if time is not None and pressure is not None:
                        channels.append({
                            'channel_id': channel_id,
                            'sampling_frequency': sampling_frequency,
                            'time': time,
                            'pressure': pressure
                        })

                if channels:
                    rpm_data_set.append({
                        'propeller_type': propeller_type,
                        'targeted_rpm': rpm,
                        'measurement_initialization_time': pkl_data['measurement_initialization_time'],
                        'channels': channels
                    })

        return rpm_data_set
    
    def background_overall_spl_calc(self, rpm):
        """
        Calculate the overall SPL value of the background noise for each channel at a specific RPM.

        Args:
            rpm (int): The target RPM value.

        Returns:
            list: A list of dictionaries containing the RPM and channel-specific SPL values grouped under 'channels'.
        """
        # Load the time and pressure data for the background at the given RPM
        rpm_dataset = self.load_time_and_pressure_by_propeller_and_rpm('Background', rpm)

        consolidated_results = []

        # Iterate over the dataset entries
        for entry in rpm_dataset:
            channels = []

            for channel in entry['channels']:
                time = channel['time']
                pressure = channel['pressure']

                if time is not None and pressure is not None:
                    # Perform dbfft calculation
                    freq, dbfs = self.dbfft(time, pressure)

                    # Perform dbffta calculation
                    dbffta = self.dbffta(freq, dbfs)

                    # Calculate overall SPL values for dbfs and dbfsa
                    dbfs_overall_spl_value = self.overall_spl_value_calc(dbfs)
                    dbfsa_overall_spl_value = self.overall_spl_value_calc(dbffta)

                    # Append channel-specific results
                    channels.append({
                        "channel_id": channel['channel_id'],
                        "dbfs_background_overall_spl": dbfs_overall_spl_value,
                        "dbfsa_background_overall_spl": dbfsa_overall_spl_value
                    })

            # Add consolidated results for the current dataset entry
            consolidated_results.append({
                "rpm": rpm,
                "channels": channels
            })

        return consolidated_results
    
    def apply_iso_correction(self, rpm, rpm_dataset):

        # Calculate the overall SPL mean of the background noise measurements at the specific RPM
        background_overall_spl = self.background_overall_spl_calc(rpm)
        # Calculate the differnce between the overall SPL value of the measuremnts at the specific RPM
        for entry in rpm_dataset:
            for channel in entry["channels"]:
                # Find the matching channel in background_overall_spl by channel_id
                matching_background_channel = next(
                    ch for ch in background_overall_spl[0]["channels"] if ch["channel_id"] == channel["channel_id"]
                )
                
                # Calculate spl_value_difference
                channel["spl_value_difference"] = self.spl_value_difference_calc(
                    channel["dbfsa"],
                    matching_background_channel["dbfsa_background_overall_spl"]
                )
                
                # Calculate spl_value_difference_a
                channel["spl_value_difference_a"] = self.spl_value_difference_calc(
                    channel["dbfs"],
                    matching_background_channel["dbfs_background_overall_spl"]
                )
        # Calculate the background noise correction factor at the specific RPM
        for entry in rpm_dataset:
            for channels in entry["channels"]:
                channels["background_noise_correction_factor"]= self.background_noise_correction_calc(channels["spl_value_difference"])
                channels["background_noise_correction_factor_a"]= self.background_noise_correction_calc(channels["spl_value_difference_a"])
        # Correct the overall SPL value of the noise measurements by the correction factor
        for entry in rpm_dataset:
            for channels in entry["channels"]:
                channels["background_corrected_spl_value"] = self.spl_value_background_correction_calc(channel["dbfs"], entry["background_noise_correction_factor"])
                channels["background_corrected_spl_value_a"] = self.spl_value_background_correction_calc(channel["dbfsa"], entry["background_noise_correction_factor_a"])
        # Check for ISO conformity
        for entry in rpm_dataset:
            for channels in entry["channels"]:
                if channels["spl_value_difference"] >= 15:
                    channels["background_corrected_spl_value"] = channels["dbfs"]
            # Check if difference is greater than 10 dB between 250 and 5^000 Hz # TODO
            # band_freq_mask, band_dbfs_mask = self.create_mask[channels["freq"],channels["dbfs"], 250, 5000]
        return rpm_dataset
    
    def thrust_calculation(self, propeller_type, ct, pressure, temperature, humidity):
        # Claculate saturation vapor pressure
        p_sat = 6.1078*10**(7.5*temperature/(temperature+237.3))

        # Calculate partial pressure of water vapor
        p_v = humidity*p_sat

        # Claculate partuall pressure of dry air
        p_d = pressure - p_v

        # Temperature in Kelvin
        temperature_k = temperature + 273.15

        # Define specific gas constant for dry air
        r_d = 287.05

        #Define specific gas constant for water vapor
        r_v = 461.495

        # Calculate air density
        rho = p_d/(r_d*temperature_k) + p_v/(r_v*temperature_k)

        # Calculate rounds per second
        rps = rpm/60

        # Define the propeller diameter and convert to meter
        diameter = int(propeller_type.split('x')[0])
        diameter = diameter * 0.0254

        #Calculate thrust
        T = ct * rho * rps**2 * diameter**4

        return T

    def ct_calculation(self, propeller_type, thrust, rpm, temperature, pressure, humidity):
        # Claculate saturation vapor pressure
        p_sat = 6.1078*10**(7.5*temperature/(temperature+237.3))

        # Calculate partial pressure of water vapor
        p_v = humidity*p_sat

        # Claculate partuall pressure of dry air
        p_d = pressure - p_v

        # Temperature in Kelvin
        temperature_k = temperature + 273.15

        # Define specific gas constant for dry air
        r_d = 287.05

        #Define specific gas constant for water vapor
        r_v = 461.495

        # Calculate air density
        rho = p_d/(r_d*temperature_k) + p_v/(r_v*temperature_k)
        
        # Calculate rounds per second
        rps = rpm/60
        
        # Calculate diameter
        diameter = int(propeller_type.split('x')[0])
        diameter = diameter * 0.0254

        ct = thrust/(rho * rps**2 * diameter**4)

        return ct
        

if __name__ == "__main__":
    # Set the directory containing the pickle files
    data_directory = "./output"

    # Initialize the processor with the directory
    signal_processor = Data_Processor(data_directory)

    # Load and calculate thrust offset for the specific file
    file_path = "./output/FORCE_2024-11-28-10-11-01_RPM_3000.pkl"
    signal_processor.thrust_sensor_offset = signal_processor.load_and_calculate_thrust_offset(file_path)
    print(f"Thrust Sensor Offset: {signal_processor.thrust_sensor_offset}")

    # Process all pickle files
    signal_processor.process_pkl_files()

    # Load time and pressure data for all files with the same propeller type and RPM
    propeller_type = "PropellerA"
    rpm = 3000
    time_pressure_data = signal_processor.load_time_and_pressure_by_propeller_and_rpm(propeller_type, rpm)
    print(f"Loaded time and pressure data for propeller type {propeller_type} and RPM {rpm}:")
    for data in time_pressure_data:
        print(f"File: {data['file_name']}, Channel: {data['channel_id']}")
