import pickle
import numpy as np
import matplotlib.pyplot as plt
import ast
import os
import logging
import pandas as pd
import h5py

# Configure logging for detailed error tracking and information
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class DataLoader:
    def __init__(self, pickle_dir, info_dir, measurement_dir):
        """
        Initialize the loader for signal data.
        Args:
            pickle_dir: Directory containing the pickle files.
            info_dir: Directory containing the corresponding info files.
        """
        self.pickle_dir = pickle_dir
        self.info_dir = info_dir
        self.measurement_dir = measurement_dir
        self.measurement_window = []  # Stores the RPM and time range information
        #self.sample_rate = 0.0  # Sample rate of the loaded signal
    
    @staticmethod
    def load_pickle_file(file_path):
        """
        Load and deserialize data from a pickle file.
        Args:
            file_path: Path to the pickle file.
        Returns:
            Deserialized data as a dictionary, or None if the file doesn't exist.
        Raises:
            FileNotFoundError: If the pickle file is not found.
        """
        if not os.path.exists(file_path):
            logging.error(f"Pickle file {file_path} does not exist.")
            return None
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    
    @staticmethod
    def load_sensor_file(measurement_path):
        """
        Opens a sensor data csv file.
        Args:
            file_path: Path to the measurement file.
        Returns:
           a dataframe containing all data
        """
        return pd.read_csv(measurement_path, delimiter=';')

    @staticmethod
    def extract_propeller_name_and_timestamp(filename):
        """
        Extract propeller name and timestamp from the given filename.
        Args:
            filename: Filename of the pickle file.
        Returns:
            propeller_name: Name of the propeller.
            timestamp: Timestamp extracted from the filename.
        """
        parts = filename.split('_')
        if len(parts) < 2:
            raise ValueError("Filename format is incorrect. Expected format: <propeller_name>_<timestamp>_...")
        propeller_name = parts[0]
        timestamp = parts[1]
        return propeller_name, timestamp

        
    def load_time_signal(self, pickle_file, info_file, rpm, signal_id):
        """
        Extract and filter a time-domain signal for a specific RPM and measurement window.
        Args:
            pickle_file: Path to the pickle file containing signal data.
            info_file: Path to the info file containing measurement metadata.
            rpm: Desired RPM value for filtering.
            signal_id: Integer identifier for the specific signal in the pickle file.
        Returns:
            A dictionary containing:
                - "closest_rpm": Closest RPM value from the measurement window.
                - "start_time": Start time of the selected measurement window.
                - "stop_time": Stop time of the selected measurement window.
                - "time_axis": Time axis for the filtered signal.
                - "filtered_scaled_samples": Filtered signal values in the specified time range.
        Raises:
            ValueError: If the info file or pickle file is invalid or empty.
            KeyError: If the specified signal_id is not found in the signal data.
        """
        try:
            # Validate the info file
            if os.stat(info_file).st_size == 0:
                raise ValueError(f"Info file {info_file} is empty or invalid.")

            # Load the signal data from the pickle file
            signal_data = self.load_pickle_file(pickle_file)
            if signal_data is None:
                raise ValueError(f"Pickle file {pickle_file} is empty or invalid.")

            # Read the measurement window from the info file
            with open(info_file, 'r') as info:
                self.measurement_window = info.readline().strip()
                if not self.measurement_window:
                    raise ValueError(f"Invalid first line in info file {info_file}.")
                self.measurement_window = ast.literal_eval(self.measurement_window)

            # Find the closest RPM value in the measurement window
            rpm_array = np.array(self.measurement_window[::3])
            idx = np.argmin(np.abs(rpm_array - rpm))


            closest_rpm = self.measurement_window[3 * idx]
            start_time = self.measurement_window[3 * idx + 1]
            stop_time = self.measurement_window[3 * idx + 2]

            if signal_id not in signal_data:
                raise KeyError(f"Signal ID '{signal_id}' not found in signal data.")
            
            # Extract and filter the signal within the specified time range
            value = list(signal_data.items())[signal_id - 1][1]
            scaled_samples = np.array(value["scaled_samples"])
            sample_rate = value["sample_rate"]
            time_axis = np.arange(len(scaled_samples)) / sample_rate
            time_mask = (time_axis >= start_time) & (time_axis <= stop_time)
            filtered_time_axis = time_axis[time_mask]
            filtered_scaled_samples = scaled_samples[time_mask]

            # Remove DC offset
            filtered_scaled_samples -= np.mean(filtered_scaled_samples)

            return {
                "closest_rpm": closest_rpm,
                "start_time": start_time,
                "stop_time": stop_time,
                "sampling_frequency": sample_rate,
                "signal_id": signal_id,
                "time": filtered_time_axis,
                "pressure": filtered_scaled_samples
            }
        except Exception as e:
            logging.error(f"Failed to load time signal for signal_id {signal_id}: {e}")
            raise
    
    def data_structure(self, propeller_type, rpm, timestamp, sensor_data, pickle_file, info_file):
        """
        Build the data structure for a single RPM and propeller type.
        Args:
            propeller_type (str): Name of the propeller type.
            rpm (int): Target RPM for the data structure.
            timestamp (str): Timestamp for the measurement.
            sensor_data (DataFrame): DataFrame containing sensor data.
            pickle_file (str): Path to the pickle file.
            info_file (str): Path to the info file.
        Returns:
            dict: The constructed data structure for the propeller measurement.
        """
        print(f"Processing propeller: {propeller_type} at rpm: {rpm}")
        

        # Initialize the main data structure
        Propeller_Measurement = {
            'propeller_type': propeller_type,
            'targeted_rpm': rpm,
            'measurement_initialization_time': timestamp,
            "sound": {
                'sequence_start_time': None,  # Default value, updated later
                'sequence_stop_time': None,   # Default value, updated later
                'channels': {}                # Will be populated later (id, position (x,y,z) in meter, samplingfrequency, time array, pressure array)
            },
            "auxilliary_sensors": {
                'measurement_time': None,
                'rx_throttle': None,
                'actual_throttle': None,
                'electric_rpm': None,
                'pulse_rpm': None,
                'pwm_output': None,
                'busbar_voltage': None,
                'busbar_current': None,
                'phase_line_current': None,
                'temperature': None,
                'air_pressure': None,
                'altitude': None,
                'thrust': None
            }
        }

        channel_positions = [
            [-1.887, 0, 0.844],
            [0.933, -1.617, 1.033],
            [0.914, 1.585, 1.222],
            [-0.884, 1.532, 1.411],
            [-0.844, -1.460, 1.6],
            [1.577, 0, 1.789],
            [0.718, 1.243, 1.978],
            [-1.249, 0, 2.167],
            [0.497, -0.861, 2.356],
            [0.589, 0, 2.544]
        ]

        # Populate sound channels
        for i in range(1, 11):
            print(f"Processing Signal {i}")
            try:
                signal_data = self.load_time_signal(pickle_file, info_file, rpm, i)
                if 'start_time' in signal_data and 'stop_time' in signal_data and 'start_time' not in Propeller_Measurement['sound'] and 'stop_time' not in Propeller_Measurement:
                    Propeller_Measurement["sound"]['sequence_start_time'] = signal_data['start_time']
                    Propeller_Measurement["sound"]['sequence_stop_time'] = signal_data['stop_time']

                Propeller_Measurement['sound']['channels'][str(i)] = {
                    "channel_id": i,
                    "channel_position": channel_positions[i-1],
                    'sampling_frequency': signal_data["sampling_frequency"],
                    "time": signal_data["time"],
                    "pressure": signal_data["pressure"],
                }
            except Exception as e:
                logging.warning(f"Failed to process channel {i}: {e}")
                # Assign None for failed channels
                Propeller_Measurement['sound']['channels'][str(i)] = {
                    "time": None,
                    "pressure": None,
                }

        # Skip processing if start_time or stop_time is not available
        if not Propeller_Measurement["sound"].get("sequence_start_time") or not Propeller_Measurement["sound"].get("sequence_stop_time"):
            logging.error("Measurement window (start_time, stop_time) could not be determined. Skipping processing.")
            return None  # Or raise an exception, depending on desired behavior

        # Populate sensor data
        sensor_data = sensor_data[(sensor_data['Time'] >= Propeller_Measurement["sound"]['sequence_start_time']) & (sensor_data['Time'] <= Propeller_Measurement["sound"]['sequence_stop_time'])]

        Propeller_Measurement['auxilliary_sensors'].update({
            'measurement_time': sensor_data['Time'].tolist() if not sensor_data.empty else None,
            'rx_throttle': sensor_data['RXThrottle'].tolist() if not sensor_data.empty else None,
            'actual_throttle': sensor_data['actualThrottle'].tolist() if not sensor_data.empty else None,
            'electric_rpm': sensor_data['ElectricRPM'].tolist() if not sensor_data.empty else None,
            'pulse_rpm': sensor_data['PulseRPM'].tolist() if not sensor_data.empty else None,
            'pwm_output': sensor_data['Output'].tolist() if not sensor_data.empty else None,
            'busbar_voltage': sensor_data['BusbarVoltage'].tolist() if not sensor_data.empty else None,
            'busbar_current': sensor_data['BusbarCurrent'].tolist() if not sensor_data.empty else None,
            'phase_line_current': sensor_data['PhaseLineCurrent'].tolist() if not sensor_data.empty else None,
            'temperature': sensor_data['Temperature'].tolist() if not sensor_data.empty else None,
            'air_pressure': sensor_data['Pressure'].tolist() if not sensor_data.empty else None,
            'altitude': sensor_data['Altitude'].tolist() if not sensor_data.empty else None,
            'thrust': sensor_data['AdcVoltage'].tolist() if not sensor_data.empty else None,
        })
        return Propeller_Measurement


    def save_to_pickle(self, data, file_path):
        """
        Save the given data to a pickle file.

        Parameters:
            data (dict): The data structure to be saved.
            file_path (str): The path of the file to save the data to.
        """
        with open(file_path, 'wb') as file:
            pickle.dump(data, file)

    def save_to_hdf5(self, propeller_measurement, file_path):
        """
        Save the propeller measurement data to an HDF5 file.
        Args:
            propeller_measurement (dict): The propeller measurement data structure.
            file_path (str): Path to the HDF5 file where data will be saved.
        """
        import h5py
        
        with h5py.File(file_path, 'w') as hdf:
            # Create groups for the propeller measurement data
            sound_group = hdf.create_group('sound')
            aux_sensors_group = hdf.create_group('auxilliary_sensors')

            # Save sound data
            sound_group.attrs['sequence_start_time'] = propeller_measurement['sound']['sequence_start_time']
            sound_group.attrs['sequence_stop_time'] = propeller_measurement['sound']['sequence_stop_time']

            channels_group = sound_group.create_group('channels')
            for channel_id, channel_data in propeller_measurement['sound']['channels'].items():
                channel_group = channels_group.create_group(channel_id)
                channel_group.attrs['channel_id'] = channel_data['channel_id']
                channel_group.create_dataset('channel_position', data=channel_data['channel_position'])
                channel_group.create_dataset('time', data=channel_data['time'] if channel_data['time'] is not None else [])
                channel_group.create_dataset('pressure', data=channel_data['pressure'] if channel_data['pressure'] is not None else [])

            # Save auxiliary sensors data
            for key, value in propeller_measurement['auxilliary_sensors'].items():
                aux_sensors_group.create_dataset(key, data=value if value is not None else [])

if __name__ == "__main__":
    pickle_dir = "./pickel"
    info_dir = "./misc"
    measurement_dir = "./misc"
    data_loader = DataLoader(pickle_dir, info_dir, measurement_dir)

    for pickle_filename in os.listdir(pickle_dir):
        if pickle_filename.endswith(".pkl"):
            propeller_name, timestamp = data_loader.extract_propeller_name_and_timestamp(pickle_filename)
            matching_info_files = []
            matching_measurement_files = []

        # Search for matching measurement files
        for measurement_filename in os.listdir(measurement_dir):
            if (
                propeller_name in measurement_filename 
                and timestamp in measurement_filename 
                and "measurement_output" in measurement_filename
            ):
                matching_measurement_files.append(os.path.join(measurement_dir, measurement_filename))


        # Search for matching info files
        for info_filename in os.listdir(info_dir):
            if (
                propeller_name in info_filename 
                and timestamp in info_filename 
                and "misc_info" in info_filename
            ):
                matching_info_files.append(os.path.join(info_dir, info_filename))

        # Iterate through matching info and measurement files to find pairs
        for matching_info_file in matching_info_files:
            for matching_measurement_file in matching_measurement_files:
                sensor_data = data_loader.load_sensor_file(matching_measurement_file)
                pickle_file_path = os.path.join(pickle_dir, pickle_filename)
                for rpm in range(3000, 7000, 500):
                    # Define output file paths
                    output_pickle_path = f"./output/{propeller_name}_{timestamp}_RPM_{rpm}.pkl"
                    output_hdf5_path = f"./output/{propeller_name}_{timestamp}_RPM_{rpm}.h5"

                    # Check if both files already exist
                    if os.path.exists(output_pickle_path) and os.path.exists(output_hdf5_path):
                        logging.info(f"Files for RPM {rpm} already exist: Skipping processing.")
                        continue  # Skip to the next RPM

                    # Load input files and process data
                    sensor_data = data_loader.load_sensor_file(matching_measurement_file)
                    pickle_file_path = os.path.join(pickle_dir, pickle_filename)
                    try:
                        data_structure = data_loader.data_structure(
                            propeller_name,
                            rpm,
                            timestamp,
                            sensor_data,
                            pickle_file_path,
                            matching_info_file
                        )
                        # Ensure the directory for the output path exists
                        output_dir = os.path.dirname(output_pickle_path)
                        os.makedirs(output_dir, exist_ok=True)

                        # Save the pickle file if it doesn't already exist
                        if not os.path.exists(output_pickle_path):
                            data_loader.save_to_pickle(data_structure, output_pickle_path)
                            logging.info(f"Saved processed data to {output_pickle_path}")

                        # Save the HDF5 file if it doesn't already exist
                        if not os.path.exists(output_hdf5_path):
                            data_loader.save_to_hdf5(data_structure, output_hdf5_path)
                            logging.info(f"Saved processed data to {output_hdf5_path}")

                    except Exception as e:
                        logging.error(f"Error processing RPM {rpm}: {e}")
