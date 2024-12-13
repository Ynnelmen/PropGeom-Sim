from os import path, listdir
from os.path import join
from os import makedirs
from datetime import datetime, timezone
import argparse
import numpy as np
import pickle
from kaitai.python.openapi_message import OpenapiMessage

class StreamProcessor:
    def __init__(self, folder_path):
        """
        Initialize the StreamProcessor with a folder containing .stream files. 
        Args:
            folder_path: Path to the folder containing .stream files.
        """
        if not path.isdir(folder_path):
            raise ValueError(f"Invalid folder path: '{folder_path}' is not a directory.")
        self.folder_path = folder_path

    @staticmethod
    def calc_time(t):
        """
        Convert an Open API 'Time' structure to a numeric value.
        Args:
            t: An Open API 'Time' instance.
        Returns:
            The time as a numeric type.
        """
        try:
            family = 2 ** t.time_family.k * 3 ** t.time_family.l * 5 ** t.time_family.m * 7 ** t.time_family.n
            return t.time_count * (1 / family)
        except AttributeError as e:
            raise ValueError(f"Invalid time structure: {e}")

    @staticmethod
    def get_quality_strings(validity_list):
        """
        Generate descriptive strings for data quality issues.
        Args:
            validity_list: List of validity objects from Open API messages.
        Returns:
            A list of strings describing the data quality at specific timestamps.
        """
        strings = []
        for validity in validity_list:
            quality_str, prefix = "", ""
            if validity["flags"].invalid:
                quality_str += prefix + "Invalid Data"
                prefix = ", "
            if validity["flags"].overload:
                quality_str += prefix + "Overload"
                prefix = ", "
            if validity["flags"].overrun:
                quality_str += prefix + "Gap In Data"
                prefix = ", "
            if not quality_str:
                quality_str = "OK"
            quality_str = f'{validity["time"]}: ' + quality_str
            strings.append(quality_str)
        return strings

    def process_stream_file(self, file_path):
        """
        Process a single .stream file to extract and save data.
        Args:
            file_path: Path to the .stream file to process.
        """
        if not path.isfile(file_path) or not file_path.endswith(".stream"):
            raise ValueError(f"Invalid file: '{file_path}' is not a valid .stream file.")

        file_name = path.basename(file_path)
        file_name = path.splitext(file_name)[0]

        print(f'Reading streaming data from file "{file_path}"...')
        try:
            file_size = path.getsize(file_path)
            file_stream = open(file_path, 'rb')
        except (OSError, IOError) as e:
            raise RuntimeError(f"Error reading file '{file_path}': {e}")

        data = {}
        plot_data = {}

        try:
            while True:
                try:
                    msg = OpenapiMessage.from_io(file_stream)
                except EOFError:
                    print("")
                    break

                if msg.header.message_type == OpenapiMessage.Header.EMessageType.e_interpretation:
                    for interpretation in msg.message.interpretations:
                        if interpretation.signal_id not in data:
                            data[interpretation.signal_id] = {}
                        data[interpretation.signal_id][interpretation.descriptor_type] = interpretation.value

                elif msg.header.message_type == OpenapiMessage.Header.EMessageType.e_signal_data:
                    for signal in msg.message.signals:
                        if "start_time" not in data[signal.signal_id]:
                            start_time = datetime.fromtimestamp(self.calc_time(msg.header.time), timezone.utc)
                            data[signal.signal_id]["start_time"] = start_time
                        if "samples" not in data[signal.signal_id]:
                            data[signal.signal_id]["samples"] = np.array([])
                        more_samples = np.array(list(map(lambda x: x.calc_value, signal.values)))
                        data[signal.signal_id]["samples"] = np.append(data[signal.signal_id]["samples"], more_samples)

                elif msg.header.message_type == OpenapiMessage.Header.EMessageType.e_data_quality:
                    for quality in msg.message.qualities:
                        if "validity" not in data[quality.signal_id]:
                            data[quality.signal_id]["validity"] = []
                        dt = datetime.fromtimestamp(self.calc_time(msg.header.time), timezone.utc)
                        data[quality.signal_id]["validity"].append({"time": dt, "flags": quality.validity_flags})

                print(f'{int(100 * file_stream.tell() / file_size)}%', end="\r")

        except Exception as e:
            raise RuntimeError(f"Error processing file '{file_path}': {e}")
        finally:
            file_stream.close()

        # Process and scale data for plotting
        try:
            for key, value in data.items():
                samples = value["samples"]
                scale_factor = value[OpenapiMessage.Interpretation.EDescriptorType.scale_factor]
                scaled_samples = (samples * scale_factor) / 2 ** 23
                sample_rate = 1 / self.calc_time(value[OpenapiMessage.Interpretation.EDescriptorType.period_time])
                unit = value[OpenapiMessage.Interpretation.EDescriptorType.unit]
                if unit.data == "Pa":
                    plot_data[key] = {
                        "scaled_samples": scaled_samples.tolist(),
                        "sample_rate": sample_rate,
                    }

            # Save processed data to a pickle file
            output_dir = ".\\pickel"
            makedirs(output_dir, exist_ok=True)
            with open(join(output_dir, f"{file_name}.pkl"), 'wb') as pickle_file:
                pickle.dump(plot_data, pickle_file)
            print(f'Plotted data has been saved to "{file_name}.pkl"')
        except Exception as e:
            raise RuntimeError(f"Error saving data for file '{file_name}': {e}")
        
    def process_all_stream_files(self):
        """
        Process all .stream files in the initialized folder.
        """
        files = [f for f in listdir(self.folder_path) if f.endswith(".stream")]
        if not files:
            print(f"No .stream files found in folder '{self.folder_path}'")
            return

        for file_name in files:
            file_path = join(self.folder_path, file_name)
            try:
                self.process_stream_file(file_path)
            except Exception as e:
                print(f"Error processing file '{file_name}': {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process .stream files to extract and save data.")
    parser.add_argument("folder", help="Path to the folder containing .stream files")
    args = parser.parse_args()

    processor = StreamProcessor(args.folder)
    try:
        processor.process_all_stream_files()
    except Exception as e:
        print(f"Error: {e}")

