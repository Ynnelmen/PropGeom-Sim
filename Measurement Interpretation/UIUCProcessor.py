import os

class UIUCProcessor:
    """
    A class to process UIUC propeller data files.

    Attributes:
        apc_propeller_data (str): Path to the directory containing APC propeller data files.
    """

    def __init__(self, apc_propeller_data):
        """
        Initialize the UIUCProcessor with the directory containing propeller data files.

        Args:
            apc_propeller_data (str): Path to the directory containing APC propeller data files.
        """
        if not os.path.isdir(apc_propeller_data):
            raise ValueError(f"The path {apc_propeller_data} is not a valid directory.")
        self.apc_propeller_data = apc_propeller_data

    def find_uiuc_data(self, propeller_type):
        """
        Find the filename of the UIUC data corresponding to the given propeller type.

        Args:
            propeller_type (str): The type of the propeller (e.g., "10x7E").

        Returns:
            str: The filename of the matching UIUC data file.

        Raises:
            FileNotFoundError: If no matching data file is found.
        """
        # Determine the attribute prefix based on propeller type
        attribute = 'apce' if propeller_type.lower().endswith('e') else 'apc'
        propeller_type = propeller_type.replace('.', '').lower().replace('e', '')
        print(propeller_type)

        # Iterate through files in the directory
        for filename in os.listdir(self.apc_propeller_data):
            filename_lower = filename.lower().replace('.', '')
            if filename_lower.startswith(attribute) and 'static' in filename_lower and propeller_type in filename_lower:
                return filename

        # Raise an error if no matching file is found
        raise FileNotFoundError(f"No matching UIUC data found for propeller type '{propeller_type}' in {self.apc_propeller_data}.")

    def load_uiuc_data(self, filename):
        """
        Load and parse the UIUC data file.

        Args:
            filename (str): The name of the UIUC data file.

        Returns:
            dict: A dictionary containing the propeller type and data (RPM, Ct, Cp values).

        Raises:
            FileNotFoundError: If the specified file does not exist.
            ValueError: If the file format is invalid or contains insufficient data.
        """
        # Construct the full file path
        file_path = os.path.join(self.apc_propeller_data, filename)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file '{filename}' does not exist in the directory '{self.apc_propeller_data}'.")

        # Extract the propeller type from the filename
        parts = filename.split("_")
        if len(parts) < 2:
            raise ValueError(f"The filename '{filename}' does not follow the expected format.")

        propeller_type = parts[1]

        # Initialize lists to store data
        rpm_values = []
        ct_values = []
        cp_values = []

        try:
            with open(file_path, 'r') as file:
                lines = file.readlines()

                # Skip the header and parse the remaining lines
                for line in lines[1:]:
                    if line.strip():  # Skip empty lines
                        parts = line.split()
                        if len(parts) < 3:
                            raise ValueError(f"Invalid data format in line: {line.strip()}")

                        rpm_values.append(float(parts[0]))
                        ct_values.append(float(parts[1]))
                        cp_values.append(float(parts[2]))

        except ValueError as e:
            raise ValueError(f"Error processing file '{filename}': {e}")

        return {
            'propeller_type': propeller_type,
            'data': {
                'rpm': rpm_values,
                'ct': ct_values,
                'cp': cp_values
            }
        }

if __name__ == "__main__":
    processor = UIUCProcessor('./uiuc')
    filename = processor.find_uiuc_data('10x7E')
    data = processor.load_uiuc_data(filename)

    print(data)
