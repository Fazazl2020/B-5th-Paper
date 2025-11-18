import h5py

def read_tr_list_and_print_sph_stats(tr_list_path):
    """
    Reads the tr_list.txt file, accesses each .ex file,
    and computes aggregate statistics for the 'sph' dataset:
    - Smallest # of samples
    - Largest # of samples
    - Average # of samples
    - # of files with samples less than average
    - # of files with samples greater than average

    Args:
        tr_list_path (str): Path to the tr_list.txt file containing paths to .ex HDF5 files.
    """
    try:
        # Read the list of .ex file paths
        with open(tr_list_path, 'r') as file:
            ex_file_paths = [line.strip() for line in file if line.strip()]

        if not ex_file_paths:
            print("No .ex file paths found in the provided list.")
            return

        print(f"Processing {len(ex_file_paths)} .ex files from: {tr_list_path}\n")
        
        # List to store the number of samples found in 'sph' for each valid file
        sph_samples_list = []

        # Iterate over each .ex file
        for ex_file in ex_file_paths:
            try:
                with h5py.File(ex_file, 'r') as hdf:
                    if 'sph' in hdf:
                        sph_dataset = hdf['sph']
                        num_samples = sph_dataset.shape[0]  # Number of samples in 'sph'
                        sph_samples_list.append(num_samples)
                    # If 'sph' is not present, we skip the file
            except Exception as e:
                # If there's an error (file not found, corrupted, etc.), skip
                print(f"Warning: Could not process file {ex_file}: {e}")

        # If no valid 'sph' samples found, stop here
        if not sph_samples_list:
            print("No valid 'sph' datasets found in any of the listed files.")
            return

        # Compute required statistics
        smallest_num_samples = min(sph_samples_list)
        largest_num_samples = max(sph_samples_list)
        avg_num_samples = sum(sph_samples_list) / len(sph_samples_list)

        # Count how many files fall below or above this average
        num_less_than_avg = sum(1 for x in sph_samples_list if x < avg_num_samples)
        num_greater_than_avg = sum(1 for x in sph_samples_list if x > avg_num_samples)

        # Print the statistics
        print(f"Smallest # of samples in 'sph': {smallest_num_samples}")
        print(f"Largest  # of samples in 'sph': {largest_num_samples}")
        print(f"Average  # of samples in 'sph': {avg_num_samples:.2f}")
        print(f"Number of files with samples < average: {num_less_than_avg}")
        print(f"Number of files with samples > average: {num_greater_than_avg}")

    except FileNotFoundError:
        print(f"Error: The tr_list.txt file was not found at: {tr_list_path}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Example usage
if __name__ == "__main__":
    tr_list_path = "/ghome/fewahab/My_4th_Pap/Final-Models/N1/scripts/tr_list.txt"
    read_tr_list_and_print_sph_stats(tr_list_path)
