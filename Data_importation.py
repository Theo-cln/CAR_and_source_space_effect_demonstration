import os

def import_edf_files(folder):
    """
    Retrieves and categorizes .edf files from a specified folder into training and test sets.

    Parameters:
    - folder (str): Path to the folder containing the .edf files.

    Returns:
    - A list containing:
        - file_train (list of str): List of file paths for training data.
        - file_test_1 (list of str): List of file paths for the first set of test data.
        - file_test_2 (list of str): List of file paths for the second set of test data (last two files).
    """
    file_train = []
    file_test_1 = []

    # Determine the prefix for training files based on the folder path
    train_prefix = "mi" if "Braccio_Connect" in folder.split(os.sep) else "train"

    # Iterate over all files in the specified folder
    for filename in os.listdir(folder):
        if filename.endswith('.edf'):
            # Extract the first word of the file name to determine its category
            first_word = filename.split('-')[0].lower()

            # Categorize files into training or test based on the first word
            if first_word == train_prefix:
                file_train.append(os.path.join(folder, filename))
            elif first_word == 'test':
                file_test_1.append(os.path.join(folder, filename))

    # Sort the test files and split the last two into a separate test set
    file_test_1 = sorted(file_test_1)
    """
    Distribute test files between test set 1 and test set 2 based on the number of test files that can be different 
    for Braccio connect dataset
    """
    try:
        if len(file_test_1) == 5:
            test_files_2 = file_test_1[-2:]
            test_files_1 = file_test_1[:-2]
        elif len(file_test_1) == 10:
            test_files_2 = file_test_1[-4:]
            test_files_1 = file_test_1[:-4]
        else:
            raise ValueError("Unexpected number of test files found. Expected 5 or 10 test files.")
    except Exception as e:
        print("Error in distributing test files:", e)
        raise

    return [file_train, test_files_1, test_files_2]



