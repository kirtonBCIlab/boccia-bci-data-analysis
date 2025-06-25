import numpy as np
import pandas as pd
import pyxdf
import os
import re
import json
from sklearn.metrics import confusion_matrix, accuracy_score

class BocciaDataAnalysis:
    """
    Class for analyzing data collected during Boccia P300 tests.
    """
    def get_eeg_files(self, folder):
        """
        Retrieve EEG data files from a given folder.

        Parameters
        ----------
        folder : str
            Path to the folder containing EEG data files

        Returns
        -------
        eeg_files : dict
            Dictionary mapping trial numbers to EEG data file paths
        """
        # Locate EEG data files from the folder using regular expression
        eeg_files = {}
        with os.scandir(folder) as entries:
            for entry in entries:
                if entry.is_file() and entry.name.endswith(".xdf"):
                    match = re.search(r'run-(\d{3})', entry.name)
                    if match:
                        trial_number = int(match.group(1))
                        eeg_files[trial_number] = entry.path
        return eeg_files
    
    def get_settings_files(self, folder):
        """
        Retrieve settings files from a given folder.

        Parameters
        ----------
        folder : str
            Path to the folder containing settings files

        Returns
        -------
        settings_files : dict
            Dictionary mapping trial numbers to Trial Settings file paths
        """
        # Locate settings files from the folder using regular expression
        settings_files = {}
        with os.scandir(folder) as entries:
            for entry in entries:
                if entry.is_file() and entry.name.endswith(".json"):
                    match = re.search(r'Trial_(\d+)', entry.name)
                    if match:
                        trial_number = int(match.group(1))
                        settings_files[trial_number] = entry.path
        return settings_files
    
    def get_complete_trial_data(self, eeg_files, settings_files):
        """
        Gets a complete set of trial data by matching EEG and settings files by trial number.

        Parameters
        ----------
        eeg_files : dict
            Dictionary mapping trial numbers to EEG data file paths
        settings_files : dict
            Dictionary mapping trial numbers to Trial Settings file paths

        Returns
        -------
        organized_data : dict
            Dictionary containing EEG and settings data for each trial, organized by trial number.
        """
        trial_data = {}
        for trial_number in sorted(set(eeg_files.keys()) & set(settings_files.keys())):
            trial_data[trial_number] = {
                'eeg_file': eeg_files[trial_number],
                'settings_file': settings_files[trial_number]
            }

        return trial_data
    
    def process_settings_file(self, settings_file):
        """
        Process the trial settings JSON file to determine the trial ID.
        
        Parameters
        ----------
        settings_file : str
            Path to the trial settings JSON file

        Returns
        -------
        trial_ID : int
            Trial ID
        """
        with open(settings_file, 'r') as f:
            # Load the JSON file
            settings = json.load(f)
            # Get the relevant settings
            num_cols = settings["coarseFanSettings"]["_nColumns"]
            num_rows = settings["coarseFanSettings"]["_nRows"]
            stim_type = settings["P300Settings"]["Test"]["StimulusType"]
            separate_buttons = settings["P300Settings"]["SeparateButtons"]

        # Determine the trial ID based on the settings values
        trial_ID = None

        # Check stimulus type and return if it was "Face Sprite" or "Gradient"
        if stim_type == 1:
            trial_ID = 4
            return trial_ID
        if stim_type == 2:
            trial_ID = 5
            return trial_ID
        
        # Check if separate buttons were used, return if so
        if separate_buttons:
            trial_ID = 6
            return trial_ID
        
        # Check the number of columns
        if num_cols == 7:
            trial_ID = 1
        elif num_cols == 3:
            trial_ID = 2
        elif num_cols == 5:
            trial_ID = 3

        # If none of the above conditions were met, return None
        else:
            print("Trial ID could not be determined.")

        return trial_ID

    def initialize_trial_ID_dict(self):
        """
        Initialize a dictionary mapping trial IDs to descriptions.
        
        Parameters
        ----------
        None

        Returns
        -------
        trial_dict : dict
            Dictionary mapping trial IDs to descriptions
        """
        trial_dict = {
            1: "Fan segments 7x5",
            2: "Fan segments 3x3",
            3: "Solid color stimulus",
            4: "Gradient stimulus",
            5: "Face sprite stimulus",
            6: "Separate buttons",
        }
        return trial_dict
    
    def load_eeg_file(self, trial_data, trial_number):
        """
        Load the EEG data file for a specific trial.
        
        Parameters
        ----------
        trial_data : dict
            Dictionary containing EEG and settings data for the trials.
        trial_number : int
            The trial number for which to load the EEG data.    
        
        Returns
        -------
        eeg_file : str
            Path to the EEG data file for the specified trial.
        streams : list
            List of streams in the EEG data file.
        """
        eeg_file = trial_data[trial_number]["eeg_file"]

        streams, _ = pyxdf.load_xdf(eeg_file)

        return eeg_file, streams
    
    def get_stream_data(self, streams, stream_name):
        """
        Extract data from a specific LSL stream in an EEG data file.

        Parameters
        ----------
        streams : list
            List of dictionaries representing the LSL streams.
        stream_name : str
            Name of the stream to extract data from.

        Returns
        -------
        stream_markers : list
            List of markers from the stream
        stream_time : list
            List of timestamps from the stream
        """
        # Find the index of the stream based on its name
        stream_index = next((i for i, stream in enumerate(streams) if stream_name in stream['info']['name'][0]), None)
        # Set the stream of interest
        stream = streams[stream_index]

        # Get markers and timestamps from the stream
        stream_markers = stream['time_series']
        stream_time = stream['time_stamps']

        return stream_markers, stream_time
    
    def get_actual_targets(self, markers):
        """
        Extract the actual target numbers from the target element stream markers.
        
        Currently, this method assumes that the target markers are in the following format:
        "ObjectID: <target_object_id> iSPO: <target_iSPO>" where iSPO is the Selectable Pool Index.
        Because Boccia-Unity currently uses iSPO to identify targets, this method extracts the iSPO number.

        Parameters
        ----------
        markers : list
            List of markers from the target element stream.

        Returns
        -------
        actual_targets : list
            List of actual target numbers extracted from the stream markers.
        """
        # Extract a list of target element numbers
        actual_targets = [int(item[1].split(": ")[1]) for item in markers]
        return actual_targets
    
    def get_predicted_targets(self, markers):
        """
        Extract the predicted target numbers from the Python response stream markers.

        This method gets the predicted target numbers from the Python response stream markers.
        It does this by removing 'ping' and 'marker received' messages from the stream markers.
        Based on how Bessy-Python outputs predictions, the remaining markers are assumed to be the predicted target numbers.

        Parameters
        ----------
        markers : list
            List of markers from the predicted target stream.

        Returns
        -------
        predicted_targets : list
            List of predicted target numbers extracted from the stream markers.

        """
        # Isolate predictions from Python response stream by removing 'ping' and 'marker received' messages
        extracted_predictions = [
            s for sublist in markers for s in sublist
            if 'ping' not in str(s).lower() and 'marker received' not in str(s).lower()
        ]

        # Convert the list of strings to a list of integers
        predicted_targets = [int(s.strip('[]')) for s in extracted_predictions]
        return predicted_targets
    
    def compare_predictions(self, actual_targets, predicted_targets):
        """
        Compare the actual and predicted target numbers.
        Calculates accuracy and determines the confusion matrix.
        
        Parameters
        ----------
        actual_targets : list
            List of actual target numbers.
        predicted_targets : list
            List of predicted target numbers.

        Returns
        -------
        accuracy : float
            The accuracy of the predictions as a percentage.
        cm : np.ndarray
            The confusion matrix.
        """
        # Check that the number of predictions matches number of actual targets
        assert len(predicted_targets) == len(actual_targets)

        # Convert to binary classification: 1 if the target matches, 0 if it doesn't match
        binary_preds = [1 if pred == true else 0 for pred, true in zip(predicted_targets, actual_targets)]
        binary_truths = [1] * len(actual_targets)  # All actual targets are correct

        accuracy = accuracy_score(binary_truths, binary_preds) * 100
        cm = confusion_matrix(binary_truths, binary_preds, labels=[1, 0])

        return accuracy, cm
    
    def run_analysis(self, folder_path, target_stream_name, predictions_stream_name):
        """
        Run the complete analysis on the participant data from a given folder.

        Parameters
        ----------
        folder_path : str
            Path to the folder containing EEG data files.
        target_stream_name : str
            Name of the target element stream.
        predictions_stream_name : str
            Name of the Python response stream.

        Returns
        -------

        """
        # Get EEG and settings files
        eeg_files = self.get_eeg_files(folder_path)
        settings_files = self.get_settings_files(folder_path)

        # Get complete trial data
        trial_data = self.get_complete_trial_data(eeg_files, settings_files)

        # Initialize trial ID dictionary
        trial_ID_dict = self.initialize_trial_ID_dict()

        accuracies = []
        trial_IDs = []
        trial_descriptions = []

        # Process each trial based on trial number
        for trial_number in trial_data.keys():
            print(f"\nProcessing trial {trial_number}")

            # Load EEG data file
            eeg_file, streams = self.load_eeg_file(trial_data, trial_number)

            # Get stream data
            target_stream_markers, _ = self.get_stream_data(streams, target_stream_name)
            predicted_stream_markers, _ = self.get_stream_data(streams, predictions_stream_name)

            # Get actual and predicted targets
            actual_targets = self.get_actual_targets(target_stream_markers)
            predicted_targets = self.get_predicted_targets(predicted_stream_markers)

            # print("Predicted targets:")
            # print(predicted_targets)
            # print("Actual targets:")
            # print(actual_targets)

            # Compare predictions and calculate accuracy and confusion matrix
            accuracy, cm = self.compare_predictions(actual_targets, predicted_targets)
            # self.print_cm(cm, accuracy, trial_number)
            accuracies.append(accuracy)

            # Process settings file to get the trial ID and description
            trial_ID = self.process_settings_file(trial_data[trial_number]["settings_file"])
            trial_IDs.append(trial_ID)
            trial_descriptions.append(trial_ID_dict[trial_ID])

        # self.print_results(trial_descriptions, trial_IDs, accuracies)

        self.results_dataframe(trial_descriptions, trial_IDs, accuracies)

    def print_cm(self, cm, accuracy, trial_number):
        """
        Print the confusion matrix for a given trial.

        Parameters
        ----------
        cm : np.ndarray
            The confusion matrix.
        trial_number : int
            The trial number for which the confusion matrix is printed.

        Returns
        -------
        None
        """
        print(f"Trial {trial_number} confusion matrix:")
        print(cm)
        print(f"Accuracy: {accuracy:.2f}%\n")

    def print_results(self, trial_descriptions, trial_IDs, accuracies):
        """
        Print the results of the analysis.

        Parameters
        ----------
        trial_descriptions : list
            List of trial descriptions.
        trial_IDs : list
            List of trial IDs.
        accuracies : list
            List of accuracies for each trial.

        Returns
        -------
        None
        """
        print("\nTrial Descriptions:")
        for desc in trial_descriptions:
            print(desc)

        print("\nTrial IDs:")
        for ID in trial_IDs:
            print(ID)

        print("\nAccuracies:")
        for acc in accuracies:
            print(f"{acc:.2f}%")

    def results_dataframe(self, trial_descriptions, trial_IDs, accuracies):
        """
        Create a DataFrame to store the results of the analysis.

        Parameters
        ----------
        trial_descriptions : list
            List of trial descriptions.
        trial_IDs : list
            List of trial IDs.
        accuracies : list
            List of accuracies for each trial.

        Returns
        -------
        None
        """
        # Create a DataFrame from the results
        df = pd.DataFrame({
            'Trial Description': trial_descriptions,
            'Trial ID': trial_IDs,
            'Prediction Accuracy (%)': accuracies
        })

        print("\nResults DataFrame:")
        print(df)
        df.to_clipboard(index=False, sep="\t")

        sorted_df = df.sort_values(by='Trial ID').reset_index(drop=True)
        print("\nSorted Accurcies by Trial ID:")  
        for acc in sorted_df["Prediction Accuracy (%)"]:
            print(f"{acc:.2f}%")    
    
def main():
    # Create instance of the BocciaDataAnalysis class
    analysis = BocciaDataAnalysis()

    # Path to the folder containing the data to analyze
    folder_path = "D:/Daniella Bourque/Boccia Validation/Participant-Data/250428_Participant_7_Data"

    # Initialize stream names
    target_stream_name = "TargetElementStream_VirtualPlay"
    predictions_stream_name = "PythonResponse"

    analysis.run_analysis(folder_path, target_stream_name, predictions_stream_name)

if __name__ == "__main__":
    main()