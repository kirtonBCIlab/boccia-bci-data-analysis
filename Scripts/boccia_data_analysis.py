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
    def get_xdf_files(self, folder):
        """
        Retrieve xdf data files from a given folder.

        Parameters
        ----------
        folder : str
            Path to the folder containing xdf data files

        Returns
        -------
        xdf_files : dict
            Dictionary mapping trial numbers to xdf file paths
        """
        # Locate xdf files from the folder using regular expression
        xdf_files = {}
        with os.scandir(folder) as entries:
            for entry in entries:
                if entry.is_file() and entry.name.endswith(".xdf"):
                    match = re.search(r'run-(\d{3})', entry.name)
                    if match:
                        trial_number = int(match.group(1))
                        xdf_files[trial_number] = entry.path
        return xdf_files
    
    def get_settings_files(self, folder):
        """
        Retrieve trial settings files from a given folder.

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
    
    def get_complete_trial_data(self, xdf_files, settings_files=None):
        """
        Gets trial data by matching xdf and optionally settings files to trial number.

        Parameters
        ----------
        xdf_files : dict
            Dictionary mapping trial numbers to xdf data file paths
        settings_files : dict *optional, default is None*
            Dictionary mapping trial numbers to Trial Settings file paths

        Returns
        -------
        organized_data : dict
            Dictionary containing xdf and optionally settings data for each trial, organized by trial number.
        """
        trial_data = {}
        if settings_files is not None:
            for trial_number in sorted(set(xdf_files.keys()) & set(settings_files.keys())):
                trial_data[trial_number] = {
                    'xdf_file': xdf_files[trial_number],
                    'settings_file': settings_files[trial_number]
                }
        
        else:
            for trial_number in sorted(xdf_files.keys()):
                trial_data[trial_number] = {
                    'xdf_file': xdf_files[trial_number]
                }

        return trial_data
    
    def process_settings_file(self, settings_file):
        """
        Processes a single trial settings JSON file to determine the trial ID.
        
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
    
    def load_xdf_file(self, trial_data, trial_number):
        """
        Load the xdf file for a specific trial.
        
        Parameters
        ----------
        trial_data : dict
            Dictionary containing xdf and settings data for the trials.
        trial_number : int
            The trial number for which to load the xdf data.    
        
        Returns
        -------
        xdf_file : str
            Path to the xdf file for the specified trial.
        streams : list
            List of streams in the xdf file.
        """
        xdf_file = trial_data[trial_number]["xdf_file"]

        streams, _ = pyxdf.load_xdf(xdf_file)

        return xdf_file, streams
    
    def get_stream_data(self, streams, stream_name):
        """
        Extract data from a specific LSL stream in an xdf file.

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
        for [s, stream] in enumerate(streams):
            print(f"Stream {s}: {stream['info']['name'][0]}")
        
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
    
    def run_analysis(self, folder_path, target_stream_name, predictions_stream_name, process_settings=True):
        """
        Run the complete analysis on the participant data from a given folder.

        Parameters
        ----------
        folder_path : str
            Path to the folder containing xdf files.
        target_stream_name : str
            Name of the target element stream.
        predictions_stream_name : str
            Name of the Python response stream.
        process_settings : bool *optional, default is True*
            Whether to process the settings files to determine trial ID and description.

        Returns
        -------

        """
        # Get xdf files
        xdf_files = self.get_xdf_files(folder_path)

        if process_settings:
            # Get settings files
            settings_files = self.get_settings_files(folder_path)

            # Initialize trial ID dictionary
            trial_ID_dict = self.initialize_trial_ID_dict()

            # Get complete trial data with settings files included
            trial_data = self.get_complete_trial_data(xdf_files, settings_files)

            trial_IDs = []
            trial_descriptions = []
        
        else:
            # Get trial data with xdf files only
            trial_data = self.get_complete_trial_data(xdf_files)

        accuracies = []
        # Process each trial based on trial number
        for trial_number in trial_data.keys():
            print(f"\nProcessing trial {trial_number}")

            # Load streams from the xdf file
            xdf_file, streams = self.load_xdf_file(trial_data, trial_number)

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

            if process_settings:
                # Process settings file to get the trial ID and description
                trial_ID = self.process_settings_file(trial_data[trial_number]["settings_file"])
                trial_IDs.append(trial_ID)
                trial_descriptions.append(trial_ID_dict[trial_ID])

        # Output results
        if process_settings:
            # Print results
            # self.print_results(trial_descriptions, trial_IDs, accuracies)

            # Create the results DataFrame
            self.results_dataframe(trial_descriptions, trial_IDs, accuracies)

        else:
            # Only output accuracy results
            self.print_results(accuracies=accuracies)

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

    def print_results(self, trial_descriptions=None, trial_IDs=None, accuracies=None):
        """
        Print the results of the analysis:
        Prints the trial descriptions, trial IDs, and accuracies depending on which parameters are provided.

        Parameters
        ----------
        trial_descriptions : list *optional*
            List of trial descriptions.
        trial_IDs : list *optional*
            List of trial IDs.
        accuracies : list *optional*
            List of accuracies for each trial.

        Returns
        -------
        None
        """
        if trial_descriptions is not None:
            print("\nTrial Descriptions:")
            for desc in trial_descriptions:
                print(desc)

        if trial_IDs is not None:
            print("\nTrial IDs:")
            for ID in trial_IDs:
                print(ID)

        if accuracies is not None:
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

    analysis.run_analysis(folder_path, target_stream_name, predictions_stream_name, process_settings=True)

if __name__ == "__main__":
    main()