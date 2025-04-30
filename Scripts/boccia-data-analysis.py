import numpy as np
import pandas as pd
import pyxdf
import os
import re
import json
from sklearn.metrics import confusion_matrix, accuracy_score

class BocciaDataAnalysis:
    def __init__(self, folder_path, stream_name):
        """
        Initialize BocciaDataAnalysis object.

        Parameters
        ----------
        folder_path : str
            Path to the folder containing the data to analyze
        stream_name : str
            Name of the target element stream (if Play or VirtualPlay was used)  
        """
        self.folder_path = folder_path # Path to the folder containing the data
        self.eeg_files = None # Stores EEG data files
        self.trial_settings_files = None # Stores trial settings files

        self.print_confusion_matrix = False # Flag to print confusion matrix
        self.trial_count = 0 # Counter for the number of processed trials

        # Names of the relevant streams
        self.target_stream_name = stream_name 
        self.python_response_stream_name = "PythonResponse"

        # Initialize target element stream variables
        self.target_markers = None
        self.target_time = None

        # Initialize python response stream variables
        self.python_response_markers = None
        self.python_response_time = None

        # Initialize trial ID dictionary
        self.trial_ID_dict = self.initialize_trial_dict()

        # Initialize prediction accuracy and trial ID lists
        self.prediction_accuracies = []
        self.trial_IDs = []
        
        # Get trial data to process and analyze
        self.trial_data = self.get_trial_data()

        # Initialize results dataframe
        self.results_df = pd.DataFrame(columns=["Trial Description", "Trial ID", "Prediction Accuracy"])

    def initialize_trial_dict(self):
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
    
    def get_trial_data(self):
        """
        Get trial data and organize it based on trial number.

        This method retrieves EEG data and trial settings files. 
        It pairs an EEG file with its corresponding trial settings file based on trial number.

        Parameters
        ----------
        None

        Returns
        -------
        trial_data : dict
            Dictionary mapping trial numbers to trial data 
        """
        # Call the methods to retrieve EEG data and trial settings
        self.eeg_files = self.retrieve_eeg_files()
        self.trial_settings_files = self.retrieve_trial_settings_files()
        
        # Pair EEG data and trial settings based on trial number
        trial_data = {}
        for trial_number in sorted(set(self.eeg_files.keys()) & set(self.trial_settings_files.keys())):
            trial_data[trial_number] = {
                "eeg_data": self.eeg_files[trial_number],
                "trial_settings": self.trial_settings_files[trial_number]
            }

        return trial_data
    
    def retrieve_eeg_files(self):
        """
        Retrieve EEG data files from the folder path.

        Parameters
        ----------
        None

        Returns
        -------
        eeg_files : dict
            Dictionary mapping trial numbers to EEG data file paths
        """
        # Locate EEG data files from the folder using regular expression
        eeg_files = {}
        with os.scandir(self.folder_path) as entries:
            for entry in entries:
                if entry.is_file() and entry.name.endswith(".xdf"):
                    match = re.search(r'run-(\d{3})', entry.name)
                    if match:
                        trial_number = int(match.group(1))
                        eeg_files[trial_number] = entry.path
        return eeg_files
    
    def retrieve_trial_settings_files(self):
        """
        Retrieve trial settings files from the folder path.

        Parameters
        ----------
        None

        Returns
        -------
        trial_settings : dict
            Dictionary mapping trial numbers to trial settings file paths
        """
        # Locate trial settings files from the folder using regular expression
        trial_settings = {}
        with os.scandir(self.folder_path) as entries:
            for entry in entries:
                if entry.is_file() and entry.name.endswith(".json"):
                    match = re.search(r'Trial_(\d+)', entry.name)
                    if match:
                        trial_number = int(match.group(1))
                        trial_settings[trial_number] = entry.path
        return trial_settings
    
    def process_trial(self, trial_number):
        """
        Process a single trial.
        Calls the process_streams method to process the LSL streams from the EEG data.
        Calls the process_settings method to process the trial settings file.

        Parameters
        ----------
        trial_number : int
            Trial number

        Returns
        -------
        None
        """
        # Process the EEG data
        xdf_file = self.trial_data[trial_number]["eeg_data"]
        self.process_streams(xdf_file)

        # Process the trial settings
        settings_file = self.trial_data[trial_number]["trial_settings"]
        trial_ID = self.process_settings(settings_file)
        self.trial_IDs.append(trial_ID)
    
    def process_streams(self, xdf_file):
        """
        Process the LSL streams from the EEG data.

        Parameters
        ----------
        xdf_file : str
            Path to the xdf file

        Returns
        -------
        None
        """
        # Load the xdf file
        streams, fileheader = pyxdf.load_xdf(xdf_file)

        # Get data from streams of interest (target stream and python response stream)
        self.target_markers, self.target_time = self.get_stream_data(streams, self.target_stream_name)
        self.python_response_markers, self.python_response_time = self.get_stream_data(streams, self.python_response_stream_name)

        # Get list of actual target elements
        target_element_iSPOs = self.get_target_elements(self.target_markers)

        # Get list of predicted targets from Python
        predicted_targets = self.get_predictions(self.python_response_markers)

        # Compare predicted targets to actual targets
        self.compare_predictions(predicted_targets, target_element_iSPOs)
    
    def get_stream_data(self, streams, stream_name):
        """
        Extract data from a specific LSL stream.

        Parameters
        ----------
        streams : list
            List of dictionaries representing the LSL streams
        stream_name : str
            Name of the stream to extract data from

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
    
    def get_target_elements(self, target_markers):
        """
        Extract target numbers from target element stream.

        Parameters
        ----------
        target_markers : list
            List of markers from the target element stream

        Returns
        -------
        target_elements : list
            List of target numbers extracted from the stream
        """
        # Extract a list of target element numbers
        target_elements = [int(item[1].split(": ")[1]) for item in target_markers]
        return target_elements
    
    def get_predictions(self, python_response_markers):
        """
        Extract predictions from Python response stream.

        Parameters
        ----------
        python_response_markers : list
            List of markers from the Python response stream

        Returns
        -------
        predicted_targets : list
            List of predicted target numbers extracted from the stream
        """
        # Isolate predictions from Python response stream by removing 'ping' and 'marker received' messages
        extracted_predictions = [
            s for sublist in python_response_markers for s in sublist
            if 'ping' not in str(s).lower() and 'marker received' not in str(s).lower()
        ]

        # Convert the list of strings to a list of integers
        predicted_targets = [int(s.strip('[]')) for s in extracted_predictions]
        return predicted_targets

    def compare_predictions(self, predicted_targets, target_element_iSPOs):
        """
        Compare predicted targets to actual targets.
        Prints confusion matrix and accuracy if print_confusion_matrix is True.

        Parameters
        ----------
        predicted_targets : list
            List of predicted targets
        target_element_iSPOs : list
            List of actual targets

        Returns
        -------
        None
        """
        # Check that the number of predictions matches number of actual targets
        assert len(predicted_targets) == len(target_element_iSPOs)

        # Convert to binary classification: 1 if the target matches, 0 if it doesn't match
        binary_preds = [1 if pred == true else 0 for pred, true in zip(predicted_targets, target_element_iSPOs)]
        binary_truths = [1] * len(target_element_iSPOs)  # All actual targets are correct

        # Compute confusion matrix
        cm = confusion_matrix(binary_truths, binary_preds, labels=[1, 0])
        accuracy = accuracy_score(binary_truths, binary_preds) * 100

        # Print confusion matrix and accuracy if print_confusion_matrix is True
        if self.print_confusion_matrix:
            self.trial_count += 1
            print(f"Trial {self.trial_count} confusion matrix:")
            print(cm)
            print(f"Accuracy: {accuracy:.2f}%\n")

        # Add the accuracy (percentage of correct predictions) to the list
        self.prediction_accuracies.append(accuracy)

    def process_settings(self, settings_file):
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
        
        # Finally, check the number of columns
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

    def print_results(self):
        """
        Print the results of the data analysis.
        
        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # Print trial descriptions from the results dataframe
        print("\nTrial Descriptions:")
        for i in range(len(self.results_df)):
            print(self.results_df["Trial Description"][i])

        # Print trial IDs for each trial
        print("\nTrial IDs:")
        for i in range(len(self.results_df)):
            print(self.results_df["Trial ID"][i])

        # Print accuracy (percentage of correct predictions) for each trial
        print("\nPrediction Accuracies:")
        for i in range(len(self.results_df)):
            print(f"{self.results_df["Prediction Accuracy"][i]:.2f}%")
        
    def output_results_dataframe(self):
        """
        Store the results to a pandas DataFrame and print it.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # Create a DataFrame from the results
        self.results_df["Trial Description"] = [self.trial_ID_dict[trial_ID] for trial_ID in self.trial_IDs]
        self.results_df["Trial ID"] = self.trial_IDs
        self.results_df["Prediction Accuracy"] = self.prediction_accuracies

        print("\nResults DataFrame:")
        print(self.results_df)
        self.results_df.to_clipboard(index=False, sep="\t")

    def sort_accuracies_by_trial(self):
        """
        Order the accuracies by trial ID and print the ordered list.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        sorted_df = self.results_df.sort_values(by="Trial ID")
        print("\nSorted Accuracies by Trial ID:")
        for i in range(len(sorted_df)):
            print(f"{sorted_df["Prediction Accuracy"][i]:.2f}%")

def main():
    # Path to the folder containing the data to analyze
    folder_path = "D:/Daniella Bourque/Boccia Validation/Participant-Data/250428_Participant_7_Data"

    # Name of the target element stream: "TargetElementStream_Play" or "TargetElementStream_VirtualPlay"
    # Depending on whether Play Boccia or Virtual Play mode was used when collecting the data
    target_stream_name = "TargetElementStream_VirtualPlay"

    # Initialize the BocciaDataAnalysis class
    boccia_data_analysis = BocciaDataAnalysis(folder_path, target_stream_name)

    # Process the trials
    for trial_number in boccia_data_analysis.trial_data:
        boccia_data_analysis.process_trial(trial_number)

    # Output the results
    boccia_data_analysis.output_results_dataframe()
    boccia_data_analysis.sort_accuracies_by_trial()
    # boccia_data_analysis.print_results()

if __name__ == "__main__":
    main()