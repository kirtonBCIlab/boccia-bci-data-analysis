import numpy as np
import pyxdf
import os

class BocciaDataAnalysis:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.files = self.retrieve_files()

        self.target_stream_name = "TargetElementStream_VirtualPlay"
        self.python_response_stream_name = "PythonResponse"

        # Initialize target element stream variables
        self.target_markers = None
        self.target_time = None

        # Initialize python response stream variables
        self.python_response_markers = None
        self.python_response_time = None
    
    def retrieve_files(self):
        with os.scandir(self.folder_path) as entries:
            files = [entry.path for entry in entries if entry.is_file() and entry.name.endswith(".xdf")]
        return files
    
    def process_streams(self, xdf_file):
        # Load the xdf file
        streams, fileheader = pyxdf.load_xdf(xdf_file)

        # Get data from streams of interest
        self.target_markers, self.target_time = self.get_stream_data(streams, self.target_stream_name)
        print(self.target_markers)
        self.python_response_markers, self.python_response_time = self.get_stream_data(streams, self.python_response_stream_name)

        # Get list of actual target elements
        target_element_iSPOs = self.get_target_elements(self.target_markers)

        # Get list of predicted targets from Python
        predicted_targets = self.get_predictions(self.python_response_markers)

        # Compare predicted targets to actual targets
        self.compare_predictions(predicted_targets, target_element_iSPOs)
        
    
    def get_stream_data(self, streams, stream_name):
        stream_index = next((i for i, stream in enumerate(streams) if stream_name in stream['info']['name'][0]), None)
        print(f"Stream {stream_index}: {streams[stream_index]['info']['name'][0]}")
        stream = streams[stream_index]

        stream_markers = stream['time_series']
        stream_time = stream['time_stamps']

        return stream_markers, stream_time
    
    def get_target_elements(self, target_markers):
        target_elements = [int(item[1].split(": ")[1]) for item in target_markers]
        return target_elements
    
    def get_predictions(self, python_response_markers):
        # Extract predictions from Python response stream by removing 'ping' and 'marker received' messages
        extracted_predictions = [
            s for sublist in python_response_markers for s in sublist
            if 'ping' not in str(s).lower() and 'marker received' not in str(s).lower()
        ]

        predicted_targets = [int(s.strip('[]')) for s in extracted_predictions] # Format as list of integers
        return predicted_targets

    def compare_predictions(self, predicted_targets, target_element_iSPOs):
        # Make sure number of predictions matches number of actual targets
        assert len(predicted_targets) == len(target_element_iSPOs)

        # Determine number of correct predictions
        num_correct = 0
        num_total = len(predicted_targets)
        for i in range(num_total):
            if predicted_targets[i] == target_element_iSPOs[i]:
                num_correct += 1

        percent_correct = (num_correct / num_total) * 100

        print(f"Number of correct predictions: {num_correct}")
        print(f"Number of total predictions: {num_total}")
        print(f"Percentage correct: {percent_correct:.2f}%\n")

def main():
    folder_path = "D:/Daniella Bourque/3-28_Testing_Data" # Path to the folder containing the data

    boccia_data_analysis = BocciaDataAnalysis(folder_path)

    count = 0
    for file in boccia_data_analysis.files:
        count += 1
        print("Trial " + str(count) + " results:")
        boccia_data_analysis.process_streams(file)

if __name__ == "__main__":
    main()