import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class BocciaDataPlotter:
    def __init__(self, file_path):
        self.file_path = file_path
        self.boccia_data = None

        # Initialize empty DataFrames for average accuracy
        self.average_accuracy_per_condition = pd.DataFrame()
        self.average_accuracy_per_participant = pd.DataFrame()

    def load_data(self):
        try:
            # Load the data from the CSV file
            self.boccia_data = pd.read_csv(self.file_path)
            # print(self.boccia_data)  # Display the first few rows of the data
        except FileNotFoundError:
            print(f"Error: File not found at {self.file_path}")
        except pd.errors.EmptyDataError:
            print("Error: The file is empty or invalid.")

    def handle_missing_data(self):
        # Check if data is loaded
        if self.boccia_data is None:
            raise ValueError("No data loaded yet.")

        # Replace missing values with 0
        self.boccia_data.fillna(0, inplace=True)

    def calculate_average_accuracy(self):
        # Check if data is loaded
        if self.boccia_data is None:
            raise ValueError("No data loaded yet.")

        # Calculate the average accuracy for each condition
        self.average_accuracy_per_condition['Condition Number'] = self.boccia_data.iloc[:, 0]
        self.average_accuracy_per_condition['Average Accuracy'] = self.boccia_data.iloc[:, 2:10].mean(axis=1)

        # Calculate the average accuracy for each participant
        self.average_accuracy_per_participant['Participant'] = range(1, 9)
        self.average_accuracy_per_participant['Average Accuracy'] = self.boccia_data.iloc[:, 2:10].mean(axis=0).values

    def plot_accuracy_per_participant(self):
        # Check if data is loaded
        if self.boccia_data is None:
            raise ValueError("No data loaded yet.")

        fig, axes = plt.subplots(2, 4, figsize=(10, 5))
        fig.suptitle('Inference Accuracy for Each Participant', fontsize=16)

        for i, column in enumerate(self.boccia_data.columns[2:10]):
            ax = axes[i // 4, i % 4]
            ax.bar(self.boccia_data[self.boccia_data.columns[0]], self.boccia_data[column])
            ax.set_title(column)
            ax.set_xlabel(self.boccia_data.columns[0])
            ax.set_xticks(self.boccia_data[self.boccia_data.columns[0]])
            ax.set_ylabel("Accuracy (%)")
            ax.set_ylim(0, 100)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    def plot_accuracy_per_condition(self):
        # Check if data is loaded
        if self.boccia_data is None:
            raise ValueError("No data loaded yet.")

        fig, axes = plt.subplots(2, 3, figsize=(10, 5))
        fig.suptitle('Inference Accuracy for Each Condition', fontsize=16)

        for i, (index, row) in enumerate(self.boccia_data.iterrows()):
            ax = axes[i // 3, i % 3]
            x_labels = range(1, 9) 
            y_values = row[2:10]  # Use row values (2 onwards) as y-axis values
            ax.bar(x_labels, y_values)
            ax.set_title(f"Condition {row[self.boccia_data.columns[0]]}")  # Use the first column as the title
            ax.set_xlabel("Participant Number")
            ax.set_ylabel("Accuracy (%)")
            ax.set_xticks(x_labels)
            ax.set_xticklabels(x_labels)
            ax.set_ylim(0, 100)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    def plot_participant_accuracy_single_plot(self):
        # Check if data is loaded
        if self.boccia_data is None:
            raise ValueError("No data loaded yet.")

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.suptitle('Boccia Validation Inference Accuracy per Participant', fontsize=16)

        num_participants = len(self.boccia_data.columns[2:10])
        num_conditions = len(self.boccia_data)

        x_positions = np.arange(num_participants)

        bar_width = 0.10  # Width of each bar

        for i in range(num_conditions):
            y_values = self.boccia_data.iloc[i, 2:10]
            ax.bar(x_positions + i * bar_width, y_values, width=bar_width)
        
        center = x_positions + (num_conditions - 1) * bar_width / 2
        ax.set_xticks(center)
        ax.set_xticklabels(range(1, num_participants + 1))
        ax.set_xlabel("Participant Number")
        ax.set_ylabel("Accuracy (%)")

        # Add legend
        legend_labels = [f"{i + 1}" for i in range(num_conditions)]
        ax.legend(legend_labels, title="Condition", bbox_to_anchor=(1.05, 1))

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    def plot_condition_accuracy_single_plot(self):
        # Check if data is loaded
        if self.boccia_data is None:
            raise ValueError("No data loaded yet.")

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.suptitle('Inference Accuracy per Participant per Condition', fontsize=20, y = 0.92)

        num_conditions = len(self.boccia_data)
        num_participants = len(self.boccia_data.columns[2:10])

        x_positions = np.arange(num_conditions)  

        bar_width = 0.10  # Width of each bar

        for i in range(num_participants):
            y_values = self.boccia_data.iloc[:, i + 2]
            ax.bar(x_positions + i * bar_width, y_values, width=bar_width)

        center = x_positions + (num_participants - 1) * bar_width / 2
        ax.set_xticks(center)
        ax.set_xticklabels(range(1, num_conditions + 1))
        ax.set_xlabel("Condition Number", fontsize=16)
        ax.set_ylabel("Accuracy (%)", fontsize=16)

        # Adjust tick label font size
        ax.tick_params(axis='x', labelsize=16)  # Set font size for x-axis tick labels
        ax.tick_params(axis='y', labelsize=16)  # Set font size for y-axis tick labels

        # Add legend
        legend_labels = [f"{i + 1}" for i in range(num_participants)]
        ax.legend(legend_labels, title="Participants", bbox_to_anchor=(1.02, 1))

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    def plot_average_accuracy_per_condition(self):
        # Check if average accuracy is calculated
        if self.average_accuracy_per_condition.empty:
            raise ValueError("Average accuracy per condition not calculated yet.")

        fig, ax = plt.subplots(figsize=(7, 5))
        fig.suptitle('Average Inference Accuracy per Condition', fontsize=20, y = 0.92)

        ax.bar(self.average_accuracy_per_condition['Condition Number'], self.average_accuracy_per_condition['Average Accuracy'])
        ax.set_xlabel("Condition Number", fontsize=16)
        ax.set_ylabel("Average Accuracy (%)", fontsize=16)
        ax.set_ylim(0, 100)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    def plot_average_accuracy_per_participant(self):
        # Check if average accuracy is calculated
        if self.average_accuracy_per_participant.empty:
            raise ValueError("Average accuracy per participant not calculated yet.")

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.suptitle('Average Inference Accuracy per Participant', fontsize=16, y = 0.92)

        ax.bar(self.average_accuracy_per_participant['Participant'], self.average_accuracy_per_participant['Average Accuracy'])
        ax.set_xlabel("Participant Number")
        ax.set_ylabel("Average Accuracy (%)")
        ax.set_ylim(0, 100)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

def main():
    # Create an instance of the BocciaDataPlotter class
    boccia_plotter = BocciaDataPlotter('D:/Daniella Bourque/Boccia Validation/Boccia_Validation_Data.csv')
    
    # Load the data
    boccia_plotter.load_data()
    boccia_plotter.handle_missing_data()

    boccia_plotter.calculate_average_accuracy()

    # boccia_plotter.plot_accuracy_per_participant()
    # boccia_plotter.plot_accuracy_per_condition()
    # boccia_plotter.plot_participant_accuracy_single_plot()
    boccia_plotter.plot_condition_accuracy_single_plot()
    boccia_plotter.plot_average_accuracy_per_condition()
    boccia_plotter.plot_average_accuracy_per_participant()

if __name__ == "__main__":
    main()