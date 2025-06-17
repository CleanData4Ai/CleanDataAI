"""
EdaApp Class

This class is part of the Exploratory Data Analysis (EDA) framework and is designed to streamline the process of analyzing datasets. 
The EdaApp class extends the functionality of the EdaCoreFunctions module and provides a user-friendly interface for performing 
comprehensive EDA workflows. 

Key Features:
- Handles datasets in pandas DataFrame format.
- Supports a variety of analyses, including summary statistics, missing value analysis, and correlation analysis.
- Allows for time series analysis when a timestamp column is provided.
- Saves results in a structured output folder, customizable by the user.
- Leverages user-defined settings via a YAML configuration file to customize the EDA process.

Usage:
- Initialize the class by providing the dataset, target column, and optional parameters like a timestamp column and user settings file path.
- Run the EDA workflow using the `run_app` method, which executes the user-defined EDA functions.

This class is designed to make EDA easier, faster, and more accessible for data analysts and scientists by automating repetitive tasks 
and providing customizable, efficient functionality.
"""

import os
import pandas as pd

# Print the current working directory (useful for debugging and relative paths)
print(os.getcwd())

# If working in the same folder structure, use the relative import like below:
# from data_analysis_exploration.pandas_eda_lib.eda_utilities.eda_core_functions import EdaCoreFunctions

# Otherwise, use the full path for importing EdaCoreFunctions
from eda_lib.data_analysis_exploration.pandas_eda_lib.eda_utilities.eda_core_functions import EdaCoreFunctions


class EdaApp(EdaCoreFunctions):
    """
    EdaApp class that inherits from EdaCoreFunctions to provide high-level EDA functionality.
    """

    def __init__(self,
                 data: pd.DataFrame,
                 targetColumn: str,
                 base_path: str,
                 project_root_name: str,
                 yaml_user_settings_path: str,
                 timestampColumn: str = None,
                 output_path: str = "data_analysis_exploration/output_folder",
                 project_folder_name: str = "default folder name",
                 project_name: str = "default project name"):
        """
        Initializes the EdaApp class with the required parameters.

        Parameters:
        - data (pd.DataFrame): Input dataset.
        - targetColumn (str): Name of the target column.
        - base_path (str): Base path where project folders and outputs will be stored.
        - project_root_name (str): Root folder name for the project.
        - yaml_user_settings_path (str): Path to the YAML config file for user-defined settings.
        - timestampColumn (str, optional): Name of the timestamp column for time series analysis.
        - output_path (str, optional): Path where the output will be saved.
        - project_folder_name (str, optional): Subfolder name for organizing project outputs.
        - project_name (str, optional): Name of the project.
        """

        # Initialize the base class with all required arguments
        super().__init__(
            data=data,
            targetColumn=targetColumn,
            timestampColumn=timestampColumn,
            project_folder_name=project_folder_name,
            base_path=base_path,
            project_root_name=project_root_name,
            output_path=output_path,
            project_name=project_name,
            yaml_user_settings_path=yaml_user_settings_path
        )

    def run_app(self):
        """
        Executes the user-defined EDA workflow as specified in the YAML config.
        """
        self.user_eda_functions()
