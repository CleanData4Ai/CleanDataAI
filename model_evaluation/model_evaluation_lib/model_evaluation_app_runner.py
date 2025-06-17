"""
MlEvaluationApp Class

Purpose:
The `MlEvaluationApp` class serves as the entry point for initializing and executing 
the model evaluation framework. Built on top of the `MlEvaluationCoreFunctions` class, 
it encapsulates all the essential functionalities required to evaluate machine learning 
models in a structured and automated manner.

Key Responsibilities:
- Framework Initialization: Establishes the core components of the model evaluation process
- Core Function Execution: Invokes the core evaluation functions to perform a comprehensive 
  analysis of the models' performance

Attributes:
- models (list[str]): List of model names or identifiers to be evaluated
- training_dataframes_list (list[pd.DataFrame]): List of training datasets
- validation_dataframes_list (list[pd.DataFrame]): List of validation datasets
- test_dataframes_list (list[pd.DataFrame]): List of test datasets
- project_root_name (str): Root name for the project
- base_path (str): Base directory path
- project_folder_name (str): Name of the folder for evaluation results
- output_path (str): Path where outputs will be stored
"""

# -----------------------------------------
# Imports for MlEvaluationApp
# -----------------------------------------

# If working in the same directory
# from model_evaluation.model_evaluation_lib.model_evaluation_utilities.model_evaluation_core_functions import MlEvaluationCoreFunctions

# Otherwise, use the full path as per module structure
from eda_lib.model_evaluation.model_evaluation_lib.model_evaluation_utilities.model_evaluation_core_functions import MlEvaluationCoreFunctions

import pandas as pd


class MlEvaluationApp(MlEvaluationCoreFunctions):
    def __init__(
        self, 
        models: list[str], 
        training_dataframes_list: list[pd.DataFrame],
        validation_dataframes_list: list[pd.DataFrame], 
        test_dataframes_list: list[pd.DataFrame],
        user_settings_path: str, 
        project_root_name: str,
        base_path: str,
        project_folder_name: str = "default Project Title",
        output_path: str = "model_evaluation/output_folder",
        project_name: str = "default Project Name"
    ):
        """
        Initialize the MlEvaluationApp with model and dataset information.

        Parameters:
        -----------
        models : list[str]
            List of model names to evaluate
        training_dataframes_list : list[pd.DataFrame]
            List of training datasets for each model
        validation_dataframes_list : list[pd.DataFrame]
            List of validation datasets for each model
        test_dataframes_list : list[pd.DataFrame]
            List of test datasets for each model
        user_settings_path : str
            Path to the YAML settings file for evaluation configuration
        project_root_name : str
            Root name for the project
        base_path : str
            Base directory path
        project_folder_name : str, optional
            Name of the folder for evaluation results, defaults to "default Project Title"
        output_path : str, optional
            Path where outputs will be stored, defaults to "model_evaluation/output_folder"
        project_name : str, optional
            Human-readable project name, defaults to "default Project Name"
        """

        # Initialize the parent class with all required attributes
        super().__init__(
            models=models,
            training_dataframes_list=training_dataframes_list,
            validation_dataframes_list=validation_dataframes_list,
            test_dataframes_list=test_dataframes_list,
            output_path=output_path,
            base_path=base_path,
            project_folder_name=project_folder_name,
            project_root_name=project_root_name, 
            user_settings_path=user_settings_path,
            project_name=project_name
        )
    
    def run_app(self):
        """
        Execute the model evaluation process.
        
        This method triggers the core evaluation functions which perform
        metrics calculation, comparison, and reporting for the specified models.
        """
        self.run_core_functions()
