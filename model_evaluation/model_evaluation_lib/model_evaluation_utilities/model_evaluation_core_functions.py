"""
Core class for managing and evaluating machine learning models.
This class integrates settings initialization, report generation, and HTML template creation
while saving the results to dynamically generated file paths.
"""

# -----------------------------------------
# Standard Library Imports
# -----------------------------------------
from datetime import datetime  # For working with date and time, used for timestamping folder names or analysis
import os  # For interacting with the operating system (e.g., file paths, environment variables)
from pathlib import Path  # For creating and managing file system paths

# -----------------------------------------
# Third-Party Library Imports
# -----------------------------------------
import pandas as pd  # For data manipulation and analysis, commonly used with DataFrames
import matplotlib.pyplot as plt  # For creating visualizations and plots

# -----------------------------------------
# Project-Specific Imports
# -----------------------------------------


# ---- Use the following if working in the same directory ----
# from src.graphical_data_visualizations.visualisation.css import css_code_with_black_side_bar  # Predefined CSS styles for black sidebar in visualizations
# from src.graphical_data_visualizations.visualisation.template import base_template, model_evalutation_report  # Base HTML template for report generation
# from src.file_operations_utils.file_operations import create_file_path, load_yaml, load_files_to_dataframes  # File handling and YAML config utilities
# from src.graphical_data_visualizations.visualisation.visualisation_utils import VisualisationUtils  # Graphical data visualization utilities
#from model_evaluation.model_evaluation_lib.model_evaluation_utilities.model_evalution_help_functions import MlEvaluationHelpFunctions


# ---- Use the following if using from module/package structure ----
from eda_lib.src.graphical_data_visualizations.visualisation.css import css_code_with_black_side_bar  # Predefined CSS styles for black sidebar in visualizations
from eda_lib.src.graphical_data_visualizations.visualisation.template import base_template, model_evalutation_report  # Base HTML template for report generation
from eda_lib.src.file_operations_utils.file_operations import create_file_path, load_yaml, load_files_to_dataframes  # Utilities for file handling and YAML configuration
from eda_lib.src.graphical_data_visualizations.visualisation.visualisation_utils import VisualisationUtils  # Utilities for generating graphical data visualizations
from eda_lib.model_evaluation.model_evaluation_lib.model_evaluation_utilities.model_evalution_help_functions import MlEvaluationHelpFunctions

# Optional import for activating LLMs (commented out)
# from large_language_model_llm_lib.activate_llm import ActivateLlm

# -----------------------------------------
# Built-in Module Imports
# -----------------------------------------
from collections import defaultdict  # For creating dictionaries with default values, used for efficient data handling

# Optional LLM activation module if needed (commented out)
# from src.large_language_model_llm_lib.activate_llm import ActivateLlm



class MlEvaluationCoreFunctions(MlEvaluationHelpFunctions):


    def __init__ (self, 
            models: list[str], 
            training_dataframes_list: list[pd.DataFrame],
            validation_dataframes_list: list[pd.DataFrame], 
            test_dataframes_list: list[pd.DataFrame],
            user_settings_path: str, 
            project_root_name: str,
            base_path: str,
            project_folder_name: str = "default Project Title",
            output_path: str = "model_evaluation/output_folder" ,
            project_name: str = "default_name"):
        
        
        # ✅ Ensure the parent class (MlEvaluationHelpFunctions) is initialized
        super().__init__(user_settings_path = user_settings_path , base_path= base_path , project_root_name = project_root_name)

        self.project_folder_name = project_folder_name  
        self.training_dataframes_list = training_dataframes_list
        self.validation_dataframes_list = validation_dataframes_list
        self.test_dataframes_list = test_dataframes_list 
        self.model_names = [model for model in models] 
        self.output_path = output_path
        self.base_path = base_path
        self.project_root_name = project_root_name
        self.user_settings_path = user_settings_path
        self.project_name = project_name
        self._initialize_settings() 
   

    def _initialize_settings(self):

            """

            Initializes the class settings by loading configurations, visualizations, and user-defined parameters
            from the YAML configuration file. Also configures paths and parameters for various analyses and reports.

            Attributes Set:
                llm_model (ActivateLlm): Instance of the LLM (Large Language Model) activation utility.
                gui_instance (VisualisationUtils): Instance for generating visualizations and HTML content.
                user_settings (dict): Loaded user-defined configuration settings from the YAML file.
                base_dir (str): Base output directory for storing all generated reports.
                task_type (str): Type of task, either "classification" or "regression".
                ai_assistant (bool): Whether AI assistance/feedback is enabled.
                training_dataframes_paths (list): Paths to the training data files.
                validation_dataframes_paths (list): Paths to the validation data files.
                test_dataframes_paths (list, optional): Paths to the test data files, if provided.
                time_analysis_active (bool): Whether time-based analysis is enabled.
                time_analysis_months (int): Number of months for time-based analysis.
                timestamp_column (str): Column name for timestamps in time-based analysis.
                non_time_analysis_active (bool): Whether non-time-based segmentation analysis is enabled.
                segmentation_column (str, optional): Column name for segmentation analysis.
                subcategory_threshold (int): Threshold for subcategory analysis in non-time-based settings.
                sidebar_title (str): Title for the HTML report's sidebar.
                side_bar_logo (str): Logo path for the HTML report's sidebar.
                report_paths (list): List of dynamically generated report paths.

            Raises:
                KeyError: If any required key is missing in the configuration file.
                ValueError: If invalid or inconsistent parameters are provided in the configuration file.

            """

            #self.llm_model = ActivateLlm()
        
            # Load visualization utilities for HTML generation
            self.gui_instance = VisualisationUtils()

            # Load user-defined configuration from the YAML file
            self.user_settings = load_yaml(os.path.join(self.base_path, self.project_root_name, self.user_settings_path))

            # General settings

            self.base_dir = os.path.join(self.base_path, self.project_root_name, self.output_path)
            
            self.task_type = self.user_settings["parameters"]["task_type"] # Task type: "classification" or "regression"

        
           
            #self.ai_assistant = bool(self.user_settings["parameters"]["ai_feedback"])  # List of model names
            
            # Time-based analysis settings
            self.time_analysis_active = bool(
                self.user_settings["time_based_analysis_paramters"]["activate_analysis"])  # Time-based analysis toggle
            self.time_analysis_months = int(
                self.user_settings["time_based_analysis_paramters"]["number_of_months"])  # Months for analysis
            self.timestamp_column = self.user_settings["time_based_analysis_paramters"]["timestamp_column"]  # timestamp column

            # Non-time-based analysis settings
            self.non_time_analysis_active = bool(self.user_settings["non_time_based_analysis_paramters"][
                                                    "activate_analysis"])  # Segmentation analysis toggle
            self.segmentation_column = self.user_settings["non_time_based_analysis_paramters"].get("segmentation_column")

            self.subcategory_threshold = int(self.user_settings["non_time_based_analysis_paramters"].get("subcategory_threshold"))

            # Configure output paths dynamically using timestamped folder names
            self.folder_name_time = f"{self.project_folder_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
            self.sidebar_title = self.user_settings["display_settings"]["side_bar_title"]  # Sidebar title for HTML
            self.side_bar_logo = self.user_settings["display_settings"]["side_bar_logo"]  # Sidebar logo for HTML
            self.report_paths = self._generate_report_paths()  # Generate dynamic paths for reports
            self.generate_html_files(base_template, css_code_with_black_side_bar)  # Generate the required HTML files
    

    def _generate_report_paths(self):



            """


                Generates file paths for various analytical reports based on the provided project folder
                name and timestamp. The paths are organized for easy access and storage of reports.

               

                Returns:
                    dict: A dictionary containing the generated file paths for different report types:
                        - "model_comparison_and_analysis": Path for the model comparison and analysis report.
                        - "non_time_based_segmentation": Path for the non-time-based segmentation report.
                        - "time_based_segmentation": Path for the time-based segmentation report.

                Notes:
                    - The file paths are constructed by joining the base directory, project folder name,
                    and the timestamped folder name.
                    - The `create_file_path` function is used to generate specific file paths for each report
                    type by appending the appropriate report identifier to the base path.


            """

            try  : 


                # Construct the base path by combining the base directory, project folder name, and timestamped folder name

                self.base_path = os.path.join(self.base_dir, self.project_folder_name, self.folder_name_time)

                # Return a dictionary with the paths for each report type

                return {
                    "model_comparison_and_analysis": create_file_path(self.base_path, "model_analysis_and_comparison"),
                    # Path for model comparison report
                    "non_time_based_segmentation": create_file_path(self.base_path, "non_time_segmentation_analysis"),
                    # Path for non-time-based segmentation report
                    "time_based_segmentation": create_file_path(self.base_path, "time_based_segmentation_analysis")
                    # Path for time-based segmentation report
                   
                }

            except Exception as e   : 

                print("an Error has accured  please check the function _generate_report_paths {e}")

            
    def generate_html_files(self, base_template, css_code):

            """

            Generates HTML templates for different types of analytical reports, including model comparisons,
            and both time-based and non-time-based segmentation analyses. The reports are dynamically created
            and optimized for memory usage by utilizing lazy loading.

            Args:
                base_template (str): The base HTML template used as the foundation for report structure.
                                    This template includes the overall layout and placeholders for dynamic content.
                css_code (str): The CSS styles used to ensure consistent visual presentation and formatting
                                across all generated reports.

            Notes:
                - This method constructs HTML files for different types of analyses (e.g., model comparison,
                segmentation analysis) by dynamically inserting content into the base template.
                - Lazy loading is employed to optimize memory usage by creating each report's content only when needed.
                - The content of each report type (e.g., project overview, model analysis) is predefined in the
                `html_structure` dictionary, ensuring that each report is correctly formatted.

            Process:
                1. **Prepare Common Arguments**: Shared settings such as the base template, CSS, title, and logo URL
                are gathered in a dictionary for consistent application across all reports.
                2. **Define HTML Structure**: The content and sections for each type of report (model comparison,
                non-time-based segmentation, time-based segmentation) are predefined.
                3. **Generate HTML Content**: A helper function (`generate_html`) is used to dynamically generate
                the HTML content for each report by inserting the relevant sections into the base template.
                4. **Lazy Loading**: Each report is generated and stored only when required, improving memory efficiency.


            """

            try:
                # Shared arguments for all HTML templates
                common_args = {
                    "base_template": base_template,
                    "css_code": css_code,
                    "title": self.sidebar_title,
                    "logo_url": self.side_bar_logo,
                }

                # Predefined HTML structure for different report types
                html_structure = {
                    'model_comparison_analysis': [
                        ("Project overview and key information", []),
                        ("Model Comparison", []),
                    ],
                    'non_time_based_segment_analysis': [
                        ("Project overview and key information", []),
                        ("Model Comparison", []),
                    ],
                    'time_based_segment_analysis': [
                        ("Project overview and key information", [])
                    ],
                }
                 
                if self.non_time_analysis_active  :

                    # Get unique segmentation values from training or validation data
                    unique_values = (
                        self.training_dataframes_list[0][self.segmentation_column].unique()
                        if self.training_dataframes_list
                        else self.validation_dataframes_list[0][self.segmentation_column].unique()
                    )

                    
                    # Access the "Model Comparison" list in non_time_based_segment_analysis and append the new item
                    for key in unique_values : 
                        for section in html_structure['non_time_based_segment_analysis']:
                            if section[0] == "Model Comparison":
                                section[1].append(f"Subcategory {key} : Model Comparison Table")
                                break

                # Append model-specific entries to model_comparison_analysis
                for model in self.model_names:

                    html_structure["time_based_segment_analysis"].append(
                        (
                            model , 
                            [
                                f"{model}: Time Based Model Analysis"  

                            ],
                        )


                    )

                    if self.task_type =="classification" : 

                        html_structure["model_comparison_analysis"].append(
                            
                            (
                                model,
                                [
                                    f"Target Distribution (Density Plot) {model}",
                                    f"Percentage of Predictions per Score Bin {model}",
                                    f"Positive Prediction Rate per Score Bin {model}",
                                ],
                            )
                        )

                        
                    else  : 

                        html_structure["model_comparison_analysis"].append(
                            (f"Prediction  Distribution for model : {model}",[])
                        )

                if self.non_time_analysis_active : 

                    if self.task_type =="classification" :
                        
                        # Define metrics in the desired order
                        metrics = [
                            "Target Distribution (Density Plot)",
                            "Percentage of Predictions per Score Bin",
                            "Positive Prediction Rate per Score Bin",
                        ]

                        # Create a lookup dictionary for non_time_based_segment_analysis for quick access
                        ntb_analysis = html_structure["non_time_based_segment_analysis"]
                        model_lookup = {name: idx for idx, (name, _) in enumerate(ntb_analysis)}

                        # Build and add metrics for each model under non_time_based_segment_analysis
                        for model in self.model_names:
                            # Create metric list using a list comprehension
                            metric_list = [
                                f"{metric} {model} {segmentation_value}"
                                for metric in metrics
                                for segmentation_value in unique_values
                            ]

                            if model in model_lookup:
                                # Append new metrics if the model already exists
                                idx = model_lookup[model]
                                existing_metrics = ntb_analysis[idx][1]
                                existing_set = set(existing_metrics)
                                for value in metric_list:
                                    if value not in existing_set:
                                        existing_metrics.append(value)
                                        existing_set.add(value)
                            else:
                                ntb_analysis.append((model, metric_list))
                                model_lookup[model] = len(ntb_analysis) - 1
                    else  : 
                            metrics = [
                            "Prediction  Distribution for model :"
                            ]
                            # Create a lookup dictionary for non_time_based_segment_analysis for quick access
                            ntb_analysis = html_structure["non_time_based_segment_analysis"]
                            model_lookup = {name: idx for idx, (name, _) in enumerate(ntb_analysis)}

                            # Build and add metrics for each model under non_time_based_segment_analysis
                            for model in self.model_names:
                                # Create metric list using a list comprehension
                                metric_list = [
                                    f"{metric} {model} {segmentation_value}"
                                    for metric in metrics
                                    for segmentation_value in unique_values
                                ]


                                if model in model_lookup:
                                    # Append new metrics if the model already exists
                                    idx = model_lookup[model]
                                    existing_metrics = ntb_analysis[idx][1]
                                    existing_set = set(existing_metrics)
                                    for value in metric_list:
                                        if value not in existing_set:
                                            existing_metrics.append(value)
                                            existing_set.add(value)
                                else:
                                    ntb_analysis.append((model, metric_list))
                                    model_lookup[model] = len(ntb_analysis) - 1


                # Helper function to generate HTML dynamically using the predefined structure and common arguments
                def generate_html(side_bar_menu_options):

                    """

                        Generates HTML content by inserting the predefined sections into the base template.

                        Args:
                            side_bar_menu_options (list): A list of tuples defining the sections and content for the report.

                        Returns:
                            str: The generated HTML content for the report.

                    """
                    return self.gui_instance.generate_html(
                        side_bar_menu_options,  # Content to insert into the template
                        common_args["base_template"],  # Base template for structure
                        common_args["css_code"],  # CSS for consistent styling
                        common_args["title"],  # Title for the report
                        common_args["logo_url"]  # Logo URL for the sidebar
                    )

                # Lazy-load HTML content for each report type based on the predefined structure
                self.model_comparison_analysis = generate_html(html_structure['model_comparison_analysis'])
                self.time_based_segment_analysis = generate_html(html_structure['time_based_segment_analysis'])
                self.non_time_segment_analysis = generate_html(html_structure['non_time_based_segment_analysis'])

            except Exception as e  : 

                print(f"an Error has  accured please check the function generate_html_files !!!!! {e} ")


    def generate_project_overview(self, start_time: datetime, end_time: datetime, subsection_id: str, title: str):

        """

            Generates an overview of the project by creating a DataFrame with specific project details
            and integrating it into the NCR dashboard.

            Parameters:
            -----------
            start_time : datetime
                The beginning of the time period for which the project overview is generated.
            end_time : datetime
                The end of the time period for which the project overview is generated.
            subsection_id : str
                A unique identifier for the specific subsection within the dashboard where the overview will be displayed.
            title : str
                The title to be assigned to the DataFrame within the dashboard for clear identification.

            Process:
            ---------
            1. **DataFrame Creation**: A DataFrame containing project-specific details is generated using
            the `create_project_dataframe` method, which calculates the duration of the project based
            on the provided start and end times.

            2. **Output Paths Preparation**: A list of output paths is defined to streamline the integration
            of the generated DataFrame into the dashboard, thereby reducing redundancy in the code.

            3. **Dashboard Integration**: The method iterates over the predefined output paths, integrating
            the DataFrame into the NCR dashboard using the `ncr_dashboard_dataframe` method. Each
            integration specifies the unique subsection ID and the designated title for the DataFrame.

            4. **Memory Management**: The DataFrame is deleted after its use to free up memory resources.

            Exception Handling:
            --------------------
            Any exceptions raised during execution will be caught and managed to ensure the stability
            of the application and to provide useful debugging information.

            Returns:
            --------
            None

        """
        try:

            # Generate a DataFrame with project-specific details
            df = self.create_project_dataframe(self.project_name, start_time, end_time, (end_time - start_time))

            self.model_comparison_analysis = self.gui_instance.ncr_dashboard_dataframe(
                self.model_comparison_analysis,  # Output file path for the dashboard
                subsection_id,  # Unique subsection ID for the dashboard entry
                df,  # DataFrame to be added to the dashboard
                dataframe_name=title  # Title for the DataFrame in the dashboard
            )
            self.time_based_segment_analysis = self.gui_instance.ncr_dashboard_dataframe(
                self.time_based_segment_analysis,  # Output file path for the dashboard
                subsection_id,  # Unique subsection ID for the dashboard entry
                df,  # DataFrame to be added to the dashboard
                dataframe_name=title  # Title for the DataFrame in the dashboard
            )

            self.non_time_segment_analysis = self.gui_instance.ncr_dashboard_dataframe(
                self.non_time_segment_analysis,  # Output file path for the dashboard
                subsection_id,  # Unique subsection ID for the dashboard entry
                df,  # DataFrame to be added to the dashboard
                dataframe_name=title  # Title for the DataFrame in the dashboard
            )

            del df

        except Exception as e:

            print(f"an error  accured  in the function  : generate_project_overview : {e}")

    
    def model_analysis_core_function(self):
            """
            This function performs model analysis based on the task type (classification or regression).
            
            Steps:
            1. Executes model analysis or regression analysis depending on task type.
            2. Concatenates results into a single table for inspection.
            3. Generates a workflow report and saves it as an HTML file.
            4. Updates the GUI dashboard with results and visualizations.
            
            Exception Handling:
            - Captures and prints errors that occur during execution.
            """
            
            try  : 

                    if self.task_type == "classification":
                        # Step 1: Perform model analysis and obtain the results (dataframes and figures)
                        dataframes, figures = self.model_analysis(
                            self.model_names,
                            self.training_dataframes_list,
                            self.validation_dataframes_list,
                            self.test_dataframes_list,
                            self.task_type,
                            ""
                        )
                         # Concatenate all dataframes into a single table for easy inspection
                        table = pd.concat(dataframes, ignore_index=True)
                        results = self.run(
                            regression_df=None, 
                            classification_df=table
                        )
                        
                        
                    else  : 
                        # Step 1: Perform regression analysis and obtain results (dataframes)
                        dataframes ,figures  = self.model_analysis_regression(
                            self.model_names,
                            self.training_dataframes_list,
                            self.validation_dataframes_list,
                            self.test_dataframes_list,
                            ""
                        )
                        # Concatenate all dataframes into a single table for easy inspection
                        table = pd.concat(dataframes, ignore_index=True)
                        results = self.run(
                            regression_df=table, 
                            classification_df=None
                        )
                    

                    # Step 2a: Update the dashboard with the inspection table (dataframe)
                    self.model_comparison_analysis = self.gui_instance.ncr_text(
                        self.model_comparison_analysis,
                        "Model Comparison",
                        "Model Comparison Table"
                    )

                
                    self.model_comparison_analysis = self.gui_instance.ncr_dataframe(
                        self.model_comparison_analysis,
                        "Model Comparison",
                        table,
                        ""
                    )


                    # Step 3: Add visual plots (figures) to the GUI dashboard
                    for model_name in self.model_names:

                        if self.task_type == "classification":
                          
                            # Update the dashboard with model-specific visual analysis
                            self.model_comparison_analysis = self.gui_instance.ncr_text(
                                self.model_comparison_analysis,
                                f"{model_name}",
                                f"{model_name} Analysis"
                            )

                            self.model_comparison_analysis = self.gui_instance.ncr_text(
                                self.model_comparison_analysis,
                                f"Target Distribution (Density Plot) {model_name}",
                                f"{model_name} : Target Distribution"
                            )

                            self.model_comparison_analysis = self.gui_instance.ncr_text(
                                self.model_comparison_analysis,
                                f"Percentage of Predictions per Score Bin {model_name}",
                                f"{model_name} : Percentage of Predictions per Score Bin"
                            )

                            self.model_comparison_analysis = self.gui_instance.ncr_text(
                                self.model_comparison_analysis,
                                f"Positive Prediction Rate per Score Bin {model_name}",
                                f"{model_name} : Positive Prediction Rate per Score Bin"
                            )
                        else  : 
                            
                                self.model_comparison_analysis = self.gui_instance.ncr_text(
                                self.model_comparison_analysis,
                                f"Prediction  Distribution for model : {model_name}",
                                f"Prediction  Distribution for model : {model_name}"
                            )


                    for model_name, dictionary in figures.items():
                        
                        for key, figures_list in dictionary.items():

                            for fig in figures_list:
                                
                                self.model_comparison_analysis = self.gui_instance.ncr_plot(

                                    self.model_comparison_analysis,
                                    fig,
                                    f'{key}',
                                    ""

                                )

           
                    # save the report 
                    report = self.generate_workflow_report(results)

                    path = create_file_path(os.path.join(self.base_dir, self.project_folder_name, self.folder_name_time), "Model_Evalutation_Report")
                    with open(f'{path}', 'w') as f:
                            f.write(report)
                    print(f"HTML report generated successfully at {path}")
                    
                    return  dataframes
                
            except Exception as e  : 
                print(f"an error  accured  in the function  : model_analysis_core_function : {e}")
             
    def non_time_based_analysis_for_model_analysis(self):

        """

            Performs non-time-based segmentation analysis for model comparison.
            
            This function analyzes model performance across different segments (subcategories)
            of data that are not time-based. It supports both classification and regression
            tasks, generating comparative tables and visualization plots for each segment.
            
            Parameters:
                None directly, but uses these instance attributes:
                    - self.non_time_analysis_active: Boolean flag indicating if analysis is enabled
                    - self.task_type: String indicating analysis type ("classification" or "regression")
                    - self.model_names: List of model names to analyze
                    - self.training_dataframes_list: List of training dataframes for each model
                    - self.validation_dataframes_list: List of validation dataframes for each model
                    - self.test_dataframes_list: List of test dataframes for each model
                    - self.segmentation_column: Column name to use for segmentation
                    - self.subcategory_threshold: Maximum number of subcategories allowed
                    - self.gui_instance: Instance of GUI handler for displaying results
            
            Returns:
                None. Results are displayed in the GUI through self.non_time_segment_analysis.
                
            Behavior:
                1. Checks if non-time-based analysis is enabled in settings
                2. Calls appropriate analysis method based on task type (classification/regression)
                3. Displays error message if number of subcategories exceeds threshold
                4. Generates and displays performance tables for each subcategory
                5. Creates and displays visualization plots for each model and subcategory

        """
        try   : 

            # Step 1: Check if non-time-based analysis is enabled
            if not self.non_time_analysis_active:
                self.non_time_segment_analysis = self.gui_instance.ncr_dataframe(
                    self.non_time_segment_analysis,
                    "Model Comparison",
                    pd.DataFrame({
                        "Alert": [
                            "Non-time segmentation analysis is currently disabled. "
                            "To activate it, update the 'user_settings.yaml' file by setting the "
                            "'activate' boolean flag under 'non_time_based_analysis_parameters' to 'true'."
                        ]
                    })
                )
                return
            
            # Check the task type
            track_task_type = True if self.task_type == "classification" else False
            
            # Step 2: Perform segmentation analysis based on task type
            if self.task_type == "classification":
                error_flag, dataframes, figures = self.non_time_based_segmentation_model_analysis(
                    self.model_names,
                    self.training_dataframes_list,
                    self.validation_dataframes_list,
                    self.test_dataframes_list,
                    self.segmentation_column,
                    self.task_type,
                    subcategory_threshold=self.subcategory_threshold
                )
            else:
                error_flag, dataframes, figures = self.non_time_based_segmentation_model_analysis_regression(
                    self.model_names,
                    self.training_dataframes_list,
                    self.validation_dataframes_list,
                    self.test_dataframes_list,
                    self.segmentation_column,
                    self.task_type,
                    subcategory_threshold=self.subcategory_threshold
                )

            # Step 3: Handle segmentation error if subcategories exceed the threshold
            if error_flag:
                self.non_time_segment_analysis = self.gui_instance.ncr_dataframe(
                    self.non_time_segment_analysis,
                    "Model Comparison",
                    pd.DataFrame({
                        "Alert": [
                            "The segmentation column contains too many subcategories, exceeding the defined threshold. "
                            "To address this issue, you can either: "
                            "1. Increase the subcategory threshold in the 'user_settings.yaml' file. "
                            "2. Map the subcategories into fewer classes to reduce complexity. "
                            "3. Select an alternative segmentation column with fewer subcategories."
                        ]
                    })
                )
                return

            # Step 4: Display segmentation results for each subcategory
            self.non_time_segment_analysis = self.gui_instance.ncr_text(
                self.non_time_segment_analysis,
                "Model Comparison",
                "Model Performance Evaluation",
                "h3"
            )

            # Display tables for each subcategory
            for key, value in dataframes.items():
                self.non_time_segment_analysis = self.gui_instance.ncr_text(
                    self.non_time_segment_analysis,
                    f"Subcategory {key} : Model Comparison Table",
                    f"Subcategory {key} : Model Comparison Table",
                    "h3"
                )

                table = pd.concat(value, ignore_index=True)
                self.non_time_segment_analysis = self.gui_instance.ncr_dataframe(
                    self.non_time_segment_analysis,
                    f"Subcategory {key} : Model Comparison Table",
                    table,
                    ""
                )

            # Set up appropriate label list based on task type
            if self.task_type == "classification":
                label_list = [
                    "Target Distribution (Density Plot)",
                    "Positive Prediction Rate per Score Bin",
                    "Percentage of Predictions per Score Bin"
                ]
            else:
                label_list = [
                    "Prediction Distribution for model :"
                ]
            
            # Add section headers for each model
            for model in self.model_names:
                self.non_time_segment_analysis = self.gui_instance.ncr_text(
                    self.non_time_segment_analysis,
                    f"{model}",
                    f"{model} Analysis"
                )
            
            # Step 5: Add plots for each subcategory
            label_index = 0  # Counter for cycling through label list
            
            for subcategory, value_list in figures.items():
                for dict_item in value_list:
                    for model_name, image_dict in dict_item.items():
                        for plot_type_name, image in image_dict.items():
                            for plot in image:
                                # Create appropriate title based on task type
                                if self.task_type == "classification":
                                    title = f"{model_name}/Subcategory : {subcategory} {label_list[label_index % len(label_list)]}"
                                else:
                                    title = f"{plot_type_name} {subcategory}"
                                
                                # Add text section for the plot
                                self.non_time_segment_analysis = self.gui_instance.ncr_text(
                                    self.non_time_segment_analysis,
                                    f"{plot_type_name} {subcategory}",
                                    title
                                )
                                
                                # Add the plot itself
                                self.non_time_segment_analysis = self.gui_instance.ncr_plot(
                                    self.non_time_segment_analysis,
                                    plot,
                                    f'{plot_type_name} {subcategory}',
                                    ""
                                )
                                label_index += 1  # Increment counter after each iteration
        
        except Exception as e   :
            print(f"Error in non_time_based_analysis_for_model_analysis function accured : {e}")
                        
    def time_based_segmentation_for_model_analysis(self):

        """

            Performs time-based segmentation analysis for model evaluation.

            This function analyzes model performance over different time intervals to detect 
            temporal patterns, trends, or potential drift in model predictions. It processes 
            the analysis results and updates the dashboard accordingly.

            Steps:
            1. Checks if time-based segmentation analysis is enabled.
            2. If enabled, performs the analysis using specified parameters.
            3. Updates the dashboard with segmented analysis results for each model.
            4. If the analysis is disabled, displays an alert message on the dashboard.
            5. Handles exceptions and logs errors if encountered.

            Raises:
                Exception: If any error occurs during the execution of the function.

        """
        try  : 

                # Step 1: Check if time-based segmentation analysis is enabled
                if self.time_analysis_active:


                    # Step 2: Perform time-based segmentation analysis using the specified parameter

                    result_dict = self.time_based_segmentation_model_analysis(
                        model_names=self.model_names,  # List of models to be evaluated
                        train_dataframes_list=self.training_dataframes_list,  # List of training datasets
                        validation_dataframes_list=self.validation_dataframes_list,  # List of validation datasets
                        test_dataframes_list=self.test_dataframes_list,  # List of test datasets
                        timestamp_column=self.timestamp_column,  # Column used for segmentation
                        number_of_months=self.time_analysis_months,  # Maximum allowed subcategories based on time intervals
                        analysis_type=self.task_type  # Type of analysis (e.g., classification, regression)
                    )

                    # Step 3: Process the results for each model and update the dashboard
                    for model in self.model_names:
                        # Add a header for the current model's analysis to the dashboard
                        self.time_based_segment_analysis = self.gui_instance.ncr_text(

                            self.time_based_segment_analysis,
                            f"{model}: Time Based Model Analysis",  # Section identifier # 
                            f"{model}: Time Based Model Analysis",  # Header for the model's analysis
                            "h2"  # Header level (secondary header)

                        )

                        # Step 4: Display segmented data tables for each model
                        for table in result_dict[model]:
                            # Update the dashboard with the segmented data for the current model
                            if  not table.empty : 

                                self.time_based_segment_analysis = self.gui_instance.ncr_dataframe(
                                    self.time_based_segment_analysis,  # Path to the dashboard output file
                                    f"{model}: Time Based Model Analysis",  # Section identifier
                                    table,  # DataFrame containing the results for this model
                                    ""  # Optional table name (left empty here)
                                )
                        
                else:

                    for model in self.model_names:

                        # Step 5: If time-based segmentation analysis is not active, display an alert on the dashboard
                        alert_message = (
                            "Time-based segmentation analysis is currently disabled. "
                            "To activate it, update the 'user_settings.yaml' file by setting the "
                            "'activate' boolean flag under 'time_based_analysis_parameters' to 'true'."
                        )

                        # Update the dashboard with the alert message
                        self.time_based_segment_analysis = self.gui_instance.ncr_dataframe(
                            self.time_based_segment_analysis,
                            f"{model}: Time Based Model Analysis",  # Section identifier
                            pd.DataFrame({"Alert": [alert_message]})  # Display the alert message in the dashboard
                        )
        except Exception as e  : 
            print(f"An error occurred in time_based_segmentation_for_model_analysis: {e}")


    def run_core_functions(self):

        """

            Executes the core functions for model analysis, time-based and non-time-based segmentation, 
            and generates the corresponding reports for each analysis type. This method coordinates the 
            workflow by performing the following steps:

            1. Running core model analysis for comparison, segmentation, and model evaluation.
            2. Generating a project overview with key information.
            3. Mapping each report to its corresponding content (analysis results).
            4. Writing the generated report content to HTML files and saving them to the specified paths.

            The reports are saved as HTML files, and appropriate messages are logged to indicate success or failure.

            Steps:
            1. **Core Model Analysis**: Runs the functions that perform core model analysis, non-time-based segmentation, 
            and time-based segmentation analysis.
            2. **Generate Project Overview**: Creates a project overview that includes the start and end times, along with 
            a title for the section.
            3. **Map Reports to Content**: Maps the analysis results (model comparison, segmentation analyses) to their 
            respective report generation functions.
            4. **Generate and Save Reports**: Loops through the predefined report paths and generates HTML reports with 
            the corresponding content. The reports are saved to disk, and success or failure is logged.

            Returns:
                None. The function executes the core analysis tasks, generates reports, and saves them as HTML files.

        """

        
        # Step 1: Track the start time of the process for performance tracking

        start_time = datetime.now()

                                                           # Perform the core model analysis for comparison and segmentation
                                                           # This includes model analysis, non-time-based segmentation, and time-based segmentation
        self.model_analysis_core_function()                # Core model analysis function
        self.non_time_based_analysis_for_model_analysis()  # Non-time-based segmentation analysis
        self.time_based_segmentation_for_model_analysis()  # Time-based segmentation analysis

        # Track the end time after all analyses are completed

        end_time = datetime.now()

        # Step 2: Generate a project overview, including the start and end times of the analysis process

        self.generate_project_overview(
            start_time,  # Start time of the process
            end_time,    # End time of the process
            "Project overview and key information",  # Unique identifier for the project overview report
            "Project Overview and Key Information"  # Title for the project overview section
        )

        # Step 3: Map each report type to its corresponding content that has been generated

        report_content_map = {
            "model_comparison_and_analysis": self.model_comparison_analysis,  # Content for model comparison report
            "non_time_based_segmentation": self.non_time_segment_analysis,  # Content for non-time-based segmentation report
            "time_based_segmentation": self.time_based_segment_analysis,    # Content for time-based segmentation report
        }

        # Step 4: Generate and save the reports by writing the content to HTML files

        for report_name, report_path in self.report_paths.items():
            # Check if the current report name has corresponding content to write
            if report_name in report_content_map:
                try:
                    # Open the report file in write mode and save the content as HTML
                    with open(report_path, "w") as html_file:
                        html_file.write(report_content_map[report_name])
                    print(f"HTML file successfully written to: {report_path}")  # Success message
                except IOError as e:
                    # If there is an error writing the file, log the error message
                    print(f"Error writing to {report_path}: {e}")
            else:
                # If no content is found for the report, print a warning
                print(f"Warning: No content found for report: {report_name}")






