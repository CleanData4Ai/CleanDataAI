"""
EdaCoreFunctions Class

Purpose:
The `EdaCoreFunctions` class serves as the core component of the Exploratory Data Analysis (EDA) framework, providing the foundational 
functions and utilities required for performing a wide range of EDA tasks. It acts as the backbone of the framework, equipping higher-level 
classes with essential methods for data analysis and report generation. 

Key Responsibilities:
- **Data Management**: Accepts and processes the dataset (pandas DataFrame), with options for specifying target and timestamp columns.
- **Configuration Loading**: Reads and initializes user-defined settings from a YAML configuration file to ensure flexibility and customization of the EDA workflow.
- **Report Generation**: Creates and manages structured HTML output files for storing analysis results, ensuring easy access to insights.
- **Warning Suppression**: Handles specific warnings (e.g., FutureWarnings, pandas SettingWithCopyWarning) to maintain clean and uninterrupted logs during analysis.
- **Inheritance**: Extends the `EdaHelperFunctions` class to inherit shared utilities, streamlining the implementation of advanced EDA functionalities.

Features:
1. **Customizable Output**: Allows users to define project-specific folder names and target columns for analysis.
2. **Time Series Support**: Includes optional timestamp column functionality for handling time-based analyses.
3. **Modular Design**: Designed for integration with higher-level classes, like `EdaApp`, enabling a layered and extendable framework architecture.
4. **HTML Report Management**: Prepares HTML templates and stylesheets for storing results of various analyses in an organized and readable format.

Usage:
- The `EdaCoreFunctions` class is not typically instantiated directly by end-users. Instead, it is utilized as a base class, providing core methods and properties for derived classes.
- It is critical for performing foundational tasks like loading settings, managing outputs, and applying EDA techniques at the framework's core.

In essence, this class forms the heart of the EDA framework, ensuring that all key functions and configurations are seamlessly managed and accessible to other components.
"""

# -----------------------------------------
# Imports for EdaCoreFunctions
# -----------------------------------------

import os
import warnings
from datetime import datetime
from pathlib import Path

# Print current working directory to help with debugging file path issues
print(f"Current working directory core functions : {os.getcwd()}")

# Local imports from the EDA framework modules
# Use the version that matches your current working directory setup

# When working in the same directory
# from data_analysis_exploration.pandas_eda_lib.eda_utilities.eda_helper_functions import EdaHelperFunctions
# from src.graphical_data_visualizations.visualisation.visualisation_utils_for_eda_bib import VisualisationUtils
# from src.graphical_data_visualizations.visualisation.css import css_code, css_code_with_black_side_bar
# from src.graphical_data_visualizations.visualisation.template import base_template
# from src.file_operations_utils.file_operations import *

# When working from the library/module structure
from eda_lib.data_analysis_exploration.pandas_eda_lib.eda_utilities.eda_helper_functions import EdaHelperFunctions
from eda_lib.src.graphical_data_visualizations.visualisation.visualisation_utils_for_eda_bib import VisualisationUtils
from eda_lib.src.graphical_data_visualizations.visualisation.css import css_code, css_code_with_black_side_bar
from eda_lib.src.graphical_data_visualizations.visualisation.template import base_template
from eda_lib.src.file_operations_utils.file_operations import *

# Optional: Feature selection agent import (currently commented)
# from src.ai_agents.feature_selection_agent import FeatureSelectionAgent

# Additional libraries
import matplotlib.pyplot as plt
import pandas as pd


class EdaCoreFunctions(EdaHelperFunctions):

    def __init__(self,
                 data: pd.DataFrame,
                 targetColumn: str,
                 base_path : str , 
                 project_root_name : str, 
                 yaml_user_settings_path :str ,
                 output_path : str ="data_analysis_exploration/output_folder" , 
                 timestampColumn: str = None,
                 project_folder_name: str = "default project name", 
                 project_name : str = "Default Project Name"                 ):

        

        # Suppress specific warnings to maintain cleaner logs and prevent interruption
        warnings.simplefilter(action='ignore', category=FutureWarning)
        warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

        super().__init__()  # Initialize base class components
        self.data = data  # Store the provided dataset
        self.project_folder_name = project_folder_name  # Define the project name for report paths
        self.target = targetColumn  # Define the target column for analysis
        self.timestampColumn = timestampColumn  # Define the timestamp column for time-based analysis
        self.base_path = base_path
        self.project_root_name = project_root_name
        self.output_path = output_path
        self.project_name = project_name
        self.yaml_user_settings_path = yaml_user_settings_path 
        # Load and initialize configuration settings from YAML files
        self._initialize_settings()

        # Generate initial HTML structure for report outputs across analysis types
        self.generate_html_files(base_template, css_code_with_black_side_bar)

    def _initialize_settings(self):

        """

            Initializes and configures the user interface, feature lists, and settings parameters for various
            analytical processes by loading settings files and applying conditional checks on data.

            Steps:
            ------
            1. **GUI Setup**: Instantiates `VisualisationUtils` for visual elements in the dashboard.
            3. **UI Configuration Extraction**: Extracts essential sidebar display configurations, including title and logo.
            4. **Feature Lists for Analysis**: Loads features for specific types of analysis (numerical, categorical, KDE,
               variance analysis) by calling helper functions to conditionally retrieve them as required.
            5. **Time-Series Analysis Parameters**: Loads time-series analysis parameters (e.g., `num_months`, `num_weeks`,
               `top_n`) from user-defined settings, setting default values where not specified.
            6. **Feature Inclusion for Analysis**: Configures specific numerical and categorical features to include
               in the time-series analysis, enhancing customization.
            7. **Output Directory Configuration**: Sets up a base directory path for storing generated outputs and creates
               a timestamped folder structure for organized storage and easier traceability.
            8. **EDA Function Names Retrieval**: Retrieves a list of function names for exploratory data analysis (EDA)
               from the user settings, facilitating their sequential execution.
            9. **Correlation Analysis Parameters**: Loads correlation analysis parameters, such as variance threshold
               and correlation metric, for inclusion in the final output.

            Returns:
            --------
            None

        """

        try  : 

            self.data = self.data.loc[:, self.data.nunique() != len(self.data)]

            self.gui_instance = VisualisationUtils()
           
            # Load user-defined and YData configuration settings from respective YAML files
            self.user_settings = load_yaml( os.path.join(self.base_path, self.project_root_name, self.yaml_user_settings_path))
            
            #use_ai 
            #self.use_llm = bool(self.user_settings["use_feature_selection_agent"])

            # Retrieve and store UI display settings, including sidebar title and logo
            self.sidebar_title = self.user_settings["display_settings"]["side_bar_title"]
            self.side_bar_logo = self.user_settings["display_settings"]["side_bar_logo"]

            # Extract display settings and general UI configurations
            display_settings = self.user_settings.get("display_settings", {})
            self.sidebar_title = display_settings.get("side_bar_title")
            self.side_bar_logo = display_settings.get("side_bar_logo")

            # Extract analysis feature lists conditionally, as they're needed
            self.features_for_numerical_plot = (None if self._get_features(self.user_settings, "numericalFeatureDistribution")=="None" else self._get_features(self.user_settings, "numericalFeatureDistribution")=="None")
            self.features_for_categorical_plot = (None if  self._get_features(self.user_settings, "categoricalFeatureDistribution") =="None" else  self._get_features(self.user_settings, "categoricalFeatureDistribution"))
            self.categorcial_feature_threshold = int(self.user_settings["functions"]["analysisParameters"]["categoricalFeatureDistribution"]["categorical_feature_threshold"])
            self.features_kde_plot = ( None if  self._get_features(self.user_settings, "kernelDensityEstimatePlots") =="None" else  self._get_features(self.user_settings, "kernelDensityEstimatePlots"))
            self.features_variance_analysis = (None if  self._get_features(self.user_settings, "varianceAnalysis")=="None" else self._get_features(self.user_settings, "varianceAnalysis"))

            # Time-series parameters
            time_series_params = self.user_settings["functions"]["analysisParameters"].get("time_series_analysis", {})
            self.num_months = int(time_series_params.get("num_months", 0))
            self.num_weeks = int(time_series_params.get("num_weeks", 0))
            self.top_n = int(time_series_params.get("top_n_subcategories", 0))

            # Inclusion settings for numerical and categorical features
            self.numerical_features_to_include = (
                None if time_series_params.get("numerical_featuresToInclude") == "None"
                else time_series_params.get("numerical_featuresToInclude")
            )
            self.categorical_features_to_include = (
                None if time_series_params.get("categorcial_featuresToInclude") == "None"
                else time_series_params.get("categorcial_featuresToInclude")
            )
            # Set up base directory for output and generate dynamic folder names
            self.base_dir = os.path.join(self.base_path, self.project_root_name, self.output_path)
            self.folder_name_time = f"{self.project_folder_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
            self.report_paths = self._generate_report_paths(self.folder_name_time)

            # EDA function names
            self.function_names = list(self.user_settings['functions'].get('defaultExploratoryDataAnalysis', {}).keys())

            # Correlation parameters
            correlation_params = self.user_settings['functions']["analysisParameters"].get("Correlation_Analysis", {})
            self.apply_variance  = bool(correlation_params.get('apply_variance_threshold'))
            self.variance_threshold = float(correlation_params.get('variance_threshold'))
            self.correlation_metric = correlation_params.get('correlation_metric')

            
            #if self.use_llm:
                #self.response =  FeatureSelectionAgent(self.data, self.target).run_workflow()


        except  Exception as e:
            print(f"an error occurred  in the function _initialize_settings   : ,{e}")

    def _generate_report_paths(self, folder_name_time):

        """
        Creates paths for storing various types of analysis reports, grouping them by purpose,
        and dynamically generates paths for storing outputs in a timestamped directory.

        Parameters:
        -----------
        folder_name_time : str
            The folder name with a timestamp, created based on the project name and current date-time
            for organized storage.

        Returns:
        --------
        dict
            A dictionary containing paths for different report categories, enabling easy retrieval
            and organized storage of analysis outputs.
        """
        
        try  : 

            base_path = os.path.join(self.base_dir, self.project_folder_name, folder_name_time)
            return {

                "arco_eda_framework": create_file_path(base_path,
                                                            "Exploratory_Data_Analysis/Uni_Multivariate_Analysis_Report"),
                "target_vs_time": create_file_path(base_path, "Time_Series_Analysis/Target_vs_Time_Report"),
                "numerical_vs_time": create_file_path(base_path, "Time_Series_Analysis/Numerical_vs_Time_Report"),
                "categorical_vs_time": create_file_path(base_path,
                                                            "Time_Series_Analysis/Categorical_vs_Time_Report")
                
            }

        except Exception as e  : 

             print(f"an error occurred  in the function _generate_report_paths   : ,{e}")

    def generate_html_files(self, base_template, css_code):

        """
            Configures HTML templates for various analytical views by setting shared arguments
            and defining content structures, which are generated on-demand to optimize memory usage.

            Parameters:
            -----------
            base_template : str
                The base HTML template used for the structure of each HTML file.
            css_code : str
                CSS styling for maintaining visual consistency across HTML reports.

            Notes:
            ------
            - **HTML Structure**: A dictionary defines content for each analysis type, referencing the function names
              from `self.function_names` and predefining structures for different types of analysis.
            - **Lazy Loading**: HTML generation is delayed until accessed, reducing memory consumption
              by using lambda functions to create content only when needed.

            Returns:
            --------
            None
        """
        
        try  : 


            # Save common arguments to reduce redundancy
            common_args = {
                "base_template": base_template,
                "css_code": css_code,
                "title": self.sidebar_title,
                "logo_url": self.side_bar_logo
            }


            # Define the contents to avoid duplicating function calls
            html_structure = {
                'uni_variate_eda_html': [
                    (self.function_names[0], []),
                    (self.function_names[1], []),
                    (self.function_names[2], []),
                    (self.function_names[3], [
                        "Feature Types Summary Table", "Features Non Null Count Summary Table",
                        "Non Null Count Pro Feature Bar Plot"
                    ]),
                    (self.function_names[4], []),
                    (self.function_names[5], []),
                    (self.function_names[6], []),
                    (self.function_names[7], [
                        "Statistical Summary Table", "Mean Values Per Numerical Feature",
                        "Distribution Analysis of Numerical Features via Box Plot",
                        "Variance Summary Table", "Variance Analysis via Feature Plot"
                    ]),
                    (self.function_names[8], []),
                    (self.function_names[9], []),
                    (self.function_names[10], []),
                    (self.function_names[11], [
                        "Correlation Analysis of Numeric Features Matrix",
                        "Correlation Analysis of Numeric Features Heat Map"
                    ]),
                    (self.function_names[12], [
                        "Correlation Analysis of Categorical Features Matrix",
                        "Correlation Analysis of Categorical Features Heat Map"
                    ])
                ],
                'target_time_series_html': [
                    (self.function_names[0], []),
                    (self.function_names[13], []),
                    (self.function_names[14], [])
                ],
                'numerical_time_series_html': [
                    (self.function_names[0], []),
                    (self.function_names[15], []),
                    (self.function_names[16], [])
                ],
                'categorical_time_series_html': [
                    (self.function_names[0], []),
                    (self.function_names[17], []),
                    (self.function_names[18], [])
                ]
            }

            # Function to filter the html_structure based on analysis execution
            def filter_html_structure(html_structure):
                filtered_structure = {}

                for key, options in html_structure.items():
                    filtered_options = [
                        (name, content) for name, content in options
                        if self.execute_analysis(self.user_settings, name)
                    ]
                    filtered_structure[key] = filtered_options

                return filtered_structure

            # Apply the filtering function
            html_structure = filter_html_structure(html_structure)

            # Generate and store HTMLs only when accessed
            def generate_html(side_bar_menu_options):
                return self.gui_instance.generate_html(side_bar_menu_options ,  common_args["base_template"]  ,common_args["css_code"] , common_args["title"]  , common_args["logo_url"]  )

            # Lazy generation with properties

            self.uni_variate_eda_html = generate_html (html_structure['uni_variate_eda_html'])
            self.target_time_series_html = generate_html(html_structure['target_time_series_html'])
            self.numerical_time_series_html = generate_html(html_structure['numerical_time_series_html'])
            self.categorical_time_series_html =generate_html(html_structure['categorical_time_series_html'])

        except Exception as e  :

            print(f"an error occurred  in the function generate_html_files   : ,{e}")

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

            self.uni_variate_eda_html = self.gui_instance.ncr_dashboard_dataframe(
                self.uni_variate_eda_html,  # Output file path for the dashboard
                subsection_id,  # Unique subsection ID for the dashboard entry
                df,  # DataFrame to be added to the dashboard
                dataframe_name=title  # Title for the DataFrame in the dashboard
            )
            self.target_time_series_html = self.gui_instance.ncr_dashboard_dataframe(
                self.target_time_series_html,  # Output file path for the dashboard
                subsection_id,  # Unique subsection ID for the dashboard entry
                df,  # DataFrame to be added to the dashboard
                dataframe_name=title  # Title for the DataFrame in the dashboard
            )

            self.numerical_time_series_html = self.gui_instance.ncr_dashboard_dataframe(
                self.numerical_time_series_html,  # Output file path for the dashboard
                subsection_id,  # Unique subsection ID for the dashboard entry
                df,  # DataFrame to be added to the dashboard
                dataframe_name=title  # Title for the DataFrame in the dashboard
            )

            self.categorical_time_series_html = self.gui_instance.ncr_dashboard_dataframe(
                self.categorical_time_series_html,  # Output file path for the dashboard
                subsection_id,  # Unique subsection ID for the dashboard entry
                df,  # DataFrame to be added to the dashboard
                dataframe_name=title  # Title for the DataFrame in the dashboard
            )

            del df

        except Exception as e:

            print(f"an error  accured  in the function  generate_project_overview  : {e}")

    def generate_data_overview(self):
        """
        Generates an overview of the dataset if the corresponding feature is enabled in user settings.

        This method performs the following steps:
        1. **Feature Check**: It verifies if the dataset overview feature is enabled by calling the `execute_analysis` method with user settings and the corresponding function name.
        2. **Overview Table Creation**: If the feature is enabled, it generates an overview DataFrame of the dataset using the `overview_table` method.
        3. **Integration into HTML Dashboard**: The generated overview DataFrame is integrated into the HTML dashboard by invoking the `ncr_dashboard_dataframe` method of the GUI instance. It specifies the output file path, a unique identifier for the overview, and a display name for the DataFrame within the dashboard.

        Exception Handling:
        --------------------
        Any exceptions encountered during the execution of the method are caught and logged with a descriptive message for debugging purposes. This ensures that the function fails gracefully, providing insight into issues without disrupting the overall application flow.

        Returns:
        --------
        None
        """
        try:
            if self.execute_analysis(self.user_settings, self.function_names[1]):
                overview_df = self.overview_table(self.data)

                self.uni_variate_eda_html = self.gui_instance.ncr_dashboard_dataframe(
                    self.uni_variate_eda_html,  # Path to the dashboard output file
                    format_string(self.function_names[1]),  # Unique identifier for the overview table
                    overview_df,  # DataFrame containing the overview
                    dataframe_name='datasetsummaryoverview'  # Name for the DataFrame in the dashboard
                )
                del overview_df

        except Exception as e:

            print(f"An error occurred in the function generate_data_overview  {e}")

    def perform_initial_data_inspection(self):
        """
        Conducts an initial inspection of the dataset to assess its structure and quality.

        This method executes the following steps:
        1. **Feature Check**: It verifies whether the data inspection feature is enabled by calling the `execute_analysis` method with the current user settings and the specific function name associated with data inspection.
        2. **Data Inspection**: If the feature is enabled, it performs an inspection of the dataset using the `observe_data` method. This generates a summary DataFrame containing insights about the data's structure, types, and any anomalies.
        3. **Integration into HTML Dashboard**: The resulting inspection DataFrame is then integrated into the HTML dashboard by invoking the `ncr_dataframe` method of the GUI instance. It specifies the output file path, a unique identifier for the inspection table, and allows for an optional name for display purposes.

        Exception Handling:
        --------------------
        The method includes a general exception handling mechanism that captures any errors encountered during execution. If an error occurs, it prints a descriptive message to facilitate troubleshooting, ensuring that the function's failure does not disrupt the application's overall performance.

        Returns:
        --------
        None
        """
        try:
            if self.execute_analysis(self.user_settings,
                                     self.function_names[2]):  # Check if the inspection feature is enabled
                # Generate and display the information table for the dataset
                data_inspection = self.observe_data(self.data)

                self.uni_variate_eda_html = self.gui_instance.ncr_dataframe(
                    self.uni_variate_eda_html,  # Path to the dashboard output file
                    format_string(self.function_names[2]),  # Unique identifier for the inspection table
                    data_inspection,  # DataFrame containing the inspection results
                    ""  # Optional name for the DataFrame in the dashboard
                )
                del data_inspection

        except Exception as e:

            print(f"An error occurred in the function perform_initial_data_inspection {e}")

    def create_feature_info_table(self):

        """
        Generates and processes a feature information table if enabled in the user settings.
        This method displays the table, integrates it into the HTML dashboard, saves it as a pickle file,
        creates a bar plot for non-null counts, and saves the plot as a PNG file.

        Process:
        -----------
        1. **Check Feature Activation**: Verifies if the feature information table generation is enabled in the user settings.
        2. **Generate Feature Information Table**: Creates an information table using the `info_table` function, focusing on non-null counts and data types, and displays the relevant columns.
        3. **Integrate Table into HTML Dashboard**: Adds the feature information table to the HTML dashboard using the `self.gui_instance.ncr_dataframe` function.
        4. **Create Bar Plot**: Generates a bar plot visualizing non-null counts per feature using the `self.bar_plot` function. Configures plot title, axis labels, and other settings.
        5. **Display Bar Plot**: Shows the bar plot using `plt.show`.
        6. **Add Bar Plot to HTML Dashboard**: Integrates the bar plot into the HTML dashboard with `self.gui_instance.ncr_plot`.

        Exception Handling:
        --------------------
        Catches and prints any exceptions that occur during the execution of the method, providing an error message for debugging purposes.
        """
        try:
            # Verify if the feature information table generation is enabled in user settings
            if self.execute_analysis(self.user_settings, self.function_names[3]):
                # Generate DataFrame for numeric and categorical columns
                feature_classification_dataframe = self.numeric_and_categorical_columns(self.data)

                # Integrate feature classification into the dashboard
                self.uni_variate_eda_html = self.gui_instance.ncr_dataframe(
                    self.uni_variate_eda_html,  # Path to save the dashboard
                    "featuretypessummarytable",  # Identifier for the feature types DataFrame
                    feature_classification_dataframe  # DataFrame with feature classifications
                )

                # Generate and filter the information table just once
                info_df = self.info_table(self.data, ascending=False)[["Feature Name", "Non Null Count"]]

                # Integrate the information table into the HTML dashboard
                self.uni_variate_eda_html = self.gui_instance.ncr_dataframe(
                    self.uni_variate_eda_html,  # Path to the dashboard output file
                    "featuresnonnullcountsummarytable",  # Unique identifier for the information table
                    info_df  # DataFrame containing the feature information
                )

                # Generate bar plot using the filtered information DataFrame
                bar_plot = self.bar_plot(
                    info_df["Feature Name"],  # X-axis data: Feature names
                    info_df["Non Null Count"],  # Y-axis data: Non-null counts
                    "Non Null Count Per Feature",  # Title of the plot
                    "Features",  # X-axis label
                    "Non Null Count",  # Y-axis label
                    annotation=False,  # Disable plot annotations
                    fig_size=(20, 12),  # Size of the figure
                    threshold=self.data.shape[0],  # Threshold for non-null counts
                    annotation_rotation=80  # Rotation angle for annotations
                )  # Figure containing the bar plot

                # Integrate the plot into the HTML dashboard
                self.uni_variate_eda_html = self.gui_instance.ncr_plot(
                    self.uni_variate_eda_html,  # Path to the dashboard output file
                    bar_plot,  # The bar plot figure
                    "nonnullcountprofeaturebarplot",  # Unique identifier for the plot
                    ""  # Optional name for the plot in the dashboard
                )

                # Close the figure to free up memory
                plt.close(bar_plot)

                # Optional cleanup for DataFrames
                del feature_classification_dataframe, info_df , bar_plot

        except Exception as e:
            # Print the error message for troubleshooting
            print(f"An error occurred create_feature_info_table  : {e}")

    def create_unique_values_table(self):
        """
        Generates and integrates a unique values table for each feature if enabled in user settings.
        This function optimizes memory usage by reducing redundant operations and deferring plot generation.
        """
        try:
            # Check if unique values summary feature is enabled
            if not self.execute_analysis(self.user_settings, self.function_names[4]):
                return

            # Check for categorical features
            categorical_data = self.data.select_dtypes(include="object")

            if categorical_data.shape[1] == 0:

                self.uni_variate_eda_html = self.gui_instance.ncr_dataframe(
                    self.uni_variate_eda_html,
                    format_string(self.function_names[4]),
                    pd.DataFrame({
                        "Alert": [
                            "Unique values table is not generated because no categorical features were found. "
                            "Unique values are only relevant for categorical features."
                        ]
                    })
                )
                return

            # Generate the unique values DataFrame once
            unique_counts_df = self.show_column_unique_counts(categorical_data)

            # Save table to HTML dashboard
            self.uni_variate_eda_html = self.gui_instance.ncr_dataframe(
                self.uni_variate_eda_html,
                format_string(self.function_names[4]),
                unique_counts_df
            )

            # Generate bar plot for unique values
            bar_plot = self.bar_plot(
                unique_counts_df["Feature"],
                unique_counts_df["Unique Values"],
                "Unique Values Per Feature",
                "Features",
                "Unique Values",
                rotation=90,
                annotation=True,
                log_scale=True,
                fig_size=(20, 12),
                annotation_rotation=80
            )

            # Integrate plot to HTML dashboard
            self.uni_variate_eda_html = self.gui_instance.ncr_plot(
                self.uni_variate_eda_html,
                bar_plot,
                format_string(self.function_names[4]),
                ""
            )

            # Close figure to free up memory
            plt.close(bar_plot)

            # Clean up large variables to free memory
            del unique_counts_df, bar_plot, categorical_data

        except Exception as e:
            print(f"An error occurred in the function create_unique_values_table  {e}")

    def create_missing_values_table(self):
        """
        Generates and processes a table of missing values for each feature if enabled in the user settings.
        This method displays the table, integrates it into the HTML dashboard, and saves it as a pickle file.
        Additionally, it creates bar plots illustrating missing values per feature, both before and after applying a log scale.

        Process:
        --------
        1. **Check Feature Activation**: Verifies if the generation of the missing values table is enabled in user settings.
        2. **Generate Missing Values Table**: Creates a DataFrame summarizing missing value counts for each column using the `show_column_missing_counts` function.
        3. **Integrate Table into HTML Dashboard**: Adds the missing values table to the HTML dashboard.
        4. **Create and Save Bar Plots**: Generates bar plots for missing values with log scale, and integrates them into the HTML dashboard.
        5. **Memory Management**: Deletes and closes plot figures after use to free up memory.

        Exception Handling:
        -------------------
        Catches and prints any exceptions that occur during execution for troubleshooting.
        """
        try:
            # Check if missing values table feature is enabled in user settings
            if self.execute_analysis(self.user_settings, self.function_names[5]):
                # Calculate missing values once and store it in a variable to avoid repetition
                missing_values_df = self.show_column_missing_counts(self.data)

                # Integrate the missing values table into the HTML dashboard
                self.uni_variate_eda_html = self.gui_instance.ncr_dataframe(
                    self.uni_variate_eda_html,
                    format_string(self.function_names[5]),
                    missing_values_df
                )

                # Generate bar plot for missing values with log scale applied
                fig_missing_values = self.bar_plot(
                    x=missing_values_df["Feature"],  # X-axis: Feature names
                    y=missing_values_df["Missing Values"],  # Y-axis: Missing value counts
                    title="Missing Values Per Feature",  # Plot title
                    xlabel="Features",  # X-axis label
                    ylabel="Missing Value Count",  # Y-axis label
                    fig_size=(20, 12),  # Plot size
                    log_scale=True,  # Apply log scale to Y-axis
                    annotation=True,  # Enable annotations
                    annotation_rotation=80  # Rotation angle for annotations
                )

                # Integrate the plot into the HTML dashboard
                self.uni_variate_eda_html = self.gui_instance.ncr_plot(
                    self.uni_variate_eda_html,
                    fig_missing_values,
                    format_string(self.function_names[5]),
                    ""
                )

                # Close and delete the figure to free up memory
                plt.close(fig_missing_values)
                del fig_missing_values  # Ensures that memory is fully released

                # Clear the missing values DataFrame from memory
                del missing_values_df

        except Exception as e:
            # Print the error message for troubleshooting
            print(f"An error occurred in the function create_missing_values_table  {e}")

    def analyze_zero_values(self):
        """
        Analyzes zero values in the dataset and performs processing tasks if enabled in the user settings.

        - Checks if zero values analysis is enabled.
        - Generates a table displaying zero counts for each feature.
        - Integrates the table and a bar plot of zero values into the HTML dashboard.

        Memory Management:
        -------------------
        Optimizations to reduce memory usage include caching results, avoiding redundant DataFrame creations,
        and clearing objects with `del` and `plt.close()` after use.

        Exception Handling:
        -------------------
        Catches and prints any exceptions for debugging.
        """
        try:
            # Check if the zero values analysis feature is enabled in user settings
            if self.execute_analysis(self.user_settings, self.function_names[6]):

                # Calculate zero values once and store it to avoid multiple calls
                zero_values_df = self.show_zeros_table(self.data)

                # Only proceed if there are any features with zero counts
                if zero_values_df.shape[0] > 0:

                    # Integrate the zero counts table into the HTML dashboard
                    self.uni_variate_eda_html = self.gui_instance.ncr_dataframe(
                        self.uni_variate_eda_html,
                        format_string(self.function_names[6]),
                        zero_values_df
                    )

                    # Generate and integrate the bar plot for zero counts per feature
                    fig_zero_values = self.bar_plot(
                        x=zero_values_df["feature"],  # X-axis: Feature names
                        y=zero_values_df["n_zeros"],  # Y-axis: Number of zeros
                        title="Zeros Per Feature",  # Plot title
                        xlabel="Features",  # X-axis label
                        ylabel="Zeros Count",  # Y-axis label
                        fig_size=(20, 12),  # Plot size
                        log_scale=True,  # Apply log scale to Y-axis
                        threshold=self.data.shape[0]  # Threshold for the plot
                    )

                    # Integrate the plot into the HTML dashboard
                    self.uni_variate_eda_html = self.gui_instance.ncr_plot(
                        self.uni_variate_eda_html,
                        fig_zero_values,
                        format_string(self.function_names[6]),
                        ""
                    )

                    # Close and delete the figure to free up memory
                    plt.close(fig_zero_values)
                    del fig_zero_values

                else:
                    # If no zero values are found, add an alert message to the dashboard
                    self.uni_variate_eda_html = self.gui_instance.ncr_dataframe(
                        self.uni_variate_eda_html,
                        format_string(self.function_names[6]),
                        pd.DataFrame({
                            "Alert": [
                                "No features with zero values found in the dataset. No zero value summary is available."
                            ]
                        })
                    )

                # Clear the zero values DataFrame from memory
                del zero_values_df

        except Exception as e:
            # Print the error message for troubleshooting
            print(f"An error occurred in the function analyze_zero_values  {e}")

    def summarize_numerical_features(self):
        """
        Summarizes numerical features, integrating summary statistics and visualizations into the HTML dashboard.

        Memory Optimization:
        --------------------
        - Caches results of calculations to avoid redundancy.
        - Clears memory with `del` and `plt.close()` after creating and integrating visualizations.

        Exception Handling:
        -------------------
        Catches and reports any exceptions encountered during execution.
        """
        try:
            # Check if summary statistics generation is enabled in user settings
            if self.execute_analysis(self.user_settings, self.function_names[7]):

                # Filter for numerical features
                numerical_features = self.data.select_dtypes(include=["number"])
                if numerical_features.shape[1] > 0:

                    # Calculate summary statistics once and reuse it
                    stats_df = self.calculate_statistics(self.data)

                    # Integrate summary statistics into the HTML dashboard
                    self.uni_variate_eda_html = self.gui_instance.ncr_dataframe(
                        self.uni_variate_eda_html,
                        "statisticalsummarytable",
                        stats_df
                    )

                    # Generate bar plot for mean values
                    fig_mean = self.bar_plot(
                        x=stats_df["Feature"],
                        y=stats_df["Mean"],
                        title="Mean Values Per Numerical Feature",
                        xlabel="Features",
                        ylabel="Mean Value",
                        annotation=True,
                        fig_size=(20, 12),
                        log_scale=True,
                        annotation_rotation=80,
                        annotation_as_int=False
                    )

                    # Integrate the mean values plot into the dashboard
                    self.uni_variate_eda_html = self.gui_instance.ncr_plot(
                        self.uni_variate_eda_html,
                        fig_mean,
                        "meanvaluespernumericalfeature",
                        ""
                    )

                    # Clear memory used by fig_mean
                    plt.close(fig_mean)
                    del fig_mean

                    # Add box plots to the dashboard
                    fig_boxplots = self.plot_boxplots(self.data)
                    self.uni_variate_eda_html = self.gui_instance.ncr_plot(
                        self.uni_variate_eda_html,
                        fig_boxplots,
                        "distributionanalysisofnumericalfeaturesviaboxplot",
                        ""
                    )

                    # Clear memory used by box plot
                    plt.close(fig_boxplots)
                    del fig_boxplots

                    # Perform variance analysis and create variance plot
                    variance_df, fig_var = self.variance_analysis(
                        self.data,
                        self.features_variance_analysis,
                        annotate=True,
                        annotation_rotation=80,
                        log_scale=True,
                        title=""
                    )

                    # Integrate variance summary and plot if variance analysis returns data
                    if not variance_df.empty:
                        self.uni_variate_eda_html = self.gui_instance.ncr_dataframe(
                            self.uni_variate_eda_html,
                            "variancesummarytable",
                            variance_df
                        )

                        self.uni_variate_eda_html = self.gui_instance.ncr_plot(
                            self.uni_variate_eda_html,
                            fig_var,
                            "varianceanalysisviafeatureplot",
                            ""
                        )

                        # Clear memory used by variance plot
                        plt.close(fig_var)
                        del fig_var

                    # Clear the summary statistics and variance dataframes to free up memory
                    del stats_df, variance_df

                else:
                    # If no numerical features are found, add an alert message to the dashboard
                    self.uni_variate_eda_html = self.gui_instance.ncr_dataframe(
                        self.uni_variate_eda_html,
                        "statisticalsummarytable",
                        pd.DataFrame({
                            "Alert": [
                                "No numerical features found in the dataset. Summary statistics and visualizations are not available."
                            ]
                        })
                    )

        except Exception as e:
            # Print the error message for troubleshooting
            print(f"An error occurred in the function summarize_numerical_features  {e}")

    def analyze_categorical_features_distribution(self):
        """
        Analyzes and visualizes the distribution of categorical features in the provided DataFrame based on user settings.

        Memory Optimization:
        --------------------
        - Caches results of calculations to avoid redundancy.
        - Clears memory with `del` and `plt.close()` after creating and integrating visualizations.
        """
        try:
            # Verify if the feature to analyze categorical feature distribution is enabled in user settings
            if self.execute_analysis(self.user_settings, self.function_names[8]):

                # Select categorical features
                categorical_features = self.data.select_dtypes(include=["object", "category"])
                if categorical_features.shape[1] > 0:

                    # Analyze categorical features and store the resulting DataFrames
                    categorical_dfs, skipped_features_df = self.analyze_categorical_features(self.data, target_column=self.target, threshold=self.categorcial_feature_threshold)

                    # Add the summary DataFrame of skipped features to the HTML dashboard
                    self.uni_variate_eda_html = self.gui_instance.ncr_dataframe(
                        self.uni_variate_eda_html,
                        format_string(self.function_names[8]),
                        skipped_features_df
                    )

                    # Delete `skipped_features_df` after integrating it to free memory
                    del skipped_features_df

                    # Check the number of unique values in the target for percentage plots
                    target_unique_values = self.data[self.target].nunique()

                    # Iterate through each categorical feature and its corresponding DataFrame
                    for feature, feature_df in categorical_dfs.items():
                        # Add header for each feature
                        self.uni_variate_eda_html = self.gui_instance.ncr_text(
                            self.uni_variate_eda_html,
                            format_string(self.function_names[8]),
                            f"Summary Table for: {feature}",
                            type="h2"
                        )

                        # Integrate the summary DataFrame for the current feature
                        self.uni_variate_eda_html = self.gui_instance.ncr_dataframe(
                            self.uni_variate_eda_html,
                            format_string(self.function_names[8]),
                            feature_df
                        )

                        # Generate and add the multi-metric plot to the HTML dashboard
                        fig1 = self.multi_metric_categorical_plot(
                            feature_df=feature_df,
                            feature_name=feature,
                            target_column=self.target
                        )

                        self.uni_variate_eda_html = self.gui_instance.ncr_plot(
                            self.uni_variate_eda_html,
                            fig1,
                            format_string(self.function_names[8]),
                            f"Multi-Metric Analysis for {feature}"
                        )

                        # Generate target percentage distribution plot if target has reasonable number of unique values
                        if target_unique_values <= 10 and self.data[feature].nunique() <= 20:
                            fig2 = self.plot_target_percentage_by_category(
                                data=self.data,
                                categorical_feature=feature,
                                target_feature=self.target,
                                figsize=(15, 8)
                            )

                            self.uni_variate_eda_html = self.gui_instance.ncr_plot(
                                self.uni_variate_eda_html,
                                fig2,
                                format_string(self.function_names[8]),
                                f"Target Percentage Distribution for {feature}"
                            )

                            # Close the percentage plot and free memory
                            plt.close(fig2)
                            del fig2

                        elif target_unique_values > 10:
                            # Add alert for too many target unique values
                            self.uni_variate_eda_html = self.gui_instance.ncr_dataframe(
                                self.uni_variate_eda_html,
                                format_string(self.function_names[8]),
                                pd.DataFrame({
                                    "Alert": [
                                        f"Target percentage distribution plot skipped for '{feature}' because target variable has {target_unique_values} unique values (>10). Consider grouping target values for better visualization."
                                    ]
                                })
                            )

                        elif self.data[feature].nunique() > 20:
                            # Add alert for too many categorical values
                            self.uni_variate_eda_html = self.gui_instance.ncr_dataframe(
                                self.uni_variate_eda_html,
                                format_string(self.function_names[8]),
                                pd.DataFrame({
                                    "Alert": [
                                        f"Target percentage distribution plot skipped for '{feature}' because it has {self.data[feature].nunique()} unique values (>20). Consider grouping categories for better readability."
                                    ]
                                })
                            )

                        # Free memory by closing the multi-metric plot and deleting variables
                        plt.close(fig1)
                        del fig1, feature_df

                    # Add overall section for Target Percentage Distribution summary
                    if target_unique_values <= 10:
                        suitable_features = [col for col in categorical_features.columns 
                                        if self.data[col].nunique() <= 20]
                        
                        if suitable_features:
                            self.uni_variate_eda_html = self.gui_instance.ncr_text(
                                self.uni_variate_eda_html,
                                format_string(self.function_names[8]),
                                "Target Percentage Distribution Summary",
                                type="h2"
                            )
                            
                            self.uni_variate_eda_html = self.gui_instance.ncr_dataframe(
                                self.uni_variate_eda_html,
                                format_string(self.function_names[8]),
                                pd.DataFrame({
                                    "Summary": [
                                        f"Target percentage distribution plots generated for {len(suitable_features)} categorical features. These plots show the proportion of each target class within each category, helping identify which categories are most predictive of the target variable."
                                    ]
                                })
                            )

                else:
                    # No categorical features found; add alert message to the HTML dashboard
                    self.uni_variate_eda_html = self.gui_instance.ncr_dataframe(
                        self.uni_variate_eda_html,
                        format_string(self.function_names[8]),
                        pd.DataFrame({
                            "Alert": [
                                "No categorical features found in the dataset. Distribution analysis and visualizations are not available."
                            ]
                        })
                    )

        except Exception as e:
            # Handle and report any exceptions encountered during execution
            print(f"An error occurred in the function analyze_categorical_features_distribution: {e}")

    def generate_target_feature_distribution_plot(self):
        
        """
        Generates and saves a distribution plot for the target feature if enabled in user settings.
        Optimized for memory usage by closing plots and deleting temporary objects.
        """
        try:
            # Check if target feature distribution plot generation is enabled in user settings
            if self.execute_analysis(self.user_settings, self.function_names[9]):

                # Generate and save the plot to the HTML dashboard
                fig = self.plot_target(
                    data=self.data,  # Dataset containing the target feature
                    target_column=self.target  # Target feature for plot
                )

                self.uni_variate_eda_html = self.gui_instance.ncr_plot(
                    self.uni_variate_eda_html,  # Path for saving the plot
                    fig,  # Plot figure
                    format_string(self.function_names[9]),  # Identifier for the plot in the dashboard
                    ""
                )

                # Close the figure to free memory
                plt.close(fig)
                del fig

                # Check the number of unique values in the target to provide an alert if necessary
                target_unique_values = self.data[self.target].nunique()
                if target_unique_values > 10:


                    self.uni_variate_eda_html = self.gui_instance.ncr_dataframe(
                        self.uni_variate_eda_html,  # Path for saving the alert message
                        format_string(self.function_names[9]),  # Identifier for the alert message
                        pd.DataFrame({
                            "Alert": [
                                "The target feature distribution plot "
                                "may be unclear due to having more than "
                                "10 unique values.Identify subcategories"
                                " that serve similar functions or target similar "
                                "audiences and merge them into a single broader category."
                            ]
                        })  # Alert DataFrame
                    )


        except Exception as e:
            print(f"An error occurred in the function generate_target_feature_distribution_plot  {e}")

   

    def generate_correlation_matrix(self):
        """
        Computes, displays, and saves the correlation matrix for numerical features in the dataset.
        Additionally, generates and saves a heatmap visualization of the correlation matrix.

        Returns:
        -------
        None
        """
        try:
            # Check if correlation analysis is enabled in user settings
            if not self.execute_analysis(self.user_settings, self.function_names[11]):
                return  # Exit early if analysis is disabled

            # Select numerical features
            numerical_features = self.data.select_dtypes(include=["number"])

            if numerical_features.shape[1] == 0:
                # Create an alert if no numerical features are found
                alert_df = pd.DataFrame({"Alert": ["No numerical features found in the dataset."]})
                self.uni_variate_eda_html = self.gui_instance.ncr_dataframe(self.uni_variate_eda_html,
                                                                            "correlationanalysisofnumericfeaturesmatrix",
                                                                            alert_df)
                self.uni_variate_eda_html = self.gui_instance.ncr_dataframe(self.uni_variate_eda_html,
                                                                            "correlationanalysisofnumericfeaturesheatmap",
                                                                            alert_df)
                del alert_df  # Free memory after alert is added
                return

            # Calculate the correlation matrix with feature names for display in the dashboard
            matrix = self.calculate_correlation_matrix_with_features_names(
                numerical_features,
                self.target,
                variance_threshold=self.variance_threshold,
                method=self.correlation_metric,
                apply_variance = self.apply_variance
            )

            # Display correlation matrix in the HTML dashboard
            self.uni_variate_eda_html = self.gui_instance.ncr_dataframe(
                self.uni_variate_eda_html,
                "correlationanalysisofnumericfeaturesmatrix",
                matrix
            )

            # Delete matrix after it has been used
            del matrix

            # Calculate the plain correlation matrix for plotting the heatmap
            matrix_df = self.calculate_correlation_matrix(
                numerical_features,
                self.target,
                variance_threshold=self.variance_threshold,
                method=self.correlation_metric,
                apply_variance= self.apply_variance 
            )

            # Plot heatmap and add to HTML dashboard
            heatmap_fig = self.plot_correlation_heatmap(matrix_df)
            self.uni_variate_eda_html = self.gui_instance.ncr_plot(
                self.uni_variate_eda_html,
                heatmap_fig,
                "correlationanalysisofnumericfeaturesheatmap",
                ""
            )

            # Delete large objects after use to free memory
            del numerical_features, matrix_df, heatmap_fig

        except Exception as e:
            print(f"An error occurred in the function generate_correlation_matrix : {e}")

    def display_cramers_vmatrix(self):
        """
        Computes, displays, and saves Cramér's V correlation matrix and its heatmap.
        Also generates and saves a heatmap for the target variable if it is present in the matrix.

        Returns:
        -------
        None
        """
        try:
            # Check if Cramér's V matrix display is enabled in user settings
            if not self.execute_analysis(self.user_settings, self.function_names[12]):
                return  # Exit early if analysis is disabled

            # Select categorical features
            categorical_columns = self.data.select_dtypes(include=["object", "category"])
            if categorical_columns.shape[1] == 0:
                # Early exit if there are no categorical features
                alert_df = self.create_alert_message()
                self.uni_variate_eda_html = self.gui_instance.ncr_dataframe(self.uni_variate_eda_html,
                                                                            "correlationanalysisofcategoricalfeaturesmatrix",
                                                                            alert_df)
                self.uni_variate_eda_html = self.gui_instance.ncr_dataframe(self.uni_variate_eda_html,
                                                                            "correlationanalysisofcategoricalfeaturesheatmap",
                                                                            alert_df)
                del alert_df
                return

            # Calculate Cramér's V correlation matrix with feature names
            matrix = self.categorical_correlation_matrix_feature_names(self.data, self.target)

            # Check if matrix contains all NaN values, indicating no valid correlations
            if matrix.isnull().values.all():
                alert_df = self.create_alert_message()
                self.uni_variate_eda_html = self.gui_instance.ncr_dataframe(self.uni_variate_eda_html,
                                                                            "correlationanalysisofcategoricalfeaturesmatrix",
                                                                            alert_df)
                self.uni_variate_eda_html = self.gui_instance.ncr_dataframe(self.uni_variate_eda_html,
                                                                            "correlationanalysisofcategoricalfeaturesheatmap",
                                                                            alert_df)
                del matrix, alert_df
                return

            # Display Cramér's V matrix in the HTML dashboard
            self.uni_variate_eda_html = self.gui_instance.ncr_dataframe(
                self.uni_variate_eda_html,
                "correlationanalysisofcategoricalfeaturesmatrix",
                matrix
            )

            # Calculate the simplified correlation matrix for heatmap display
            matrix_df = self.categorical_correlation_matrix(self.data, self.target)

            # Plot heatmap and add to HTML dashboard
            heatmap_fig = self.plot_heatmap(matrix_df)
            self.uni_variate_eda_html = self.gui_instance.ncr_plot(
                self.uni_variate_eda_html,
                heatmap_fig,
                "correlationanalysisofcategoricalfeaturesheatmap",
                ""
            )

            # Delete objects to free memory after use
            del categorical_columns, matrix, matrix_df, heatmap_fig

        except Exception as e:
            alert_df = self.create_alert_message()
            self.uni_variate_eda_html = self.gui_instance.ncr_dataframe(self.uni_variate_eda_html,
                                                                        "correlationanalysisofcategoricalfeaturesmatrix",
                                                                        alert_df)
            self.uni_variate_eda_html = self.gui_instance.ncr_dataframe(self.uni_variate_eda_html,
                                                                        "correlationanalysisofcategoricalfeaturesheatmap",
                                                                        alert_df)
            del alert_df  # Clean up alert data
            print(f"An error occurred in the function display_cramers_vmatrix  {e}")

    def create_alert_message(self):
        """
        Creates a detailed alert message for Cramér's V calculation errors.

        Returns:
        -------
        pd.DataFrame: A DataFrame containing the alert message.
        """
        alert_message = pd.DataFrame({
            "Alert": [
                "No valid combinations found for Cramér's V calculation. This could be due to one or more of the following reasons:\n"
                "- There are insufficient categorical features in the dataset, or they contain only one unique value.\n"
                "- All entries in the categorical features may be null (None) or missing.\n"
                "- There might be a lack of variability in the categorical columns, resulting in no meaningful combinations.\n\n"
                "Please check your data for valid categorical feature combinations:\n"
                "- Inspect categorical features to ensure they contain sufficient valid entries.\n"
                "- Review the presence of null values and consider removing or imputing them.\n"
                "- Verify that the categorical features have enough unique values to provide meaningful insights.\n\n"
                "Taking these steps will help ensure that the Cramér's V calculation can proceed successfully."
            ]
        })
        return alert_message

    def analyze_target_by_month(self):
        """
        Analyzes target trends by month and saves the output as either a DataFrame or a plot,
        depending on the analysis result.
        """
        try:
            # Check if 'Analyze Target Trends by Month' is enabled in user settings
            if not self.execute_analysis(self.user_settings, self.function_names[13]):
                return  # Exit early if the feature is disabled

            # Check if a timestamp column is specified
            if self.timestampColumn is None:
                # Early exit with an alert if no timestamp column is found
                alert_df = pd.DataFrame({
                    "Alert": [
                        "No timestamp column specified. Please provide a valid timestamp column for analysis. Example: 'Date' or 'Timestamp'."
                    ]
                })
                self.target_time_series_html = self.gui_instance.ncr_dataframe(
                    self.target_time_series_html,
                    format_string(self.function_names[13]),
                    alert_df
                )
                del alert_df  # Free memory
                return

            # Perform the target analysis by month
            result = self.plot_target_per_month(
                self.data,
                self.timestampColumn,
                self.target,
                self.num_months
            )

            # Save the result as a DataFrame or a plot, based on its type
            if isinstance(result, pd.DataFrame):
                self.target_time_series_html = self.gui_instance.ncr_dataframe(
                    self.target_time_series_html,
                    format_string(self.function_names[13]),
                    result
                )
                del result  # Free memory once saved
            else:
                self.target_time_series_html = self.gui_instance.ncr_plot(
                    self.target_time_series_html,
                    result,
                    format_string(self.function_names[13]),
                    ""
                )
                del result  # Free memory once saved

        except Exception as e:
            # Log any exceptions that occur
            print(f"An error occurred in the function analyze_target_by_month  {e}")

    def visualize_target_feature_by_week(self):
        """
        Visualizes the target feature by week and saves the output as either a DataFrame or a plot,
        depending on the analysis result.
        """
        try:
            # Exit early if 'Target Feature Visualization by Week' is disabled
            if not self.execute_analysis(self.user_settings, self.function_names[14]):
                return

            # Exit early if no timestamp column is specified
            if self.timestampColumn is None:
                # Generate and display an alert DataFrame for missing timestamp column
                alert_df = pd.DataFrame({
                    "Alert": [
                        "No timestamp column specified. Please provide a valid timestamp column for analysis. Example: 'Date' or 'Timestamp'."
                    ]
                })
                self.target_time_series_html = self.gui_instance.ncr_dataframe(
                    self.target_time_series_html,
                    format_string(self.function_names[14]),
                    alert_df
                )
                del alert_df  # Free memory used by alert DataFrame
                return

            

            # Perform the target feature visualization by week
            result = self.plot_target_per_week(
                self.data,
                self.timestampColumn,
                self.target,
                num_weeks=self.num_weeks
            )

            # Save the result as a DataFrame or a plot, based on its type
            if isinstance(result, pd.DataFrame):
                self.target_time_series_html = self.gui_instance.ncr_dataframe(
                    self.target_time_series_html,
                    format_string(self.function_names[14]),
                    result
                )
                del result  # Free memory used by result DataFrame
            else:
                self.target_time_series_html = self.gui_instance.ncr_plot(
                    self.target_time_series_html,
                    result,
                    format_string(self.function_names[14]),
                    ""
                )
                del result  # Free memory used by result plot

        except Exception as e:
            # Handle any exceptions that occur and log an error message
            print(f"An error occurred in the function visualize_target_feature_by_week  {e}")

    def analyze_numerical_time_series_by_month(self):
        """
        Analyzes numerical time series data by month and saves the output as either DataFrames or plots.
        """
        try:
            # Exit early if 'Numerical Time Series Analysis by Month' is disabled in user settings
            if not self.execute_analysis(self.user_settings, self.function_names[15]):
                return

            # Exit early if no timestamp column is specified
            if self.timestampColumn is None:
                return

            # Perform numerical time series analysis by month
            dataframes, figures, error_flag = self.numerical_time_series_analysis_per_month(
                self.data,
                timestamp_col=self.timestampColumn,
                target_col=self.target,
                subset_features=self.numerical_features_to_include,
                num_months=self.num_months
            )

            # Save each DataFrame and corresponding plot if no errors occurred
            if not error_flag:
                for key, dataframe in dataframes.items():
                    # Display feature name as a header
                    feature_name = dataframe["Feature"].iloc[0]
                    self.numerical_time_series_html = self.gui_instance.ncr_text(
                        self.numerical_time_series_html,
                        format_string(self.function_names[15]),
                        feature_name,
                        type="h3"
                    )
                    # Save DataFrame and corresponding plot to the HTML dashboard
                    self.numerical_time_series_html = self.gui_instance.ncr_dataframe(
                        self.numerical_time_series_html,
                        format_string(self.function_names[15]),
                        dataframe
                    )
                    del dataframe  # Free memory used by the individual DataFrame

                    self.numerical_time_series_html = self.gui_instance.ncr_plot(
                        self.numerical_time_series_html,
                        figures[key],
                        format_string(self.function_names[15]),
                        ""
                    )
                    del figures[key]  # Free memory used by the individual plot figure

            # If an error occurred during analysis, save the error message or DataFrame
            else:
                self.numerical_time_series_html = self.gui_instance.ncr_dataframe(
                    self.numerical_time_series_html,
                    format_string(self.function_names[15]),
                    dataframes
                )
                del dataframes  # Free memory in case of error

        except Exception as e:
            print(f"An error occurred in the function analyze_numerical_time_series_by_month  {e}")

    def analyze_numerical_time_series_by_week(self):

        """
        Analyzes numerical time series data by week and saves the output as either DataFrames or plots.
        """
        try:
            # Early exit if 'Numerical Time Series Analysis by Week' is not enabled
            if not self.execute_analysis(self.user_settings, self.function_names[16]):
                return

            # Early exit if no timestamp column is specified
            if self.timestampColumn is None:
                return

            # Perform numerical time series analysis by week
            dataframes, figures, error_flag = self.numerical_time_series_analysis_per_week(
                self.data,
                timestamp_col=self.timestampColumn,
                target_col=self.target,
                subset_features=self.numerical_features_to_include,
                num_weeks=self.num_weeks
            )


            # If no errors occurred during analysis
            if not error_flag:
                for key, dataframe in dataframes.items():
                    # Display feature name as a header
                    feature_name = dataframe["Feature"].iloc[0]
                    self.numerical_time_series_html = self.gui_instance.ncr_text(
                        self.numerical_time_series_html,
                        format_string(self.function_names[16]),
                        feature_name,
                        type="h3"
                    )

                    # Save the DataFrame to the HTML dashboard
                    self.numerical_time_series_html = self.gui_instance.ncr_dataframe(
                        self.numerical_time_series_html,
                        format_string(self.function_names[16]),
                        dataframe[['week', 'Week_Label', 'Feature', 'count', 'mean', 'median', 'min', 'max']]
                    )
                    del dataframe  # Free memory used by the individual DataFrame

                    # Save the plot to the HTML dashboard
                    self.numerical_time_series_html = self.gui_instance.ncr_plot(
                        self.numerical_time_series_html,
                        figures[key],
                        format_string(self.function_names[16]),
                        ""
                    )
                    del figures[key]  # Free memory used by the individual plot figure

            # If an error occurred during analysis, save the error message or DataFrame
            else:
                self.numerical_time_series_html = self.gui_instance.ncr_dataframe(
                    self.numerical_time_series_html,
                    format_string(self.function_names[16]),
                    dataframes
                )
                del dataframes  # Free memory in case of error

        except Exception as e:
            print(f"An error occurred in the function analyze_numerical_time_series_by_week {e}")

    def analyze_categorical_time_series_by_month(self):
        """
        Analyzes categorical time series data by month and saves the results as plots or dataframes.
        """
        try:
            # Early exit if analysis is not enabled in user settings
            if not self.execute_analysis(self.user_settings, self.function_names[17]):
                return

            # Early exit if no timestamp column is provided
            if self.timestampColumn is None:
                alert_message = pd.DataFrame({
                    "Alert": [
                        "No timestamp column specified. Please provide a valid timestamp column for analysis. Example: 'Date' or 'Timestamp'."
                    ]
                })
                self.categorical_time_series_html = self.gui_instance.ncr_dataframe(
                    self.categorical_time_series_html,
                    format_string(self.function_names[17]),
                    alert_message
                )
                return

            # Perform categorical time series analysis by month

            dataframes, figures, ignored_features_df, error_flag = self.categorical_time_series_analysis_per_month(
                self.data,
                self.timestampColumn,
                self.target,
                self.num_months,
                self.top_n,
                subset_features=self.categorical_features_to_include
            )

            # If no errors occurred during analysis

            if not error_flag:
                for figure in figures:
                    self.categorical_time_series_html = self.gui_instance.ncr_plot(
                        self.categorical_time_series_html,
                        figure,
                        format_string(self.function_names[17]),
                        ""
                    )
                    del figure  # Free memory used by the individual plot figure

                if ignored_features_df.shape[0] > 0:
                    self.categorical_time_series_html = self.gui_instance.ncr_dataframe(
                        self.categorical_time_series_html,
                        format_string(self.function_names[17]),
                        ignored_features_df
                    )
                    del ignored_features_df  # Free memory for ignored features DataFrame

            # If an error occurred during analysis
            else:
                
                self.categorical_time_series_html = self.gui_instance.ncr_dataframe(
                    self.categorical_time_series_html,
                    format_string(self.function_names[17]),
                    dataframes
                )
                del dataframes  # Free memory for the dataframes in case of error

        except Exception as e:
            # Handle exceptions and print the error
            print(f"an error has accured in the functioon analyze_categorical_time_series_by_month {e}")
            
    def generate_kde_plots(self):
        """
        Generates and saves Kernel Density Estimate (KDE) plots for specified features and the target variable,
        provided the feature is enabled in the user settings.
        Shows percentage distributions with lines and color fill for easy interpretation.
        Optimized for memory usage by closing plots and minimizing temporary DataFrame usage.
        """
        try:
            # Check if KDE plot generation is enabled in user settings
            if self.execute_analysis(self.user_settings, self.function_names[10]):
                
                # Check if there are any numerical features available for KDE plotting
                numerical_features = self.data.select_dtypes(include=["number"])
                if numerical_features.shape[1] > 0:
                    
                    # Generate the KDE plot with percentage distribution
                    kde_plot_fig = self.plot_bivariate_kde(
                        data=self.data,  # Dataset to analyze
                        cols_to_plot=self.features_kde_plot,  # List of features to include in the KDE plot
                        target_var=self.target,  # Target variable for the KDE plot
                        fill="Another_Variable",  # Variable used for filling the KDE plot
                        palette="Set1"  # Color palette for the plot
                    )
                    
                    # Add the KDE plot to the HTML dashboard
                    self.uni_variate_eda_html = self.gui_instance.ncr_plot(
                        self.uni_variate_eda_html,  # Path for saving the HTML dashboard
                        kde_plot_fig,  # KDE plot figure
                        format_string(self.function_names[10]),  # Identifier for the plot in the dashboard
                        ""
                    )
                    
                    # Close the plot to free memory
                    plt.close(kde_plot_fig)
                    del kde_plot_fig
                    
                    # Check the number of unique values in the target
                    target_unique_values = self.data[self.target].nunique()
                    if target_unique_values > 10:
                        # Create alert DataFrame for unique value count
                        self.uni_variate_eda_html = self.gui_instance.ncr_dataframe(
                            self.uni_variate_eda_html,
                            format_string(self.function_names[10]),
                            pd.DataFrame({
                                "Alert": [
                                    "The target feature distribution plot may be unclear due to having more than 10 unique values. Consider grouping similar categories together for better visualization."
                                ]
                            })
                        )
                else:
                    # Create and add alert for no numerical features found
                    alert_no_numerical_df = pd.DataFrame({
                        "Alert": [
                            "No numerical features found in the dataset. KDE plot generation is not applicable."
                        ]
                    })
                    
                    self.uni_variate_eda_html = self.gui_instance.ncr_dataframe(
                        self.uni_variate_eda_html,
                        format_string(self.function_names[10]),
                        alert_no_numerical_df
                    )
                    
                    # Delete alert DataFrame to free memory
                    del alert_no_numerical_df
                    
        except Exception as e:
            # Handle any exceptions and log an error message
            print(f"An error occurred in the function generate_kde_plots: {e}")

            
    def analyze_categorical_time_series_by_week(self):
        """
        Analyzes categorical time series data by week and saves the results as plots or dataframes.
        """
        try:
            # Early exit if analysis is not enabled in user settings
            if not self.execute_analysis(self.user_settings, self.function_names[18]):
                return

            # Early exit if no timestamp column is provided
            if self.timestampColumn is None:
                alert_message = pd.DataFrame({
                    "Alert": [
                        "No timestamp column specified. Please provide a valid timestamp column for analysis. Example: 'Date' or 'Timestamp'."
                    ]
                })
                self.categorical_time_series_html = self.gui_instance.ncr_dataframe(
                    self.categorical_time_series_html,
                    format_string(self.function_names[18]),
                    alert_message
                )
                return

            # Perform categorical time series analysis by week
            dataframes, figures, ignored_features_df, error_flag = self.categorical_time_series_analysis_per_week(
                self.data,  # Use original data to avoid unnecessary copying
                self.timestampColumn,
                self.target,
                self.num_weeks,
                self.top_n,
                subset_features=self.categorical_features_to_include
            )

            # If no errors occurred during analysis
            if not error_flag:
                for figure in figures:
                    self.categorical_time_series_html = self.gui_instance.ncr_plot(
                        self.categorical_time_series_html,
                        figure,
                        format_string(self.function_names[18]),
                        ""
                    )
                    del figure  # Free memory used by the individual plot figure

                if ignored_features_df.shape[0] > 0:
                    self.categorical_time_series_html = self.gui_instance.ncr_dataframe(
                        self.categorical_time_series_html,
                        format_string(self.function_names[18]),
                        ignored_features_df
                    )
                    del ignored_features_df  # Free memory for ignored features DataFrame

            # If an error occurred during analysis

            else:

                self.categorical_time_series_html = self.gui_instance.ncr_dataframe(
                    self.categorical_time_series_html,
                    format_string(self.function_names[18]),
                    dataframes
                )
                del dataframes  # Free memory for the dataframes in case of error

        except Exception as e:
            # Handle exceptions and print the error
            print(f"Error has accured in the function analyze_categorical_time_series_by_week  {e}")

    def user_eda_functions(self):
        """
        Executes exploratory data analysis (EDA) functions based on user-defined settings.
        """
        # Check if target is provided
        if self.target is None:
            raise KeyError("Target is not provided. Please provide a target column name.")

        # Record the start time for the EDA process
        uni_multi_variate_analysis_start_time = datetime.now()

        # Perform various EDA tasks in a sequence that avoids unnecessary intermediate data retention
        self.generate_data_overview()
        self.perform_initial_data_inspection()
        self.create_feature_info_table()
        self.create_unique_values_table()
        self.create_missing_values_table()
        self.analyze_zero_values()
        self.summarize_numerical_features()
        self.analyze_categorical_features_distribution()
        self.generate_kde_plots()
        self.generate_target_feature_distribution_plot()
        self.generate_correlation_matrix()
        self.display_cramers_vmatrix()

        # Perform time series analysis
        self.analyze_target_by_month()
        self.visualize_target_feature_by_week()
        self.analyze_numerical_time_series_by_month()
        self.analyze_numerical_time_series_by_week()
        self.analyze_categorical_time_series_by_month()
        self.analyze_categorical_time_series_by_week()

        # Mark categorical time series analysis end time
        categorical_time_series_analysis_end_time = datetime.now()

        self.generate_project_overview(
            uni_multi_variate_analysis_start_time,
            categorical_time_series_analysis_end_time,
            format_string(self.function_names[0]),
            "projectoverviewandkeyinformation"
        )

        # Efficiently write the reports to files, avoiding unnecessary reopening of files
        report_files = {
            "arco_eda_framework": self.uni_variate_eda_html,
            "target_vs_time": self.target_time_series_html,
            "numerical_vs_time": self.numerical_time_series_html,
            "categorical_vs_time": self.categorical_time_series_html
        }

        for report_name, report_content in report_files.items():
            with open(self.report_paths[report_name], 'w', encoding='utf-8') as file:
                file.write(report_content)

        # Clear large objects after use to free up memory, if necessary
        del report_files  # Remove the reference to the report files dictionary
        print("EDA completed successfully.")
        print(f"Files Saved to following path : {self.output_path}" ) 
        """
        if self.use_llm : 

            path = os.path.join(self.base_dir, self.project_folder_name, self.folder_name_time)
            path = create_file_path(path,"FeatureSelectionAgentReports/agent_report")
            self.response  =self.generate_feature_selection_report(self.response["feature_recommendation_response"].data.selected_features, self.response["feature_recommendation_response"].data.selected_features_justification, 
                                        self.response["feature_recommendation_response"].data.excluded_features,  self.response["feature_recommendation_response"].data.excluded_features_justification,   self.response["feature_recommendation_response"].data.preprocessing_recommendations,  self.response["feature_recommendation_response"].data.feature_importance_ranking ,path)
        """