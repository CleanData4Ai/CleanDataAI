
"""

    EdaHelperFunctions Class

    Purpose:
    The `EdaHelperFunctions` class provides a collection of utility methods essential for performing detailed and structured Exploratory Data Analysis (EDA). It acts as a supporting component within the framework, offering reusable functions that streamline data inspection, analysis, and report generation. This class is designed to complement the `EdaCoreFunctions` class by handling specific tasks and ensuring flexibility in the analysis process.

    Key Responsibilities:
    - **Execution of Analysis Settings**: Facilitates the execution of user-specified or default analysis functions based on settings configurations.
    - **Data Profiling**: Generates comprehensive profile reports for datasets using configurable parameters (e.g., variables, missing data, correlations).
    - **Data Observation**: Provides functionality to observe subsets of the dataset (e.g., top or bottom rows) for quick inspection.
    - **Overview Generation**: Creates summary tables that highlight the dataset's key characteristics, enabling a high-level understanding of the data.

    Features:
    1. **Configurable Analysis**:
    - The `execute_analysis` method allows dynamic execution of specific analysis functions based on user-defined YAML settings. This flexibility ensures that the framework adapts to varying requirements.
    
    2. **Data Profiling**:
    - The `generate_profile_report` method generates an HTML report summarizing dataset characteristics such as variable distributions, missing values, and correlations. The configuration is customizable via YAML files, ensuring tailored profiling.

    3. **Data Inspection**:
    - The `observe_data` method enables quick inspection of the top or bottom rows of the dataset, allowing users to validate data formats or identify irregularities efficiently.

    4. **Dataset Overview**:
    - The `overview_table` method produces a concise summary of the dataset's structure and contents, including the number of rows, columns, and data types.

    Usage:
    - **Integration**: This class is not intended for standalone use but is designed to be leveraged by core components (like `EdaCoreFunctions`) to perform repetitive and modular tasks.
    - **Customization**: Many methods rely on user-defined configurations, making them highly customizable to different datasets and analysis needs.

    Examples:
    1. Generate a Profile Report:
    ```python
    helper = EdaHelperFunctions()
    helper.generate_profile_report(data=df, config=my_config, output_path="profile_report.html")

"""

import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import chi2_contingency
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import zscore #help function
from typing import List, Optional, Union


class EdaHelperFunctions:

    def __init__(self):
        pass
    

    def generate_feature_selection_report(self , selected_features, selected_features_justification, 
                                      excluded_features, excluded_features_justification,
                                      preprocessing_recommendations, feature_importance_ranking, 
                                      output_path):
        
        # Create the HTML content
        html_report = f"""

                <!DOCTYPE html>
                <html lang="en">
                <head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <title>Feature Selection Report</title>
                    <style>
                        body {{
                            font-family: Arial, sans-serif;
                            line-height: 1.6;
                            margin: 40px;
                            color: #333;
                        }}
                        h1, h2 {{
                            color: #007BFF;
                        }}
                        .section {{
                            margin-bottom: 20px;
                        }}
                        .section p {{
                            margin: 10px 0;
                        }}
                        .code-block {{
                            background-color: #f8f9fa;
                            border: 1px solid #ddd;
                            padding: 10px;
                            border-radius: 4px;
                            font-family: "Courier New", Courier, monospace;
                            white-space: pre-wrap;
                        }}
                    </style>
                </head>
                <body>
                    <h1>Feature Selection Report</h1>

                    <div class="section">
                        <h2>1. Selected Features</h2>
                        <p>The following features were recommended for selection:</p>
                        <div class="code-block">
                            {', '.join(selected_features)}
                        </div>
                        <p><strong>Justification for Selected Features:</strong></p>
                        <div class="code-block">
                            {selected_features_justification}
                        </div>
                    </div>

                    <div class="section">
                        <h2>2. Excluded Features</h2>
                        <p>The following features were excluded from the selection:</p>
                        <div class="code-block">
                            {', '.join(excluded_features)}
                        </div>
                        <p><strong>Justification for Excluded Features:</strong></p>
                        <div class="code-block">
                            {excluded_features_justification}
                        </div>
                    </div>

                    <div class="section">
                        <h2>3. Preprocessing Recommendations</h2>
                        <p>The following preprocessing steps are recommended:</p>
                        <div class="code-block">
                            {preprocessing_recommendations}
                        </div>
                    </div>

                    <div class="section">
                        <h2>4. Feature Importance Ranking</h2>
                        <p>The ranking of features based on their importance for the model is as follows:</p>
                        <div class="code-block">
                            {feature_importance_ranking}
                        </div>
                    </div>

                </body>
                </html>

        """

        # Save the generated HTML report to the specified path
        with open(output_path, "w") as file:
            file.write(html_report)

        print(f"Feature selection report has been saved to {output_path}")

    def execute_analysis(self , settings, function_name):
        # Perform the operation with the provided function name index
        return settings["functions"]["defaultExploratoryDataAnalysis"][function_name]

    def generate_profile_report(self, data: pd.DataFrame, config: str, output_path: str = "output.html") -> None:
        """
        Generates a profile report for the given dataset using configuration from a YAML file.

        Parameters:
        data : DataFrame
            The input data for which the profile report is to be generated.
        config_path : str
            The path to the YAML configuration file.
        output_path : str, optional
            The path where the HTML report will be saved. Default is 'output.html'.

        Returns:
        None
        """

        # Suppress all warnings to ensure a clean output
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Generate profile report with configuration from YAML file
            profile = data.profile_report(

                vars=config['vars'],
                missing_diagrams=config['missing_diagrams'],
                correlations=config['correlations'],
                interactions=config['interactions'],
                html=config['html'],
                minimal=config['minimal']

            )

            # Export the generated profile report to an HTML file at the specified path
            profile.to_file(output_path)
    

    def chi_square_test(self, data: pd.DataFrame, col1: str, col2: str, print_results: bool = False):
        """
        Perform a Chi-Square test of independence between two categorical variables in the provided DataFrame.
        
        The Chi-Square test assesses whether there is a statistically significant association between
        two categorical variables. The null hypothesis (H0) assumes the variables are independent,
        while the alternative hypothesis (H1) posits that they are dependent.
        
        Parameters:
        ----------
        data : pd.DataFrame
            The DataFrame containing the data for analysis.
        col1 : str
            The name of the first categorical column to be tested.
        col2 : str
            The name of the second categorical column to be tested.
        print_results : bool, default=True
            Whether to print the test results.
        
        Returns:
        -------
        dict
            A dictionary containing the test statistic, p-value, degrees of freedom, and the contingency table.
        """
        # Generate the contingency table for the two categorical columns
        contingency_table = pd.crosstab(data[col1], data[col2])
        
        # Execute the Chi-Square test of independence
        chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
        
        # Create results dictionary
        results = {
            'chi2_stat': chi2_stat,
            'p_value': p_value,
            'dof': dof,
            'contingency_table': contingency_table,
            'expected': expected
        }
        
        # Output the results if requested
        if print_results:
            print(f"Chi-Square Test of Independence between '{col1}' and '{col2}':")
            print(f"--------------------------------------------")
            print(f"Chi-Square Statistic: {chi2_stat:.4f}")
            print(f"Degrees of Freedom: {dof}")
            print(f"p-value: {p_value:.4f}")
            
            # Interpret the results based on the p-value
            if p_value < 0.05:
                print(f"\nResult: Reject the null hypothesis (H0). There is a significant association between '{col1}' and '{col2}'. \n")
            else:
                print(f"\nResult: Fail to reject the null hypothesis (H0). There is no significant association between '{col1}' and '{col2}'. \n")
        
        return results
    

    
    def analyze_categorical_associations(self, data: pd.DataFrame, target: str, alpha: float = 0.05):
        """
        Analyze associations between categorical features and a target variable.
        
        Parameters:
        ----------
        data : pd.DataFrame
            The DataFrame containing the data for analysis.
        target : str
            The name of the target variable column.
        alpha : float, default=0.05
            Significance level for hypothesis testing.
            
        Returns:
        -------
        pd.DataFrame
            A DataFrame summarizing the chi-square test results for each feature.
        """
        # Identify all categorical columns except the target variable
        # This includes both object and categorical dtypes
        columns = [col for col in data.columns 
                if (col != target) and 
                (data[col].dtype.name == 'object' or 
                    pd.api.types.is_categorical_dtype(data[col]) or
                    (data[col].nunique() < 10 and pd.api.types.is_numeric_dtype(data[col])))]
        
        results = []
        
        # Perform Chi-Square test for each categorical column against the target variable
        for col in columns:
            test_result = self.chi_square_test(data, col, target, print_results=False)
            
            # Add to results
            results.append({
                'Feature': col,
                'Chi-Square': test_result['chi2_stat'],
                'p-value': test_result['p_value'],
                'DoF': test_result['dof'],
                'Significant': test_result['p_value'] < alpha
            })
        
        # Create and sort results DataFrame
        results_df = pd.DataFrame(results).sort_values('p-value')
        
        # Print summary
        """
        print(f"Chi-Square Test Results for Association with '{target}'")
        print(f"----------------------------------------------------")
        print(results_df.to_string(index=False))
        """
        
        return results_df
    
    
    def calculate_feature_metrics(self , data :pd.DataFrame):
        """
        Calculate various statistical metrics for each numerical feature in the DataFrame. These metrics
        include the following:
        - Lower Bound (using IQR method for outlier detection)
        - Upper Bound (using IQR method for outlier detection)
        - Outlier Count (number of outliers detected based on the bounds)
        - Z-Score Mean (mean of Z-scores)
        - Variance (measure of spread)
        - Standard Deviation (measure of spread)
        - Skewness (measure of asymmetry in the distribution)
        - Percentiles (25%, 50%, 75%)
        - Mean (average value of the feature)
        - Mode (most frequent value of the feature)

        Parameters:
        ----------
        data : pandas DataFrame
            The DataFrame containing the numerical features for which the metrics are calculated.

        Returns:
        -------
        pandas DataFrame
            A DataFrame containing the calculated metrics for each numerical feature in the input DataFrame.
        """
        metrics = []  # List to store the metrics for each feature

        # Iterate over each column in the DataFrame
        for column in data.columns:

            # Proceed only if the column is numeric
            if pd.api.types.is_numeric_dtype(data[column]):

                # Calculate the 25th and 75th percentiles for the Interquartile Range (IQR)
                Q1 = data[column].quantile(0.25)  # 25th percentile (Q1)
                Q3 = data[column].quantile(0.75)  # 75th percentile (Q3)
                IQR = Q3 - Q1  # Interquartile Range (IQR)

                # Calculate the lower and upper bounds for detecting outliers based on IQR
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                # Identify outliers using the bounds
                outliers = data[column][(data[column] < lower_bound) | (data[column] > upper_bound)]
                outlier_count = len(outliers)  # Count the number of outliers

                # Calculate Z-scores for the column to assess the standardization of the data
                z_scores = zscore(data[column])
                mean_zscore = np.mean(z_scores)  # Compute the mean of the Z-scores

                # Calculate variance (spread of the data) and standard deviation (spread around the mean)
                variance = data[column].var()  # Variance
                std_deviation = data[column].std()  # Standard deviation

                # Calculate skewness (measure of symmetry of the data distribution)
                skewness = data[column].skew()

                # Calculate percentiles (25%, 50%, and 75%)
                percentile_25 = data[column].quantile(0.25)
                percentile_50 = data[column].quantile(0.50)  # Median (50th percentile)
                percentile_75 = data[column].quantile(0.75)

                # Calculate mean and mode for the feature
                mean = data[column].mean()  # Mean of the feature
                mode = data[column].mode()[0]  # Mode of the feature (taking the first mode if there are multiple)

                # Store all the calculated metrics for this column in the metrics list
                metrics.append({
                    'Feature': column,
                    'Lower Bound': lower_bound,
                    'Upper Bound': upper_bound,
                    '25% Percentile': percentile_25,
                    '50% Percentile (Median)': percentile_50,
                    '75% Percentile': percentile_75,
                    'Mean': mean,
                    'Mode': mode,
                    'Outlier Count': outlier_count,
                    'Z-Score Mean': mean_zscore,
                    'Variance': variance,
                    'Standard Deviation': std_deviation,
                    'Skewness': skewness
                })

        # Convert the list of feature metrics into a DataFrame for better presentation
        metrics_df =metrics

        return metrics_df  # Return the DataFrame containing the feature metrics


    def observe_data(self, data: pd.DataFrame, n: int = 5, head: bool = True) -> pd.DataFrame:
        """
        Observes the top or bottom 'n' rows of a pandas DataFrame.

        Parameters:
        -----------
        data : pd.DataFrame
            The DataFrame to observe.
        n : int, optional
            Number of rows to display. Default is 5.
        head : bool, optional
            If True, displays the first 'n' rows; if False, displays the last 'n' rows. Default is True.

        Returns:
        --------
        pd.DataFrame
            DataFrame containing the observed rows.

        Examples:
        ---------
        # Display the first 5 rows of the DataFrame
        observe_data(data)

        # Display the last 3 rows of the DataFrame
        observe_data(data, n=3, head=False)
        """
        if head:
            return data.head(n)
        else:
            return data.tail(n)

    def overview_table(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generates an overview table summarizing key characteristics of the provided DataFrame.

        Parameters:
        ----------
        data : pd.DataFrame
            The DataFrame for which the summary information is to be generated.

        Returns:
        -------
        pd.DataFrame
            A DataFrame containing the following summary statistics:
            - 'number_samples': Total number of rows in the dataset.
            - 'number_features': Total number of columns in the dataset.
            - 'n_numeric_features': Number of numerical features in the dataset.
            - 'n_categorical_features': Number of categorical features in the dataset.
            - 'n_timestamp_features': Number of features with timestamp data type.
            - 'dataset_total_missing_values': Total number of missing values in the dataset.
            - 'number_duplicates': Total number of duplicate rows in the dataset.
            - 'missing_values_per_feature': A Series with the count of missing values for each feature.

        Notes:
        -----
        - Numerical features are identified using the data types "number".
        - Categorical features are identified using the data types "object" and "category".
        - Timestamp features are identified using the data type "datetime".
        """
        # Calculate the number of samples (rows) and features (columns)
        number_samples = data.shape[0]
        number_features = data.shape[1]

        # Calculate the number of numerical features
        number_num = len(data.select_dtypes(include=["number"]).columns)

        # Calculate the number of categorical features
        number_categorical = len(data.select_dtypes(include=["object", "category"]).columns)

        # Calculate the number of timestamp features
        number_timestamp = len(data.select_dtypes(include=["datetime"]).columns)

        # Calculate the total number of duplicate rows
        number_duplicates = data.duplicated().sum()

        # Create a summary table
        table = {
            "Dataset Number of Samples / Number of Rows": [number_samples],
            "Dataset Number of Features / Number of Columns": [number_features],
            "Number of Numerical Features": [number_num],
            "Number of Categorical Features": [number_categorical],
            "Number of Timestamp Features": [number_timestamp],
            "Total Number of Missing Values (None Values) in the Dataset": [data.isna().sum().sum()],
            "Number of Duplicated Rows in the Dataset": [number_duplicates]
        }

        # Convert the summary table to a DataFrame
        table = pd.DataFrame(table)

        return table

    def info_table(self, data: pd.DataFrame, sort_by: str = "Non Null Count", ascending: bool = False) -> pd.DataFrame:
        """
        Generate a summary DataFrame for the given DataFrame, with an option to sort the result.

        Parameters:
        data (pd.DataFrame): The DataFrame for which to generate the summary.
        sort_by (str): The column name by which to sort the summary DataFrame. Default is 'features_name'.
        ascending (bool): Whether to sort in ascending order. Default is True.

        Returns:
        pd.DataFrame: A summary DataFrame containing the column names, non-null counts, and data types.
        """
        # Create the summary DataFrame
        summary_df = pd.DataFrame({
            'Feature Name': data.columns,
            'Non Null Count': data.notnull().sum(),
            'Dtype': data.dtypes
        })

        # Sort the DataFrame based on the specified column
        summary_df = summary_df.sort_values(by=sort_by, ascending=ascending)

        return summary_df

    def dtypes_table(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Display information about the data types of columns in a pandas DataFrame.

        Parameters:
        -----------
        data : pd.DataFrame
            The DataFrame for which column information is to be displayed.

        Returns:
        --------
        pd.DataFrame
            DataFrame containing column data types.

        Examples:
        ---------
        # Display column information for a DataFrame
        show_column_info(data)
        """
        result = {"features": data.columns.to_list(), "data_types": data.dtypes.to_list()}
        return pd.DataFrame(result)

    def show_column_unique_counts(self, data: pd.DataFrame, sort_by: str = "Unique Values",
                                  ascending: bool = False) -> pd.DataFrame:
        """
        Calculate and display the count of unique values in each categorical column of a DataFrame.

        Parameters:
        -----------
        data : pd.DataFrame
            The DataFrame for which unique value counts are to be calculated.
        sort_by : str, optional
            The column name to sort the result by ('Feature' or 'Unique Values').
            Default is 'Unique Values'.
        ascending : bool, optional
            If True, sort in ascending order; if False, sort in descending order.
            Default is True (ascending).

        Returns:
        --------
        pd.DataFrame
            DataFrame containing the count of unique values for each categorical column.
            The index of the DataFrame represents column names, and columns are:
            - 'Feature': The name of the column (feature) in the DataFrame.
            - 'Unique Values': Number of unique values in each categorical column.

        Examples:
        ---------
        # Display unique value counts for each categorical column in a DataFrame
        show_column_unique_counts(data)

        # Display sorted by unique values in descending order
        show_column_unique_counts(data, sort_by='Unique Values', ascending=False)
        """
        # Filter to only categorical columns
        categorical_data = data.select_dtypes(include=['object', 'category'])

        # Calculate unique values for categorical columns
        unique_values = pd.DataFrame(categorical_data.nunique(), columns=["Unique Values"])

        # Add the feature (column name) as a new column
        unique_values['Feature'] = unique_values.index

        # Reset the index so that 'Feature' becomes a column
        unique_values = unique_values.reset_index(drop=True)

        # Reorder the columns to place 'Feature' first
        unique_values = unique_values[['Feature', 'Unique Values']]

        # Sort the table if sort_by is provided
        if sort_by in ['Feature', 'Unique Values']:
            unique_values = unique_values.sort_values(by=sort_by, ascending=ascending)

        return unique_values

    def show_column_missing_counts(self, data: pd.DataFrame, decimal_places: int = 2, sort_by: str = "Missing Values",
                                   ascending: bool = False) -> pd.DataFrame:
        """
        Calculate and display the count and ratio of missing values (NaN) in each column of a DataFrame.

        Parameters:
        -----------
        data : pd.DataFrame
            The DataFrame for which missing value counts are to be calculated.
        decimal_places : int, optional
            Number of decimal places to round the ratios to (default is 2).
        sort_by : str, optional
            The column name to sort the result by ('Feature', 'Missing Values', or 'Ratio').
            Default is None, which means no sorting.
        ascending : bool, optional
            If True, sort in ascending order; if False, sort in descending order.
            Default is True (ascending).

        Returns:
        --------
        pd.DataFrame
            DataFrame containing the count and ratio of missing values for each column.
            The index of the DataFrame represents column names, and columns are:
            - 'Feature': The name of the column (feature) in the DataFrame
            - 'Missing Values': Number of missing values in each column
            - 'Ratio': Percentage of missing values relative to the total number of rows in each column, rounded to the specified decimal places.

        Examples:
        ---------
        # Display missing value counts and ratios for each column in a DataFrame
        show_column_missing_counts(data)

        # Display sorted by missing values in descending order
        show_column_missing_counts(data, sort_by='Missing Values', ascending=False)
        """
        # Calculate missing values and their ratios
        missing_values = pd.DataFrame(data.isnull().sum(), columns=["Missing Values"])
        ratio = pd.DataFrame((data.isnull().sum() / data.shape[0]) * 100, columns=["Ratio"])

        # Round the ratios to the specified number of decimal places
        ratio["Ratio"] = ratio["Ratio"].round(decimal_places)

        # Combine the missing values and ratios into one DataFrame
        table = pd.concat([missing_values, ratio], axis=1)

        # Add the feature (column name) as a new column
        table['Feature'] = table.index

        # Reset the index so that 'Feature' becomes a column
        table = table.reset_index(drop=True)

        # Reorder the columns to place 'Feature' first
        table = table[['Feature', 'Missing Values', 'Ratio']]

        # Sort the table if sort_by is provided
        if sort_by in ['Feature', 'Missing Values', 'Ratio']:
            table = table.sort_values(by=sort_by, ascending=ascending)

        return table



    def show_zeros_table(self, data: pd.DataFrame, threshold: float = 0.1, decimal_places: int = 2,
                         sort_by: str = "n_zeros", ascending: bool = False) -> pd.DataFrame:
        """
        Generate a table displaying columns with zero values exceeding a specified threshold.

        Parameters:
        -----------
        data : pd.DataFrame
            The DataFrame to analyze for zero values.
        threshold : float, optional
            The threshold above which the ratio of zeros in a column is considered significant. Default is 0.1.
        decimal_places : int, optional
            Number of decimal places to round the ratio of zeros to (default is 2).
        sort_by : str, optional
            The column name to sort the result by ('feature', 'n_zeros', or 'ratio').
            Default is 'n_zeros'.
        ascending : bool, optional
            If True, sort in ascending order; if False, sort in descending order.
            Default is False (descending).

        Returns:
        --------
        pd.DataFrame
            DataFrame containing columns with zero values exceeding the specified threshold.
            The DataFrame includes columns:
            - 'feature': Name of the column
            - 'n_zeros': Number of zero values in the column
            - 'ratio': Percentage of zero values relative to the total number of rows, rounded to the specified decimal places.
        """

        # Identify columns with zero values exceeding the threshold
        zero_cols = [col for col in data if data[col].dtype in ['number', 'int64', 'float64'] and
                     (data.loc[data[col] == 0, col].count() / data.shape[0]) > threshold]

        zero_data = []

        for col in zero_cols:
            n_zeros = data.loc[data[col] == 0, col].count()
            ratio = (n_zeros / data.shape[0]) * 100
            zero_data.append({
                "feature": col,
                "n_zeros": n_zeros,
                "ratio": round(ratio, decimal_places)
            })

        # Create a DataFrame from the collected zero data
        zero_df = pd.DataFrame(zero_data)

        # Check if zero_df is empty before sorting
        if zero_df.empty:
            return zero_df

        # Sort the DataFrame if the sort_by column exists
        if sort_by in zero_df.columns:
            zero_df = zero_df.sort_values(by=sort_by, ascending=ascending)

        return zero_df

    def numeric_and_categorical_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Separate numerical, categorical, and timestamp columns from a DataFrame and return them in a single DataFrame.

        Parameters:
        -----------
        data : pd.DataFrame
            The DataFrame for which numerical, categorical, and timestamp columns are to be identified.

        Returns:
        --------
        pd.DataFrame
            DataFrame containing three columns:
            - 'Numerical Features': Names of numerical columns
            - 'Categorical Features': Names of categorical columns
            - 'Timestamp Features': Names of timestamp columns

        Examples:
        ---------
        # Get a DataFrame with numerical, categorical, and timestamp columns from a DataFrame
        features_df = numeric_and_categorical_columns(data)
        """
        # Get numerical columns
        numerical_features = data.select_dtypes(include=["number"]).columns.tolist()

        # Get categorical columns
        categorical_features = data.select_dtypes(include=['object', 'category']).columns.tolist()

        # Get timestamp columns
        timestamp_features = data.select_dtypes(include=['datetime']).columns.tolist()

        # Create a DataFrame to hold the features
        features_df = pd.DataFrame({
            "Numerical Features": pd.Series(numerical_features),
            "Categorical Features": pd.Series(categorical_features),
            "Timestamp Features": pd.Series(timestamp_features)
        })

        return features_df

    def calculate_statistics(self, data: pd.DataFrame, low_quantile: float = 0.25, high_quantile: float = 0.75,
                             sort_by: str = "Mean", ascending: bool = False) -> pd.DataFrame:
        """
        Calculates outliers and various statistical metrics for each numerical column.

        Parameters:
        -----------
        data : pd.DataFrame
            The input dataframe.
        low_quantile : float, optional
            The low quantile value for IQR calculation. Default is 0.25.
        high_quantile : float, optional
            The high quantile value for IQR calculation. Default is 0.75.
        sort_by : str, optional
            The column name to sort the result by ('Feature', 'Outlier Count', 'Min', 'Max', 'Median', 'Mean', 'Standard Deviation', '1st Quantile', '25th Quantile', '75th Quantile', '95th Quantile', '99th Quantile').
            Default is None, which means no sorting.
        ascending : bool, optional
            If True, sort in ascending order; if False, sort in descending order.
            Default is True (ascending).

        Returns:
        --------
        pd.DataFrame
            A DataFrame with column names as index and statistical metrics as columns.

        Examples:
        ---------
        # Calculate and display statistics without sorting
        calculate_statistics(data)

        # Calculate and display statistics sorted by 'Mean' in descending order
        calculate_statistics(data, sort_by='Mean', ascending=False)
        """
        stats = {}
        numeric_data = data.select_dtypes(include=['number'])
        for column in numeric_data.columns:
            Q1 = data[column].quantile(low_quantile)
            Q3 = data[column].quantile(high_quantile)
            Q4 = data[column].quantile(0.95)
            Q01 = data[column].quantile(0.01)
            Q99 = data[column].quantile(0.99)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # If the lower_bound and upper_bound are the same, set outlier count to 0
            if lower_bound == upper_bound:
                outlier_count = 0
            else:
                outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
                outlier_count = outliers[column].count()

            # Calculating statistical metrics
            stats[column] = {
                'Feature': column,
                'Outlier Count': outlier_count,
                'Min': data[column].min(),
                'Max': data[column].max(),
                'Median': data[column].median(),
                'Mean': data[column].mean(),
                'Standard Deviation': data[column].std(),
                '1st Quantile': Q01,
                '25th Quantile': Q1,
                '75th Quantile': Q3,
                '95th Quantile': Q4,
                '99th Quantile': Q99
            }

        # Convert the dictionary into a DataFrame
        stats_df = pd.DataFrame.from_dict(stats, orient='index')

        # Sort the DataFrame if sort_by is provided
        if sort_by in stats_df.columns:
            stats_df = stats_df.sort_values(by=sort_by, ascending=ascending)

        return stats_df

    def analyze_categorical_features(self, df: pd.DataFrame, target_column: str = None, threshold: int = 30) -> dict:
        
        """
        Analyze categorical features in a DataFrame.

        This function processes the categorical columns of the given DataFrame,
        generating a dictionary of DataFrames. Each DataFrame contains the count
        and ratio of each unique value for a categorical feature, excluding those
        with more than the specified threshold of unique values. Additionally,
        if a target column is provided, it calculates the ratio of target class=1
        within each categorical subcategory. Returns a DataFrame with features 
        that were skipped due to exceeding the unique values threshold.

        Parameters:
        df (pd.DataFrame): The input DataFrame containing the data to analyze.
        target_column (str): The name of the target column to analyze distribution for.
        threshold (int): The maximum number of unique values allowed for a feature to be included.

        Returns:
        tuple: A tuple containing:
            - Dictionary of DataFrames for each categorical feature
            - DataFrame with information about skipped features
        """

        # Dictionary to store DataFrames for each categorical feature
        categorical_dfs = {}

        # List to keep track of skipped features
        skipped_features = []

        # Identify categorical columns
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns

        # Iterate over each categorical column
        for column in categorical_columns:
            # Skip the target column if it's the same as the current column
            if column == target_column:
                continue
                
            # Check the number of unique values in the column
            unique_values_count = df[column].nunique()

            # Skip features with more than the specified threshold of unique values
            if unique_values_count > threshold:
                # Add skipped feature information to the list
                skipped_features.append({
                    'Feature': column,
                    'Alert': f"Feature '{column}' was skipped because it has {unique_values_count} unique values, exceeding the threshold of {threshold}."
                })
                continue

            # Get the counts of each unique value in the column
            value_counts = df[column].value_counts()

            # Calculate the ratio of each unique value
            total_count = len(df[column])
            value_ratios = (value_counts / total_count) * 100

            # Create a DataFrame for the current feature
            feature_df = pd.DataFrame({
                'Feature': column,
                'Subgroup': value_counts.index,
                'Count': value_counts.values,
                'Ratio': value_ratios.values.round(3)
            })
            
            # If target column is provided, calculate target ratio for each subgroup
            if target_column is not None and target_column in df.columns:
                # For each subgroup in the current feature
                for subgroup in value_counts.index:
                    # Filter data for current subgroup
                    subgroup_data = df[df[column] == subgroup]
                    
                    # Calculate ratio of target=1 within this subgroup
                    target_positive_count = len(subgroup_data[subgroup_data[target_column] == 1])
                    target_ratio = (target_positive_count / len(subgroup_data)) * 100 if len(subgroup_data) > 0 else 0
                    
                    # Add target ratio column with readable name
                    target_ratio_col_name = f"{target_column}_Ratio"
                    
                    # Add this value to the row for this subgroup
                    feature_df.loc[feature_df['Subgroup'] == subgroup, target_ratio_col_name] = round(target_ratio, 3)

            # Add the DataFrame to the dictionary
            categorical_dfs[column] = feature_df

        # Convert skipped features list to DataFrame
        skipped_features_df = pd.DataFrame(skipped_features) if skipped_features else pd.DataFrame(columns=['Feature', 'Alert'])

        return categorical_dfs, skipped_features_df

    def plot_bivariate_kde(self, data: pd.DataFrame, cols_to_plot: list = None, target_var: str = None,
                       fill: str = True, palette: str = None, **kwargs) -> None:
       
        """
        Plot bivariate kernel density estimation plots for selected numerical columns with percentage distributions.
        Uses histograms with KDE overlay and percentage scaling for easy interpretation.

        Parameters:
            data (pd.DataFrame): The DataFrame containing the data.
            cols_to_plot (list, optional): List of specific numerical columns to plot.
            target_var (str, optional): The target variable to hue the plots.
            fill (str, optional): Variable name in `data` to map plot aspects to different fill colors.
            palette (str or sequence, optional): Colors to use for the different levels of the hue variable.
            **kwargs: Additional keyword arguments.

        Returns:
            matplotlib.figure.Figure: The generated figure object
        """
     
        
        # Filter out numerical columns
        data_numeric = data.select_dtypes(include=['number']).copy()

        # Use the provided list of columns to plot or default to all numerical columns
        if cols_to_plot is None:
            cols_to_plot = data_numeric.columns.difference([target_var])
        else:
            # Validate provided columns
            cols_to_plot = [col for col in cols_to_plot if col in data_numeric.columns]
            if not cols_to_plot:
                raise ValueError("None of the specified columns are numerical or exist in the DataFrame.")

        # Number of rows and columns for subplots
        n_cols = min(2, len(cols_to_plot))  # Max 2 columns for better readability
        n_rows = (len(cols_to_plot) + n_cols - 1) // n_cols  # ceiling division

        # Create the figure and subplots with better spacing
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 12, n_rows * 6))
        fig.suptitle('Distribution Analysis - Percentage View', fontsize=16, fontweight='bold', y=0.98)

        # Ensure axes is always an array
        if n_rows * n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if hasattr(axes, '__len__') else [axes]
        else:
            axes = axes.flatten()

        # Define colors for consistent plotting
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        # Iterate over numerical columns
        for i, col in enumerate(cols_to_plot):
            ax = axes[i]
            
            # Check if the column has non-zero variance
            if data[col].var() > 0:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    
                    # Create percentage-based distribution plot
                    if target_var and data[target_var].nunique() <= 10:
                        # Get unique categories in target variable
                        categories = data[target_var].unique()
                        
                        # Plot histogram with percentage for each category
                        for idx, category in enumerate(categories):
                            subset_data = data[data[target_var] == category][col].dropna()
                            if len(subset_data) > 0:
                                # Calculate histogram with percentage
                                counts, bins = np.histogram(subset_data, bins=30, density=False)
                                # Convert counts to percentages
                                percentages = (counts / len(data)) * 100
                                
                                # Plot histogram bars
                                ax.hist(subset_data, bins=30, alpha=0.6, 
                                    color=colors[idx % len(colors)], 
                                    label=f'{category}', 
                                    weights=np.ones(len(subset_data)) / len(data) * 100,
                                    edgecolor='white', linewidth=0.5)
                                
                                # Add KDE line overlay
                                try:
                                    sns.kdeplot(data=subset_data, ax=ax, color=colors[idx % len(colors)],
                                            linewidth=2.5, alpha=0.8)
                                    # Scale KDE to match percentage scale
                                    lines = ax.get_lines()
                                    if lines:
                                        latest_line = lines[-1]
                                        y_data = latest_line.get_ydata()
                                        # Scale KDE to match histogram percentage scale
                                        y_scaled = y_data * (percentages.max() / y_data.max()) if y_data.max() > 0 else y_data
                                        latest_line.set_ydata(y_scaled)
                                except:
                                    pass  # Skip KDE if it fails
                    else:
                        # Plot single distribution
                        col_data = data[col].dropna()
                        if len(col_data) > 0:
                            # Plot histogram with percentage
                            ax.hist(col_data, bins=30, alpha=0.7, color='steelblue',
                                weights=np.ones(len(col_data)) / len(col_data) * 100,
                                edgecolor='white', linewidth=0.5)
                            
                            # Add KDE line overlay
                            try:
                                sns.kdeplot(data=col_data, ax=ax, color='darkblue',
                                        linewidth=2.5, alpha=0.9)
                                # Scale KDE to match percentage scale
                                lines = ax.get_lines()
                                if lines:
                                    latest_line = lines[-1]
                                    y_data = latest_line.get_ydata()
                                    hist_max = ax.patches[0].get_height() if ax.patches else 1
                                    y_scaled = y_data * (hist_max / y_data.max()) if y_data.max() > 0 else y_data
                                    latest_line.set_ydata(y_scaled)
                            except:
                                pass  # Skip KDE if it fails
                    
                    # Customize the plot appearance
                    ax.set_title(f'Distribution of {col}', fontsize=14, fontweight='bold', pad=20)
                    ax.set_xlabel(f'{col}', fontsize=12, fontweight='semibold')
                    ax.set_ylabel('Percentage of Total Data (%)', fontsize=12, fontweight='semibold')
                    
                    # Add grid for better readability
                    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
                    ax.set_facecolor('white')
                    
                    # Format y-axis to show percentage
                    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1f}%'))
                    
                    # Customize legend if present
                    if target_var and data[target_var].nunique() <= 10:
                        legend = ax.legend(
                            title=target_var,
                            title_fontsize=11,
                            fontsize=10,
                            loc='upper right',
                            frameon=True,
                            fancybox=True,
                            shadow=True,
                            framealpha=0.9
                        )
                        legend.get_title().set_fontweight('bold')
                    
                    # Add statistics text box
                    mean_val = data[col].mean()
                    std_val = data[col].std()
                    median_val = data[col].median()
                    stats_text = f'Mean: {mean_val:.2f}\nMedian: {median_val:.2f}\nStd: {std_val:.2f}'
                    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                        fontsize=9, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
                    
            else:
                # Handle columns with zero variance
                ax.text(0.5, 0.5, f'No variation in {col}\n(constant value)', 
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=12, bbox=dict(boxstyle='round', facecolor='lightyellow'))
                ax.set_title(f'{col} - No Variation', fontsize=14)

        # Hide any unused subplots
        for i in range(len(cols_to_plot), len(axes)):
            fig.delaxes(axes[i])

        # Adjust layout for better spacing
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        return fig
    def plot_target(self, data: pd.DataFrame, target_column: str, plot_type: str = "count",
                    figsize: tuple[int, int] = (10, 6),
                    title: str = "Target Distribution", xlabel: str = "", ylabel: str = "") -> plt.Figure:
            """
            Optimized function to plot the target column data based on the specified plot type ('pie' or 'count').

            Parameters:
            ----------
            - data: DataFrame containing the data to be plotted.
            - target_column: Name of the column in `data` to be plotted.
            - plot_type: Type of plot to generate ('pie' or 'count').
            - figsize: Tuple specifying the width and height of the figure (default: (10, 6)).
            - title: Optional title for the plot (default: "Target Distribution").
            - xlabel: Label for the x-axis (default: "").
            - ylabel: Label for the y-axis (default: "").

            Returns:
            -------
            - fig: Matplotlib Figure object containing the generated plot.
            """

            fig, ax = plt.subplots(figsize=figsize)

            # Generate a count plot
            sns.countplot(data=data, y=target_column, palette="bright", ax=ax)

            total_count = len(data)
            percentages = data[target_column].value_counts(normalize=True) * 100

            # Add percentages as text on the bars
            for i, v in enumerate(percentages):
                ax.text(v + 0.5, i, f'{v:.1f}%', color='black', va='center')

            ax.set_title(title if title else f"Target column: {target_column} visualization", fontsize=16,
                        fontweight='bold', color='darkblue')
            ax.set_xlabel(xlabel, fontsize=14, fontweight='bold', color='gray')
            ax.set_ylabel(ylabel, fontsize=14, fontweight='bold', color='gray')

            plt.tight_layout()  # Adjust layout to prevent overlaps

            return fig


    def calculate_correlation_matrix(self, df: pd.DataFrame, target: str = None, 
                                 with_feature_names: bool = True, variance_threshold: float = 0.1, 
                                 method: str = 'pearson' , apply_variance  : bool = True ) -> pd.DataFrame:
        """
        Calculate the correlation matrix for numerical columns in a DataFrame, applying a variance threshold, 
        and set default display settings. Supports different correlation methods: Pearson, Kendall, or Spearman.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame containing the data.
        target : str, optional
            The target variable to consider during numeric conversion.
        with_feature_names : bool, optional
            If True, the correlation matrix will include feature names as both index and columns. Default is True.
        variance_threshold : float, optional
            The threshold for variance below which features will be excluded from the correlation matrix. Default is 0.1.
        method : str, optional
            The method to use for calculating correlation ('pearson', 'kendall', 'spearman'). Default is 'pearson'.

        Returns:
        --------
        pd.DataFrame
            The correlation matrix, including feature names as both index and columns.

        Examples:
        ---------
        # Calculate and display the correlation matrix including feature names using Pearson correlation
        calculate_correlation_matrix(df)

        # Calculate the correlation matrix using Spearman correlation
        calculate_correlation_matrix(df, method='spearman')

        # Calculate the correlation matrix with a specific variance threshold and Kendall correlation
        calculate_correlation_matrix(df, variance_threshold=0.1, method='kendall')
        """
        # Create a deep copy of the DataFrame to avoid modifying the original data
        copy = df.copy(deep=True)

        # Convert to numeric data types if necessary
        copy = self.convert_to_numeric(copy, target)

        # Select numerical columns
        numerical_data = copy.select_dtypes(include=["number"])
        
        
        # Calculate the variance for each numerical feature
        variances = numerical_data.var()
        # Filter features based on the variance threshold
        high_variance_features = variances[variances > variance_threshold].index
        filtered_numerical_data = numerical_data[high_variance_features]

        if apply_variance:

            # Calculate the correlation matrix using the specified method
            correlation_matrix = filtered_numerical_data.corr(method=method)
        else  : 
            correlation_matrix = numerical_data.corr(method=method)



        # Optionally format the matrix to emphasize feature names (though feature names are included by default)
        if not with_feature_names:
            correlation_matrix = correlation_matrix.reset_index()
            correlation_matrix.columns = ['Feature'] + list(correlation_matrix.columns[1:])

        return correlation_matrix



    def calculate_correlation_matrix_with_features_names(self, df: pd.DataFrame, target: str,
                                                        with_feature_names: bool = True, variance_threshold: float = 0.0,
                                                        method:str = "pearson" ,  apply_variance  : bool = True) -> pd.DataFrame:
        """
        Calculate the correlation matrix for numerical columns in a DataFrame and apply default display settings,
        with a variance threshold for filtering features.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame containing the data.
        target : str
            The target variable to consider during numeric conversion.
        with_feature_names : bool, optional
            If True, the correlation matrix will include feature names as both index and columns. Default is True.
        variance_threshold : float, optional
            The threshold for variance below which features will be excluded from the correlation matrix. Default is 0.01.

        Returns:
        --------
        pd.DataFrame
            The correlation matrix, including feature names as both index and columns.

        Examples:
        ---------
        # Calculate and display the correlation matrix including feature names
        calculate_correlation_matrix_with_features_names(df, target)

        # Calculate the correlation matrix, explicitly ensuring feature names are included
        calculate_correlation_matrix_with_features_names(df, target, with_feature_names=True)

        # Calculate the correlation matrix with a specific variance threshold
        calculate_correlation_matrix_with_features_names(df, target, variance_threshold=0.1)
        """
        # Create a deep copy of the DataFrame to avoid modifying the original data
        copy = df.copy(deep=True)

        # Convert to numeric data types if necessary
        copy = self.convert_to_numeric(copy, target)

        # Select numerical columns
        numerical_data = copy.select_dtypes(include=["number"])

        # Calculate the variance for each numerical feature
        variances = numerical_data.var()

        # Filter features based on the variance threshold
        high_variance_features = variances[variances > variance_threshold].index
        filtered_numerical_data = numerical_data[high_variance_features]

        if apply_variance:
            # Calculate the correlation matrix using the specified method
            correlation_matrix = filtered_numerical_data.corr(method=method)
        else  : 
            correlation_matrix = numerical_data.corr(method=method)


        # Add feature names column if required
        if with_feature_names:
            correlation_matrix = correlation_matrix.reset_index()
            correlation_matrix.columns = ['Feature'] + list(correlation_matrix.columns[1:])
        else:
            correlation_matrix = correlation_matrix.reset_index()
            correlation_matrix.columns = ['Feature'] + list(correlation_matrix.columns[1:])

        return correlation_matrix
    

    

    def convert_to_categorical(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """
        Converts a numeric target column to a categorical type based on its unique values.

        Parameters:
        -----------
        df (pd.DataFrame): The DataFrame containing the target column.
        target_column (str): The name of the target column to be converted.

        Returns:
        --------
        pd.DataFrame: DataFrame with the target column converted to categorical if it was numeric.
        """
        # Check if the target_column exists in the DataFrame
        if target_column not in df.columns:
            raise ValueError(f"Column '{target_column}' does not exist in the DataFrame.")

        # Check if the target column is numeric
        if pd.api.types.is_numeric_dtype(df[target_column]):
            # Convert to categorical type
            df[target_column] = df[target_column].astype('category')

        return df

    def convert_to_numeric(self, df, target_column):

        """
        Converts a categorical target column to numeric codes.
        
        Parameters:
        df (pd.DataFrame): The DataFrame containing the target column.
        target_column (str): The name of the target column to be converted.
        
        Returns:
        pd.DataFrame: DataFrame with the target column converted to numeric codes if it was categorical.
        """

        # Check if the target column is either categorical or of type 'object' (string categories)
        if pd.api.types.is_categorical_dtype(df[target_column]) or pd.api.types.is_object_dtype(df[target_column]):
            # Convert the column to 'category' dtype and then use `.cat.codes` to assign numeric codes
            df[target_column] = df[target_column].astype('category').cat.codes

        return df

    def plot_correlation_heatmap(self, correlation_matrix, title='Correlation Heatmap', cmap='coolwarm',
                                 annot=True) -> plt.Figure:
        """
        Generates a heatmap from a correlation matrix.
        
        Parameters:
        correlation_matrix (pd.DataFrame): A Pandas DataFrame containing the correlation matrix.
        title (str): Title of the heatmap plot. Default is 'Correlation Heatmap'.
        cmap (str): Colormap to use for the heatmap. Default is 'coolwarm'.
        annot (bool): Whether to annotate the heatmap with correlation coefficients. Default is True.
        
        Returns:
        plt.Figure: The matplotlib Figure object containing the heatmap plot.
        """

        # Check if the input is a Pandas DataFrame
        if not isinstance(correlation_matrix, pd.DataFrame):
            raise TypeError("The correlation matrix must be a Pandas DataFrame.")

        # Verify that the DataFrame is square
        if correlation_matrix.shape[0] != correlation_matrix.shape[1]:
            raise ValueError("The correlation matrix must be square.")

        # Create the heatmap
        fig = plt.figure(figsize=(20, 16))  # Set the figure size and capture the figure object
        sns.heatmap(correlation_matrix,
                    annot=annot,
                    cmap=cmap,
                    fmt='.2f',  # Format for annotations
                    linewidths=0.5,  # Width of the lines that will divide each cell
                    linecolor='black')  # Color of the lines dividing cells

        # Set plot title and labels
        plt.title(title, size=15)
        plt.xlabel('Variables')
        plt.ylabel('Variables')
        plt.tight_layout()  # Adjust layout to prevent overlap

        # Return the figure object
        return fig  # Return the captured figure object

    def cramers_v(self, x, y):

        """
        Computes Cramér's V statistic for association between two categorical variables.
        
        Parameters:
        - x (pd.Series): The first categorical variable.
        - y (pd.Series): The second categorical variable.
        
        Returns:
        - float: Cramér's V statistic.
        """
        crosstab = pd.crosstab(x, y)
        chi2_stat, p_value, dof, expected = chi2_contingency(crosstab, correction=False)
        n = crosstab.sum().sum()
        return np.sqrt(chi2_stat / (n * min(crosstab.shape) - 1))

    def categorical_correlation_matrix(self, data: pd.DataFrame, target: str) -> pd.DataFrame:

        """
        Creates a correlation matrix for categorical features in the DataFrame `data` using Cramér's V.
        
        Parameters:
        - data (pd.DataFrame): The input DataFrame containing the categorical data.
        
        Returns:
        - pd.DataFrame: A DataFrame containing Cramér's V statistics for each pair of categorical features.
        """
        copy = self.convert_to_categorical(data, target)
        # Select categorical columns
        categorical_features = copy.select_dtypes(include=['object', 'category'])

        # Initialize an empty DataFrame for the correlation matrix
        corr_matrix = pd.DataFrame(index=categorical_features.columns, columns=categorical_features.columns)

        # Compute Cramér's V for each pair of categorical features
        for i in range(len(categorical_features.columns)):
            for j in range(i, len(categorical_features.columns)):
                feature_i = categorical_features.columns[i]
                feature_j = categorical_features.columns[j]
                if feature_i == feature_j:
                    corr_matrix.loc[feature_i, feature_j] = np.nan  # Avoid self-correlation
                else:
                    corr_value = self.cramers_v(categorical_features[feature_i], categorical_features[feature_j])
                    corr_matrix.loc[feature_i, feature_j] = corr_value
                    corr_matrix.loc[feature_j, feature_i] = corr_value  # Symmetric matrix

        # Convert all values to numeric, coercing errors to NaNs, then fill NaNs with 0
        corr_matrix = corr_matrix.apply(pd.to_numeric, errors='coerce')
        corr_matrix.fillna(1, inplace=True)

        return corr_matrix

    def categorical_correlation_matrix_feature_names(self, data: pd.DataFrame, target) -> pd.DataFrame:
        """
        Creates a correlation matrix for categorical features in the DataFrame `data` using Cramér's V.
        
        Parameters:
        - data (pd.DataFrame): The input DataFrame containing the categorical data.
        
        Returns:
        - pd.DataFrame: A DataFrame containing Cramér's V statistics for each pair of categorical features.
        """
        copy = self.convert_to_categorical(data, target)

        # Select categorical columns
        categorical_features = data.select_dtypes(include=['object', 'category'])

        # Initialize an empty DataFrame for the correlation matrix
        corr_matrix = pd.DataFrame(index=categorical_features.columns, columns=categorical_features.columns)

        # Compute Cramér's V for each pair of categorical features
        for i in range(len(categorical_features.columns)):
            for j in range(i, len(categorical_features.columns)):
                feature_i = categorical_features.columns[i]
                feature_j = categorical_features.columns[j]
                if feature_i == feature_j:
                    corr_matrix.loc[feature_i, feature_j] = np.nan  # Avoid self-correlation
                else:
                    corr_value = self.cramers_v(categorical_features[feature_i], categorical_features[feature_j])
                    corr_matrix.loc[feature_i, feature_j] = corr_value
                    corr_matrix.loc[feature_j, feature_i] = corr_value  # Symmetric matrix

        # Convert all values to numeric, coercing errors to NaNs, then fill NaNs with 0
        corr_matrix = corr_matrix.apply(pd.to_numeric, errors='coerce')
        corr_matrix.fillna(1, inplace=True)

        # Add a column for feature names
        corr_matrix = corr_matrix.reset_index()
        corr_matrix.columns.name = None  # Remove the name of the columns axis for better clarity
        corr_matrix.insert(0, 'Feature Name', corr_matrix['index'])
        corr_matrix = corr_matrix.drop(columns='index')

        return corr_matrix

    def plot_heatmap(self, matrix: pd.DataFrame, title: str = 'Categorical Correlation Heatmap') -> plt.Figure:
        """
        Plots a heatmap of the correlation matrix.

        Parameters:
        ----------
        - matrix (pd.DataFrame): The correlation matrix to plot.
        - title (str): The title of the heatmap.

        Returns:
        -------
        - plt.Figure: The matplotlib Figure object containing the heatmap plot.
        """
        fig = plt.figure(figsize=(20, 14))  # Create the figure
        sns.heatmap(matrix, annot=True, cmap='coolwarm', center=0, vmin=0, vmax=1, fmt='.2f', linewidths=0.5)
        plt.title(title, fontsize=20)
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.yticks(rotation=0, fontsize=12)
        plt.tight_layout()  # Adjust layout

        plt.close(fig)  # Close the figure to free memory

        return fig  # Return the figure object




    def bar_plot(self, x, y, title="Cool Bar Plot", xlabel="X-axis", ylabel="Y-axis", rotation=80, annotation=True,
                 annotation_as_int=True, round_decimals=3, fig_size=(10, 6), threshold=None, log_scale=True,
                 annotation_rotation=45):
            """
            Creates and displays a bar plot with cool styling using matplotlib and handles both positive and negative values.
            """


            # Create a figure and axis with a custom size
            fig, ax = plt.subplots(figsize=fig_size)

            # Separate positive and negative values for handling
            y_positive = [val if val > 0 else 0 for val in y]
            y_negative = [val if val < 0 else 0 for val in y]

            # Create bar plots for positive and negative values separately
            bars_positive = ax.bar(x, y_positive, color='#1f77b4', edgecolor='black', label='Positive')
            bars_negative = ax.bar(x, y_negative, color='red', edgecolor='black', label='Negative')

            # Add labels, title, and grid with cool styling
            ax.set_title(title, fontsize=20, fontweight='bold', color='#333333')
            ax.set_xlabel(xlabel, fontsize=16, fontweight='bold', color='#555555', fontfamily='sans-serif')
            ax.set_ylabel(ylabel, fontsize=16, fontweight='bold', color='#555555', fontfamily='sans-serif')

            # Customize the grid and background
            ax.grid(True, which='both', axis='y', linestyle='--', linewidth=0.7, color='#aaaaaa')
            ax.set_facecolor('#f0f0f0')

            # Rotate x-axis labels
            plt.xticks(rotation=rotation, fontsize=12, fontweight='bold', fontfamily='sans-serif')

            # Apply log scale to the y-axis for positive values if specified
            if log_scale and np.any(np.array(y_positive) > 0):
                ax.set_yscale('symlog', linthresh=0.1)  # Use symlog for handling both negative and positive values

            # Add annotations if the parameter is set to True
            if annotation:
                for bar in bars_positive:
                    height = bar.get_height()
                    if height > 0:  # Only annotate positive bars
                        label = f'{int(height)}' if annotation_as_int else f'{round(height, round_decimals)}'
                        ax.annotate(label,
                                    xy=(bar.get_x() + bar.get_width() / 2, height),
                                    xytext=(0, 3),
                                    textcoords="offset points",
                                    ha='center', va='bottom',
                                    fontsize=12, fontweight='bold', color='#333333',
                                    rotation=annotation_rotation)

                for bar in bars_negative:
                    height = bar.get_height()
                    if height < 0:  # Only annotate negative bars
                        label = f'{int(height)}' if annotation_as_int else f'{round(height, round_decimals)}'
                        ax.annotate(label,
                                    xy=(bar.get_x() + bar.get_width() / 2, height),
                                    xytext=(0, -10),
                                    textcoords="offset points",
                                    ha='center', va='top',
                                    fontsize=12, fontweight='bold', color='#333333',
                                    rotation=annotation_rotation)

            # Plot the threshold line if the parameter is provided
            if threshold is not None:
                ax.axhline(y=threshold, color='green', linestyle='--', linewidth=3, label=f'Threshold: {threshold}')
                ax.legend()

            # Adjust layout to prevent overlaps
            plt.tight_layout()

            return fig
    
    
    def multi_metric_categorical_plot(self, feature_df, feature_name, target_column):
        """
        Creates a multi-subplot plot showing Count, Sum, and Mean of target variable for categorical features.
        
        Parameters:
        -----------
        feature_df : DataFrame
            DataFrame containing the categorical feature analysis results
        feature_name : str
            Name of the categorical feature being analyzed
        target_column : str
            Name of the target column
            
        Returns:
        --------
        matplotlib.figure.Figure
            The created figure object
        """
        # Create figure with 3 subplots (3 rows, 1 column)
        fig, axes = plt.subplots(3, 1, figsize=(20, 15))
        fig.suptitle(f'Multi-Metric Analysis of {feature_name}', fontsize=16, fontweight='bold')
        
        # Extract data from feature_df
        categories = feature_df["Subgroup"]
        counts = feature_df["Count"]
        
        # Calculate sum and mean if target-related columns exist
        if f"Sum_{target_column}" in feature_df.columns:
            sums = feature_df[f"Sum_{target_column}"]
        else:
            sums = feature_df["Count"]  # Fallback to counts if sum not available
        
        if f"Mean_{target_column}" in feature_df.columns:
            means = feature_df[f"Mean_{target_column}"]
        else:
            means = [1.0] * len(categories)  # Fallback to 1.0 if mean not available
        
        # Color schemes for different metrics
        colors = ['#FF4444', '#44AA44', '#4444FF']
        
        # Plot 1: Count
        bars1 = axes[0].bar(categories, counts, color=colors[0], alpha=0.8)
        axes[0].set_title(f'Count of {feature_name}', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Count', fontsize=12)
        axes[0].grid(True, alpha=0.3)
        
        # Add value annotations on bars
        for bar in bars1:
            height = bar.get_height()
            axes[0].annotate(f'{int(height)}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom',
                            rotation=80, fontsize=10)
        
        # Plot 2: Sum
        bars2 = axes[1].bar(categories, sums, color=colors[1], alpha=0.8)
        axes[1].set_title(f'Sum of {target_column}', fontsize=14, fontweight='bold')
        axes[1].set_ylabel(f'Sum of {target_column}', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        
        # Add value annotations on bars
        for bar in bars2:
            height = bar.get_height()
            axes[1].annotate(f'{int(height)}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom',
                            rotation=80, fontsize=10)
        
        # Plot 3: Mean (as line plot to match your image)
        axes[2].bar(categories, means, color=colors[2], alpha=0.8)
        axes[2].set_title(f'Mean of {target_column}', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Subgroups', fontsize=12)
        axes[2].set_ylabel(f'Rate of {target_column}', fontsize=12)
        axes[2].grid(True, alpha=0.3)
        axes[2].set_ylim(0, max(means) * 1.1)  # Set y-limit for better visualization
        
        # Rotate x-axis labels for all subplots
        for ax in axes:
            ax.tick_params(axis='x', rotation=45, labelsize=10)
            ax.tick_params(axis='y', labelsize=10)
        
        # Adjust layout to prevent overlap
        plt.tight_layout()
        
        return fig

    def plot_boxplots(self, dataframe):
        """
        Plots box plots for all numerical columns in the dataframe to visualize outliers.

        Parameters:
        dataframe (pd.DataFrame): The input dataframe containing the data.

        Returns:
        matplotlib.figure.Figure: The figure object containing the box plots.
        """
        # Select only numerical columns
        numeric_columns = dataframe.select_dtypes(include=['number']).columns

        # Define the number of subplots needed
        num_columns = len(numeric_columns)
        num_rows = (num_columns + 2) // 3  # 3 columns per row

        # Create a figure for the plots
        fig, axes = plt.subplots(num_rows, 3, figsize=(15, 5 * num_rows))

        # Flatten axes for easier iteration
        axes = axes.flatten()

        for i, column in enumerate(numeric_columns):
            sns.boxplot(y=dataframe[column], ax=axes[i])
            axes[i].set_title(f'Boxplot of {column}')

        # Hide any unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()  # Adjust subplots to fit into figure area.

        # Close the figure to prevent warnings about too many open figures
        plt.close(fig)

        return fig  # Return the figure object

    def variance_analysis(self, df, features=None, annotate=False, annotation_rotation: int = 80,
                          log_scale: bool = False, fig_size: tuple = (20, 12),
                          title: str = "Variance Analysis for Numerical Features"):
        """
        Analyzes the variance of specified numerical features in the given DataFrame.
        If no features are specified, analyzes all numerical features.

        Parameters:
        df (pd.DataFrame): The input DataFrame containing numerical features.
        features (list of str, optional): A list of the names of the numerical features to analyze. If None, analyzes all numerical features.
        annotate (bool): Whether to add annotations to the plot.

        Returns:
        pd.DataFrame: A DataFrame with columns 'Feature' and 'Variance' showing the variance of each numerical feature, or just the specified features.
        plt.Figure: The figure object for the plot.
        """

        # Ensure that the DataFrame contains numerical features only
        numerical_features = df.select_dtypes(include=[np.number])

        if features:
            # Validate that all specified features are in the DataFrame
            missing_features = [feature for feature in features if feature not in numerical_features.columns]
            if missing_features:
                raise ValueError(f"Features {missing_features} not found in the DataFrame.")
            # Calculate variance for the specified features only
            variance_series = numerical_features[features].var()
            variance_df = pd.DataFrame({
                'Feature': variance_series.index,
                'Variance': variance_series.values
            })
        else:
            # Calculate variance for each numerical feature
            variance_series = numerical_features.var()

            # Create a DataFrame to store the variance of each feature
            variance_df = pd.DataFrame({
                'Feature': variance_series.index,
                'Variance': variance_series.values
            })

            # Sort the DataFrame by variance in descending order for better readability
            variance_df = variance_df.sort_values(by='Variance', ascending=False).reset_index(drop=True)

        # Use the bar_plot function to create the plot with customized styling
        fig = self.bar_plot(

            x=variance_df['Feature'],
            y=variance_df['Variance'],
            title=title,
            xlabel='Feature',
            ylabel='Variance',
            rotation=90,
            annotation=annotate,
            fig_size=(20, 12),
            log_scale=log_scale,  # Set to True if variance values span several orders of magnitude
            annotation_rotation=annotation_rotation,
            annotation_as_int=False

        )

        return variance_df, fig
    
    def plot_target_percentage_by_category(self, data, categorical_feature, target_feature, figsize=(15, 8)):
        """
        Creates a bar plot showing the percentage distribution of target feature categories
        for each subcategory of the categorical feature.
        
        Parameters:
        -----------
        data : DataFrame
            The dataset containing the features
        categorical_feature : str
            Name of the categorical feature to analyze
        target_feature : str
            Name of the target feature
        figsize : tuple
            Figure size (width, height)
            
        Returns:
        --------
        matplotlib.figure.Figure
            The created figure object
        """
        # Create a copy of relevant data to avoid modifying original
        plot_data = data[[categorical_feature, target_feature]].copy()
        
        # Remove any missing values
        plot_data = plot_data.dropna()
        
        # Calculate percentage distribution for each category
        crosstab = pd.crosstab(plot_data[categorical_feature], plot_data[target_feature])
        percentage_df = crosstab.div(crosstab.sum(axis=1), axis=0) * 100
        
        # Create the figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get unique target values and create color palette
        target_values = sorted(plot_data[target_feature].unique())
        colors = plt.cm.Set1(np.linspace(0, 1, len(target_values)))
        
        # Create stacked bar plot
        bottom = np.zeros(len(percentage_df.index))
        
        for i, target_val in enumerate(target_values):
            if target_val in percentage_df.columns:
                values = percentage_df[target_val].values
                bars = ax.bar(percentage_df.index, values, bottom=bottom, 
                            label=f'{target_feature}_{target_val}', 
                            color=colors[i], alpha=0.8, edgecolor='white', linewidth=0.5)
                
                # Add percentage labels on bars (only if percentage > 5% to avoid clutter)
                for j, (bar, value) in enumerate(zip(bars, values)):
                    if value > 5:  # Only show label if percentage > 5%
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., bottom[j] + height/2.,
                            f'{value:.1f}%', ha='center', va='center', 
                            fontweight='bold', fontsize=10, color='white')
                
                bottom += values
        
        # Customize the plot
        ax.set_title(f'Target Distribution Percentage by {categorical_feature}', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel(f'{categorical_feature} Categories', fontsize=12, fontweight='bold')
        ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
        
        # Set y-axis to show 0-100%
        ax.set_ylim(0, 100)
        
        # Add horizontal grid lines for better readability
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Customize legend
        ax.legend(title=f'{target_feature} Values', 
                bbox_to_anchor=(1.05, 1), loc='upper left',
                frameon=True, fancybox=True, shadow=True)
        
        # Rotate x-axis labels if there are many categories
        plt.xticks(rotation=45, ha='right')
        
        # Add percentage reference lines
        for y in [25, 50, 75]:
            ax.axhline(y=y, color='gray', linestyle=':', alpha=0.5, linewidth=0.8)
        
        # Adjust layout to prevent legend cutoff
        plt.tight_layout()
        
        return fig

    def plot_percentage_distribution(self, data: pd.DataFrame, cols_to_plot: List[str] = None, 
                               target_var: str = None, palette: str = "Set1", 
                               bins: int = 50, **kwargs) -> plt.Figure:
        """
        Plot percentage distribution histograms for selected numerical columns.
        
        Parameters:
            data (pd.DataFrame): The DataFrame containing the data
            cols_to_plot (list, optional): List of numerical columns to plot. 
                                        If None, all numerical columns will be used
            target_var (str, optional): Target variable for grouping/coloring
            palette (str, optional): Color palette for different groups
            bins (int, optional): Number of bins for histogram. Default is 50
            **kwargs: Additional arguments for customization
        
        Returns:
            plt.Figure: The matplotlib figure object
        """
        
        # Get numerical columns only
        numeric_data = data.select_dtypes(include=['number']).copy()
        
        # Determine columns to plot
        if cols_to_plot is None:
            cols_to_plot = [col for col in numeric_data.columns if col != target_var]
        else:
            # Validate columns exist and are numerical
            cols_to_plot = [col for col in cols_to_plot if col in numeric_data.columns]
            if not cols_to_plot:
                raise ValueError("No valid numerical columns found in the specified list")
        
        # Setup subplot grid
        n_cols = 2 if len(cols_to_plot) > 1 else 1
        n_rows = (len(cols_to_plot) + n_cols - 1) // n_cols
        
        # Create figure with better sizing
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 8, n_rows * 6))
        
        # Handle single subplot case
        if n_rows * n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
        else:
            axes = axes.flatten()
        
        # Plot each column
        for idx, col in enumerate(cols_to_plot):
            ax = axes[idx]
            
            # Check if column has variation
            if data[col].var() == 0:
                ax.text(0.5, 0.5, f'No variation in {col}', 
                    ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title(f'{col} - No Variation')
                continue
            
            # Remove missing values for plotting
            plot_data = data[[col, target_var]].dropna() if target_var else data[[col]].dropna()
            
            if target_var and target_var in data.columns:
                # Plot with target variable grouping
                unique_targets = plot_data[target_var].unique()
                
                # Limit number of categories for clarity
                if len(unique_targets) > 10:
                    # Get top 10 most frequent categories
                    top_categories = plot_data[target_var].value_counts().head(10).index
                    plot_data = plot_data[plot_data[target_var].isin(top_categories)]
                    ax.text(0.02, 0.98, f'Showing top 10 categories only', 
                        transform=ax.transAxes, fontsize=8, va='top',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
                
                # Create percentage histograms for each group
                for i, category in enumerate(plot_data[target_var].unique()):
                    subset = plot_data[plot_data[target_var] == category][col]
                    weights = np.ones(len(subset)) / len(subset) * 100  # Convert to percentage
                    
                    ax.hist(subset, bins=bins, alpha=0.6, 
                        weights=weights, label=str(category),
                        color=plt.cm.get_cmap(palette)(i / len(plot_data[target_var].unique())))
            else:
                # Plot without grouping
                weights = np.ones(len(plot_data[col])) / len(plot_data[col]) * 100
                ax.hist(plot_data[col], bins=bins, alpha=0.7, weights=weights, 
                    color='steelblue', edgecolor='black', linewidth=0.5)
            
            # Customize the plot
            ax.set_xlabel(f'{col}', fontsize=11, fontweight='bold')
            ax.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
            ax.set_title(f'Distribution of {col}', fontsize=12, fontweight='bold', pad=15)
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # Add percentage formatting to y-axis
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}%'))
            
            # Add legend if target variable is used
            if target_var and target_var in data.columns:
                ax.legend(title=target_var, loc='upper right', fontsize=9)
            
            # Add statistics text box
            mean_val = plot_data[col].mean()
            std_val = plot_data[col].std()
            stats_text = f'Mean: {mean_val:.2f}\nStd: {std_val:.2f}'
            ax.text(0.02, 0.85, stats_text, transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='lightblue', alpha=0.8))
        
        # Hide unused subplots
        for i in range(len(cols_to_plot), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout(pad=2.0)
        return fig

    def create_project_dataframe(self, project_name, start_analysis_time, end_time, total_analysis_time):

        """

        Creates a pandas DataFrame containing project information such as the project name,
        analysis start time, end time, and the total analysis duration.

        Parameters:
        -----------
        project_name : str
            The name of the project.
        start_analysis_time : datetime
            The timestamp when the analysis started.
        end_time : datetime
            The timestamp when the analysis ended.
        total_analysis_time : timedelta
            The total duration of the analysis.

        Returns:
        --------
        pd.DataFrame
            A DataFrame containing the project information, including the project name,
            analysis start and end times, and the total time taken for exploratory data analysis (EDA).
        """

        # Define a dictionary to hold the project information
        project_data = {
            "Project Name": [project_name],  # Name of the project
            "Analysis Start Time": [start_analysis_time],  # Time the analysis started
            "Analysis End Time": [end_time],  # Time the analysis ended
            "Total Analysis Duration": [total_analysis_time]  # Total duration of the analysis
        }

        # Create a pandas DataFrame from the dictionary
        project_df = pd.DataFrame(project_data)

        # Return the DataFrame containing the project details
        return project_df

    def create_datetime_features(self, df, datetime_column, user_preference=["Year", "Month", "Week", "Day"]):
        """
        Extracts specified datetime features from a column in a pandas DataFrame.

        Parameters:
        df (pd.DataFrame): The DataFrame containing the datetime column.
        datetime_column (str): The name of the column containing datetime values.
        user_preference (list): A list of features to extract, such as "Year", "Month", "Week", "Day".

        Returns:
        pd.DataFrame: The DataFrame with new datetime features added.
        """
        # Convert the column to datetime format
        df[datetime_column] = pd.to_datetime(df[datetime_column])

        # Initialize a dictionary to hold the new features
        features = {}

        # Add features based on user preferences
        if 'Year' in user_preference:
            features['Year'] = df[datetime_column].dt.year
        if 'Month' in user_preference:
            features['Month'] = df[datetime_column].dt.month
        if 'Week' in user_preference:
            features['Week'] = (df[datetime_column].dt.day - 1) // 7 + 1
        if 'Day' in user_preference:
            features['Day'] = df[datetime_column].dt.day

        # Add the features to the DataFrame
        for key, value in features.items():
            df[key] = value

        return df


    def plot_target_per_month(self, df, timestamp_col, target_col, num_months=0,
                              log_scale=True, ascending=True):
        """
        Generates a line plot showing the count of occurrences for a specified target column, grouped by month.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame containing the data to be plotted.
        timestamp_col : str
            The name of the column containing timestamp information.
        target_col : str
            The column whose occurrences are to be counted and plotted.
        num_months : int, optional
            The number of most recent months to include in the plot. Defaults to 0 (all available months).
        log_scale : bool, optional
            If True, applies logarithmic scaling to the y-axis if there are any non-positive counts. Defaults to False.
        ascending : bool, optional
            If True, sorts the x-axis in ascending order. Defaults to False.

        Returns:
        --------
        plotly.graph_objects.Figure
            The generated plot figure.
        """

        # Copying the DataFrame to avoid modifying the original data
        df = df.copy()

        if timestamp_col is None or timestamp_col not in df.columns:
            return pd.DataFrame({"Alert": [
                f"Error: The timestamp column '{timestamp_col}' is not provided or does not exist in the DataFrame."]})

        # Ensure the timestamp column is in datetime format
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')

        # Extract year and month for grouping
        df['year_month'] = df[timestamp_col].dt.to_period('M').astype(str)

        # Retrieve available months in descending order
        available_months = sorted(df['year_month'].unique(), reverse=True)
        total_available_months = len(available_months)

        # Validate `num_months` parameter
        if num_months < 0:
            return pd.DataFrame({"alert": ["The number of months cannot be negative."]})
        if num_months > total_available_months:
            return pd.DataFrame({"alert": ["The number of months exceeds the available months in the dataset."]})

        # Determine months to include in the plot
        selected_months = available_months if num_months == 0 else available_months[:num_months]

        # Filter the DataFrame for the selected months
        df_filtered = df[df['year_month'].isin(selected_months)]

        # Count occurrences of the target column per month
        monthly_counts = df_filtered.groupby(['year_month', target_col]).size().reset_index(name='count')

        # Sort by year_month in descending order for consistent plotting
        monthly_counts.sort_values(by='year_month', ascending=ascending, inplace=True)

        # Create the line plot using Plotly
        fig = px.line(
            monthly_counts,
            x='year_month',
            y='count',
            color=target_col,
            markers=True,
            title=f"Count of {target_col} Over Months",
            labels={'year_month': 'Month', 'count': 'Count'}
        )

        # Set the figure layout for dark mode
        fig.update_layout(
            title_font=dict(size=20, family='Arial, sans-serif', color='white'),
            xaxis_title_font=dict(size=14, family='Arial, sans-serif', color='white'),
            yaxis_title_font=dict(size=14, family='Arial, sans-serif', color='white'),
            legend_title_font=dict(size=14, family='Arial, sans-serif', color='white'),
            legend=dict(
                title='Target Categories',
                orientation='h',  # Horizontal legend
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1,
                font=dict(color='white')  # Legend text color
            ),

            plot_bgcolor='rgba(30, 30, 30, 1)',  # Dark plot background
            paper_bgcolor='rgba(20, 20, 20, 1)',  # Dark paper background
            xaxis=dict(
                showgrid=True,  # Show grid lines
                gridcolor='rgba(100, 100, 100, 0.3)',  # Light gray grid lines for contrast
                zerolinecolor='rgba(255, 255, 255, 0.5)',  # Light line for the zero line
                tickfont=dict(color='white'),  # Change x-axis tick label color to white
                tickangle=-80

            ),
            # Reduced x-axis label size

            yaxis=dict(
                showgrid=True,  # Show grid lines
                gridcolor='rgba(100, 100, 100, 0.3)',  # Light gray grid lines for contrast
                zerolinecolor='rgba(255, 255, 255, 0.5)',  # Light line for the zero line
                tickfont=dict(color='white')  # Change x-axis tick label color to white

            ),
            hovermode='x unified',  # Unified hover mode
            hoverlabel=dict(
                bgcolor='rgba(50, 50, 50, 0.8)',  # Dark background for hover label
                font=dict(color='white')  # White text in hover label
            )
        )

        # Apply logarithmic scale if there are non-positive counts and log_scale is True
        if log_scale:
            fig.update_yaxes(type='log', title_text='Log Count', title_font=dict(size=14, color='white'))

        # Update x-axis order
        fig.update_xaxes(categoryorder='array', categoryarray=monthly_counts['year_month'])

        return fig


    def plot_target_per_week(self, df, timestamp_col, target_col, num_weeks=0,
                             log_scale=False, ascending=True , annotation:bool = False):
        """
        Generates a line plot showing the count of occurrences for a specified target column, grouped by week.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame containing the data to be plotted.
        timestamp_col : str
            The name of the column containing timestamp information.
        target_col : str
            The column whose occurrences are to be counted and plotted.
        num_weeks : int, optional
            The number of most recent weeks to include in the plot. Defaults to 0 (all available weeks).
        log_scale : bool, optional
            If True, applies logarithmic scaling to the y-axis. Defaults to False.
        ascending : bool, optional
            If True, sorts the x-axis in ascending order. Defaults to False.

        Returns:
        --------
        plotly.graph_objects.Figure
            The generated plot figure.
        """

        # Copying the DataFrame to avoid modifying the original data
        df = df.copy()

        if timestamp_col is None or timestamp_col not in df.columns:
            return pd.DataFrame({"Alert": [
                f"Error: The timestamp column '{timestamp_col}' is not provided or does not exist in the DataFrame."]})

            # Ensure the timestamp column is in datetime format
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')

        # Extract year and week for grouping
        df['year_week'] = df[timestamp_col].dt.to_period('W').astype(str)

        # Retrieve available weeks in descending order
        available_weeks = sorted(df['year_week'].unique(), reverse=True)
        total_available_weeks = len(available_weeks)

        # Validate `num_weeks` parameter
        if num_weeks < 0:
            return pd.DataFrame({"alert": ["The number of weeks cannot be negative."]})
        if num_weeks > total_available_weeks:
            return pd.DataFrame({"alert": ["The number of weeks exceeds the available weeks in the dataset."]})

        # Determine weeks to include in the plot
        selected_weeks = available_weeks if num_weeks == 0 else available_weeks[:num_weeks]

        # Filter the DataFrame for the selected weeks
        df_filtered = df[df['year_week'].isin(selected_weeks)]

        # Count occurrences of the target column per week
        weekly_counts = df_filtered.groupby(['year_week', target_col]).size().reset_index(name='count')

        # Extract the year from 'year_week' and store it in a temporary Series
        years = weekly_counts['year_week'].apply(lambda x: x.split('-')[0])

        # Create the 'new_week' column using an f-string and the index for numbering
        weekly_counts['new_week'] = [f"woche{i + 1}_{year}" for i, year in enumerate(years)]

        # Extract the year from 'year_week' and create a new column for year
        weekly_counts['year'] = weekly_counts['year_week'].apply(lambda x: x.split('-')[0])

        # Generate the new week number starting from 1 for each year
        weekly_counts['new_week'] = weekly_counts.groupby('year').cumcount() + 1

        # Create the formatted 'new_week' string with reset week number per year
        weekly_counts['new_week'] = [f"woche{week}_{year}" for week, year in
                                     zip(weekly_counts['new_week'], weekly_counts['year'])]

        # Ensure that 'new_week' is treated as a categorical variable with specific order
        weekly_counts['new_week'] = pd.Categorical(weekly_counts['new_week'], ordered=True)

        # Plotting using Plotly
        fig = px.line(
            weekly_counts,
            x='new_week',
            y='count',
            color=target_col,
            title=f"Count of {target_col} Over Weeks",
            labels={'new_week': 'Week','year_week' : 'exact_week ' ,'count': 'Count'},
            markers=True,
            line_shape='linear'  # Can also use 'spline' for smoother lines
        )

        # Customize line styles and markers for dark mode
        for trace in fig.data:
            trace.line.width = 2.5  # Thicker lines
            trace.marker.size = 8  # Larger markers
            trace.marker.symbol = 'circle'  # Marker shape
            trace.marker.color = 'rgba(255, 255, 255, 1)'  # White markers

        # Set log scale if required
        if log_scale:
            fig.update_yaxes(type="log", title_text='Log Count', title_font=dict(size=14, color='white'))

        fig.update_xaxes(categoryorder='array', categoryarray=weekly_counts["new_week"])

        # Update layout for dark mode aesthetics
        fig.update_layout(
            title_font=dict(size=20, family='Arial, sans-serif', color='white'),
            xaxis_title_font=dict(size=14, family='Arial, sans-serif', color='white'),
            yaxis_title_font=dict(size=14, family='Arial, sans-serif', color='white'),
            legend_title_font=dict(size=14, family='Arial, sans-serif', color='white'),
            legend=dict(
                title='Target Categories',
                orientation='h',  # Horizontal legend
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1,
                font=dict(color='white')  # Legend text color
            ),
            plot_bgcolor='rgba(30, 30, 30, 1)',  # Dark plot background
            paper_bgcolor='rgba(20, 20, 20, 1)',  # Dark paper background
            xaxis=dict(
                showgrid=True,  # Show grid lines
                gridcolor='rgba(100, 100, 100, 0.3)',  # Light gray grid lines for contrast
                zerolinecolor='rgba(255, 255, 255, 0.5)',  # Light line for the zero line
                tickfont=dict(color='white'),  # Change x-axis tick label color to white
                tickangle=-80

            ),
            yaxis=dict(
                showgrid=True,  # Show grid lines
                gridcolor='rgba(100, 100, 100, 0.3)',  # Light gray grid lines for contrast
                zerolinecolor='rgba(255, 255, 255, 0.5)',  # Light line for the zero line
                tickfont=dict(color='white')  # Change x-axis tick label color to white

            ),
            hovermode='x unified',  # Unified hover mode
            hoverlabel=dict(
                bgcolor='rgba(50, 50, 50, 0.8)',  # Dark background for hover label
                font=dict(color='white')  # White text in hover label
            )
        )
        
        # the annotation will be displayed only if the annotation is set to true 
        if annotation   : 
                
            # Add annotations for key points (example)
            for index, row in weekly_counts.iterrows():
                if row['count'] > 5:  # Example condition for adding an annotation
                    fig.add_annotation(
                        x=row['new_week'],
                        y=row['count'],
                        text=f"{row['count']}",
                        showarrow=True,
                        arrowhead=2,
                        ax=0,
                        ay=-40,
                        font=dict(color='white', size=12),
                        bgcolor='rgba(30, 30, 30, 0.9)',  # Dark background for annotations
                        bordercolor='gray',
                        borderwidth=1,
                        borderpad=4,
                        opacity=0.9
                    )

        # Show the figure
        return fig

    def numerical_time_series_analysis_per_month(self, df, timestamp_col, target_col=None, subset_features=None,
                                                 num_months=0):
        """
        Generates monthly summary statistics for each numerical feature in the DataFrame,
        excluding the target and timestamp columns, along with a plot for the mean values of each feature.

        Parameters:
        - df (pd.DataFrame): The input DataFrame containing time series data.
        - timestamp_col (str): The name of the timestamp column in the DataFrame.
        - target_col (str, optional): The name of the target column to exclude from analysis.
        - subset_features (list of str, optional): List of specific features to include in the analysis.
        If None, all numerical features will be included, except the target and timestamp columns.
        - num_months (int): The number of months to include in the analysis, starting from the most recent month
        (0 includes all available months).

        Returns:
        - summary_tables (dict): A dictionary where keys are feature names and values are DataFrames
        containing the monthly statistics for each feature.
        - figures (dict): A dictionary where keys are feature names and values are Plotly figures
        representing the mean values over time.
        """

        df = df.copy()
        error_flag = False

        # Check if timestamp column is provided and valid
        if timestamp_col is None or timestamp_col not in df.columns:
            return pd.DataFrame({"Alert": [
                f"Error: The timestamp column '{timestamp_col}' is not provided or does not exist in the DataFrame."]}), None, True

        # Check if num_months is a non-negative integer
        if not isinstance(num_months, int) or num_months < 0:
            return pd.DataFrame({"Alert": [
                f"Error: The specified number of months '{num_months}' is invalid. It must be a non-negative integer."]}), None, True

        # Ensure the timestamp column is in datetime format
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])

        # Create a 'Month_Label' column for grouping by month
        df['Month_Label'] = df[timestamp_col].dt.to_period('M').astype(str)

        # Extract numerical columns from the DataFrame, excluding timestamp and target columns
        numerical_cols = df.select_dtypes(include='number').columns.difference([target_col]).tolist()

        # Filter numerical columns if a subset of features is specified
        if subset_features is not None:
            numerical_cols = [col for col in numerical_cols if col in subset_features]

        # Initialize dictionaries for summary statistics and figures
        summary_tables = {}
        figures = {}

        # Determine the most recent month in the dataset
        recent_month = df['Month_Label'].max()
        recent_month_period = pd.Period(recent_month)

        # Calculate the number of available months in the dataset
        available_months = df['Month_Label'].nunique()

        # Check if the specified num_months exceeds the available months
        if num_months > available_months:
            return pd.DataFrame({"Alert": [
                f"Error: The specified number of months '{num_months}' exceeds the available months '{available_months}' in the dataset."]}), None, True

        # Filter the DataFrame for the specified number of recent months, if applicable
        if num_months > 0:
            month_labels = pd.date_range(end=recent_month_period.start_time, periods=num_months, freq='M').to_period(
                'M').astype(str)
            df = df[df['Month_Label'].isin(month_labels)]

        # Calculate monthly statistics for each numerical column
        for col in numerical_cols:
            # Group by 'Month_Label' and compute summary statistics
            summary_df = df.groupby('Month_Label')[col].agg(
                count='count',
                mean='mean',
                median='median',
                min='min',
                max='max'
            ).reset_index()

            # Add the feature name to the summary DataFrame
            summary_df['Feature'] = col

            # Store the summary DataFrame in the dictionary
            summary_tables[col] = summary_df

            # Create a Plotly figure for the mean values with dark mode and high contrast
            fig = go.Figure()

            # Add a line for the mean values
            fig.add_trace(go.Scatter(
                x=summary_df['Month_Label'],
                y=summary_df['mean'],
                mode='lines+markers',
                name='Mean Value',
                line=dict(color='cyan', width=2),  # Bright line color for contrast
                marker=dict(size=8, symbol='circle', color='cyan', line=dict(color='white', width=2))
            ))

            fig.update_xaxes(categoryorder='array', categoryarray=summary_tables[col]["Month_Label"],
                             tickfont=dict(color='white'))  # X-axis tick label color

            # Customize layout for dark mode and high contrast
            fig.update_layout(
                title=f'Monthly Mean of {col}',
                xaxis_title='Month',
                yaxis_title='Mean Value',

                legend=dict(title='Metrics', font=dict(color='white')),  # Legend text color
                template='plotly_dark',  # Use dark background theme
                margin=dict(l=40, r=40, t=40, b=40),
                hovermode='x unified',  # Improve hover interaction
                xaxis=dict(showgrid=True,gridcolor='lightgray', tickangle=-80, tickfont=dict(size=12)),  # Reduced x-axis label size
                yaxis=dict(showgrid=True, gridcolor='lightgray'),  # Light gray gridlines
                title_font=dict(color='white')  # Title color in white
            )

            # Store the figure in the figures dictionary
            figures[col] = fig

        return summary_tables, figures, error_flag

    def numerical_time_series_analysis_per_week(self, df, timestamp_col, target_col=None, subset_features=None,
                                                num_weeks=0):
        """
        Generates weekly summary statistics for each numerical feature in the DataFrame,
        excluding the target and timestamp columns, along with a plot for the mean values of each feature.

        Parameters:
        - df (pd.DataFrame): The input DataFrame containing time series data.
        - timestamp_col (str): The name of the timestamp column in the DataFrame.
        - target_col (str, optional): The name of the target column to exclude from analysis.
        - subset_features (list of str, optional): List of specific features to include in the analysis.
        If None, all numerical features will be included, except the target and timestamp columns.
        - num_weeks (int): The number of weeks to include in the analysis, starting from the most recent week
        (0 includes all available weeks).

        Returns:
        - summary_tables (dict): A dictionary where keys are feature names and values are DataFrames
        containing the weekly statistics for each feature.
        - figures (dict): A dictionary where keys are feature names and values are Plotly figures
        representing the mean values over time.
        """

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            df = df.copy()
            error_flag = False

            # Check if timestamp column is provided and valid
            if timestamp_col is None or timestamp_col not in df.columns:
                return pd.DataFrame({"Alert": [
                    f"Error: The timestamp column '{timestamp_col}' is not provided or does not exist in the DataFrame."]}), None, True

            # Check if num_weeks is a non-negative integer
            if not isinstance(num_weeks, int) or num_weeks < 0:
                return pd.DataFrame({"Alert": [
                    f"Error: The specified number of weeks '{num_weeks}' is invalid. It must be a non-negative integer."]}), None, True

            # Ensure the timestamp column is in datetime format
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])

            # Create a 'Week_Label' column for grouping by week
            df['Week_Label'] = df[timestamp_col].dt.to_period('W').astype(str)

            # Extract numerical columns from the DataFrame, excluding timestamp and target columns
            numerical_cols = df.select_dtypes(include='number').columns.difference([target_col]).tolist()

            # Filter numerical columns if a subset of features is specified
            if subset_features is not None:
                numerical_cols = [col for col in numerical_cols if col in subset_features]

            # Initialize dictionaries for summary statistics and figures
            summary_tables = {}
            figures = {}

            # Determine the most recent week in the dataset
            recent_week = df['Week_Label'].max()
            recent_week_period = pd.Period(recent_week)

            # Calculate the number of available weeks in the dataset
            available_weeks = df['Week_Label'].nunique()

            # Check if the specified num_weeks exceeds the available weeks
            if num_weeks > available_weeks:
                return pd.DataFrame({"Alert": [
                    f"Error: The specified number of weeks '{num_weeks}' exceeds the available weeks '{available_weeks}' in the dataset."]}), None, True

            # Filter the DataFrame for the specified number of recent weeks, if applicable
            if num_weeks > 0:
                week_labels = pd.date_range(end=recent_week_period.start_time, periods=num_weeks, freq='W').to_period(
                    'W').astype(str)
                df = df[df['Week_Label'].isin(week_labels)]

            # Calculate weekly statistics for each numerical column
            for col in numerical_cols:
                # Group by 'Week_Label' and compute summary statistics
                summary_df = df.groupby('Week_Label')[col].agg(
                    count='count',
                    mean='mean',
                    median='median',
                    min='min',
                    max='max'
                ).reset_index()

                # Add the feature name to the summary DataFrame
                summary_df['Feature'] = col

                # Store the summary DataFrame in the dictionary
                summary_tables[col] = summary_df

                # Extract the year from 'year_week' and store it in a temporary Series
                years = summary_tables[col]['Week_Label'].apply(lambda x: x.split('-')[0])
                # Create the 'new_week' column using an f-string and the index for numbering
                summary_tables[col]['week'] = [f"woche{i + 1}_{year}" for i, year in enumerate(years)]

                # Extract the year from 'year_week' and create a new column for year
                summary_tables[col]['year'] = summary_tables[col]['Week_Label'].apply(lambda x: x.split('-')[0])

                # Generate the new week number starting from 1 for each year
                summary_tables[col]['week'] = summary_tables[col].groupby('year').cumcount() + 1

                # Create the formatted 'new_week' string with reset week number per year
                summary_tables[col]['week'] = [f"woche{week}_{year}" for week, year in
                                               zip(summary_tables[col]['week'], summary_tables[col]['year'])]

                # Ensure that 'new_week' is treated as a categorical variable with specific order
                summary_tables[col]['week'] = pd.Categorical(summary_tables[col]['week'], ordered=True)

                # Create a Plotly figure for the mean values with enhanced design
                fig = go.Figure()

                # Add a line for the mean values
                fig.add_trace(go.Scatter(
                    x=summary_df['week'],
                    y=summary_df['mean'],
                    mode='lines+markers',
                    name='Mean Value',
                    line=dict(color='cyan', width=2),
                    marker=dict(size=8, symbol='circle', color='white', line=dict(color='white', width=2))

                ))

                # Update x-axis to use categorical order
                fig.update_xaxes(categoryorder='array', categoryarray=summary_tables[col]["week"],
                                 tickfont=dict(color='white'))  # Change x-axis tick label color to white

                # Customize layout for better aesthetics in dark mode
                fig.update_layout(
                    title=f'Weekly Mean of {col}',
                    title_font=dict(size=20, color='white'),
                    xaxis_title='Week',
                    xaxis_title_font=dict(size=14, color='white'),
                    yaxis_title='Mean Value',
                    yaxis_title_font=dict(size=14, color='white'),
                    legend=dict(title='Metrics', font=dict(color='white')),
                    margin=dict(l=40, r=40, t=40, b=40),
                    hovermode='x unified',
                    plot_bgcolor='rgba(30, 30, 30, 1)',  # Dark plot background
                    paper_bgcolor='rgba(20, 20, 20, 1)',  # Dark paper background
                    xaxis=dict(showgrid=True, gridcolor='lightgray', tickangle=-80, tickfont=dict(size=12)),
                    # Reduced x-axis label size
                    yaxis=dict(showgrid=True, gridcolor='rgba(255, 255, 255, 0.3)', tickfont=dict(color='white'))


                )

                # Store the figure in the figures dictionary
                figures[col] = fig

        return summary_tables, figures, error_flag

    def categorical_time_series_analysis_per_month(self, df, timestamp_col, target, num_months, top_n=3,
                                                   consider_null: bool = True, annotation: bool = False,
                                                   log_scale: bool = True, ascending: bool = True,
                                                   subset_features: list = None):

        """
        Analyzes categorical time series data on a monthly basis and generates separate plots for the top N most frequent
        subcategories for each categorical feature, over the specified months using Plotly.

        Parameters:
        - df (pd.DataFrame): The input DataFrame containing time series data.
        - timestamp_col (str): The name of the timestamp column in the DataFrame.
        - target (str): The name of the target column to be excluded from analysis.
        - num_months (int): The number of months to include in the analysis, starting from the most recent month.
        - top_n (int): The number of top unique subcategories to be included in the analysis.
        - consider_null (bool): Whether to include null values as a category.
        - annotation (bool): Whether to annotate counts on the plot.
        - log_scale (bool): Whether to use a logarithmic scale on the y-axis.
        - ascending (bool): Sorting direction for categories.
        - subset_features (list): Optional list of specific categorical features to consider in the analysis.
                                  If None, all categorical features are analyzed.

        Returns:
        - dataframes (list of pd.DataFrame): List of DataFrames containing aggregated counts for each categorical feature.
        - figures (list of plotly.graph_objects.Figure): List of Plotly figures containing plots of the counts over the specified months.
        - ignored_features_df (pd.DataFrame): DataFrame containing the names of features that were ignored and the reason why.
        - error_flag (bool): Indicator of whether an error occurred during processing.
        """

        # Suppress warnings
        warnings.filterwarnings("ignore")
        df = df.copy()
        ignore_features = {target}  # Using a set for faster membership testing
        figures = []
        dataframes = []
        error_flag = False

        # Validate input DataFrame
        if timestamp_col not in df.columns:
            error_flag = True
            return pd.DataFrame({
                "Alert": [
                    f"Error: The '{timestamp_col}' column is required in the DataFrame.\n"
                    f"Current DataFrame does not contain this column.\n"
                    f"Please ensure that you are passing the correct DataFrame.\n"
                    f"Suggested action: Verify the input data to include the '{timestamp_col}' column."
                ]
            }), None, pd.DataFrame(), True

        if not isinstance(num_months, int) or num_months < 0:
            error_flag = True
            return pd.DataFrame({
                "Alert": [
                    "Error: The value provided for 'months' is invalid.\n"
                    "Please ensure that the number of months is a non-negative integer (0 or greater).\n"
                    "A negative number or a non-integer value is not acceptable.\n"
                    "Example of valid input: 0, 1, 2, ... or any positive integer."
                ]
            }), None, pd.DataFrame(), True

        # Convert timestamp column to datetime and create 'YearMonth'
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        df['YearMonth'] = df[timestamp_col].dt.to_period('M')

        # Sort DataFrame and filter by the most recent months
        df_sorted = df.sort_values(by=timestamp_col, ascending=False)
        latest_date = df_sorted[timestamp_col].max()
        total_available_months = df_sorted['YearMonth'].nunique()

        if num_months == 0:
            num_months = total_available_months

        if num_months > total_available_months:
            error_flag = True
            return pd.DataFrame({
                "Alert": [
                    f"Error: The specified number of months exceeds the available months in the dataset.\n"
                    f"Currently, the dataset contains {total_available_months} available months.\n"
                    "Please check the input value and ensure it does not exceed the available months.\n"
                    "Consider reducing the number of months to a value less than or equal to the available months."
                ]
            }), None, pd.DataFrame(), True

        # Filter DataFrame for the required months
        cut_off_date = latest_date - pd.DateOffset(months=num_months)
        df_filtered = df_sorted[df_sorted[timestamp_col] >= cut_off_date]

        # Select categorical columns, considering user-specified subset if provided
        if subset_features:
            categorical_cols = set(subset_features).intersection(
                df_filtered.select_dtypes(include=['object', 'category']).columns)
        else:
            categorical_cols = df_filtered.select_dtypes(include=['object', 'category']).columns.difference(
                ignore_features)

        # Process each categorical column
        for col in categorical_cols:
            # Group and aggregate data
            if consider_null:
                if df_filtered[col].isnull().any():
                    if 'Missing Values' not in df_filtered[col].unique():
                        df_filtered[col] = df_filtered[col].astype('category').cat.add_categories('Missing Values')
                    df_filtered[col].fillna('Missing Values', inplace=True)

            # Group data by YearMonth and the categorical column
            agg_df = df_filtered.groupby(['YearMonth', col]).size().unstack(fill_value=0)

            # Get top N categories
            top_categories = agg_df.sum().nlargest(top_n).index
            filtered_agg_df = agg_df[top_categories]

            # Melt the DataFrame for easier plotting
            melted_df = filtered_agg_df.reset_index().melt(id_vars='YearMonth', var_name='Category', value_name='Count')

            # Convert YearMonth to string for Plotly compatibility
            melted_df['YearMonth'] = melted_df['YearMonth'].astype(str)

            # Append melted DataFrame for later use
            dataframes.append(melted_df)

            # Sort melted DataFrame by 'YearMonth' in descending order
            melted_df.sort_values(by='YearMonth', ascending=True, inplace=True)

            # Plotting the results for each top category using Plotly
            fig = px.line(
                melted_df,
                x='YearMonth',
                y='Count',
                color='Category',
                markers=True,
                title=f"Top {top_n} Categories in '{col}' Over Months",
                template='plotly_dark'  # Dark mode template
            )

            # Update x-axis to use categorical order and make tick labels white
            fig.update_xaxes(categoryorder='array', categoryarray=melted_df["YearMonth"],
                             tickfont=dict(color='white'))

            # Update layout for high contrast and fixed size
            fig.update_layout(
                xaxis_title="Month-Year",
                yaxis_title="Count (Log Scale)" if log_scale else "Count",
                legend_title_text='Categories',
                title_font=dict(size=18, color='white'),  # White title font
                font=dict(color='white'),  # White font for all text
                xaxis=dict(showgrid=True, tickangle=-80, tickfont=dict(size=12)),  # Reduced x-axis label size
                yaxis=dict(showgrid=True, gridcolor='grey'),
                margin=dict(l=50, r=50, t=80, b=150),  # Add space for legend
                legend=dict(
                    orientation="v",  # Vertical orientation
                    yanchor="middle",
                    y=0.5,  # Center vertically
                    xanchor="left",
                    x=1.05,  # Move to the right of the plot
                    title_font=dict(size=14, color='white'),
                    font=dict(size=12, color='white'),
                    bgcolor='rgba(0,0,0,0.5)',
                )
            )

            if log_scale:
                fig.update_yaxes(type='log', title_standoff=10)  # Apply logarithmic scale to y-axis

            # Add annotations if requested
            if annotation:
                for category in melted_df['Category'].unique():
                    category_data = melted_df[melted_df['Category'] == category]
                    for i, row in category_data.iterrows():
                        fig.add_annotation(
                            x=row['YearMonth'],
                            y=row['Count'],
                            text=str(row['Count']),
                            showarrow=True,
                            arrowhead=2,
                            ax=0,
                            ay=-40,
                            font=dict(color='white', size=10),  # White annotation text
                            bgcolor='black'  # Black background for contrast
                        )

            figures.append(fig)

        # Return the dataframes, figures, ignored features, and error flag
        ignored_features_df = pd.DataFrame(list(ignore_features), columns=['Ignored_Feature'])
        return dataframes, figures, ignored_features_df, error_flag

    def categorical_time_series_analysis_per_week(self, df, timestamp_col, target, num_weeks, top_n=3,
                                                  consider_null: bool = True, annotation: bool = False,
                                                  log_scale: bool = True, ascending: bool = True,
                                                  subset_features: list = None):
        """
        Analyzes categorical time series data on a weekly basis and generates separate plots for the top N most frequent
        subcategories for each categorical feature, over the specified weeks using Plotly.

        Parameters:
        - df (pd.DataFrame): The input DataFrame containing time series data.
        - timestamp_col (str): The name of the timestamp column in the DataFrame.
        - target (str): The name of the target column to be excluded from analysis.
        - num_weeks (int): The number of weeks to include in the analysis, starting from the most recent week.
        - top_n (int): The number of top unique subcategories to be included in the analysis.
        - consider_null (bool): Whether to consider null values in the analysis by treating them as a category.
        - annotation (bool): Whether to annotate counts on the plot.
        - log_scale (bool): Whether to apply a logarithmic scale to the y-axis.
        - ascending (bool): Whether to sort values in ascending order.
        - subset_features (list): List of specific categorical columns to analyze. If None, all categorical columns are considered.

        Returns:
        - dataframes (list of pd.DataFrame): List of DataFrames containing aggregated counts for each categorical feature.
        - figures (list of plotly.graph_objects.Figure): List of Plotly figures containing plots of the counts over the specified weeks.
        - ignored_features_df (pd.DataFrame): DataFrame containing the names of features that were ignored and the reason why.
        - error_flag (bool): Indicator of whether an error occurred during processing.
        """

        # Ignore FutureWarnings
        warnings.filterwarnings("ignore", category=FutureWarning)
        df = df.copy()
        ignore_features = {target}  # Using a set for faster membership testing
        figures = []
        dataframes = []
        error_flag = False

        # Validate input DataFrame
        if timestamp_col not in df.columns:
            return pd.DataFrame({"Alert": [f"Error: The '{timestamp_col}' column is required in the DataFrame.\n"
                                           f"Current DataFrame does not contain this column.\n"
                                           f"Please ensure that you are passing the correct DataFrame.\n"
                                           f"Suggested action: Verify the input data to include the '{timestamp_col}' column."]}), None, pd.DataFrame(), True

        if not isinstance(num_weeks, int) or num_weeks < 0:
            return pd.DataFrame({"Alert": ["Error: The value provided for 'weeks' is invalid.\n"
                                           "Please ensure that the number of weeks is a non-negative integer (0 or greater).\n"
                                           "A negative number or a non-integer value is not acceptable.\n"
                                           "Example of valid input: 0, 1, 2, ... or any positive integer."]}), None, pd.DataFrame(), True

        # Convert timestamp column to datetime and create 'YearWeek'
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        df['YearWeek'] = df[timestamp_col].dt.to_period('W')

        # Sort DataFrame and filter by the most recent weeks
        df_sorted = df.sort_values(by=timestamp_col, ascending=False)
        latest_date = df_sorted[timestamp_col].max()
        total_available_weeks = df_sorted['YearWeek'].nunique()

        if num_weeks == 0:
            num_weeks = total_available_weeks

        if num_weeks > total_available_weeks:
            return pd.DataFrame(
                {"Alert": [f"Error: The specified number of weeks exceeds the available weeks in the dataset.\n"
                           f"Currently, the dataset contains {total_available_weeks} available weeks.\n"
                           "Please check the input value and ensure it does not exceed the available Weeks.\n"
                           "Consider reducing the number of Weeks to a value less than or equal to the available Weeks."]}), None, pd.DataFrame(), True

        # Filter DataFrame for the required weeks
        cut_off_date = latest_date - pd.DateOffset(weeks=num_weeks)
        df_filtered = df_sorted[df_sorted[timestamp_col] >= cut_off_date]

        # Select categorical columns, considering only the specified subset if provided
        if subset_features:
            categorical_cols = pd.Index(subset_features).intersection(
                df_filtered.select_dtypes(include=['object', 'category']).columns).difference(ignore_features)
        else:
            categorical_cols = df_filtered.select_dtypes(include=['object', 'category']).columns.difference(
                ignore_features)

        # Process each categorical column
        for col in categorical_cols:
            # Group and aggregate data
            if consider_null:
                # Only fill NaN with 'Missing Values' if there are any NaN values
                if df_filtered[col].isnull().any():
                    if 'Missing Values' not in df_filtered[col].unique():
                        df_filtered[col] = df_filtered[col].astype('category').cat.add_categories('Missing Values')
                    df_filtered[col].fillna('Missing Values', inplace=True)

            # Group data by YearWeek and the categorical column
            agg_df = df_filtered.groupby(['YearWeek', col]).size().unstack(fill_value=0)

            # Get top N categories
            top_categories = agg_df.sum().nlargest(top_n).index
            filtered_agg_df = agg_df[top_categories]

            # Melt the DataFrame for easier plotting
            melted_df = filtered_agg_df.reset_index().melt(id_vars='YearWeek', var_name='Category', value_name='Count')

            # Convert YearWeek to string for Plotly compatibility
            melted_df['YearWeek'] = melted_df['YearWeek'].astype(str)

            # Append melted DataFrame for later use
            dataframes.append(melted_df)

            # Sort melted DataFrame by 'YearWeek' in ascending order
            melted_df.sort_values(by='YearWeek', ascending=True, inplace=True)

            # Extract the year from 'year_week' and store it in a temporary Series
            years = melted_df['YearWeek'].apply(lambda x: x.split('-')[0])
            # Create the 'new_week' column using an f-string and the index for numbering
            melted_df['week'] = [f"woche{i + 1}_{year}" for i, year in enumerate(years)]

            # Extract the year from 'year_week' and create a new column for year
            melted_df['year'] = melted_df['YearWeek'].apply(lambda x: x.split('-')[0])

            # Generate the new week number starting from 1 for each year
            melted_df['week'] = melted_df.groupby('year').cumcount() + 1

            # Create the formatted 'new_week' string with reset week number per year
            melted_df['week'] = [f"woche{week}_{year}" for week, year in
                                           zip(melted_df['week'], melted_df['year'])]


            # Ensure that 'new_week' is treated as a categorical variable with specific order
            melted_df['week'] = pd.Categorical(melted_df['week'], ordered=True)

            fig = px.line(
                melted_df,
                x='week',
                y='Count',
                color='Category',
                markers=True,
                title=f"Top {top_n} Categories in '{col}' Over Weeks",
                template='plotly_dark'  # Using a dark background theme
            )

            # Update x-axis, y-axis, and layout for high contrast
            fig.update_xaxes(categoryorder='array', categoryarray=melted_df["week"],
                             tickfont=dict(color='white') ,tickangle=90  # Rotate labels from right to left
)
            fig.update_yaxes(type='log' if log_scale else 'linear', title="Count", tickfont=dict(color='white'))

            fig.update_layout(
                title_font=dict(size=18, color='white'),  # White title font
                xaxis=dict(showgrid=True, tickangle=-80, tickfont=dict(size=12)),  # Reduced x-axis label size
                yaxis=dict(showgrid=True),
                margin=dict(l=50, r=50, t=80, b=200),  # Increased bottom margin further to 200
                legend=dict(
                    orientation="v",  # Vertical orientation
                    yanchor="middle",
                    y=0.5,  # Center vertically
                    xanchor="left",
                    x=1.05,  # Move to the right of the plot
                    title_font=dict(size=14, color='white'),
                    font=dict(size=12, color='white'),
                    bgcolor='rgba(0,0,0,0.5)',
                )
            )

            # Add annotations if requested
            if annotation:
                for category in melted_df['Category'].unique():
                    category_data = melted_df[melted_df['Category'] == category]
                    for i, row in category_data.iterrows():
                        fig.add_annotation(
                            x=row['week'],
                            y=row['Count'],
                            text=str(row['Count']),
                            showarrow=True,
                            arrowhead=2,
                            ax=0,
                            ay=-40,
                            font=dict(color='white', size=10),
                            bgcolor='black'
                        )

            figures.append(fig)

        ignored_features_df = pd.DataFrame(list(ignore_features), columns=['Ignored_Feature'])
        return dataframes, figures, ignored_features_df, error_flag
    def get_used_features(self, function_name: str):

        """
        Retrieves the list of features used for a specific type of analysis based on user settings.

        This helper method extracts the feature list from the user settings for the specified
        analysis function. It is designed to support various analysis types by querying
        the settings associated with the given function name.

        Parameters:
        - function_name (str): The name of the analysis function for which features are to be retrieved.

        Returns:
        - list or None:
            A list of features if they are defined in the user settings for the given function name;
            otherwise, returns None if no features are specified.
        """

        feature = self.user_settings["functions"]["defaultExploratoryDataAnalysis"][function_name]

        return feature

    def _get_features(self, settings, analysis_type: str):
        """
        Helper method to extract feature lists for different analysis types from settings.

        Parameters:
        - analysis_type (str): The specific type of analysis for which features are to be retrieved.

        Returns:
        - list or None: A list of features if specified in the settings; otherwise, None.
        """
        features = settings["functions"]["analysisParameters"].get(analysis_type, {}).get("featuresToInclude")
        return features if features != "None" else None




    



    