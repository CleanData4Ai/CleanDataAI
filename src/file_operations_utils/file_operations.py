import yaml  # For reading and processing YAML configuration files, facilitating configuration management.
import os
import pandas as pd
import json
import csv
from pyhive import hive

def create_file_path( base_dir: str, title: str, file_type: str = "html") -> str:
    """
    Constructs a valid file path by sanitizing the title, ensuring any missing directories exist,
    and combining it with the base directory. The default file type is HTML.

    Parameters:
    - base_dir (str): The base directory where the file will be created.
    - title (str): The title of the file, which will be sanitized and used as the file name.
    - file_type (str): The type of the file to be created (e.g., 'csv', 'pickle', 'html'). Defaults to 'html'.

    Returns:
    - str: The full path to the newly created file, with spaces in the title replaced by underscores
            and the specified file extension added.
    """

    # Replace spaces in the title with underscores to ensure it's a valid file name
    sanitized_title = title.replace(" ", "_")

    # Determine the file extension based on the user-specified file type
    if file_type.lower() == "csv":
        extension = ".csv"
    elif file_type.lower() == "pickle":
        extension = ".pkl"
    else:  # Default to HTML
        extension = ".html"

    # Construct the full file path by joining the base directory and the sanitized title with the determined extension
    file_path = os.path.join(base_dir, f"{sanitized_title}{extension}")

    # Ensure that the directory exists, create it if it does not
    dir_name = os.path.dirname(file_path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    return file_path


def format_string(input_string):
    """
    Formats the input string by converting it to lowercase and removing all spaces.

    Parameters:
    input_string (str): The string to be formatted.

    Returns:
    str: The formatted string, in lowercase and without spaces.
    """
    formatted_string = input_string.lower().replace(" ", "")
    return formatted_string


def load_yaml(file_path: str):
    """
    Load data from a YAML file.

    Parameters:
    file_path (str): The path to the YAML file.

    Returns:
    dict: The data loaded from the YAML file.
    """
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)

    return data





import os
import pandas as pd
import json
import csv
from pyhive import hive
import mlflow
import mlflow.pyfunc

def load_files_to_dataframes(file_paths, hive_config=None, mlflow_model_config=None):
    """
    Load multiple data files into a list of pandas DataFrames.

    Parameters:
    file_paths (list of str): List of paths to the data files.
    hive_config (dict, optional): Configuration for connecting to Hive. Should contain keys like 'host', 'port', 'username', and 'database'.
    mlflow_model_config (dict, optional): Configuration for loading data from an MLflow model. Should contain keys like 'model_uri' and optionally 'input_data'.

    Returns:
    list of pd.DataFrame: List of loaded DataFrames (or None for files that failed to load).
    """
    dataframes = []

    for path in file_paths:
        try:
            # Check the file extension to determine the file type
            _, file_extension = os.path.splitext(path)

            if file_extension == ".pkl":
                # Load pickle file
                df = pd.read_pickle(path)
            elif file_extension in [".csv"]:
                # Load CSV file
                df = pd.read_csv(path)
            elif file_extension in [".json"]:
                # Load JSON file
                with open(path, 'r') as f:
                    json_data = json.load(f)
                    df = pd.DataFrame(json_data)
            elif file_extension in [".xlsx"]:
                # Load Excel file
                df = pd.read_excel(path)
            elif file_extension in [".parquet"]:
                # Load Parquet file
                df = pd.read_parquet(path)
            elif file_extension == "hive":
                if hive_config is None:
                    raise ValueError("Hive configuration is required to load data from Hive.")
                connection = hive.Connection(
                    host=hive_config.get('host', 'localhost'),
                    port=hive_config.get('port', 10000),
                    username=hive_config.get('username', None),
                    database=hive_config.get('database', 'default')
                )
                query = hive_config.get('query', 'SELECT * FROM some_table')
                df = pd.read_sql(query, connection)
            elif file_extension == "mlflow":
                if mlflow_model_config is None:
                    raise ValueError("MLflow model configuration is required to load data from an MLflow model.")
                model_uri = mlflow_model_config.get('model_uri')
                if not model_uri:
                    raise ValueError("Model URI is required in MLflow model configuration.")
                model = mlflow.pyfunc.load_model(model_uri)
                input_data = mlflow_model_config.get('input_data', None)
                if input_data is not None:
                    # If input data is provided, ensure it's in DataFrame format
                    if not isinstance(input_data, pd.DataFrame):
                        raise ValueError("Input data for MLflow model must be a pandas DataFrame.")
                    df = model.predict(input_data)
                else:
                    df = pd.DataFrame([{"model_loaded": True}])  # Example placeholder if no input data
            else:
                print(f"Unsupported file type for file: {path}")
                df = None

            dataframes.append(df)

        except Exception as e:
            print(f"Error loading file at {path}: {e}")
            dataframes.append(None)

    return dataframes

# Example usage
# file_paths = ["data1.pkl", "data2.csv", "data3.json", "data4.xlsx", "data5.parquet", "mlflow"]
# hive_config = {"host": "localhost", "port": 10000, "username": "user", "database": "default", "query": "SELECT * FROM table"}
# mlflow_model_config = {"model_uri": "runs:/<run_id>/model", "input_data": pd.DataFrame(...)}
# loaded_dataframes = load_files_to_dataframes(file_paths, hive_config=hive_config, mlflow_model_config=mlflow_model_config)
