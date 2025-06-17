# Importing evaluation metrics for classification models
from sklearn.metrics import confusion_matrix, roc_auc_score, matthews_corrcoef, f1_score

# Importing evaluation metrics for regression models
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score

# Importing standard libraries for numerical operations and data handling
import numpy as np
import pandas as pd

# Importing libraries for data visualization
import matplotlib.pyplot as plt
import plotly.graph_objects as go  # For interactive visualizations
import seaborn as sns  # For statistical visualizations

# Additional classification metric
from sklearn.metrics import brier_score_loss

# Importing tools for building computation graphs (possibly for ML pipelines or workflows)
import langgraph.graph as lg

# Typing support for better code clarity and type hinting
from typing import Dict, List, Union

# For handling date and time
from datetime import datetime

# Import templates and utility functions if working in another directory
from eda_lib.src.graphical_data_visualizations.visualisation.template import base_template, model_evalutation_report  # HTML templates for report generation
from eda_lib.src.file_operations_utils.file_operations import create_file_path, load_yaml, load_files_to_dataframes  # File and YAML utility functions

# If not working in another directory, use these imports instead:
# from src.graphical_data_visualizations.visualisation.template import base_template, model_evalutation_report
# from src.file_operations_utils.file_operations import create_file_path, load_yaml, load_files_to_dataframes

# For operating system level file operations
import os



class MlEvaluationHelpFunctions:

    def __init__(self, user_settings_path : str , project_root_name : str  , base_path:str ):
        
        user_settings = load_yaml(os.path.join(base_path, project_root_name, user_settings_path))
        params = user_settings["model_selection_parameters"]

        reg_mae_weight = params["reg_mae_weight"]
        reg_mse_weight = params["reg_mse_weight"]
        reg_rmse_weight = params["reg_rmse_weight"]
        reg_r2_weight = params["reg_r2_weight"]
        
        reg_explained_variance_weight = params["reg_explained_variance_weight"]

        clf_precision_weight = params["clf_precision_weight"]
        clf_recall_weight = params["clf_recall_weight"]
        clf_accuracy_weight = params["clf_accuracy_weight"]
        clf_auc_weight = params["clf_auc_weight"]
        clf_f1_weight = params["clf_f1_weight"]
        clf_phi_weight = params["clf_phi_weight"]
        clf_fpr_weight = params["clf_fpr_weight"]
        clf_fnr_weight = params["clf_fnr_weight"]

        # Initialize the graph
        self.workflow = lg.StateGraph(dict)
        
        # Store weights as instance variables
        # Regression weights
        self.reg_weights = {
            'mae': reg_mae_weight,
            'mse': reg_mse_weight,
            'rmse': reg_rmse_weight,
            'r2': reg_r2_weight,
            'explained_variance': reg_explained_variance_weight
        }
        
        # Classification weights
        self.clf_weights = {
            'precision': clf_precision_weight,
            'recall': clf_recall_weight,
            'accuracy': clf_accuracy_weight,
            'auc': clf_auc_weight,
            'f1': clf_f1_weight,
            'phi': clf_phi_weight,
            'fpr': clf_fpr_weight,
            'fnr': clf_fnr_weight
        }
        
        # Validate weights sum to 1.0
        self._validate_weights()
        
        self.setup_workflow()

    def _validate_weights(self):
        """
        Validate that weights sum to approximately 1.0
        """
        reg_sum = sum(self.reg_weights.values())
        clf_sum = sum(self.clf_weights.values())
        
        if not (0.99 <= reg_sum <= 1.01):
            raise ValueError(f"Regression weights must sum to 1.0, got {reg_sum}")
        
        if not (0.99 <= clf_sum <= 1.01):
            raise ValueError(f"Classification weights must sum to 1.0, got {clf_sum}")

    def setup_workflow(self):
        """
        Set up the LangGraph workflow for model ranking
        """
        # Add nodes for each processing step
        self.workflow.add_node("normalize_metrics", self.normalize_metrics)
        self.workflow.add_node("aggregate_scores", self.aggregate_scores)
        self.workflow.add_node("rank_models", self.rank_models)

        # Define edges
        self.workflow.add_edge("normalize_metrics", "aggregate_scores")
        self.workflow.add_edge("aggregate_scores", "rank_models")
        
        # Set entry and finish points
        self.workflow.set_entry_point("normalize_metrics")
        self.workflow.set_finish_point("rank_models")

        
    def normalize_metrics(self, state: Dict) -> Dict:
        """
        Normalize metrics for both regression and classification models
        
        :param state: Dictionary containing input DataFrames
        :return: Updated state with normalized metrics
        """
        reg_df = state.get('regression_df')
        clf_df = state.get('classification_df')
        
        normalized_reg_df = None
        normalized_clf_df = None

        # Regression metrics normalization
        if reg_df is not None:
            # Create a copy to avoid modifying the original
            normalized_reg_df = reg_df.copy()
            
            # Normalize MAE, MSE, RMSE (lower is better) by taking inverse
            normalized_reg_df['normalized_MAE'] = 1 / (normalized_reg_df['Mean Absolute Error (MAE)'] + 1e-10)
            normalized_reg_df['normalized_MSE'] = 1 / (normalized_reg_df['Mean Squared Error (MSE)'] + 1e-10)
            normalized_reg_df['normalized_RMSE'] = 1 / (normalized_reg_df['Root Mean Squared Error (RMSE)'] + 1e-10)
            
            # R² and Explained Variance are already high-is-better
            normalized_reg_df['normalized_R2'] = normalized_reg_df['R-squared (R²)']
            normalized_reg_df['normalized_ExplainedVariance'] = normalized_reg_df['Explained Variance']

        # Classification metrics normalization
        if clf_df is not None:
            # Create a copy to avoid modifying the original
            normalized_clf_df = clf_df.copy()
            
            # Higher is better metrics - keep as is
            normalized_clf_df['normalized_TPR'] = normalized_clf_df['TPR']
            normalized_clf_df['normalized_TNR'] = normalized_clf_df['TNR']
            normalized_clf_df['normalized_Precision'] = normalized_clf_df['Precision']
            normalized_clf_df['normalized_Recall'] = normalized_clf_df['Recall']
            normalized_clf_df['normalized_Accuracy'] = normalized_clf_df['Accuracy']
            normalized_clf_df['normalized_AUC'] = normalized_clf_df['AUC']
            normalized_clf_df['normalized_F1Score'] = normalized_clf_df['F1-Score']
            normalized_clf_df['normalized_PhiCoefficient'] = normalized_clf_df['Phi Coefficient']

            # Lower is better metrics - take inverse
            normalized_clf_df['normalized_FPR'] = 1 / (normalized_clf_df['FPR'] + 1e-10)
            normalized_clf_df['normalized_FNR'] = 1 / (normalized_clf_df['FNR'] + 1e-10)
            normalized_clf_df['normalized_BrierScoreLoss'] = 1 / (normalized_clf_df['Brier Score Loss'] + 1e-10)

        return {
            **state, 
            'normalized_regression_df': normalized_reg_df, 
            'normalized_classification_df': normalized_clf_df
        }

    def aggregate_scores(self, state: Dict) -> Dict:
        """
        Compute aggregate scores for models using the provided weights
        
        :param state: Dictionary with normalized DataFrames
        :return: Updated state with model scores
        """
        reg_df = state.get('normalized_regression_df')
        clf_df = state.get('normalized_classification_df')
        
        scored_reg_df = None
        scored_clf_df = None

        # Regression score computation
        if reg_df is not None:
            # Create a copy to avoid modifying the original
            scored_reg_df = reg_df.copy()
            
            # Weighted aggregation of regression metrics using adjustable weights
            scored_reg_df['aggregate_score'] = (
                self.reg_weights['mae'] * scored_reg_df['normalized_MAE'] +
                self.reg_weights['mse'] * scored_reg_df['normalized_MSE'] +
                self.reg_weights['rmse'] * scored_reg_df['normalized_RMSE'] +
                self.reg_weights['r2'] * scored_reg_df['normalized_R2'] +
                self.reg_weights['explained_variance'] * scored_reg_df['normalized_ExplainedVariance']
            )

        # Classification score computation
        if clf_df is not None:
            # Create a copy to avoid modifying the original
            scored_clf_df = clf_df.copy()
            
            # Weighted aggregation of classification metrics using adjustable weights
            scored_clf_df['aggregate_score'] = (
                self.clf_weights['precision'] * scored_clf_df['normalized_Precision'] +
                self.clf_weights['recall'] * scored_clf_df['normalized_Recall'] +
                self.clf_weights['accuracy'] * scored_clf_df['normalized_Accuracy'] +
                self.clf_weights['auc'] * scored_clf_df['normalized_AUC'] +
                self.clf_weights['f1'] * scored_clf_df['normalized_F1Score'] +
                self.clf_weights['phi'] * scored_clf_df['normalized_PhiCoefficient'] +
                self.clf_weights['fpr'] * scored_clf_df['normalized_FPR'] +
                self.clf_weights['fnr'] * scored_clf_df['normalized_FNR']
            )

        return {
            **state, 
            'scored_regression_df': scored_reg_df, 
            'scored_classification_df': scored_clf_df
        }

    def rank_models(self, state: Dict) -> Dict:
        """
        Rank models based on their aggregate scores
        
        :param state: Dictionary with scored DataFrames
        :return: Ranked model results
        """
        reg_df = state.get('scored_regression_df')
        clf_df = state.get('scored_classification_df')

        rankings = {}

        # Rank regression models
        if reg_df is not None:
            # Create a copy to avoid modifying the original
            ranking_reg_df = reg_df.copy()
            
            # Group by model name (removing phase)
            ranking_reg_df['model_name'] = ranking_reg_df['model_name/phase'].apply(lambda x: x.split(' / ')[0])
            reg_rankings = ranking_reg_df.groupby('model_name')['aggregate_score'].mean().reset_index()
            reg_rankings = reg_rankings.sort_values('aggregate_score', ascending=False)
            rankings['regression_rankings'] = reg_rankings

        # Rank classification models
        if clf_df is not None:
            # Create a copy to avoid modifying the original
            ranking_clf_df = clf_df.copy()
            
            # Group by model name (removing phase)
            ranking_clf_df['model_name'] = ranking_clf_df['model_name/phase'].apply(lambda x: x.split(' / ')[0])
            clf_rankings = ranking_clf_df.groupby('model_name')['aggregate_score'].mean().reset_index()
            clf_rankings = clf_rankings.sort_values('aggregate_score', ascending=False)
            rankings['classification_rankings'] = clf_rankings

        return {**state, 'model_rankings': rankings}

    def run(self, regression_df: pd.DataFrame = None, classification_df: pd.DataFrame = None):
        """
        Execute the full model ranking workflow
        
        :param regression_df: DataFrame with regression model metrics
        :param classification_df: DataFrame with classification model metrics
        :return: Model rankings
        """
        # Compile the graph
        workflow_app = self.workflow.compile()

        # Initial state
        initial_state = {
            'regression_df': regression_df,
            'classification_df': classification_df
        }

        # Run the workflow
        result = workflow_app.invoke(initial_state)
        return result['model_rankings']

    def generate_workflow_report(self, result: Dict) -> str:
        """
        Generate a detailed HTML report with modern, clean styling and timestamp
        
        :param result: The result dictionary from the model ranking process
        :return: Comprehensive HTML-formatted report
        """
        # Get current date and time
        generation_timestamp = datetime.now().strftime("%B %d, %Y at %I:%M:%S %p")

        model_evalutation_report = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Model Ranking Workflow Report</title>
    <style>
        body {{
            font-family: 'Arial', sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f4f4f4;
        }}
        .container {{
            background-color: white;
            padding: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            border-radius: 8px;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            text-align: center;
        }}
        .timestamp {{
            text-align: center;
            color: #7f8c8d;
            font-size: 0.9em;
            margin-bottom: 20px;
            font-style: italic;
        }}
        h2 {{
            color: #2980b9;
            margin-top: 25px;
        }}
        .section {{
            background-color: #f9f9f9;
            border-left: 4px solid #3498db;
            padding: 15px;
            margin: 15px 0;
        }}
        .insights {{
            background-color: #e8f4f8;
            border-left: 4px solid #2ecc71;
            padding: 15px;
            margin: 15px 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
            box-shadow: 0 2px 3px rgba(0,0,0,0.1);
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        tr:hover {{
            background-color: #e6f2ff;
        }}
        .weights-section {{
            background-color: #f5f9fa;
            border-left: 4px solid #9b59b6;
            padding: 15px;
            margin: 15px 0;
        }}
        .weight-table {{
            width: 100%;
            border-collapse: collapse;
        }}
        .weight-table th {{
            background-color: #9b59b6;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 Model Ranking Workflow Report</h1>
        
        <div class="timestamp">
            Generated on: {generation_timestamp}
        </div>
        
        <div class="section">
            <h2>Workflow Overview</h2>
            <p>The model ranking process follows a structured workflow using LangGraph, which involves three key stages:</p>
            <ol>
                <li>Metric Normalization</li>
                <li>Score Aggregation</li>
                <li>Model Ranking</li>
            </ol>
        </div>
        
        <div class="weights-section">
            <h2>Current Weights Configuration</h2>
            
            <h3>Regression Metric Weights</h3>
            <table class="weight-table">
                <tr>
                    <th>Metric</th>
                    <th>Weight</th>
                </tr>
                <tr>
                    <td>Mean Absolute Error (MAE)</td>
                    <td>{self.reg_weights['mae'] * 100}%</td>
                </tr>
                <tr>
                    <td>Mean Squared Error (MSE)</td>
                    <td>{self.reg_weights['mse'] * 100}%</td>
                </tr>
                <tr>
                    <td>Root Mean Squared Error (RMSE)</td>
                    <td>{self.reg_weights['rmse'] * 100}%</td>
                </tr>
                <tr>
                    <td>R-squared (R²)</td>
                    <td>{self.reg_weights['r2'] * 100}%</td>
                </tr>
                <tr>
                    <td>Explained Variance</td>
                    <td>{self.reg_weights['explained_variance'] * 100}%</td>
                </tr>
            </table>
            
            <h3>Classification Metric Weights</h3>
            <table class="weight-table">
                <tr>
                    <th>Metric</th>
                    <th>Weight</th>
                </tr>
                <tr>
                    <td>Precision</td>
                    <td>{self.clf_weights['precision'] * 100}%</td>
                </tr>
                <tr>
                    <td>Recall</td>
                    <td>{self.clf_weights['recall'] * 100}%</td>
                </tr>
                <tr>
                    <td>Accuracy</td>
                    <td>{self.clf_weights['accuracy'] * 100}%</td>
                </tr>
                <tr>
                    <td>AUC</td>
                    <td>{self.clf_weights['auc'] * 100}%</td>
                </tr>
                <tr>
                    <td>F1-Score</td>
                    <td>{self.clf_weights['f1'] * 100}%</td>
                </tr>
                <tr>
                    <td>Phi Coefficient</td>
                    <td>{self.clf_weights['phi'] * 100}%</td>
                </tr>
                <tr>
                    <td>False Positive Rate (FPR)</td>
                    <td>{self.clf_weights['fpr'] * 100}%</td>
                </tr>
                <tr>
                    <td>False Negative Rate (FNR)</td>
                    <td>{self.clf_weights['fnr'] * 100}%</td>
                </tr>
            </table>
        </div>
        
        <div class="section">
            <h2>1. Metric Normalization Stage</h2>
            <h3>Purpose</h3>
            <p>The normalization stage transforms raw metrics into a comparable scale, ensuring fair evaluation across different models and metrics.</p>
            
            <h4>Regression Metrics Normalization</h4>
            <ul>
                <li>Metrics like MAE, MSE, RMSE are inverted (1 / metric) since lower values are better</li>
                <li>R² and Explained Variance are kept as-is, as higher values indicate better performance</li>
                <li>A small epsilon (1e-10) is added to prevent division by zero</li>
            </ul>
            
            <h4>Classification Metrics Normalization</h4>
            <ul>
                <li>High-is-better metrics (Precision, Recall, Accuracy, AUC) are normalized directly</li>
                <li>Low-is-better metrics (FPR, FNR, Brier Score) are inverted</li>
            </ul>
        </div>
        
        <div class="section">
            <h2>2. Score Aggregation Stage</h2>
            <h3>Purpose</h3>
            <p>Compute a single aggregate score for each model by applying user-defined weighted combinations of normalized metrics.</p>
        </div>
        
        <div class="section">
            <h2>3. Model Ranking Stage</h2>
            <h3>Purpose</h3>
            <p>Rank models based on their aggregate scores, grouping by base model name and calculating mean performance across phases.</p>
        """

        # Add rankings to the report
        if 'regression_rankings' in result:
            model_evalutation_report += """
            <h3>Regression Model Rankings</h3>
            """
            model_evalutation_report += result['regression_rankings'].to_html(index=False, classes='rankings')

        if 'classification_rankings' in result:
            model_evalutation_report += """
            <h3>Classification Model Rankings</h3>
            """
            model_evalutation_report += result['classification_rankings'].to_html(index=False, classes='rankings')

        model_evalutation_report += """
        </div>
        
        <div class="insights">
            <h2>🔍 Key Insights</h2>
            <ul>
                <li>The workflow provides a customizable, weighted evaluation of model performance</li>
                <li>Normalization ensures fair comparison across different metrics and scales</li>
                <li>User-defined weights allow for domain-specific prioritization of metrics</li>
                <li>Aggregate scoring captures multiple performance aspects according to user priorities</li>
            </ul>
        </div>
    </div>
</body>
</html>
        """
        
        return model_evalutation_report
    def calculate_classification_metrics(self, y_true, y_pred, Y_pred_proba_y1, round_decimals=3):
        
        """
        Calculate various classification evaluation metrics based on true and predicted labels.
        
        Args:
            y_true (array-like): True class labels or ground truth values.
            y_pred (array-like): Predicted class labels generated by the model.
            Y_pred_proba_y1 (array-like): Predicted probabilities for class 1.
            round_decimals (int, optional): Number of decimal places to round the computed metrics to.
                                            Default is 3. If None, no rounding will be applied.

        Returns:
            dict: A dictionary containing the computed classification metrics, including descriptive statistics.
        """

        # Compute confusion matrix components (True Positive, False Positive, False Negative, True Negative)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        # Calculate individual metrics using confusion matrix components and other formulas
        total = len(y_true)
        tpr = tp / (tp + fn) if (tp + fn) != 0 else 0
        tnr = tn / (tn + fp) if (tn + fp) != 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) != 0 else 0
        fnr = fn / (tp + fn) if (tp + fn) != 0 else 0
        precision_adj = tp / (tp + fp) if (tp + fp) != 0 else 0
        recall = tpr
        accuracy = (tp + tn) / total
        
        auc = roc_auc_score(y_true, y_pred) if len(set(y_true)) > 1 else 0
        phi = matthews_corrcoef(y_true, y_pred)
        
        f1 = f1_score(y_true, y_pred) if (tp + fp) != 0 and (tp + fn) != 0 else 0
        brier = brier_score_loss(y_true, y_pred) if y_pred is not None else None

        # Calculate descriptive statistics for the predicted probabilities
        mean_y1 = np.mean(Y_pred_proba_y1)
        median_y1 = np.median(Y_pred_proba_y1)
        std_y1 = np.std(Y_pred_proba_y1)
        percentile_25_y1 = np.percentile(Y_pred_proba_y1, 25)
        percentile_50_y1 = np.percentile(Y_pred_proba_y1, 50)
        percentile_75_y1 = np.percentile(Y_pred_proba_y1, 75)
        percentile_95_y1 = np.percentile(Y_pred_proba_y1, 95)

        # Construct a dictionary of metrics for output
        metrics = {
            'Total': total,
            'TPR': tpr,
            'TNR': tnr,
            'FPR': fpr,
            'FNR': fnr,
            'Precision': precision_adj,
            'Recall': recall,
            'Accuracy': accuracy,
            'AUC': auc,
            'Phi Coefficient': phi,
            'F1-Score': f1,
            'TP': tp,
            'FP': fp,
            'FN': fn,
            'TN': tn,
            'Brier Score Loss': brier,
            
            # Descriptive statistics for predicted probabilities
            'Mean Probability Class 1': mean_y1,
            'Median Probability Class 1': median_y1,
            'Std Dev Probability Class 1': std_y1,
            '25th Percentile Probability Class 1': percentile_25_y1,
            '50th Percentile Probability Class 1': percentile_50_y1,
            '75th Percentile Probability Class 1': percentile_75_y1,
            '95th Percentile Probability Class 1': percentile_95_y1,
        }

        # Round the calculated metrics to the specified number of decimal places if requested
        if round_decimals is not None:
            metrics = {key: round(value, round_decimals) if isinstance(value, (int, float)) else value
                    for key, value in metrics.items()}

        return metrics


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
    
    # the regression  function should  also contains average , standard deviation and other statistics metrics this  
    # will  be added if and only if the users have asked for that otherwise it will remain unchanged
    def calculate_regression_metrics(self,y_true, y_pred, round_decimals=3):
        """
        Calculate key regression metrics including Mean Absolute Error (MAE),
        Mean Squared Error (MSE), Root Mean Squared Error (RMSE), R-squared (R²),
        and Explained Variance.

        Args:
            y_true (array-like): The true target values or ground truth values.
            y_pred (array-like): The predicted target values from the regression model.
            round_decimals (int, optional): The number of decimal places to round the computed metrics.
                                             Default is None, meaning no rounding will occur.

        Returns:
            dict: A dictionary containing the computed regression metrics, with the metric names as keys
                  and the corresponding computed values as values.
        """
        # Compute individual regression metrics
        mae = mean_absolute_error(y_true, y_pred)  # Mean Absolute Error (MAE): average absolute difference
        mse = mean_squared_error(y_true, y_pred)  # Mean Squared Error (MSE): average squared difference
        rmse = np.sqrt(mse)  # Root Mean Squared Error (RMSE): square root of MSE
        r2 = r2_score(y_true, y_pred)  # R-squared (R²): proportion of variance explained by the model
        explained_variance = explained_variance_score(y_true,
                                                      y_pred)  # Explained Variance: variability captured by the model

        # Store all metrics in a dictionary
        metrics = {
            'Mean Absolute Error (MAE)': mae,  # Captures average magnitude of errors without considering direction
            'Mean Squared Error (MSE)': mse,  # Penalizes larger errors due to squaring
            'Root Mean Squared Error (RMSE)': rmse,  # Provides errors in the same units as the target variable
            'R-squared (R²)': r2,  # Indicates goodness-of-fit (1.0 is perfect fit, lower values mean worse fit)
            'Explained Variance': explained_variance  # Reflects how well model captures target variability
        }

        # Round metrics if a round_decimals value is provided
        if round_decimals is not None:
            # Rounds each value in the dictionary to the specified decimal places
            metrics = {key: round(value, round_decimals) if isinstance(value, (int, float)) else value
                       for key, value in metrics.items()}

        # Return the dictionary containing all the metrics
        return metrics
    
    
    def get_metrics(self, analysis_type, y_true, y_pred, Y_pred_proba_y1=None):
        """
        Compute and return performance metrics for a given model based on the type of analysis (classification or regression).

        Args:
            model: The trained machine learning model used for prediction.
            analysis_type (str): The type of analysis being performed. Accepts:
                                - "classification" for classification problems.
                                - Any other value for regression problems.
            X (array-like): Input features used for generating predictions.
            Y (array-like): Ground truth target values.

        Returns:
            dict: A dictionary containing the computed metrics based on the analysis type.
                For classification, it includes metrics such as precision, recall, etc.
                For regression, it includes metrics such as MAE, MSE, etc.
        """
        # Generate predictions using the input model
        # Check if the analysis type is classification
        if analysis_type == "classification":

            # Compute classification metrics if the analysis type is "classification"
            return self.calculate_classification_metrics(y_true, y_pred, Y_pred_proba_y1)
        else:
            # Compute regression metrics for other types of analysis
            return self.calculate_regression_metrics(y_true, y_pred)

    

    def sort_table_by_column(self,table_list, column_name):

        """

            Sort the evaluation table by a specified column in ascending order.

            Args:
                table_list (list of pd.DataFrame): List of evaluation result DataFrames.
                column_name (str): The column to sort the table by.

            Returns:
                list of pd.DataFrame: Sorted list of DataFrames.
                

        """
        sorted_tables = []
        for table in table_list:
            # Ensure the column exists in the DataFrame
            if column_name in table.columns:
                sorted_table = table.sort_values(by=column_name, ascending=True).reset_index(drop=True)
                sorted_tables.append(sorted_table)
            else:
                print(f"Column '{column_name}' not found in table. Returning original table.")
                sorted_tables.append(table)
        return sorted_tables

    
    # works perfectly fine and user correctly in the code 

    def generate_score_bin_analysis(self  , model_name, train_df, val_df=None, test_df=None, bins=10,
                                    customized_label="" , default_rate : bool  = True ):
        
        """

            Generates a score bin analysis chart for a classification model using Plotly.

            Args:
                model_name (str): The name of the model, used in the plot title.
                train_df (DataFrame): DataFrame for training data with columns 'prediction' and 'target'.
                val_df (DataFrame, optional): DataFrame for validation data. Default is None.
                test_df (DataFrame, optional): DataFrame for testing data. Default is None.
                bins (int, optional): Number of bins to group predicted probabilities into. Default is 10.
                customized_label (str, optional): Additional label for the plot. Default is "".

            Returns:
                plotly.graph_objects.Figure: The Plotly figure object for the plot.

        """

        # Extract predicted probabilities // get the probalities y(=1)
        train_probs = train_df.iloc[:, -1] if train_df is not None else None 
        val_probs = val_df.iloc[:, -1] if val_df is not None else None
        test_probs = test_df.iloc[:, -1] if test_df is not None else None

        # Create bins for predicted probabilities
        bin_edges = np.linspace(0, 1, bins + 1)
        # train validation and test can be empty lists
        train_bin_counts = np.histogram(train_probs, bins=bin_edges)[0] if train_probs is not None  else  None 
        val_bin_counts = np.histogram(val_probs, bins=bin_edges)[0] if val_probs is not None else None
        test_bin_counts = np.histogram(test_probs, bins=bin_edges)[0] if test_probs is not None else None
        
        # count the total number of  predictio n equal to 1   and falls within  each bin  

        prediction_y1_train   =train_df.iloc[:, -1][train_df.iloc[: , -2] ==1 ] if train_df is not None else None
        prediction_y1_val  = val_df.iloc[:, -1][val_df.iloc[: , -2] ==1 ] if val_df is not None else None
        prediction_y1_test = test_df.iloc[:, -1][test_df.iloc[: , -2] ==1 ] if test_df is not None else None

         
        # count the total numbre of samples that fall within eahc bin and equal to 1 in   for the trainset  
        bin_counts_filtered_train = np.histogram(prediction_y1_train , bin_edges)[0] if prediction_y1_train is not None  else  None 
        bin_counts_filtered_validation = np.histogram(prediction_y1_val , bin_edges)[0] if prediction_y1_val is not None  else  None 
        bin_counts_filtered_test = np.histogram(prediction_y1_test , bin_edges)[0] if prediction_y1_test is not None  else  None 
        

        # if default rate is set to true then in that case will cauclate the percentage of prediction equal 1 within each bin otherwise just the total
        # number of sample or prediction that fall within each bin devided on the total number of predcition or samples 

        # train bin count contains the total number of samples within each  bin 
        # bin count filtered contains the total number of sample  the y=1 within each bin 
        #

        if  default_rate == False   :

            # Normalize counts to get percentages

            
            train_bin_percents = train_bin_counts / len(train_probs) * 100 if train_probs is not None else None
            val_bin_percents = val_bin_counts / len(val_probs) * 100 if val_probs is not None else None
            test_bin_percents = test_bin_counts / len(test_probs) * 100 if test_probs is not None else None
        else   :
            #the problem was that i was deviding on the  len  not on the variable it self 




            train_bin_percents = bin_counts_filtered_train / train_bin_counts * 100 if train_bin_counts is not None else None
            val_bin_percents = bin_counts_filtered_validation / val_bin_counts * 100 if val_bin_counts is not None else None
            test_bin_percents = bin_counts_filtered_test / test_bin_counts* 100 if test_probs is not None else None 
            

        # Create a DataFrame for plotting
        plot_data = {
            "Score Bins": [f"{round(bin_edges[i], 2)}-{round(bin_edges[i + 1], 2)}" for i in range(bins)],
            "Train": train_bin_percents
        }
        if val_bin_percents is not None:
            plot_data["Validation"] = val_bin_percents
        if test_bin_percents is not None:
            plot_data["Test"] = test_bin_percents

        plot_df = pd.DataFrame(plot_data)

        # Create Plotly bar plot
        fig = go.Figure()

        # Add bars for each dataset (Train, Validation, Test)
        if "Train" in plot_data:
            fig.add_trace(go.Bar(
                x=plot_df['Score Bins'],
                y=plot_df['Train'],
                name='Train',
                marker_color="#a1d0ff"
            ))

        if "Validation" in plot_data:
            fig.add_trace(go.Bar(
                x=plot_df['Score Bins'],
                y=plot_df['Validation'],
                name='Validation',
                marker_color="#ffcccb"
            ))

        if "Test" in plot_data:
            fig.add_trace(go.Bar(
                x=plot_df['Score Bins'],
                y=plot_df['Test'],
                name='Test',
                marker_color="#90ee90"
            ))

        # Update layout for better aesthetics and interactivity
        if default_rate   : 

            fig.update_layout(
                title=f"",
                xaxis_title="Score Bins",
                yaxis_title="% of Samples",
                barmode='group',  # Changed from 'stack' to 'group' for side-by-side bars
                xaxis_tickangle=45,
                legend_title="Dataset",
                template='plotly_white'
            )
        else   : 
              fig.update_layout(
                title=f"",
                xaxis_title="Score Bins",
                yaxis_title="% of Samples",
                barmode='group',  # Changed from 'stack' to 'group' for side-by-side bars
                xaxis_tickangle=45,
                legend_title="Dataset",
                template='plotly_white'
            )


        return fig
    
    

    def model_analysis(self , model_names, train_dataframes, validation_dataframes=None, test_dataframes=None,
                       analysis_type="classification", customized_label=""):
        

        """

                Perform model analysis by evaluating metrics on train, validation, and test datasets.

                Args:
                    model_names (list): List of model names as strings.
                    train_dataframes (list): List of dataframes, one per model, containing 'target' and 'prediction' columns.
                    validation_dataframes (list, optional): List of dataframes for validation data. Defaults to None.
                    test_dataframes (list, optional): List of dataframes for test data. Defaults to None.
                    analysis_type (str, optional): Type of analysis ("classification" or "regression"). Defaults to "classification".
                    customized_label (str, optional): A label for custom use, such as figure annotations. Defaults to "".

                Returns:
                    tuple: A list of result tables and a list of generated figures for each model.


        """


        results_list = []

        
        figures_list = {model: [] for model in model_names}

        # Initialize figures_list with a correct dictionary structure
        figures_list = {model: {f"Target Distribution (Density Plot) {model}": [],
                            f"Percentage of Predictions per Score Bin {model}": [], 
                            f"Positive Prediction Rate per Score Bin {model}": []}
                    for model in model_names}
 



        for idx, model_name in enumerate(model_names):
            # Extract train, validation, and test dataframes for the current model
            # simple check if the dataframes list is not empty then in that case take the current dataframe idx else None
            train_df = train_dataframes[idx] if train_dataframes else None 
            val_df = validation_dataframes[idx] if validation_dataframes else None
            test_df = test_dataframes[idx] if test_dataframes else None

            # Initialize a list to store metrics for the current model
            model_results = []

            # Compute metrics for the training phase
            if train_df is not  None  : 
                train_metrics = self.get_metrics(analysis_type, train_df.iloc[:, -3] ,  train_df.iloc[:, -2], train_df.iloc[:, -1])
                train_metrics["model_name/phase"] = f"{model_name} / training_phase"
                model_results.append(train_metrics)

            # Compute metrics for the validation phase, if available

            if val_df is not None:
                val_metrics = self.get_metrics(analysis_type, val_df.iloc[:, -3] , val_df.iloc[:, -2], val_df.iloc[:, -1])
                val_metrics["model_name/phase"] = f"{model_name} / validation_phase"
                model_results.append(val_metrics)

            # Compute metrics for the test phase, if available

            if test_df is not None:

                test_metrics = self.get_metrics(analysis_type, test_df.iloc[:, -3] , test_df.iloc[:, -2], test_df.iloc[:, -1])
                test_metrics["model_name/phase"] = f"{model_name} / test_phase"
                model_results.append(test_metrics)

            # Generate score bin analysis plot

            fig = self.generate_score_bin_analysis(
                model_name,
                train_df,
                val_df,
                test_df,
                customized_label=customized_label
            )

            # solution just add  another plot  but set the boolean to false 
            fig_2 = self.generate_score_bin_analysis(
                model_name,
                train_df,
                val_df,
                test_df,
                customized_label=customized_label , 
                default_rate = False # this will  plot the total number of samples within each bin  
            )



            density_plots = self.generate_density_plots(model_name, train_df,val_df , test_df ,  customized_label= customized_label)


            # Define the desired column order, ensuring 'model_name/phase' is the first column
            column_order = ["model_name/phase"] + [col for col in pd.DataFrame(model_results).columns if
                                                   col != "model_name/phase"]

            # Convert results to a DataFrame and append to the results list
            results_list.append(pd.DataFrame(model_results, columns=column_order))

            # add images to the list 
            for figure in density_plots : 

                figures_list[model_name][f"Target Distribution (Density Plot) {model_name}"].append(figure)

            figures_list[model_name][f"Positive Prediction Rate per Score Bin {model_name}"].append(fig)
            #append the second plot  

            figures_list[model_name][f"Percentage of Predictions per Score Bin {model_name}"].append(fig_2)

    
        return results_list, figures_list

    # model analysis function for  regression task //
    def model_analysis_regression(self , model_names, train_dataframes, validation_dataframes=None, test_dataframes=None,
                       analysis_type="regression", customized_label=""):
        

        """

                Perform model analysis by evaluating metrics on train, validation, and test datasets.

                Args:
                    model_names (list): List of model names as strings.
                    train_dataframes (list): List of dataframes, one per model, containing 'target' and 'prediction' columns.
                    validation_dataframes (list, optional): List of dataframes for validation data. Defaults to None.
                    test_dataframes (list, optional): List of dataframes for test data. Defaults to None.
                    analysis_type (str, optional): Type of analysis ("classification" or "regression"). Defaults to "classification".
                    customized_label (str, optional): A label for custom use, such as figure annotations. Defaults to "".

                Returns:
                    tuple: A list of result tables and a list of generated figures for each model.


        """

        results_list = []


        figures_list = {model: [] for model in model_names}

        # Initialize figures_list with a correct dictionary structure
        figures_list = {model: {f"Prediction  Distribution for model : {model}": []}
                        for model in model_names}

        for idx, model_name in enumerate(model_names):
            # Extract train, validation, and test dataframes for the current model
            # simple check if the dataframes list is not empty then in that case take the current dataframe idx else None
            train_df = train_dataframes[idx] if train_dataframes else None 
            val_df = validation_dataframes[idx] if validation_dataframes else None
            test_df = test_dataframes[idx] if test_dataframes else None

            # Initialize a list to store metrics for the current model
            model_results = []

            # Compute metrics for the training phase
            if train_df is not  None  : 
                train_metrics = self.get_metrics(analysis_type,  train_df.iloc[:, -2], train_df.iloc[:, -1])
                train_metrics["model_name/phase"] = f"{model_name} / training_phase"
                model_results.append(train_metrics)

            # Compute metrics for the validation phase, if available

            if val_df is not None:
                val_metrics = self.get_metrics(analysis_type, val_df.iloc[:, -2], val_df.iloc[:, -1])
                val_metrics["model_name/phase"] = f"{model_name} / validation_phase"
                model_results.append(val_metrics)

            # Compute metrics for the test phase, if available

            if test_df is not None:

                test_metrics = self.get_metrics(analysis_type, test_df.iloc[:, -2], test_df.iloc[:, -1])
                test_metrics["model_name/phase"] = f"{model_name} / test_phase"
                model_results.append(test_metrics)

            # Define the desired column order, ensuring 'model_name/phase' is the first column
            column_order = ["model_name/phase"] + [col for col in pd.DataFrame(model_results).columns if
                                                   col != "model_name/phase"]

            # Convert results to a DataFrame and append to the results list
            results_list.append(pd.DataFrame(model_results, columns=column_order))

            density_plots = self.generate_histograms_regression_use_case(model_name, train_df,val_df , test_df ,  customized_label= customized_label)
            
            # add images to the list 
            for figure in density_plots : 

                figures_list[model_name][f"Prediction  Distribution for model : {model_name}"].append(figure)


        return  results_list, figures_list


    def generate_density_plots(self, model_name, train_df, val_df=None, test_df=None, customized_label=""):

        """
        Dynamically generates density plots based on the available datasets.
        """


        # Filter available datasets
        datasets = {
            "Train": train_df,
            "Validation": val_df,
            "Test": test_df
        }
        available_datasets = {key: df for key, df in datasets.items() if df is not None}

        # Initialize an empty list to store the generated plots
        plots = []

        # Dynamically create subplots based on the number of available datasets
        num_datasets = len(available_datasets)
        if num_datasets == 0:
            print("No datasets available to plot.")
            return plots  # Return an empty list if no datasets are available

        # Create a figure with dynamic subplots
        fig, axs = plt.subplots(num_datasets, 1, figsize=(14, 4 * num_datasets))  # Adjust height for readability
        if num_datasets == 1:
            axs = [axs]  # Ensure axs is always iterable

        # Loop through available datasets and plot
        for ax, (name, df) in zip(axs, available_datasets.items()):
            sns.kdeplot(df, x="Prediction_Probabilities y(1)", hue="Prediction", fill=True, ax=ax)
            ax.set_title(f"Density Plot of Predictions ({name}) - {model_name}/{customized_label}")
            ax.set_xlabel("Predicted Probability")
            ax.set_ylabel("Density")
            ax.grid(axis="y", linestyle="--", alpha=0.7)
            # Annotate with the number of data points
            ax.text(0.5, 0.7, f'N={len(df)}', transform=ax.transAxes, ha='center', va='center', fontsize=12, color='black')

        # Adjust layout
        plt.tight_layout()
        plots.append(fig)

        return plots

    def generate_histograms_regression_use_case(self, model_name, train_df, val_df=None, test_df=None, customized_label=""):
        """
          Dynamically generates histograms based on the available datasets.
        """

        # Filter available datasets
        datasets = {
            "Train": train_df,
            "Validation": val_df,
            "Test": test_df
        }

        available_datasets = {key: df for key, df in datasets.items() if df is not None}

        # Initialize an empty list to store the generated plots
        plots = []

        # Dynamically create subplots based on the number of available datasets
        num_datasets = len(available_datasets)
        if num_datasets == 0:
            print("No datasets available to plot.")
            return plots  # Return an empty list if no datasets are available

        # Create a figure with dynamic subplots
        fig, axs = plt.subplots(num_datasets, 1, figsize=(14, 4 * num_datasets))  # Adjust height for readability
        if num_datasets == 1:
            axs = [axs]  # Ensure axs is always iterable

        # Loop through available datasets and plot
        for ax, (name, df) in zip(axs, available_datasets.items()):
            # Plot histogram for predicted probabilities (Prediction_Probabilities y(1))
            sns.histplot( df.iloc[:, -1], kde=False, bins=30, ax=ax, color='skyblue', edgecolor='black')
            ax.set_title(f"Prediction Distribution ({name}) - {model_name}/{customized_label}")
            ax.set_xlabel("Predicted Probability")
            ax.set_ylabel("Frequency")
            ax.grid(axis="y", linestyle="--", alpha=0.7)
            # Annotate with the number of data points
            ax.text(0.5, 0.7, f'N={len(df)}', transform=ax.transAxes, ha='center', va='center', fontsize=12, color='black')

        # Adjust layout
        plt.tight_layout()
        plots.append(fig)

        return plots

    

    def generate_score_bin_analysis_model_comparison(self, model_dict, train_dataframes, validation_dataframes=None,
                                      test_dataframes=None, bins=10, customized_label=""):
        """
        Generate a score bin analysis for different models on training, validation, and test dataframes.
        The analysis will include interactive bar charts with Plotly.

        Args:
            model_dict (dict): A dictionary of models and their corresponding dataframes.
            train_dataframes (dict): A dictionary of training dataframes for each model.
            validation_dataframes (dict, optional): A dictionary of validation dataframes for each model.
            test_dataframes (dict, optional): A dictionary of test dataframes for each model.
            bins (int, optional): Number of bins to use for the histogram. Default is 10.
            customized_label (str, optional): Custom label to append to the plot titles.

        Returns:
            tuple: Plotly figures for training, validation, and test data analyses.
        """

        # Initialize a DataFrame to hold the bin percentages
        bin_edges = np.linspace(0, 1, bins + 1)
        plot_data = pd.DataFrame({
            "Score Bins": [f"{bin_edges[i]:.2f}-{bin_edges[i + 1]:.2f}" for i in range(len(bin_edges) - 1)]
        })

        # Example colors for different models
        base_colors = ['#ffcccb', '#a1d0ff', '#f0e68c', '#d3ffb3']
        colors = base_colors * ((len(model_dict) // len(base_colors)) + 1)

        # Loop through each model in the dictionary
        for (model_name, model), color in zip(model_dict.items(), colors):
            # Get the predicted probabilities for train, validation, and test sets
            train_probs = train_dataframes[model_name].iloc[:, -1]
            val_probs = validation_dataframes[model_name].iloc[:, -1] if validation_dataframes else None
            test_probs = test_dataframes[model_name].iloc[:, -1] if test_dataframes else None

            # Calculate histogram counts for each dataset
            train_bin_counts, _ = np.histogram(train_probs, bins=bin_edges)
            val_bin_counts = np.histogram(val_probs, bins=bin_edges)[0] if val_probs is not None else None
            test_bin_counts = np.histogram(test_probs, bins=bin_edges)[0] if test_probs is not None else None

            # Convert counts to percentages
            train_bin_percents = train_bin_counts / len(train_probs) * 100
            val_bin_percents = val_bin_counts / len(val_probs) * 100 if val_probs is not None else np.zeros(bins)
            test_bin_percents = test_bin_counts / len(test_probs) * 100 if test_probs is not None else np.zeros(bins)

            # Add the percentages to the DataFrame for each model
            plot_data[model_name + " Train"] = train_bin_percents
            if val_probs is not None:
                plot_data[model_name + " Validation"] = val_bin_percents
            if test_probs is not None:
                plot_data[model_name + " Test"] = test_bin_percents

        # Create Plotly bar plots for each data set (training, validation, test)
        fig_train = go.Figure()
        for model_name, color in zip(model_dict.keys(), colors[:len(model_dict)]):
            fig_train.add_trace(go.Bar(
                x=plot_data['Score Bins'],
                y=plot_data[model_name + " Train"],
                name=f'{model_name} Train',
                marker_color=color
            ))

        fig_train.update_layout(
            title=f"Score Bin Analysis  ( y=1 rate within each bin )  - Training/{customized_label}",
            xaxis_title="Score Bins",
            yaxis_title="% of Samples",
            barmode='group',  # Changed from 'stack' to 'group' for side-by-side bars
            xaxis_tickangle=45,
            legend_title="Model",
            template='plotly_white'
        )

        fig_val = None
        if validation_dataframes:
            fig_val = go.Figure()
            for model_name, color in zip(model_dict.keys(), colors[:len(model_dict)]):
                fig_val.add_trace(go.Bar(
                    x=plot_data['Score Bins'],
                    y=plot_data[model_name + " Validation"],
                    name=f'{model_name} Validation',
                    marker_color=color
                ))

            fig_val.update_layout(
                title=f"Score Bin Analysis  ( y=1 rate within each bin ) - Validation/{customized_label}",
                xaxis_title="Score Bins",
                yaxis_title="% of Samples",
                barmode='group',  # Changed from 'stack' to 'group' for side-by-side bars
                xaxis_tickangle=45,
                legend_title="Model",
                template='plotly_white'
            )

        fig_test = None
        if test_dataframes:
            fig_test = go.Figure()
            for model_name, color in zip(model_dict.keys(), colors[:len(model_dict)]):
                fig_test.add_trace(go.Bar(
                    x=plot_data['Score Bins'],
                    y=plot_data[model_name + " Test"],
                    name=f'{model_name} Test',
                    marker_color=color
                ))

            fig_test.update_layout(
                title=f"Score Bin Analysis ( y=1 rate within each bin ) - Test/{customized_label}",
                xaxis_title="Score Bins",
                yaxis_title="% of Samples",
                barmode='group',  # Changed from 'stack' to 'group' for side-by-side bars
                xaxis_tickangle=45,
                legend_title="Model",
                template='plotly_white'
            )
            

        return fig_train, fig_val, fig_test
    
    #this can be removed not used in the code above 

    def model_comparison(self , model_names, train_dataframes, validation_dataframes=None, test_dataframes=None,
                         analysis_type="classification", customized_label=""):
        """
        Evaluates a list of machine learning model predictions on training, validation, and optional test datasets,
        providing detailed evaluation metrics and score bin analysis for each model.

        Args:
            model_names (list): A list of names corresponding to the machine learning models.
            train_dataframes (list): A list of DataFrames, each containing `target` and `predictions` columns for training data.
            validation_dataframes (list, optional): A list of DataFrames, each containing `target` and `predictions` columns for validation data. Default is None.
            test_dataframes (list, optional): A list of DataFrames, each containing `target` and `predictions` columns for test data. Default is None.
            analysis_type (str, optional): Specifies the type of analysis to be performed, either 'classification' or 'regression'. Default is "classification".
            customized_label (str, optional): A custom label for score bin analysis plots.

        Returns:
            tuple: A tuple containing:
                - list of DataFrames: Each DataFrame contains evaluation metrics for the models across train, validation, and test datasets.
                - list of matplotlib figures: Plots of the score bin analysis for train, validation, and test datasets.
        """


        # Initialize dictionaries to store the metrics for each phase (train, validation, test)
        train_metrics = {}
        validation_metrics = {}
        test_metrics = {}

        # Validate input lengths
        if len(model_names) != len(train_dataframes):
            raise ValueError("The number of model names must match the number of train dataframes.")
        if validation_dataframes and len(model_names) != len(validation_dataframes):
            raise ValueError("The number of model names must match the number of validation dataframes.")
        if test_dataframes and len(model_names) != len(test_dataframes):
            raise ValueError("The number of model names must match the number of test dataframes.")

        # Loop through the models and compute metrics
        for i, model_name in enumerate(model_names):
            # Extract the training metrics

            if train_dataframes:

                train_df = train_dataframes[i]
                train_metrics[model_name] = self.get_metrics(analysis_type , train_df.iloc[:, -3] ,  train_df.iloc[:, -2], train_df.iloc[:, -1] )

            # Extract the validation metrics if available

            if validation_dataframes:

                val_df = validation_dataframes[i]
                validation_metrics[model_name] = self.get_metrics(analysis_type, val_df.iloc[:, -3] , val_df.iloc[:, -2] ,  val_df.iloc[:, -1])

            # Extract the test metrics if available

            if test_dataframes:

                test_df = test_dataframes[i]
                test_metrics[model_name] = self.get_metrics(analysis_type,  test_df.iloc[:, -3] , test_df.iloc[:, -2], test_df.iloc[:, -1])

        # Convert the metrics dictionaries to DataFrames
        train_df = pd.DataFrame(train_metrics).transpose()
        validation_df = pd.DataFrame(validation_metrics).transpose() if validation_metrics else None
        test_df = pd.DataFrame(test_metrics).transpose() if test_metrics else None

        # Add a prefix for the dataset and phase in the index (metrics) to make it clearer
        train_df["model_name/phase"] = [f"{model_name} /train" for model_name in train_df.index]
        if validation_df is not None:
            validation_df["model_name/phase"] = [f"{model_name} /validation" for model_name in validation_df.index]
        if test_df is not None:
            test_df["model_name/phase"] = [f"{model_name} /test" for model_name in test_df.index]

        # Ensure consistent column ordering
        order = ["model_name/phase"] + [col for col in train_df.columns if col != "model_name/phase"]
        result_tables = [pd.DataFrame(table[order]) for table in [train_df, validation_df, test_df] if
                         table is not None]

        # Prepare data for score bin analysis
        train_data_dict = {model_names[i]: train_dataframes[i] for i in range(len(model_names))}
        validation_data_dict = {model_names[i]: validation_dataframes[i] for i in
                                range(len(model_names))} if validation_dataframes else None
        test_data_dict = {model_names[i]: test_dataframes[i] for i in
                          range(len(model_names))} if test_dataframes else None

        # Generate score bin analysis
        fig_train, fig_val, fig_test = self.generate_score_bin_analysis_model_comparison(
            model_dict={model_name: None for model_name in model_names},  # Models aren't used in this function
            train_dataframes=train_data_dict,
            validation_dataframes=validation_data_dict,
            test_dataframes=test_data_dict,
            bins=10,
            customized_label=customized_label
        )

        # Return the result tables and figures
        return result_tables, [fig_train, fig_val, fig_test]
    
    def non_time_based_segmentation_model_analysis(self, model_names, train_dataframes_list, validation_dataframes_list,
                                                   test_dataframes_list=None, subcategory_column=None,
                                                   task_type="classification", subcategory_threshold=100):
        """
        Perform non-time-based segmentation analysis for models and return results or a boolean flag if subcategories exceed a threshold.

        Args:
            model_names (list): List of model names to evaluate.
            train_dataframes_list (list): List of training dataframes, one for each model.
            validation_dataframes_list (list): List of validation dataframes, one for each model.
            test_dataframes_list (list, optional): List of test dataframes, one for each model. Default is None.
            subcategory_column (str): The column used for segmentation.
            task_type (str): Type of task for analysis (e.g., "classification" or "regression"). Default is "classification".
            subcategory_threshold (int): Maximum number of subcategories before returning only a boolean flag. Default is 100.

        Returns:
            tuple or bool: If the number of subcategories is below the threshold, returns:
                           - subcategory_results (dict): Segmented dataframes for each subcategory.
                           - figures_results (dict): Segmented figures for each subcategory.
                           If the number of subcategories exceeds the threshold, returns:
                           - bool: `True`.
        """
        # Ensure subcategory_column is provided
        if subcategory_column is None:
            raise ValueError("A subcategory_column must be specified to split the dataset.")

        # Identify unique subcategories in the training data (based on one model's train dataframe)
        # the train proportion will include the most of the data this is why im getting first the unique values from the 
        # train list then the unique values from the validation list  

        if train_dataframes_list : 
        
            unique_subcategories = train_dataframes_list[0][subcategory_column].unique()
        else  : 
            unique_subcategories = validation_dataframes_list[0][subcategory_column].unique()


        # Check if the number of subcategories exceeds the threshold
        if len(unique_subcategories) > subcategory_threshold:
            return True ,None ,None

        # Initialize a dictionary to hold results for each subcategory
        subcategory_results = {}
        figures_results = {}

        # Iterate over each subcategory
        for subcategory in unique_subcategories:
            # Initialize lists to store results for each model
            results = []
            figures = []

            # Iterate over each model
            for i, model_name in enumerate(model_names):
                
                # make sure that the list not empty
                sub_X_train, sub_Y_train = None, None

                if train_dataframes_list:
                    # Filter training, validation, and testing data by subcategory for the current model
                    sub_X_train = train_dataframes_list[i][
                        train_dataframes_list[i][subcategory_column] == subcategory].drop(columns=[subcategory_column])
                    sub_Y_train = train_dataframes_list[i][
                        train_dataframes_list[i][subcategory_column] == subcategory].drop(columns=[subcategory_column])

                # Validate and test data (if provided)
                sub_X_val, sub_Y_val = None, None
                if validation_dataframes_list:
                    sub_X_val = validation_dataframes_list[i][
                        validation_dataframes_list[i][subcategory_column] == subcategory].drop(
                        columns=[subcategory_column])
                    sub_Y_val = validation_dataframes_list[i][
                        validation_dataframes_list[i][subcategory_column] == subcategory]

                sub_X_test, sub_Y_test = None, None
                if test_dataframes_list :
                    sub_X_test = test_dataframes_list[i][
                        test_dataframes_list[i][subcategory_column] == subcategory].drop(columns=[subcategory_column])
                    sub_Y_test = test_dataframes_list[i][test_dataframes_list[i][subcategory_column] == subcategory]

                # Perform model analysis for the current subcategory and model
                model_results, model_figures = self.model_analysis(
                    model_names=[model_name],  # Only passing one model name
                    train_dataframes=[sub_X_train],  # List of dataframes for the model
                    validation_dataframes=[sub_X_val] if sub_X_val is not None else None,
                    test_dataframes=[sub_X_test] if sub_X_test is not None else None,
                    analysis_type=task_type,
                    customized_label=subcategory
                )
                

                # Store the results for each model
                for j, table in enumerate(model_results):
                    # Add 'subcategory' column and rearrange the order
                    table["subcategory"] = subcategory
                    order = ["model_name/phase"] + ["subcategory"] + [col for col in table.columns if
                                                                      col != "subcategory" and col != "model_name/phase"]
                    model_results[j] = table[order]

                # Append the results and figures for this model
                results.extend(model_results)
                figures.append(model_figures)

            # Store the results for the subcategory
            subcategory_results[subcategory] = results
            figures_results[subcategory] = figures

        return False , subcategory_results, figures_results
    

    def non_time_based_segmentation_model_analysis_regression(self, model_names, train_dataframes_list, validation_dataframes_list,
                                                   test_dataframes_list=None, subcategory_column=None,
                                                   task_type="regression", subcategory_threshold=100):
        """
        Perform non-time-based segmentation analysis for models and return results or a boolean flag if subcategories exceed a threshold.

        Args:
            model_names (list): List of model names to evaluate.
            train_dataframes_list (list): List of training dataframes, one for each model.
            validation_dataframes_list (list): List of validation dataframes, one for each model.
            test_dataframes_list (list, optional): List of test dataframes, one for each model. Default is None.
            subcategory_column (str): The column used for segmentation.
            task_type (str): Type of task for analysis (e.g., "classification" or "regression"). Default is "classification".
            subcategory_threshold (int): Maximum number of subcategories before returning only a boolean flag. Default is 100.

        Returns:
            tuple or bool: If the number of subcategories is below the threshold, returns:
                           - subcategory_results (dict): Segmented dataframes for each subcategory.
                           - figures_results (dict): Segmented figures for each subcategory.
                           If the number of subcategories exceeds the threshold, returns:
                           - bool: `True`.
        """
        # Ensure subcategory_column is provided
        if subcategory_column is None:
            raise ValueError("A subcategory_column must be specified to split the dataset.")

        # Identify unique subcategories in the training data (based on one model's train dataframe)
        # the train proportion will include the most of the data this is why im getting first the unique values from the 
        # train list then the unique values from the validation list  

        if train_dataframes_list : 
        
            unique_subcategories = train_dataframes_list[0][subcategory_column].unique()
        else  : 
            unique_subcategories = validation_dataframes_list[0][subcategory_column].unique()


        # Check if the number of subcategories exceeds the threshold
        if len(unique_subcategories) > subcategory_threshold:
            return True ,None ,None

        # Initialize a dictionary to hold results for each subcategory
        subcategory_results = {}
        figures_results = {}

        # Iterate over each subcategory
        for subcategory in unique_subcategories:
            # Initialize lists to store results for each model
            results = []
            figures = [] 


            # Iterate over each model
            for i, model_name in enumerate(model_names):
                
                # make sure that the list not empty
                sub_X_train, sub_Y_train = None, None

                if train_dataframes_list:
                    # Filter training, validation, and testing data by subcategory for the current model
                    sub_X_train = train_dataframes_list[i][
                        train_dataframes_list[i][subcategory_column] == subcategory].drop(columns=[subcategory_column])
                    sub_Y_train = train_dataframes_list[i][
                        train_dataframes_list[i][subcategory_column] == subcategory].drop(columns=[subcategory_column])

                # Validate and test data (if provided)
                sub_X_val, sub_Y_val = None, None
                if validation_dataframes_list:
                    sub_X_val = validation_dataframes_list[i][
                        validation_dataframes_list[i][subcategory_column] == subcategory].drop(
                        columns=[subcategory_column])
                    sub_Y_val = validation_dataframes_list[i][
                        validation_dataframes_list[i][subcategory_column] == subcategory]

                sub_X_test, sub_Y_test = None, None
                if test_dataframes_list :
                    sub_X_test = test_dataframes_list[i][
                        test_dataframes_list[i][subcategory_column] == subcategory].drop(columns=[subcategory_column])
                    sub_Y_test = test_dataframes_list[i][test_dataframes_list[i][subcategory_column] == subcategory]

                # Perform model analysis for the current subcategory and model
                model_results , plots = self.model_analysis_regression(
                    model_names=[model_name],  # Only passing one model name
                    train_dataframes=[sub_X_train],  # List of dataframes for the model
                    validation_dataframes=[sub_X_val] if sub_X_val is not None else None,
                    test_dataframes=[sub_X_test] if sub_X_test is not None else None,
                    analysis_type=task_type,
                    customized_label=subcategory
                )
                

                # Store the results for each model
                for j, table in enumerate(model_results):
                    # Add 'subcategory' column and rearrange the order
                    table["subcategory"] = subcategory
                    order = ["model_name/phase"] + ["subcategory"] + [col for col in table.columns if
                                                                      col != "subcategory" and col != "model_name/phase"]
                    model_results[j] = table[order]

                # Append the results and figures for this model
                results.extend(model_results)
                figures.append(plots)


            # Store the results for the subcategory
            subcategory_results[subcategory] = results
            figures_results[subcategory] = figures


        return False , subcategory_results , figures_results

    def non_time_based_segmentation_model_comparison(
            self,
            model_names,
            train_dataframes_list,
            validation_dataframes_list=None,
            test_dataframes_list=None,
            subcategory_column=None,
            task_type="classification",
            subcategory_threshold=10  # Added threshold parameter
    ):
        """
        Perform segmentation-based model comparison for multiple models.

        **Purpose**:
        This function segments the dataset based on a specified subcategory column and evaluates the performance
        of different models on each subcategory. It supports training, validation, and optional test datasets.

        **Parameters**:
        - model_names: List of model names to evaluate.
        - train_dataframes_list: List of training dataframes for each model.
        - validation_dataframes_list: (Optional) List of validation dataframes for each model.
        - test_dataframes_list: (Optional) List of test dataframes for each model.
        - subcategory_column: Column name used for segmentation.
        - task_type: Type of task, e.g., "classification" or "regression" (default is "classification").
        - subcategory_threshold: Maximum number of unique subcategories allowed for analysis (default is 10).

        **Returns**:
        - subcategory_results: Dictionary containing evaluation results and visualizations for each subcategory.
        """
        # Ensure subcategory_column is provided
        if subcategory_column is None:
            raise ValueError("A subcategory_column must be specified to split the dataset.")

        # Identify unique subcategories in the training data (based on the first model's training dataframe)
        unique_subcategories = train_dataframes_list[0][subcategory_column].unique()

        # Check if the number of subcategories exceeds the threshold
        if len(unique_subcategories) > subcategory_threshold:
            return True ,None

        # Initialize a dictionary to hold results for each subcategory
        subcategory_results = {}

        # Iterate over each subcategory
        for subcategory in unique_subcategories:
            # Segment the data for all models based on the subcategory
            segmented_X_train_list = []
            segmented_Y_train_list = []
            segmented_X_val_list = []
            segmented_Y_val_list = []
            segmented_X_test_list = []
            segmented_Y_test_list = []

            # Process training data for all models
            for i in range(len(model_names)):
                
                
                # Filter training data for the current subcategory
                if train_dataframes_list  is not None:

                    sub_train = train_dataframes_list[i][train_dataframes_list[i][subcategory_column] == subcategory]
                    segmented_X_train_list.append(sub_train.drop(columns=[subcategory_column]))
                    segmented_Y_train_list.append(
                        sub_train[subcategory_column])  # Assuming the target column is the same as the subcategory column

                # Filter validation data for the current subcategory (if provided)
                if validation_dataframes_list is not None:
                    sub_val = validation_dataframes_list[i][
                        validation_dataframes_list[i][subcategory_column] == subcategory]
                    segmented_X_val_list.append(sub_val.drop(columns=[subcategory_column]))
                    segmented_Y_val_list.append(sub_val[subcategory_column])

                # Filter test data for the current subcategory (if provided)
                if test_dataframes_list is not None:
                    sub_test = test_dataframes_list[i][test_dataframes_list[i][subcategory_column] == subcategory]
                    segmented_X_test_list.append(sub_test.drop(columns=[subcategory_column]))
                    segmented_Y_test_list.append(sub_test[subcategory_column])

            # Call model_comparison with all models and their segmented data for the current subcategory
            model_results, model_figures = self.model_comparison(
                model_names=model_names,  # Pass all model names at once
                train_dataframes=segmented_X_train_list,  # Segmented training data for all models
                validation_dataframes=segmented_X_val_list if validation_dataframes_list is not None else None,
                test_dataframes=segmented_X_test_list if test_dataframes_list is not None else None,
                analysis_type=task_type,
                customized_label=subcategory  # Label results with the subcategory
            )

            # Add subcategory column to the results and restructure the data
            for j, table in enumerate(model_results):
                table["subcategory"] = subcategory
                order = ["model_name/phase"] + ["subcategory"] + [col for col in table.columns if
                                                                  col not in {"subcategory", "model_name/phase"}]
                model_results[j] = table[order]

            # Store the results and figures for the current subcategory
            subcategory_results[subcategory] = {
                "results": model_results,
                "figures": model_figures
            }

        return False ,subcategory_results
    
    # modidification made on this  : is that the the function will be dependent from the validation list acutually

    def time_based_segmentation_model_analysis(
        self,
        model_names,
        train_dataframes_list=None,
        validation_dataframes_list=None,
        test_dataframes_list=None,
        timestamp_column=None,
        number_of_months=4,
        analysis_type="classification"
    ):
        """
        Evaluates machine learning models for different months in the dataset and generates structured results
        for each phase (train, validation, test) for each model and month, each in a separate DataFrame.

        Args:
            model_names (list): A list of model names corresponding to the models in the 'train_dataframes_list'.
            train_dataframes_list (list, optional): A list of training DataFrames with 'Target' and 'Prediction' columns for each model. Default is [].
            validation_dataframes_list (list, optional): A list of validation DataFrames with 'Target' and 'Prediction' columns for each model. This is mandatory and should be provided.
            test_dataframes_list (list, optional): A list of test DataFrames with 'Target' and 'Prediction' columns for each model. Default is [].
            timestamp_column (str, optional): The name of the timestamp column. Default is None.
            number_of_months (int): Number of months to consider for analysis from the most recent month. Default is 4.
            analysis_type (str, optional): Specifies the type of analysis, either 'classification' or 'regression'. Default is "classification".

        Returns:
            dict: A dictionary of DataFrames where each key is a model name and each value is the corresponding model's metrics.
        """
        # Input validation
        if len(validation_dataframes_list) ==  0 :
            raise ValueError("Validation dataframes list must be provided.")

        if timestamp_column is None:
            raise ValueError("A timestamp_column must be specified to filter the dataset by date.")

        if number_of_months <= 0:
            raise ValueError("The number_of_months must be a positive integer.")

        # Default to empty DataFrames if train or test lists are not provided
        if len(train_dataframes_list) == 0 :
            train_dataframes_list = [pd.DataFrame() for _ in model_names]

        if len(test_dataframes_list) == 0 :
            test_dataframes_list = [pd.DataFrame() for _ in model_names]

        # Ensure timestamp column is in datetime format
        for df_list in [train_dataframes_list, validation_dataframes_list, test_dataframes_list]:
            for df in df_list:
                if not df.empty:
                    df[timestamp_column] = pd.to_datetime(df[timestamp_column])

        # Determine the latest date and months to consider based on training or validation data
        if any(not df.empty for df in train_dataframes_list):
            latest_date = max((df[timestamp_column].max() for df in train_dataframes_list if not df.empty))
        else:
            latest_date = max((df[timestamp_column].max() for df in validation_dataframes_list if not df.empty))

        months_to_consider = pd.date_range(end=latest_date, periods=number_of_months, freq='M')
        # Initialize results dictionary
        model_metrics_dict = {}

        # Iterate over models and their corresponding DataFrames
        for model_name, train_df, val_df, test_df in zip(model_names, train_dataframes_list, validation_dataframes_list, test_dataframes_list):

            training_metrics, validation_metrics, testing_metrics = [], [], []

            for month in months_to_consider:
                month_str = month.strftime('%Y-%m')

                # Filter data for the current month
                train_month_df = train_df[train_df[timestamp_column].dt.to_period('M') == month.to_period('M')] if not train_df.empty else pd.DataFrame()
                val_month_df = val_df[val_df[timestamp_column].dt.to_period('M') == month.to_period('M')]
                test_month_df = test_df[test_df[timestamp_column].dt.to_period('M') == month.to_period('M')] if not test_df.empty else pd.DataFrame()

                # Compute metrics for each phase if data is available

                if not train_month_df.empty:
                    if analysis_type == "classification":
                        train_metrics = self.get_metrics(
                            analysis_type,
                            train_month_df.iloc[:, -3],  # Target column
                            train_month_df.iloc[:, -2],  # Prediction column 1
                            train_month_df.iloc[:, -1]   # Additional column 2
                        )
                        train_metrics.update({"model_name/phase": f"{model_name} / train", "month": month_str})
                        training_metrics.append(train_metrics)
                    else  : 
                        train_metrics = self.get_metrics(
                            analysis_type,
                            train_month_df.iloc[:, -2],  # Target column
                            train_month_df.iloc[:, -1]   # Prediction  column
                        )
                        train_metrics.update({"model_name/phase": f"{model_name} / train", "month": month_str})
                        training_metrics.append(train_metrics)

                if not val_month_df.empty:
                    if analysis_type == "classification":
                        val_metrics = self.get_metrics(
                            analysis_type,
                            val_month_df.iloc[:, -3],
                            val_month_df.iloc[:, -2],
                            val_month_df.iloc[:, -1]
                        )
                        val_metrics.update({"model_name/phase": f"{model_name} / validation", "month": month_str})
                        validation_metrics.append(val_metrics)
                    else  : 
                        val_metrics = self.get_metrics(
                            analysis_type,
                            val_month_df.iloc[:, -2],
                            val_month_df.iloc[:, -1]
                        )
                        val_metrics.update({"model_name/phase": f"{model_name} / validation", "month": month_str})
                        validation_metrics.append(val_metrics)


                if not test_month_df.empty:
                    if analysis_type == "classification":
                        test_metrics = self.get_metrics(
                                analysis_type,
                                test_month_df.iloc[:, -3],
                                test_month_df.iloc[:, -2],
                                test_month_df.iloc[:, -1]
                            )
                        test_metrics.update({"model_name/phase": f"{model_name} / test", "month": month_str})
                        testing_metrics.append(test_metrics)
                    else  : 
                            test_metrics = self.get_metrics(
                                analysis_type,
                                test_month_df.iloc[:, -2],
                                test_month_df.iloc[:, -1]
                            )
                            test_metrics.update({"model_name/phase": f"{model_name} / test", "month": month_str})
                            testing_metrics.append(test_metrics)


            # Convert metrics lists to DataFrames
            train_df = pd.DataFrame(training_metrics)
            validation_df = pd.DataFrame(validation_metrics)
            test_df = pd.DataFrame(testing_metrics)

            # Order columns for consistency
            order = ["model_name/phase", "month"] + [col for col in validation_df.columns if col not in ["month", "model_name/phase"]]
            model_metrics_dict[model_name] = [
                train_df[order] if not train_df.empty else pd.DataFrame(columns=order),
                validation_df[order] if not validation_df.empty else pd.DataFrame(columns=order),
                test_df[order] if not test_df.empty else pd.DataFrame(columns=order),
            ]

        return model_metrics_dict



   



