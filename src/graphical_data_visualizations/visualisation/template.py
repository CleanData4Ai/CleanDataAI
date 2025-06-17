"""
Base HTML Template for Generating Reports

This base HTML template is used to generate dynamic reports with embedded user-defined CSS and content sections. It leverages Bootstrap for responsive design and Font Awesome for icons. The template is structured to include a navigation bar and a main content area, allowing for a customizable and flexible report layout.

Template Structure:
1. **DOCTYPE Declaration**: Defines the document type and version of HTML.
2. **HTML Head Section**:
   - **Meta Tags**: Includes character set and viewport settings for responsiveness.
   - **CSS Links**: Imports Bootstrap and Font Awesome stylesheets.
   - **Inline CSS**: Allows for custom CSS styling via `{{css_code}}`.
3. **HTML Body Section**:
   - **Navigation Bar**:
     - **Header**: Displays the logo and title, with placeholders for dynamic content (`{{logo_section}}`, `{{title}}`).
     - **Content**: Lists navigation options with a placeholder for dynamic content (`{{options_list}}`).
   - **Main Content Area**: The primary section where dynamic report content is inserted (`{{sections_html}}`).
4. **JavaScript**: Includes Bootstrap's JavaScript bundle for interactive components.

Usage:
- **`{{css_code}}`**: Placeholder for custom CSS code to be applied within the `<style>` tags.
- **`{{logo_section}}`**: Placeholder for the logo section HTML.
- **`{{title}}`**: Placeholder for the navigation bar title.
- **`{{options_list}}`**: Placeholder for a list of navigation options.
- **`{{sections_html}}`**: Placeholder for the main content HTML sections.

Note: Ensure no extra spaces or characters are included in the placeholders to maintain proper formatting.
"""

base_template = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <!-- Bootstrap CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <!-- Font Awesome CSS -->
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.1.1/css/all.min.css" rel="stylesheet">
    <style>
        {{css_code}} <!-- Placeholder for custom CSS; ensure no extra spaces or characters -->
    </style>
</head>
<body>
    <div class="container-fluid">
        <div class="row">
            <!-- Navigation Bar -->
            <div id="nav-bar" class="col-auto">
                <div id="nav-header" class="d-flex align-items-center">
                    {{logo_section}} <!-- Placeholder for logo section; ensure no extra spaces or characters -->
                    <h2 id="nav-title">{{title}}</h2> <!-- Placeholder for navigation title -->
                </div>
                <hr>
                <div id="nav-content">
                    <ul class="list-unstyled">
                        {{options_list}} <!-- Placeholder for navigation options list; ensure no extra spaces or characters -->
                    </ul>
                </div>
            </div>
            <!-- Main Content Area -->
            <div class="col">
                {{sections_html}} <!-- Placeholder for main content sections; ensure no extra spaces or characters -->
            </div>
        </div>
    </div>
    <!-- Bootstrap JavaScript Bundle -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
'''


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
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 Model Ranking Workflow Report</h1>
        
        <div class="timestamp">
            Generated on: {{generation_timestamp}}
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
            <p>Compute a single aggregate score for each model by applying weighted combinations of normalized metrics.</p>
            
            <h4>Regression Score Computation</h4>
            <p>Weighted aggregation with the following weights:</p>
            <ul>
                <li>MAE: 20%</li>
                <li>MSE: 20%</li>
                <li>RMSE: 20%</li>
                <li>R²: 30%</li>
                <li>Explained Variance: 10%</li>
            </ul>
            
            <h4>Classification Score Computation</h4>
            <p>Weighted aggregation with the following weights:</p>
            <ul>
                <li>Precision: 15%</li>
                <li>Recall: 15%</li>
                <li>Accuracy: 10%</li>
                <li>AUC: 15%</li>
                <li>F1-Score: 15%</li>
                <li>Phi Coefficient: 10%</li>
                <li>FPR: 10%</li>
                <li>FNR: 10%</li>
            </ul>
        </div>
        
        <div class="section">
            <h2>3. Model Ranking Stage</h2>
            <h3>Purpose</h3>
            <p>Rank models based on their aggregate scores, grouping by base model name and calculating mean performance across phases.</p>
        """
