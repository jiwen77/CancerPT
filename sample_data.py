"""
This file contains simulated data for the head and neck cancer survival prediction app
In a real-world scenario, this would be replaced with actual patient data
"""

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from sklearn.preprocessing import StandardScaler
import pickle
import os
import io
import base64
import plotly.graph_objects as go

# Set random seed for reproducibility
np.random.seed(123)

def generate_sample_data(n=500):
    """Generate simulated patient data for head and neck cancer"""
    
    # Demographics
    age = np.round(np.random.normal(62, 10, n))
    sex = np.random.choice(["Male", "Female"], n, p=[0.7, 0.3])
    race = np.random.choice(
        ["White", "Black", "Asian or Pacific Islander", "American Indian/Alaska Native", "Other"],
        n, p=[0.7, 0.15, 0.1, 0.03, 0.02]
    )
    marital_status = np.random.choice(
        ["Single", "Married", "Divorced", "Separated", "Widowed", "Unknown"], n
    )
    
    # Tumor characteristics
    primary_site = np.random.choice([
        "Lip", "Tongue", "Salivary Gland", "Floor of Mouth", "Gum and Other Mouth",
        "Nasopharynx", "Tonsil", "Oropharynx", "Hypopharynx", "Other Oral Cavity and Pharynx"
    ], n)
    
    histology = np.random.choice(
        ["Squamous Cell Carcinoma", "Adenocarcinoma", "Other"], n, p=[0.85, 0.1, 0.05]
    )
    
    grade = np.random.choice([
        "Well differentiated", "Moderately differentiated",
        "Poorly differentiated", "Undifferentiated", "Unknown"
    ], n)
    
    laterality = np.random.choice(["Right", "Left", "Bilateral", "Not applicable"], n)
    
    # Staging
    t_stage = np.random.choice(
        ["T1", "T2", "T3", "T4a", "T4b", "TX"], n, p=[0.2, 0.25, 0.25, 0.15, 0.1, 0.05]
    )
    
    n_stage = np.random.choice(
        ["N0", "N1", "N2", "N3", "NX"], n, p=[0.4, 0.25, 0.2, 0.1, 0.05]
    )
    
    m_stage = np.random.choice(["M0", "M1", "MX"], n, p=[0.8, 0.15, 0.05])
    
    ajcc_stage = np.random.choice(
        ["Stage I", "Stage II", "Stage III", "Stage IVA", "Stage IVB", "Stage IVC"], n
    )
    
    # Treatment
    surgery = np.random.choice(["Yes", "No"], n, p=[0.7, 0.3])
    radiation = np.random.choice(["Yes", "No"], n, p=[0.6, 0.4])
    chemotherapy = np.random.choice(["Yes", "No"], n, p=[0.5, 0.5])
    immunotherapy = np.random.choice(["Yes", "No"], n, p=[0.2, 0.8])
    
    # Additional factors
    comorbidities = np.random.choice(
        ["None", "Mild", "Moderate", "Severe"], n, p=[0.3, 0.4, 0.2, 0.1]
    )
    
    # Create the DataFrame
    data = pd.DataFrame({
        'age': age,
        'sex': sex,
        'race': race,
        'marital_status': marital_status,
        'primary_site': primary_site,
        'histology': histology,
        'grade': grade,
        'laterality': laterality,
        't_stage': t_stage,
        'n_stage': n_stage,
        'm_stage': m_stage,
        'ajcc_stage': ajcc_stage,
        'surgery': surgery,
        'radiation': radiation,
        'chemotherapy': chemotherapy,
        'immunotherapy': immunotherapy,
        'comorbidities': comorbidities
    })
    
    # Calculate risk score for each patient based on their features (fictional model)
    risk_score = (
        0.02 * (data['age'] - 60) +
        0.3 * (data['sex'] == "Male") +
        0.2 * (data['primary_site'].isin(["Hypopharynx", "Other Oral Cavity and Pharynx"])) +
        0.4 * (data['histology'] == "Other") +
        0.25 * (data['grade'] == "Poorly differentiated") +
        0.4 * (data['grade'] == "Undifferentiated") +
        0.4 * (data['t_stage'] == "T3") +
        0.7 * (data['t_stage'] == "T4a") +
        1.0 * (data['t_stage'] == "T4b") +
        0.5 * (data['n_stage'] == "N1") +
        0.8 * (data['n_stage'] == "N2") +
        1.2 * (data['n_stage'] == "N3") +
        1.5 * (data['m_stage'] == "M1") +
        -0.5 * (data['surgery'] == "Yes") +
        -0.3 * (data['radiation'] == "Yes") +
        -0.2 * (data['chemotherapy'] == "Yes") +
        -0.1 * (data['immunotherapy'] == "Yes") +
        0.2 * (data['comorbidities'] == "Moderate") +
        0.4 * (data['comorbidities'] == "Severe")
    )
    
    # Generate survival times based on risk scores
    lambda_param = np.exp(risk_score) / 50
    time = np.round(np.random.exponential(scale=1/lambda_param))
    
    # Cap survival time at 120 months (10 years)
    time[time > 120] = 120
    
    # Generate censoring
    censoring_time = np.round(np.random.uniform(10, 120, n))
    
    # Observed time is the minimum of survival time and censoring time
    observed_time = np.minimum(time, censoring_time)
    
    # Status: 1 = event (death), 0 = censored
    status = (time <= censoring_time).astype(int)
    
    # Add survival data to the DataFrame
    data['time'] = observed_time
    data['status'] = status
    
    return data

def prepare_model_data(data):
    """
    Prepare data for modeling by converting categorical variables to dummy variables
    """
    # Create dummy variables for categorical features
    categorical_cols = [
        'sex', 'race', 'marital_status', 'primary_site', 'histology', 'grade',
        'laterality', 't_stage', 'n_stage', 'm_stage', 'ajcc_stage',
        'surgery', 'radiation', 'chemotherapy', 'immunotherapy', 'comorbidities'
    ]
    
    model_data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
    return model_data

def train_survival_model(data):
    """Train a Cox Proportional Hazards model on the sample data"""
    # Prepare data for modeling
    model_data = prepare_model_data(data)
    
    # Initialize and fit the CoxPH model
    cph = CoxPHFitter()
    cph.fit(model_data, duration_col='time', event_col='status')
    
    return cph, model_data.columns

def get_variable_importance(model):
    """Get variable importance from the Cox model"""
    # Extract coefficients
    importance = model.summary['coef'].abs().sort_values(ascending=False)
    
    # Clean variable names for display
    var_names = []
    for name in importance.index:
        if name.startswith('age'):
            clean_name = 'Age'
        elif name.startswith('sex'):
            clean_name = 'Sex: ' + name.split('_')[-1]
        elif name.startswith('primary_site'):
            clean_name = 'Site: ' + name.split('_', 1)[1]
        elif name.startswith('histology'):
            clean_name = 'Histology: ' + name.split('_', 1)[1]
        elif name.startswith('grade'):
            clean_name = 'Grade: ' + name.split('_', 1)[1]
        elif name.startswith('t_stage'):
            clean_name = 'T Stage: ' + name.split('_')[-1]
        elif name.startswith('n_stage'):
            clean_name = 'N Stage: ' + name.split('_')[-1]
        elif name.startswith('m_stage'):
            clean_name = 'M Stage: ' + name.split('_')[-1]
        elif name.startswith('surgery'):
            clean_name = 'Surgery: ' + name.split('_')[-1]
        elif name.startswith('radiation'):
            clean_name = 'Radiation: ' + name.split('_')[-1]
        elif name.startswith('chemotherapy'):
            clean_name = 'Chemotherapy: ' + name.split('_')[-1]
        elif name.startswith('immunotherapy'):
            clean_name = 'Immunotherapy: ' + name.split('_')[-1]
        elif name.startswith('comorbidities'):
            clean_name = 'Comorbidities: ' + name.split('_', 1)[1]
        else:
            clean_name = name
        var_names.append(clean_name)
    
    # Create importance DataFrame
    var_importance = pd.DataFrame({
        'variable': var_names,
        'importance': importance.values
    })
    
    # Take top 10 variables
    var_importance = var_importance.iloc[:min(10, len(var_importance))]
    
    return var_importance

def prepare_patient_data(patient_data, model_columns):
    """
    Convert patient input data to the format expected by the model
    """
    # Create a DataFrame from patient data
    patient_df = pd.DataFrame({
        'age': [patient_data['age']],
        'sex': [patient_data['sex']],
        'primary_site': [patient_data['primary_site']],
        'histology': [patient_data['histology']],
        'grade': [patient_data['grade']],
        't_stage': [patient_data['t_stage']],
        'n_stage': [patient_data['n_stage']],
        'm_stage': [patient_data['m_stage']],
        'surgery': [patient_data['surgery']],
        'radiation': [patient_data['radiation']],
        'chemotherapy': [patient_data['chemotherapy']],
        'immunotherapy': [patient_data['immunotherapy']],
        'comorbidities': [patient_data['comorbidities']]
    })
    
    # Create dummy variables
    patient_dummies = pd.get_dummies(patient_df)
    
    # Ensure all columns in model_columns are present
    for col in model_columns:
        if col not in patient_dummies.columns and col not in ['time', 'status']:
            patient_dummies[col] = 0
    
    # Keep only columns that are in model_columns
    patient_dummies = patient_dummies[[c for c in model_columns if c not in ['time', 'status']]]
    
    return patient_dummies

def predict_survival(model, patient_data, model_columns, times=None):
    """
    Predict survival probability for a patient at specified times
    """
    if times is None:
        times = [0, 12, 24, 36, 48, 60]  # Default prediction times (months)
    
    # Prepare patient data for prediction
    patient_df = prepare_patient_data(patient_data, model_columns)
    
    # Make prediction
    survival_func = model.predict_survival_function(patient_df)
    
    # Extract survival probabilities at the specified times
    survival_prob = []
    for t in times:
        # Find closest time in the index
        closest_time = min(survival_func.index, key=lambda x: abs(x - t))
        survival_prob.append(float(survival_func.iloc[survival_func.index.get_loc(closest_time)].values[0]))
    
    # Create DataFrame with results
    results = pd.DataFrame({
        'time': times,
        'survival': survival_prob
    })
    
    return results

def predict_with_confidence_intervals(model, patient_data, model_columns, times=None):
    """
    Predict survival probability with confidence intervals
    """
    if times is None:
        times = [0, 12, 24, 36, 48, 60]  # Default prediction times (months)
    
    # Prepare patient data for prediction
    patient_df = prepare_patient_data(patient_data, model_columns)
    
    # Make prediction with confidence intervals
    try:
        survival_func = model.predict_survival_function(patient_df, alpha=0.05)
        lower_ci = model.predict_survival_function(patient_df, alpha=0.05).lower_bound()
        upper_ci = model.predict_survival_function(patient_df, alpha=0.05).upper_bound()
        
        # Extract survival probabilities at the specified times
        survival_prob = []
        lower_bounds = []
        upper_bounds = []
        
        for t in times:
            # Find closest time in the index
            closest_time = min(survival_func.index, key=lambda x: abs(x - t))
            idx = survival_func.index.get_loc(closest_time)
            
            survival_prob.append(float(survival_func.iloc[idx].values[0]))
            lower_bounds.append(float(lower_ci.iloc[idx].values[0]))
            upper_bounds.append(float(upper_ci.iloc[idx].values[0]))
        
    except Exception:
        # Fall back to regular prediction and add simulated confidence intervals
        prediction = predict_survival(model, patient_data, model_columns, times)
        survival_prob = prediction['survival'].values
        
        # Simulate confidence intervals (for demonstration purposes)
        lower_bounds = np.maximum(0, survival_prob - 0.05)
        upper_bounds = np.minimum(1, survival_prob + 0.05)
    
    # Create DataFrame with results
    results = pd.DataFrame({
        'time': times,
        'survival': survival_prob,
        'lower_ci': lower_bounds,
        'upper_ci': upper_bounds
    })
    
    return results

def create_kaplan_meier_plot(data, stratify_by=None):
    """
    Create a Kaplan-Meier plot, optionally stratified by a categorical variable
    """
    kmf = KaplanMeierFitter()
    
    if stratify_by is None:
        # Single curve
        kmf.fit(data['time'], data['status'], label='Overall')
        fig = go.Figure()
        
        # Add survival curve
        fig.add_trace(go.Scatter(
            x=kmf.timeline,
            y=kmf.survival_function_.values.flatten(),
            mode='lines',
            name='Overall'
        ))
        
        # Add confidence intervals
        if hasattr(kmf, 'confidence_interval_'):
            fig.add_trace(go.Scatter(
                x=kmf.timeline.tolist() + kmf.timeline.tolist()[::-1],
                y=kmf.confidence_interval_['KM_estimate_upper_0.95'].values.tolist() + 
                  kmf.confidence_interval_['KM_estimate_lower_0.95'].values.tolist()[::-1],
                fill='toself',
                fillcolor='rgba(0,0,255,0.1)',
                line=dict(color='rgba(0,0,255,0)'),
                hoverinfo="skip",
                name='95% Confidence Interval'
            ))
    else:
        # Stratified curves
        groups = data[stratify_by].unique()
        fig = go.Figure()
        
        for group in groups:
            group_data = data[data[stratify_by] == group]
            kmf.fit(
                group_data['time'], 
                group_data['status'], 
                label=str(group)
            )
            
            # Add survival curve
            fig.add_trace(go.Scatter(
                x=kmf.timeline,
                y=kmf.survival_function_.values.flatten(),
                mode='lines',
                name=f'{stratify_by}={group}'
            ))
    
    # Update layout
    fig.update_layout(
        title='Kaplan-Meier Survival Curve',
        xaxis_title='Time (months)',
        yaxis_title='Survival Rate',
        yaxis=dict(tickformat=".0%", range=[0, 1]),
        hovermode='x unified'
    )
    
    return fig

def process_uploaded_data(uploaded_file):
    """
    Process uploaded CSV or Excel file
    """
    try:
        # Determine file type and read accordingly
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            data = pd.read_excel(uploaded_file)
        else:
            return None, "Unsupported file format. Please upload a CSV or Excel file."
        
        # Check if required columns exist
        required_cols = ['time', 'status']
        missing_cols = [col for col in required_cols if col not in data.columns]
        
        if missing_cols:
            return None, f"Missing required columns: {', '.join(missing_cols)}"
        
        # Basic validation
        if data['time'].min() < 0:
            return None, "Invalid time values. Time must be non-negative."
        
        if not all(status in [0, 1] for status in data['status'].unique()):
            return None, "Invalid status values. Status must be 0 (censored) or 1 (event)."
        
        return data, None
    except Exception as e:
        return None, f"Error processing file: {str(e)}"

def generate_report_html(patient_data, prediction_df, var_importance):
    """
    Generate an HTML report for the patient
    """
    # Create HTML content for the report
    html = f"""
    <html>
    <head>
        <title>Head and Neck Cancer Survival Prediction Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #2C3E50; }}
            h2 {{ color: #2980B9; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f2f2f2; }}
            .metric {{ background-color: #3498DB; color: white; padding: 15px; margin-bottom: 10px; border-radius: 5px; }}
            .disclaimer {{ font-style: italic; color: #7F8C8D; margin-top: 30px; }}
        </style>
    </head>
    <body>
        <h1>Head and Neck Cancer Survival Prediction Report</h1>
        <p>Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}</p>
        
        <h2>Patient Information</h2>
        <table>
            <tr><th>Factor</th><th>Value</th></tr>
    """
    
    # Add patient information
    for key, value in patient_data.items():
        html += f"<tr><td>{key.replace('_', ' ').title()}</td><td>{value}</td></tr>"
    
    html += """
        </table>
        
        <h2>Survival Predictions</h2>
    """
    
    # Add survival metrics
    one_year = float(prediction_df.loc[prediction_df['time'] == 12, 'survival'].values[0]) * 100
    three_year = float(prediction_df.loc[prediction_df['time'] == 36, 'survival'].values[0]) * 100
    five_year = float(prediction_df.loc[prediction_df['time'] == 60, 'survival'].values[0]) * 100
    
    html += f"""
        <div class="metric">
            <h3>1-Year Survival: {one_year:.1f}%</h3>
        </div>
        <div class="metric">
            <h3>3-Year Survival: {three_year:.1f}%</h3>
        </div>
        <div class="metric">
            <h3>5-Year Survival: {five_year:.1f}%</h3>
        </div>
        
        <h2>Survival Table</h2>
        <table>
            <tr><th>Time (months)</th><th>Survival Rate (%)</th></tr>
    """
    
    # Add survival table
    for _, row in prediction_df.iterrows():
        html += f"<tr><td>{int(row['time'])}</td><td>{row['survival'] * 100:.1f}%</td></tr>"
    
    html += """
        </table>
        
        <h2>Important Risk Factors</h2>
        <table>
            <tr><th>Factor</th><th>Relative Importance</th></tr>
    """
    
    # Add variable importance
    for _, row in var_importance.iterrows():
        html += f"<tr><td>{row['variable']}</td><td>{row['importance']:.3f}</td></tr>"
    
    html += """
        </table>
        
        <p class="disclaimer">
            This report is for research purposes only and should not replace professional medical advice, 
            diagnosis, or treatment. The prediction model is based on simulated data and should be replaced 
            with a real clinical data-based model before use in actual research settings.
        </p>
    </body>
    </html>
    """
    
    return html

def create_downloadable_report(patient_data, prediction_df, var_importance):
    """
    Create a downloadable HTML report
    """
    # Generate report HTML
    report_html = generate_report_html(patient_data, prediction_df, var_importance)
    
    # Convert to bytes
    b64 = base64.b64encode(report_html.encode()).decode()
    
    # Create download link
    href = f'<a href="data:text/html;base64,{b64}" download="survival_report.html">Download HTML Report</a>'
    
    return href

def initialize_model():
    """
    Initialize or load the survival model. If model doesn't exist, create and save it.
    """
    model_path = 'survival_model.pkl'
    
    if os.path.exists(model_path):
        # Load existing model
        with open(model_path, 'rb') as f:
            model, model_columns = pickle.load(f)
    else:
        # Generate sample data
        sample_data = generate_sample_data(500)
        
        # Train the model
        model, model_columns = train_survival_model(sample_data)
        
        # Save the model
        with open(model_path, 'wb') as f:
            pickle.dump((model, model_columns), f)
    
    return model, model_columns

if __name__ == "__main__":
    # This code runs if the file is run directly
    # Generate sample data
    sample_data = generate_sample_data(500)
    
    # Train the model
    model, model_columns = train_survival_model(sample_data)
    
    # Get variable importance
    var_importance = get_variable_importance(model)
    
    # Example of prediction
    patient = {
        'age': 60,
        'sex': 'Male',
        'primary_site': 'Tongue',
        'histology': 'Squamous Cell Carcinoma',
        'grade': 'Moderately differentiated',
        't_stage': 'T2',
        'n_stage': 'N0',
        'm_stage': 'M0',
        'surgery': 'Yes',
        'radiation': 'Yes',
        'chemotherapy': 'No',
        'immunotherapy': 'No',
        'comorbidities': 'None'
    }
    
    pred = predict_survival(model, patient, model_columns)
    print("Survival Prediction:")
    print(pred) 