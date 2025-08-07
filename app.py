"""
Head and Neck Cancer Survival Prediction App
Built with Streamlit
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import time
import base64
from io import BytesIO

# Import our prediction model functions
from sample_data import (
    initialize_model, predict_survival, get_variable_importance, 
    process_uploaded_data, create_kaplan_meier_plot,
    predict_with_confidence_intervals, create_downloadable_report
)

# Page configuration
st.set_page_config(
    page_title="Nasopharyngeal Carcinoma Survival Prediction Tool",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize the model when the app starts
@st.cache_resource
def load_model():
    return initialize_model()

model, model_columns = load_model()

# App title and description
def render_header():
    st.title("Nasopharyngeal Carcinoma Survival Prediction Tool")
    st.markdown(
        "This tool predicts survival rates for nasopharyngeal carcinoma patients. Please enter patient information, tumor characteristics, and treatment details to see the prediction results."
    )
    st.markdown(
        "**Note:** This tool is for research purposes only and should not replace professional medical advice, diagnosis, or treatment. The prediction model is based on simulated data and should be replaced with a real clinical data-based model before use in actual research settings."
    )

# Create tabs for the application
def render_tabs():
    tabs = st.tabs(["Patient Data Input", "Survival Prediction", "Comparison", "Risk Factors", "Data Analysis", "About"])
    return tabs

# Input form for patient data
def render_patient_input(tab):
    with tab:
        st.header("Patient Data Input")
        
        col1, col2 = st.columns(2)
        
        # Get loaded profile data if available
        loaded_data = st.session_state.get('loaded_profile', {})
        
        # Demographics
        with col1:
            st.subheader("Patient Demographics")
            age = st.number_input("Age:", min_value=0, max_value=120, value=loaded_data.get('age', 60))
            
            sex_options = ["Male", "Female"]
            sex_index = sex_options.index(loaded_data.get('sex', 'Male')) if loaded_data.get('sex') in sex_options else 0
            sex = st.selectbox("Sex:", sex_options, index=sex_index)
            
            race_options = ["White", "Black", "Asian or Pacific Islander", "American Indian/Alaska Native", "Other"]
            race_index = race_options.index(loaded_data.get('race', 'White')) if loaded_data.get('race') in race_options else 0
            race = st.selectbox("Race:", race_options, index=race_index)
            
            marital_options = ["Single", "Married", "Divorced", "Separated", "Widowed", "Unknown"]
            marital_index = marital_options.index(loaded_data.get('marital_status', 'Single')) if loaded_data.get('marital_status') in marital_options else 0
            marital_status = st.selectbox("Marital Status:", marital_options, index=marital_index)
        
        # Tumor characteristics
        with col2:
            st.subheader("Tumor Characteristics")
            site_options = ["Nasopharyngeal Carcinoma"]
            site_index = site_options.index(loaded_data.get('primary_site', 'Nasopharyngeal Carcinoma')) if loaded_data.get('primary_site') in site_options else 0
            primary_site = st.selectbox("Primary Site:", site_options, index=site_index)
            
            histology_options = ["Squamous Cell Carcinoma", "Adenocarcinoma", "Other"]
            histology_index = histology_options.index(loaded_data.get('histology', 'Squamous Cell Carcinoma')) if loaded_data.get('histology') in histology_options else 0
            histology = st.selectbox("Histology:", histology_options, index=histology_index)
            
            grade_options = [
                "Well differentiated", "Moderately differentiated", 
                "Poorly differentiated", "Undifferentiated", "Unknown"
            ]
            grade_index = grade_options.index(loaded_data.get('grade', 'Well differentiated')) if loaded_data.get('grade') in grade_options else 0
            grade = st.selectbox("Grade:", grade_options, index=grade_index)
            
            laterality_options = ["Right", "Left", "Bilateral", "Not applicable"]
            laterality_index = laterality_options.index(loaded_data.get('laterality', 'Right')) if loaded_data.get('laterality') in laterality_options else 0
            laterality = st.selectbox("Laterality:", laterality_options, index=laterality_index)
        
        col3, col4 = st.columns(2)
        
        # Staging information
        with col3:
            st.subheader("Staging Information")
            t_options = ["T1", "T2", "T3", "T4a", "T4b", "TX"]
            t_index = t_options.index(loaded_data.get('t_stage', 'T1')) if loaded_data.get('t_stage') in t_options else 0
            t_stage = st.selectbox("T Stage:", t_options, index=t_index)
            
            n_options = ["N0", "N1", "N2", "N3", "NX"]
            n_index = n_options.index(loaded_data.get('n_stage', 'N0')) if loaded_data.get('n_stage') in n_options else 0
            n_stage = st.selectbox("N Stage:", n_options, index=n_index)
            
            m_options = ["M0", "M1a", "M1b", "MX"]
            m_index = m_options.index(loaded_data.get('m_stage', 'M0')) if loaded_data.get('m_stage') in m_options else 0
            m_stage = st.selectbox("M Stage:", m_options, index=m_index)
            
            ajcc_options = ["Stage IA", "Stage IB", "Stage II", "Stage III", "Stage IVA", "Stage IVB"]
            ajcc_index = ajcc_options.index(loaded_data.get('ajcc_stage', 'Stage IA')) if loaded_data.get('ajcc_stage') in ajcc_options else 0
            ajcc_stage = st.selectbox("AJCC Stage (TNM-9):", ajcc_options, index=ajcc_index)
        
        # Treatment information
        with col4:
            st.subheader("Treatment Information")
            surgery_options = ["Yes", "No"]
            surgery_index = surgery_options.index(loaded_data.get('surgery', 'Yes')) if loaded_data.get('surgery') in surgery_options else 0
            surgery = st.selectbox("Surgery:", surgery_options, index=surgery_index)
            
            radiation_options = ["Yes", "No"]
            radiation_index = radiation_options.index(loaded_data.get('radiation', 'Yes')) if loaded_data.get('radiation') in radiation_options else 0
            radiation = st.selectbox("Radiation Therapy:", radiation_options, index=radiation_index)
            
            chemo_options = ["Yes", "No"]
            chemo_index = chemo_options.index(loaded_data.get('chemotherapy', 'No')) if loaded_data.get('chemotherapy') in chemo_options else 0
            chemotherapy = st.selectbox("Chemotherapy:", chemo_options, index=chemo_index)
            
            immuno_options = ["Yes", "No"]
            immuno_index = immuno_options.index(loaded_data.get('immunotherapy', 'No')) if loaded_data.get('immunotherapy') in immuno_options else 0
            immunotherapy = st.selectbox("Immunotherapy:", immuno_options, index=immuno_index)
        
        # Additional factors
        st.subheader("Additional Factors")
        comorbid_options = ["None", "Mild", "Moderate", "Severe"]
        comorbid_index = comorbid_options.index(loaded_data.get('comorbidities', 'None')) if loaded_data.get('comorbidities') in comorbid_options else 0
        comorbidities = st.selectbox("Comorbidities:", comorbid_options, index=comorbid_index)

        # Save/Load profile section
        st.subheader("Save/Load Patient Profile")
        
        # Create a container for better layout
        with st.container():
            save_col1, save_col2 = st.columns([1, 2])
            
            # Save profile section
            with save_col1:
                st.write("**Save Current Data:**")
                profile_name = st.text_input("Profile Name:", "", key="profile_name_input")
                save_button = st.button("Save Profile", use_container_width=True)
            
            # Load profile section
            with save_col2:
                st.write("**Load Saved Profile:**")
                if 'saved_profiles' in st.session_state and st.session_state.saved_profiles:
                    profile_options = list(st.session_state.saved_profiles.keys())
                    selected_profile = st.selectbox("Select Profile:", [""] + profile_options, key="profile_select")
                    
                    load_col1, load_col2 = st.columns(2)
                    load_button = load_col1.button("Load", key="load_btn", use_container_width=True)
                    clear_button = load_col2.button("Clear Form", key="clear_btn", use_container_width=True)
                else:
                    st.info("No saved profiles available")
                    selected_profile = ""
                    load_button = False
                    clear_button = False
        
        # Handle save action
        if save_button and profile_name:
            patient_profile = {
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
            }
            
            if 'saved_profiles' not in st.session_state:
                st.session_state.saved_profiles = {}
            
            st.session_state.saved_profiles[profile_name] = patient_profile
            st.success(f"✅ Profile '{profile_name}' saved successfully!")
            # Rerun to update the Load Profile dropdown
            st.rerun()
        
        # Handle load action
        if 'saved_profiles' in st.session_state and selected_profile and load_button:
            # Load the selected profile data into session state
            st.session_state.loaded_profile = st.session_state.saved_profiles[selected_profile]
            st.success(f"📂 Profile '{selected_profile}' loaded successfully!")
            st.rerun()  # This will cause the page to rerun with the loaded values
        
        # Handle clear action
        if clear_button:
            if 'loaded_profile' in st.session_state:
                del st.session_state.loaded_profile
            st.success("🔄 Form cleared to default values!")
            st.rerun()
        
        # Submit button
        submit_button = st.button("Calculate Survival Probability", type="primary", use_container_width=True)
        
        # Collect patient data
        if submit_button:
            patient_data = {
                'age': age,
                'sex': sex,
                'primary_site': primary_site,
                'histology': histology,
                'grade': grade,
                't_stage': t_stage,
                'n_stage': n_stage,
                'm_stage': m_stage,
                'surgery': surgery,
                'radiation': radiation,
                'chemotherapy': chemotherapy,
                'immunotherapy': immunotherapy,
                'comorbidities': comorbidities
            }
            
            # Store data in session state for use in prediction tab
            st.session_state.patient_data = patient_data
            st.session_state.submitted = True
            
            # Display success message
            st.success('✅ Data submitted successfully!')
            
            # Show spinner to indicate calculation
            with st.spinner('Calculating...'):
                # Add slight delay to show the spinner
                time.sleep(1)
            
            # Show notification with instruction
            st.info('📊 Calculation complete! Please click the **"Survival Prediction"** tab above to view your results.')
            
            # Auto-switch to results tab and scroll to top using JavaScript
            components.html("""
                <script>
                    setTimeout(function() {
                        try {
                            // Look for the Survival Prediction tab and click it
                            var tabs = parent.document.querySelectorAll('button[role="tab"]');
                            for (var i = 0; i < tabs.length; i++) {
                                if (tabs[i].textContent.trim() === 'Survival Prediction') {
                                    tabs[i].click();
                                    
                                    // Scroll to top after tab switch
                                    setTimeout(function() {
                                        parent.window.scrollTo(0, 0);
                                        // Also try to scroll the main content area
                                        var mainContent = parent.document.querySelector('[data-testid="stAppViewContainer"]');
                                        if (mainContent) {
                                            mainContent.scrollTop = 0;
                                        }
                                    }, 500);
                                    break;
                                }
                            }
                        } catch (e) {
                            console.log('Could not auto-switch tab or scroll:', e);
                        }
                    }, 2000);
                </script>
            """, height=0)

# Prediction results visualization
def render_prediction(tab):
    with tab:
        st.markdown('<div id="survival-prediction-results"></div>', unsafe_allow_html=True)
        
        # Add auto-scroll to top when this tab is loaded
        if 'submitted' in st.session_state and st.session_state.submitted:
            components.html("""
                <script>
                    setTimeout(function() {
                        try {
                            // Scroll to top of the page
                            parent.window.scrollTo(0, 0);
                            
                            // Also scroll the main content container
                            var containers = parent.document.querySelectorAll('[data-testid="stAppViewContainer"], .main, .block-container');
                            containers.forEach(function(container) {
                                container.scrollTop = 0;
                            });
                            
                            // Find and scroll to the results header
                            var resultHeader = parent.document.querySelector('#survival-prediction-results');
                            if (resultHeader) {
                                resultHeader.scrollIntoView({ behavior: 'smooth', block: 'start' });
                            }
                        } catch (e) {
                            console.log('Could not scroll to top:', e);
                        }
                    }, 100);
                </script>
            """, height=0)
        
        st.header("Survival Prediction Results")
        
        if 'submitted' in st.session_state and st.session_state.submitted:
            # Get patient data from session state
            patient_data = st.session_state.patient_data
            
            # Perform prediction
            with st.spinner("Calculating prediction results..."):
                prediction_df = predict_with_confidence_intervals(model, patient_data, model_columns)
                var_importance = get_variable_importance(model)
            
            # Display survival metrics in colored boxes
            col1, col2, col3 = st.columns(3)
            
            # 1-year survival probability
            one_year_prob = float(prediction_df.loc[prediction_df['time'] == 12, 'survival'].values[0]) * 100
            
            # Color based on probability
            color1 = "green" if one_year_prob >= 70 else "orange" if one_year_prob >= 40 else "red"
            
            html1 = f"""
            <div style="padding:10px;border-radius:5px;background-color:{color1};color:white;text-align:center;">
            <h3>1-Year Survival</h3>
            <h2>{one_year_prob:.1f}%</h2>
            </div>
            """
            col1.markdown(html1, unsafe_allow_html=True)
            
            # 3-year survival probability
            three_year_prob = float(prediction_df.loc[prediction_df['time'] == 36, 'survival'].values[0]) * 100
            
            # Color based on probability
            color3 = "green" if three_year_prob >= 60 else "orange" if three_year_prob >= 30 else "red"
            
            html3 = f"""
            <div style="padding:10px;border-radius:5px;background-color:{color3};color:white;text-align:center;">
            <h3>3-Year Survival</h3>
            <h2>{three_year_prob:.1f}%</h2>
            </div>
            """
            col2.markdown(html3, unsafe_allow_html=True)
            
            # 5-year survival probability
            five_year_prob = float(prediction_df.loc[prediction_df['time'] == 60, 'survival'].values[0]) * 100
            
            # Color based on probability
            color5 = "green" if five_year_prob >= 50 else "orange" if five_year_prob >= 20 else "red"
            
            html5 = f"""
            <div style="padding:10px;border-radius:5px;background-color:{color5};color:white;text-align:center;">
            <h3>5-Year Survival</h3>
            <h2>{five_year_prob:.1f}%</h2>
            </div>
            """
            col3.markdown(html5, unsafe_allow_html=True)
            
            # Create survival curve
            st.subheader("Survival Curve")
            
            fig = go.Figure()
            
            # Add survival curve
            fig.add_trace(go.Scatter(
                x=prediction_df['time'],
                y=prediction_df['survival'],
                mode='lines+markers',
                name='Survival Rate',
                line=dict(color='blue', width=2),
                marker=dict(size=8)
            ))
            
            # Add confidence intervals
            fig.add_trace(go.Scatter(
                x=prediction_df['time'].tolist() + prediction_df['time'].tolist()[::-1],
                y=prediction_df['upper_ci'].tolist() + prediction_df['lower_ci'].tolist()[::-1],
                fill='toself',
                fillcolor='rgba(0,0,255,0.1)',
                line=dict(color='rgba(0,0,255,0)'),
                hoverinfo="skip",
                name='95% Confidence Interval'
            ))
            
            # Update layout
            fig.update_layout(
                title='Predicted Survival Curve with 95% Confidence Interval',
                xaxis_title='Time (months)',
                yaxis_title='Survival Rate',
                yaxis=dict(tickformat=".0%", range=[0, 1]),
                hovermode='x unified',
                height=500,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            # Show the plot
            st.plotly_chart(fig, use_container_width=True)
            
            # Show survival table
            st.subheader("Survival Table")
            
            # Format table
            display_df = pd.DataFrame({
                'Time (months)': prediction_df['time'].astype(int),
                'Survival Rate (%)': (prediction_df['survival'] * 100).round(1),
                'Lower 95% CI (%)': (prediction_df['lower_ci'] * 100).round(1),
                'Upper 95% CI (%)': (prediction_df['upper_ci'] * 100).round(1)
            })
            
            st.dataframe(display_df, hide_index=True, use_container_width=True)
            
            # Show variable importance
            st.subheader("Variable Importance")
            
            # Create importance plot
            fig_importance = px.bar(
                var_importance,
                y='variable',
                x='importance',
                orientation='h',
                title='Variable Importance in Survival Prediction',
                labels={'importance': 'Relative Importance', 'variable': ''},
                color_discrete_sequence=['steelblue'],
                height=500
            )
            
            # Update layout
            fig_importance.update_layout(
                yaxis=dict(autorange="reversed")
            )
            
            # Show the plot
            st.plotly_chart(fig_importance, use_container_width=True)
            
            # Add download options
            st.subheader("Download Options")
            
            col1, col2 = st.columns(2)
            
            # Add a button to download the prediction results as CSV
            csv = display_df.to_csv(index=False)
            b64_csv = base64.b64encode(csv.encode()).decode()
            href_csv = f'<a href="data:file/csv;base64,{b64_csv}" download="survival_prediction.csv">Download Survival Table (CSV)</a>'
            col1.markdown(href_csv, unsafe_allow_html=True)
            
            # Add a button to download a complete HTML report
            href_html = create_downloadable_report(patient_data, prediction_df, var_importance)
            col2.markdown(href_html, unsafe_allow_html=True)
            
            # Display disclaimer
            st.warning("**Disclaimer:** This tool is for research purposes only and should not replace professional medical advice, diagnosis, or treatment. The prediction model is based on simulated data and should be replaced with a real clinical data-based model before use in actual research settings.")
            
        else:
            st.info("Please input patient information in the 'Patient Data Input' tab and click the 'Calculate Survival Probability' button first.")

# Patient comparison functionality
def render_comparison(tab):
    with tab:
        st.header("Compare Patient Scenarios")
        
        if 'saved_profiles' not in st.session_state or not st.session_state.saved_profiles:
            st.info("Please save at least two patient profiles in the 'Patient Data Input' tab to compare scenarios.")
            return
        
        # Select profiles to compare
        st.subheader("Select Profiles to Compare")
        profiles = list(st.session_state.saved_profiles.keys())
        
        col1, col2 = st.columns(2)
        
        profile1 = col1.selectbox("Profile 1:", profiles, key="profile1")
        profile2 = col2.selectbox("Profile 2:", profiles, key="profile2", index=min(1, len(profiles)-1) if len(profiles) > 1 else 0)
        
        compare_button = st.button("Compare Survival", type="primary", use_container_width=True)
        
        if compare_button:
            # Get patient data from profiles
            patient_data1 = st.session_state.saved_profiles[profile1]
            patient_data2 = st.session_state.saved_profiles[profile2]
            
            with st.spinner("Calculating comparison..."):
                # Get predictions
                prediction_df1 = predict_survival(model, patient_data1, model_columns)
                prediction_df2 = predict_survival(model, patient_data2, model_columns)
                
                # Display comparison of key metrics
                col1, col2, col3 = st.columns(3)
                
                # Get survival rates
                one_year1 = float(prediction_df1.loc[prediction_df1['time'] == 12, 'survival'].values[0]) * 100
                three_year1 = float(prediction_df1.loc[prediction_df1['time'] == 36, 'survival'].values[0]) * 100
                five_year1 = float(prediction_df1.loc[prediction_df1['time'] == 60, 'survival'].values[0]) * 100
                
                one_year2 = float(prediction_df2.loc[prediction_df2['time'] == 12, 'survival'].values[0]) * 100
                three_year2 = float(prediction_df2.loc[prediction_df2['time'] == 36, 'survival'].values[0]) * 100
                five_year2 = float(prediction_df2.loc[prediction_df2['time'] == 60, 'survival'].values[0]) * 100
                
                # Display comparison
                col1.metric(
                    "1-Year Survival", 
                    f"{one_year1:.1f}% vs {one_year2:.1f}%", 
                    f"{one_year1 - one_year2:.1f}%"
                )
                
                col2.metric(
                    "3-Year Survival", 
                    f"{three_year1:.1f}% vs {three_year2:.1f}%", 
                    f"{three_year1 - three_year2:.1f}%"
                )
                
                col3.metric(
                    "5-Year Survival", 
                    f"{five_year1:.1f}% vs {five_year2:.1f}%", 
                    f"{five_year1 - five_year2:.1f}%"
                )
                
                # Create comparison plot
                st.subheader("Survival Curve Comparison")
                
                fig = go.Figure()
                
                # Add survival curves
                fig.add_trace(go.Scatter(
                    x=prediction_df1['time'],
                    y=prediction_df1['survival'],
                    mode='lines+markers',
                    name=f'Profile 1: {profile1}',
                    line=dict(color='blue', width=2),
                    marker=dict(size=8)
                ))
                
                fig.add_trace(go.Scatter(
                    x=prediction_df2['time'],
                    y=prediction_df2['survival'],
                    mode='lines+markers',
                    name=f'Profile 2: {profile2}',
                    line=dict(color='red', width=2),
                    marker=dict(size=8)
                ))
                
                # Update layout
                fig.update_layout(
                    title='Survival Curve Comparison',
                    xaxis_title='Time (months)',
                    yaxis_title='Survival Rate',
                    yaxis=dict(tickformat=".0%", range=[0, 1]),
                    hovermode='x unified',
                    height=500,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                # Show plot
                st.plotly_chart(fig, use_container_width=True)
                
                # Show comparison table
                st.subheader("Profile Differences")
                
                # Create comparison table
                comparison_data = []
                for key in patient_data1.keys():
                    if patient_data1[key] != patient_data2[key]:
                        comparison_data.append({
                            "Factor": key.replace("_", " ").title(),
                            "Profile 1": patient_data1[key],
                            "Profile 2": patient_data2[key]
                        })
                
                if comparison_data:
                    # Convert all values to strings to avoid Arrow serialization issues
                    comparison_df = pd.DataFrame(comparison_data)
                    comparison_df = comparison_df.astype(str)
                    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                else:
                    st.info("The profiles have identical values for all factors.")

# Risk factor visualization
def render_risk_factors(tab):
    with tab:
        st.header("Risk Factor Analysis")
        
        st.markdown("""
        This tab allows you to explore how different risk factors affect survival probability.
        Select a risk factor and see how changes in that factor impact the survival curve.
        """)
        
        # Select a risk factor to analyze
        risk_factor = st.selectbox(
            "Select Risk Factor to Analyze:",
            ["Age", "T Stage", "N Stage", "M Stage", "Treatment Modality", "Comorbidities"]
        )
        
        # Create a baseline patient
        baseline_patient = {
            'age': 60,
            'sex': 'Male',
            'primary_site': 'Nasopharyngeal Carcinoma',
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
        
        with st.spinner("Calculating risk factor impact..."):
            if risk_factor == "Age":
                # Analyze impact of age
                ages = [40, 50, 60, 70, 80]
                fig = go.Figure()
                
                for age in ages:
                    modified_patient = baseline_patient.copy()
                    modified_patient['age'] = age
                    prediction = predict_survival(model, modified_patient, model_columns)
                    
                    fig.add_trace(go.Scatter(
                        x=prediction['time'],
                        y=prediction['survival'],
                        mode='lines',
                        name=f'Age {age}'
                    ))
                
                fig.update_layout(
                    title=f'Impact of {risk_factor} on Survival',
                    xaxis_title='Time (months)',
                    yaxis_title='Survival Rate',
                    yaxis=dict(tickformat=".0%", range=[0, 1]),
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Create a heatmap to show age impact
                st.subheader("Age Impact Heatmap")
                
                age_range = list(range(40, 85, 5))
                time_points = [12, 36, 60]
                
                heatmap_data = []
                for age in age_range:
                    modified_patient = baseline_patient.copy()
                    modified_patient['age'] = age
                    prediction = predict_survival(model, modified_patient, model_columns)
                    
                    row_data = {'Age': age}
                    for time_point in time_points:
                        surv_rate = float(prediction.loc[prediction['time'] == time_point, 'survival'].values[0]) * 100
                        row_data[f'{time_point} Month Survival (%)'] = surv_rate
                    
                    heatmap_data.append(row_data)
                
                heatmap_df = pd.DataFrame(heatmap_data)
                
                # Create the heatmap for survival rates
                fig_heatmap = px.imshow(
                    heatmap_df.iloc[:, 1:].values,
                    x=[f'{t} Month' for t in time_points],
                    y=heatmap_df['Age'],
                    color_continuous_scale='RdYlGn',
                    labels=dict(x="Time Point", y="Age", color="Survival Rate (%)"),
                    aspect="auto",
                    height=500
                )
                
                fig_heatmap.update_layout(title="Age vs. Survival Rate (%) Heatmap")
                st.plotly_chart(fig_heatmap, use_container_width=True)
                
            elif risk_factor == "T Stage":
                # Analyze impact of T stage
                t_stages = ["T1", "T2", "T3", "T4a", "T4b"]
                fig = go.Figure()
                
                for t_stage in t_stages:
                    modified_patient = baseline_patient.copy()
                    modified_patient['t_stage'] = t_stage
                    prediction = predict_survival(model, modified_patient, model_columns)
                    
                    fig.add_trace(go.Scatter(
                        x=prediction['time'],
                        y=prediction['survival'],
                        mode='lines',
                        name=f'{t_stage}'
                    ))
                
                fig.update_layout(
                    title=f'Impact of {risk_factor} on Survival',
                    xaxis_title='Time (months)',
                    yaxis_title='Survival Rate',
                    yaxis=dict(tickformat=".0%", range=[0, 1]),
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            elif risk_factor == "Treatment Modality":
                # Analyze impact of treatment combinations
                treatments = [
                    {"surgery": "Yes", "radiation": "Yes", "chemotherapy": "Yes", "immunotherapy": "Yes", "label": "All Treatments"},
                    {"surgery": "Yes", "radiation": "Yes", "chemotherapy": "No", "immunotherapy": "No", "label": "Surgery + Radiation"},
                    {"surgery": "Yes", "radiation": "No", "chemotherapy": "No", "immunotherapy": "No", "label": "Surgery Only"},
                    {"surgery": "No", "radiation": "Yes", "chemotherapy": "Yes", "immunotherapy": "No", "label": "Radiation + Chemo"},
                    {"surgery": "No", "radiation": "No", "chemotherapy": "No", "immunotherapy": "No", "label": "No Treatment"}
                ]
                
                fig = go.Figure()
                
                for treatment in treatments:
                    modified_patient = baseline_patient.copy()
                    modified_patient['surgery'] = treatment['surgery']
                    modified_patient['radiation'] = treatment['radiation']
                    modified_patient['chemotherapy'] = treatment['chemotherapy']
                    modified_patient['immunotherapy'] = treatment['immunotherapy']
                    
                    prediction = predict_survival(model, modified_patient, model_columns)
                    
                    fig.add_trace(go.Scatter(
                        x=prediction['time'],
                        y=prediction['survival'],
                        mode='lines',
                        name=treatment['label']
                    ))
                
                fig.update_layout(
                    title=f'Impact of {risk_factor} on Survival',
                    xaxis_title='Time (months)',
                    yaxis_title='Survival Rate',
                    yaxis=dict(tickformat=".0%", range=[0, 1]),
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Create a bar chart for 5-year survival by treatment
                st.subheader("5-Year Survival by Treatment Modality")
                
                survival_data = []
                for treatment in treatments:
                    modified_patient = baseline_patient.copy()
                    modified_patient['surgery'] = treatment['surgery']
                    modified_patient['radiation'] = treatment['radiation']
                    modified_patient['chemotherapy'] = treatment['chemotherapy']
                    modified_patient['immunotherapy'] = treatment['immunotherapy']
                    
                    prediction = predict_survival(model, modified_patient, model_columns)
                    five_year = float(prediction.loc[prediction['time'] == 60, 'survival'].values[0]) * 100
                    
                    survival_data.append({
                        "Treatment": treatment['label'],
                        "5-Year Survival (%)": five_year
                    })
                
                bar_df = pd.DataFrame(survival_data)
                
                fig_bar = px.bar(
                    bar_df,
                    x='Treatment',
                    y='5-Year Survival (%)',
                    color='5-Year Survival (%)',
                    color_continuous_scale='RdYlGn',
                    height=400
                )
                
                st.plotly_chart(fig_bar, use_container_width=True)
            
            else:
                # For other factors, create a simple comparison
                if risk_factor == "N Stage":
                    options = ["N0", "N1", "N2", "N3"]
                    field = "n_stage"
                elif risk_factor == "M Stage":
                    options = ["M0", "M1a", "M1b"]
                    field = "m_stage"
                elif risk_factor == "Comorbidities":
                    options = ["None", "Mild", "Moderate", "Severe"]
                    field = "comorbidities"
                
                fig = go.Figure()
                
                for option in options:
                    modified_patient = baseline_patient.copy()
                    modified_patient[field] = option
                    prediction = predict_survival(model, modified_patient, model_columns)
                    
                    fig.add_trace(go.Scatter(
                        x=prediction['time'],
                        y=prediction['survival'],
                        mode='lines',
                        name=f'{option}'
                    ))
                
                fig.update_layout(
                    title=f'Impact of {risk_factor} on Survival',
                    xaxis_title='Time (months)',
                    yaxis_title='Survival Rate',
                    yaxis=dict(tickformat=".0%", range=[0, 1]),
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)

# Data analysis page
def render_data_analysis(tab):
    with tab:
        st.header("Data Analysis")
        
        st.markdown("""
        This tab allows you to analyze your own survival data. Upload a CSV or Excel file containing survival data,
        and the app will generate Kaplan-Meier survival curves and other analyses.
        """)
        
        # File upload section
        st.subheader("Upload Data")
        
        uploaded_file = st.file_uploader(
            "Upload a CSV or Excel file containing survival data", 
            type=["csv", "xls", "xlsx"]
        )
        
        if uploaded_file is not None:
            # Process uploaded file
            data, error = process_uploaded_data(uploaded_file)
            
            if error:
                st.error(error)
            else:
                st.success(f"Successfully loaded data with {len(data)} rows.")
                
                # Store the data in session state
                st.session_state.user_data = data
                
                # Show data preview
                st.subheader("Data Preview")
                st.dataframe(data.head(10), use_container_width=True)
                
                # Data summary
                st.subheader("Data Summary")
                
                col1, col2, col3 = st.columns(3)
                
                # Calculate summary statistics
                total_patients = len(data)
                events = data['status'].sum()
                censored = total_patients - events
                median_followup = np.median(data['time'])
                
                col1.metric("Total Patients", f"{total_patients}")
                col2.metric("Events", f"{events} ({events/total_patients:.1%})")
                col3.metric("Censored", f"{censored} ({censored/total_patients:.1%})")
                
                st.metric("Median Follow-up Time", f"{median_followup:.1f} months")
                
                # Kaplan-Meier analysis
                st.subheader("Kaplan-Meier Survival Analysis")
                
                # Check if there are categorical variables to stratify by
                categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
                
                # Filter out columns that are likely not useful for stratification
                categorical_cols = [col for col in categorical_cols if col not in ['time', 'status', 'id', 'patient_id']]
                
                if categorical_cols:
                    # Allow stratification by categorical variables
                    stratify_options = ["None"] + categorical_cols
                    stratify_by = st.selectbox("Stratify by:", stratify_options)
                    
                    if stratify_by != "None":
                        # Create stratified KM plot
                        km_fig = create_kaplan_meier_plot(data, stratify_by)
                    else:
                        # Create overall KM plot
                        km_fig = create_kaplan_meier_plot(data)
                else:
                    # Create overall KM plot
                    km_fig = create_kaplan_meier_plot(data)
                
                st.plotly_chart(km_fig, use_container_width=True)
                
                # Download options
                st.subheader("Download Options")
                
                # Add a button to download the processed data
                csv = data.to_csv(index=False)
                b64_csv = base64.b64encode(csv.encode()).decode()
                href_csv = f'<a href="data:file/csv;base64,{b64_csv}" download="processed_data.csv">Download Processed Data (CSV)</a>'
                st.markdown(href_csv, unsafe_allow_html=True)
        else:
            st.info("Please upload a file to begin analysis.")
            
            # Show sample data format
            st.subheader("Sample Data Format")
            st.markdown("""
            Your data should be in a CSV or Excel format with at least the following columns:
            - `time`: Time to event or censoring (in months)
            - `status`: Event indicator (1 = event occurred, 0 = censored)
            
            Optional columns:
            - Categorical variables for stratification (e.g., treatment, stage, sex)
            - Other clinical variables
            """)
            
            # Show sample data table
            sample_data = pd.DataFrame({
                'time': [12, 24, 6, 36, 18],
                'status': [1, 0, 1, 0, 1],
                'treatment': ['A', 'A', 'B', 'B', 'A'],
                'sex': ['M', 'F', 'M', 'F', 'M'],
                'age': [65, 72, 58, 60, 75]
            })
            
            st.dataframe(sample_data, use_container_width=True)
            
            # Option to download sample data template
            csv = sample_data.to_csv(index=False)
            b64_csv = base64.b64encode(csv.encode()).decode()
            href_csv = f'<a href="data:file/csv;base64,{b64_csv}" download="sample_template.csv">Download Sample Template (CSV)</a>'
            st.markdown(href_csv, unsafe_allow_html=True)

# About page
def render_about(tab):
    with tab:
        st.header("About This Application")
        
        st.markdown("This web application provides survival prediction for nasopharyngeal carcinoma patients based on patient characteristics, tumor properties, and treatment information.")
        
        st.markdown("The application uses a Cox proportional hazards model trained on simulated data to generate survival probability estimates over time.")
        
        st.markdown("### Features:")
        st.markdown("""
        - Input patient demographic information
        - Input tumor characteristics and staging information
        - Input treatment details
        - Calculate survival probability based on input data
        - Visualize survival curves with confidence intervals
        - Compare different patient scenarios
        - Analyze risk factors and their impact on survival
        - Upload and analyze your own survival data
        - View variable importance in prediction
        - Download prediction results and reports
        """)
        
        st.markdown("### Research Use Only:")
        st.markdown("This tool is for research purposes only and should not replace professional medical advice, diagnosis, or treatment. The prediction model is based on simulated data and should be replaced with a real clinical data-based model before use in actual research settings.")
        
        st.markdown("### Technical Implementation:")
        st.markdown("""
        - Frontend: Streamlit
        - Data Processing: Pandas
        - Survival Analysis: Lifelines
        - Visualization: Plotly
        """)

        # References
        st.markdown("### References:")
        st.markdown("""
        1. Kaplan, E. L., & Meier, P. (1958). Nonparametric estimation from incomplete observations. Journal of the American Statistical Association, 53(282), 457-481.
        2. Cox, D. R. (1972). Regression Models and Life-Tables. Journal of the Royal Statistical Society. Series B (Methodological), 34(2), 187-220.
        3. Davidson-Pilon, C. (2019). Lifelines: survival analysis in Python. Journal of Open Source Software, 4(40), 1317.
        """)
        
        # Version information
        st.markdown("### Version Information:")
        st.markdown(f"""
        - App Version: 1.1.0
        - Last Updated: {pd.Timestamp.now().strftime('%Y-%m-%d')}
        """)

# Main function to run the app
def main():
    # App header
    render_header()
    
    # Create tabs
    tabs = render_tabs()
    
    # Render content for each tab
    render_patient_input(tabs[0])
    render_prediction(tabs[1])
    render_comparison(tabs[2])
    render_risk_factors(tabs[3])
    render_data_analysis(tabs[4])
    render_about(tabs[5])

if __name__ == "__main__":
    main() 