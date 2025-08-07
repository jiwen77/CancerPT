# Head and Neck Cancer Survival Prediction Tool

A comprehensive web application for predicting survival rates in head and neck cancer patients using advanced statistical modeling.

## Features

- **Patient Data Input**: Comprehensive form for entering patient demographics, tumor characteristics, and treatment information
- **Survival Prediction**: Cox proportional hazards model-based survival probability calculations
- **Interactive Visualizations**: Kaplan-Meier survival curves with confidence intervals
- **Patient Comparison**: Compare survival predictions between different patient scenarios
- **Risk Factor Analysis**: Visualize the impact of different risk factors on survival
- **Data Upload**: Analyze your own survival data with CSV/Excel upload functionality
- **Report Generation**: Download detailed survival prediction reports

## Demo

🔗 **Live Demo**: [Coming Soon - Will be deployed on Streamlit Cloud]

## Technology Stack

- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Survival Analysis**: Lifelines (Cox Proportional Hazards, Kaplan-Meier)
- **Visualization**: Plotly
- **Machine Learning**: Scikit-learn

## Installation & Local Development

### Prerequisites
- Python 3.8+
- pip

### Setup
1. Clone the repository:
```bash
git clone https://github.com/yourusername/head-neck-cancer-survival-app.git
cd head-neck-cancer-survival-app
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

4. Open your browser and navigate to `http://localhost:8501`

## Usage

### 1. Patient Data Input
- Fill in patient demographics (age, sex, race, marital status)
- Enter tumor characteristics (primary site, histology, grade, laterality)
- Provide staging information (T, N, M stages, AJCC stage)
- Specify treatment details (surgery, radiation, chemotherapy, immunotherapy)
- Add comorbidity information

### 2. Survival Prediction
- Click "Calculate Survival Probability" to generate predictions
- View 1-year, 3-year, and 5-year survival rates
- Examine detailed survival curves with confidence intervals
- Review variable importance in the prediction model

### 3. Comparison Analysis
- Save multiple patient profiles
- Compare survival predictions between different scenarios
- Identify key differences affecting outcomes

### 4. Risk Factor Analysis
- Explore how different factors impact survival
- View age-stratified analysis
- Compare treatment modalities
- Analyze staging effects

### 5. Data Upload & Analysis
- Upload your own survival data (CSV/Excel format)
- Generate Kaplan-Meier plots
- Stratify by categorical variables
- Download processed results

## Data Requirements

For uploaded data, your file should contain:
- `time`: Time to event or censoring (in months)
- `status`: Event indicator (1 = event occurred, 0 = censored)
- Optional categorical variables for stratification

## Model Information

The prediction model uses:
- **Cox Proportional Hazards Model**: For survival probability estimation
- **Kaplan-Meier Estimator**: For non-parametric survival analysis
- **Simulated Training Data**: Based on realistic head and neck cancer patient characteristics

**Important**: This tool uses simulated data for demonstration purposes and should not be used for actual clinical decision-making.

## Disclaimer

⚠️ **Research Use Only**: This application is intended for research and educational purposes only. It should not replace professional medical advice, diagnosis, or treatment. The prediction model is based on simulated data and should be replaced with a real clinical data-based model before use in actual research settings.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Survival analysis powered by [Lifelines](https://lifelines.readthedocs.io/)
- Visualizations created with [Plotly](https://plotly.com/)

## Contact

For questions or suggestions, please open an issue on GitHub.

---

**Version**: 1.1.0  
**Last Updated**: 2025-01-07 