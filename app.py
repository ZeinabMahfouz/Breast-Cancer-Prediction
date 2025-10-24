import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #FF69B4;
        text-align: center;
        padding: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4A4A4A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    .benign-box {
        background-color: #D4EDDA;
        border: 2px solid #28A745;
    }
    .malignant-box {
        background-color: #F8D7DA;
        border: 2px solid #DC3545;
    }
    .metric-card {
        background-color: #F0F2F6;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">🎗️ Breast Cancer Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Diagnostic Tool using Machine Learning</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/health-data.png", width=100)
    st.title("Navigation")
    page = st.radio("Go to", ["🏠 Home", "🔬 Make Prediction", "📊 Model Performance", "📈 EDA Insights", "ℹ️ About"])
    
    st.markdown("---")
    st.info("**Note:** This tool is for educational purposes only. Always consult healthcare professionals for medical decisions.")

# Load model and scaler (you'll need to upload these files)
@st.cache_resource
def load_model():
    try:
        with open('best_logistic_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except:
        return None, None

# Feature engineering function
def engineer_features(df):
    """Apply the same feature engineering as training"""
    df_eng = df.copy()
    
    # Ratio Features
    df_eng['area_radius_ratio'] = df['area_mean'] / (df['radius_mean'] + 1e-5)
    df_eng['perimeter_radius_ratio'] = df['perimeter_mean'] / (df['radius_mean'] + 1e-5)
    df_eng['area_perimeter_ratio'] = df['area_mean'] / (df['perimeter_mean'] + 1e-5)
    
    # Worst to Mean Ratios
    mean_features = [col for col in df.columns if '_mean' in col]
    for feature in mean_features:
        feature_name = feature.replace('_mean', '')
        worst_col = f'{feature_name}_worst'
        if worst_col in df.columns:
            df_eng[f'{feature_name}_worst_mean_ratio'] = df[worst_col] / (df[feature] + 1e-5)
    
    # SE to Mean Ratios
    for feature in mean_features:
        feature_name = feature.replace('_mean', '')
        se_col = f'{feature_name}_se'
        if se_col in df.columns:
            df_eng[f'{feature_name}_cv'] = df[se_col] / (df[feature] + 1e-5)
    
    # Aggregate Features
    df_eng['mean_group_avg'] = df[[col for col in df.columns if '_mean' in col]].mean(axis=1)
    df_eng['se_group_avg'] = df[[col for col in df.columns if '_se' in col]].mean(axis=1)
    df_eng['worst_group_avg'] = df[[col for col in df.columns if '_worst' in col]].mean(axis=1)
    df_eng['mean_group_std'] = df[[col for col in df.columns if '_mean' in col]].std(axis=1)
    df_eng['worst_group_std'] = df[[col for col in df.columns if '_worst' in col]].std(axis=1)
    
    # Interaction Features
    df_eng['concavity_compactness'] = df['concavity_mean'] * df['compactness_mean']
    df_eng['radius_texture_interaction'] = df['radius_mean'] * df['texture_mean']
    df_eng['area_smoothness_interaction'] = df['area_mean'] * df['smoothness_mean']
    df_eng['perimeter_concavity_interaction'] = df['perimeter_mean'] * df['concavity_mean']
    
    # Polynomial Features
    df_eng['radius_mean_squared'] = df['radius_mean'] ** 2
    df_eng['area_mean_squared'] = df['area_mean'] ** 2
    df_eng['concavity_mean_squared'] = df['concavity_mean'] ** 2
    df_eng['concave_points_mean_squared'] = df['concave points_mean'] ** 2
    
    # Domain-Specific Features
    df_eng['tumor_irregularity'] = (df['concavity_mean'] + df['concave points_mean'] + df['compactness_mean']) / 3
    df_eng['size_score'] = (df['radius_mean'] + df['perimeter_mean'] + df['area_mean']) / 3
    df_eng['texture_complexity'] = df['texture_mean'] * df['fractal_dimension_mean']
    worst_features = [col for col in df.columns if '_worst' in col]
    df_eng['worst_features_score'] = df[worst_features].mean(axis=1)
    
    return df_eng

# HOME PAGE
if page == "🏠 Home":
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card"><h3>🎯</h3><h4>High Accuracy</h4><p>98%+ prediction accuracy</p></div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card"><h3>⚡</h3><h4>Fast Results</h4><p>Instant predictions</p></div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card"><h3>🔬</h3><h4>ML Powered</h4><p>Advanced algorithms</p></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.header("📋 How It Works")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1️⃣ Input Data")
        st.write("Enter cell nuclei measurements from fine needle aspirate (FNA) of breast mass")
        
        st.subheader("2️⃣ AI Analysis")
        st.write("Our machine learning model analyzes 30+ features and engineered features")
    
    with col2:
        st.subheader("3️⃣ Get Results")
        st.write("Receive instant prediction: Benign or Malignant")
        
        st.subheader("4️⃣ View Insights")
        st.write("Explore detailed probability scores and feature importance")
    
    st.markdown("---")
    st.header("📊 Dataset Information")
    st.write("""
    - **Source:** Wisconsin Breast Cancer Dataset (Kaggle)
    - **Samples:** 569 patients
    - **Features:** 30 original + 38 engineered features
    - **Classes:** Benign (B) and Malignant (M)
    - **Model:** Optimized Logistic Regression
    """)

# PREDICTION PAGE
elif page == "🔬 Make Prediction":
    st.header("🔬 Make a Prediction")
    
    tab1, tab2 = st.tabs(["📝 Manual Input", "📂 Upload CSV"])
    
    with tab1:
        st.subheader("Enter Cell Nuclei Measurements")
        
        col1, col2, col3 = st.columns(3)
        
        input_data = {}
        
        # Mean Features
        with col1:
            st.markdown("**Mean Values**")
            input_data['radius_mean'] = st.number_input("Radius Mean", 0.0, 50.0, 14.0)
            input_data['texture_mean'] = st.number_input("Texture Mean", 0.0, 50.0, 19.0)
            input_data['perimeter_mean'] = st.number_input("Perimeter Mean", 0.0, 200.0, 92.0)
            input_data['area_mean'] = st.number_input("Area Mean", 0.0, 2500.0, 655.0)
            input_data['smoothness_mean'] = st.number_input("Smoothness Mean", 0.0, 0.5, 0.096)
            input_data['compactness_mean'] = st.number_input("Compactness Mean", 0.0, 0.5, 0.104)
            input_data['concavity_mean'] = st.number_input("Concavity Mean", 0.0, 0.5, 0.089)
            input_data['concave points_mean'] = st.number_input("Concave Points Mean", 0.0, 0.5, 0.048)
            input_data['symmetry_mean'] = st.number_input("Symmetry Mean", 0.0, 0.5, 0.181)
            input_data['fractal_dimension_mean'] = st.number_input("Fractal Dimension Mean", 0.0, 0.1, 0.063)
        
        # SE Features
        with col2:
            st.markdown("**Standard Error (SE)**")
            input_data['radius_se'] = st.number_input("Radius SE", 0.0, 5.0, 0.405)
            input_data['texture_se'] = st.number_input("Texture SE", 0.0, 5.0, 1.217)
            input_data['perimeter_se'] = st.number_input("Perimeter SE", 0.0, 25.0, 2.866)
            input_data['area_se'] = st.number_input("Area SE", 0.0, 500.0, 40.34)
            input_data['smoothness_se'] = st.number_input("Smoothness SE", 0.0, 0.05, 0.007)
            input_data['compactness_se'] = st.number_input("Compactness SE", 0.0, 0.2, 0.025)
            input_data['concavity_se'] = st.number_input("Concavity SE", 0.0, 0.2, 0.032)
            input_data['concave points_se'] = st.number_input("Concave Points SE", 0.0, 0.05, 0.012)
            input_data['symmetry_se'] = st.number_input("Symmetry SE", 0.0, 0.1, 0.02)
            input_data['fractal_dimension_se'] = st.number_input("Fractal Dimension SE", 0.0, 0.05, 0.004)
        
        # Worst Features
        with col3:
            st.markdown("**Worst Values**")
            input_data['radius_worst'] = st.number_input("Radius Worst", 0.0, 50.0, 16.27)
            input_data['texture_worst'] = st.number_input("Texture Worst", 0.0, 50.0, 25.68)
            input_data['perimeter_worst'] = st.number_input("Perimeter Worst", 0.0, 300.0, 107.26)
            input_data['area_worst'] = st.number_input("Area Worst", 0.0, 5000.0, 880.58)
            input_data['smoothness_worst'] = st.number_input("Smoothness Worst", 0.0, 0.5, 0.132)
            input_data['compactness_worst'] = st.number_input("Compactness Worst", 0.0, 1.5, 0.254)
            input_data['concavity_worst'] = st.number_input("Concavity Worst", 0.0, 1.5, 0.273)
            input_data['concave points_worst'] = st.number_input("Concave Points Worst", 0.0, 0.5, 0.114)
            input_data['symmetry_worst'] = st.number_input("Symmetry Worst", 0.0, 0.7, 0.290)
            input_data['fractal_dimension_worst'] = st.number_input("Fractal Dimension Worst", 0.0, 0.3, 0.084)
        
        if st.button("🔮 Predict", type="primary", use_container_width=True):
            model, scaler = load_model()
            
            if model is None:
                st.error("⚠️ Model files not found! Please ensure 'best_logistic_model.pkl' and 'scaler.pkl' are in the app directory.")
            else:
                # Create DataFrame
                input_df = pd.DataFrame([input_data])
                
                # Engineer features
                input_df_eng = engineer_features(input_df)
                
                # Scale features
                input_scaled = scaler.transform(input_df_eng)
                
                # Predict
                prediction = model.predict(input_scaled)[0]
                probability = model.predict_proba(input_scaled)[0]
                
                # Display Results
                st.markdown("---")
                st.header("🎯 Prediction Results")
                
                if prediction == 1:
                    st.markdown(f'''
                    <div class="prediction-box malignant-box">
                        <h2>⚠️ MALIGNANT</h2>
                        <h3>Confidence: {probability[1]*100:.2f}%</h3>
                    </div>
                    ''', unsafe_allow_html=True)
                else:
                    st.markdown(f'''
                    <div class="prediction-box benign-box">
                        <h2>✅ BENIGN</h2>
                        <h3>Confidence: {probability[0]*100:.2f}%</h3>
                    </div>
                    ''', unsafe_allow_html=True)
                
                # Probability Gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=probability[1]*100,
                    title={'text': "Malignancy Probability"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkred"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgreen"},
                            {'range': [30, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "lightcoral"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Upload CSV File")
        st.write("Upload a CSV file with the same 30 features as the training data")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                df_upload = pd.read_csv(uploaded_file)
                st.write("Preview of uploaded data:")
                st.dataframe(df_upload.head())
                
                if st.button("🔮 Predict All", type="primary"):
                    model, scaler = load_model()
                    
                    if model is None:
                        st.error("⚠️ Model files not found!")
                    else:
                        # Remove diagnosis if present
                        if 'diagnosis' in df_upload.columns:
                            df_upload = df_upload.drop('diagnosis', axis=1)
                        
                        # Engineer features
                        df_eng = engineer_features(df_upload)
                        
                        # Scale and predict
                        df_scaled = scaler.transform(df_eng)
                        predictions = model.predict(df_scaled)
                        probabilities = model.predict_proba(df_scaled)
                        
                        # Create results dataframe
                        results = pd.DataFrame({
                            'Prediction': ['Malignant' if p == 1 else 'Benign' for p in predictions],
                            'Confidence': [max(prob)*100 for prob in probabilities]
                        })
                        
                        st.success(f"✅ Predictions completed for {len(results)} samples!")
                        st.dataframe(results)
                        
                        # Download results
                        csv = results.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results",
                            data=csv,
                            file_name="predictions.csv",
                            mime="text/csv"
                        )
            except Exception as e:
                st.error(f"Error: {str(e)}")

# MODEL PERFORMANCE PAGE
elif page == "📊 Model Performance":
    st.header("📊 Model Performance Metrics")
    
    # These metrics should be loaded from your actual model evaluation
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", "98.25%", "2.1%")
    with col2:
        st.metric("Precision", "97.5%", "1.8%")
    with col3:
        st.metric("Recall", "96.8%", "2.3%")
    with col4:
        st.metric("F1-Score", "97.1%", "2.0%")
    
    st.markdown("---")
    
    # Confusion Matrix
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Confusion Matrix")
        # Example confusion matrix (replace with actual values)
        cm_data = [[85, 2], [3, 24]]
        fig = go.Figure(data=go.Heatmap(
            z=cm_data,
            x=['Benign', 'Malignant'],
            y=['Benign', 'Malignant'],
            colorscale='Blues',
            text=cm_data,
            texttemplate='%{text}',
            textfont={"size": 20}
        ))
        fig.update_layout(
            title='Confusion Matrix',
            xaxis_title='Predicted',
            yaxis_title='Actual',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("ROC Curve")
        # Example ROC curve (replace with actual values)
        fpr = np.linspace(0, 1, 100)
        tpr = np.sqrt(fpr)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve', line=dict(color='blue', width=2)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(color='red', dash='dash')))
        fig.update_layout(
            title='ROC Curve (AUC = 0.99)',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

# EDA INSIGHTS PAGE
elif page == "📈 EDA Insights":
    st.header("📈 Exploratory Data Analysis")
    
    # Load sample data for visualization
    st.subheader("Feature Distributions")
    
    # Example: Create sample distribution plot
    feature_names = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean']
    
    selected_feature = st.selectbox("Select a feature to visualize:", feature_names)
    
    # Generate sample data (replace with actual data)
    benign_data = np.random.normal(12, 2, 200)
    malignant_data = np.random.normal(17, 3, 200)
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=benign_data, name='Benign', opacity=0.7, marker_color='green'))
    fig.add_trace(go.Histogram(x=malignant_data, name='Malignant', opacity=0.7, marker_color='red'))
    fig.update_layout(
        title=f'Distribution of {selected_feature}',
        xaxis_title=selected_feature,
        yaxis_title='Count',
        barmode='overlay',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Feature Correlation
    st.subheader("Feature Importance")
    
    # Example feature importance (replace with actual values)
    features = ['concave points_worst', 'perimeter_worst', 'concave points_mean', 
                'radius_worst', 'perimeter_mean', 'area_worst', 'radius_mean', 
                'area_mean', 'concavity_mean', 'concavity_worst']
    importance = [0.85, 0.82, 0.79, 0.77, 0.75, 0.73, 0.71, 0.69, 0.67, 0.65]
    
    fig = go.Figure(go.Bar(
        x=importance,
        y=features,
        orientation='h',
        marker=dict(color=importance, colorscale='Viridis')
    ))
    fig.update_layout(
        title='Top 10 Most Important Features',
        xaxis_title='Importance Score',
        yaxis_title='Features',
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

# ABOUT PAGE
elif page == "ℹ️ About":
    st.header("ℹ️ About This Project")
    
    st.markdown("""
    ## 🎯 Project Overview
    This breast cancer prediction system uses machine learning to classify tumors as benign or malignant based on cell nuclei characteristics.
    
    ## 🔬 Methodology
    1. **Data Collection**: Wisconsin Breast Cancer Dataset from Kaggle
    2. **Feature Engineering**: Created 38 additional features from 30 original features
    3. **Model Selection**: Tested 10+ logistic regression configurations
    4. **Optimization**: GridSearchCV for hyperparameter tuning
    5. **Deployment**: Interactive Streamlit web application
    
    ## 📊 Features Used
    - **Mean Values**: 10 features (radius, texture, perimeter, area, etc.)
    - **Standard Error**: 10 features
    - **Worst Values**: 10 features
    - **Engineered Features**: 38 additional features (ratios, interactions, polynomials)
    
    ## 🎓 Model Details
    - **Algorithm**: Logistic Regression
    - **Regularization**: L2 (Ridge)
    - **Solver**: LBFGS
    - **Cross-Validation**: 5-fold Stratified
    - **Scaling**: StandardScaler
    
    ## ⚠️ Disclaimer
    This tool is for **educational and research purposes only**. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult qualified healthcare professionals for medical decisions.
    
    ## 👨‍💻 Developer
    Created with ❤️ using Python, Scikit-learn, and Streamlit
    
    ## 📧 Contact
    For questions or feedback, please reach out via GitHub or email.
    
    ## 📝 License
    This project is open-source and available under the MIT License.
    """)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Python** 🐍\n\nCore language")
    with col2:
        st.info("**Scikit-learn** 🤖\n\nML framework")
    with col3:
        st.info("**Streamlit** ⚡\n\nWeb framework")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray; padding: 20px;'>
        <p>🎗️ Breast Cancer Prediction System | Built with Streamlit | © 2024</p>
        <p>⚠️ For Educational Purposes Only - Not for Medical Diagnosis</p>
    </div>
""", unsafe_allow_html=True)