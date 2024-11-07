import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from factor_analyzer import FactorAnalyzer, calculate_bartlett_sphericity, calculate_kmo

def load_data(file):
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    elif file.name.endswith(('.xls', '.xlsx')):
        return pd.read_excel(file)
    else:
        st.error("Unsupported file type. Please upload a CSV or Excel file.")
        return None

def perform_pca(data, n_components):
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(data)
    explained_variance = pca.explained_variance_ratio_
    return principal_components, explained_variance

def perform_factor_analysis(data, n_factors):
    fa = FactorAnalyzer(n_factors=n_factors, rotation='varimax')
    fa.fit(data)
    loadings = fa.loadings_
    return loadings

st.title("Data Reduction and Interpretation App")
st.markdown("""
This app reads data from CSV or Excel files, performs **PCA** and **Factor Analysis**, 
and provides interpretations for better decision-making.
""")

uploaded_file = st.file_uploader("Upload your data file (CSV or Excel)", type=["csv", "xls", "xlsx"])
if uploaded_file:
    df = load_data(uploaded_file)
    if df is not None:
        st.subheader("Uploaded Data")
        st.write(df.head())
        
        st.subheader("Data Preprocessing")
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        st.write(f"Numeric Columns Detected: {numerical_cols}")
        data = df[numerical_cols].dropna()
        
        if data.shape[1] < 2:
            st.error("At least two numeric columns are required for PCA and Factor Analysis.")
        else:
            st.subheader("PCA Analysis")
            n_components = st.slider("Select Number of Principal Components", 1, min(data.shape[1], 10), 2)
            pca_results, explained_variance = perform_pca(data, n_components)
            st.write(f"Explained Variance Ratio: {explained_variance}")
            
            st.subheader("Factor Analysis")
            st.write("Conducting Bartlett’s Test of Sphericity and KMO Test...")
            chi_square_value, p_value = calculate_bartlett_sphericity(data)
            kmo_all, kmo_model = calculate_kmo(data)
            
            st.write(f"Bartlett’s Test Chi-Square: {chi_square_value:.2f}, p-value: {p_value:.4f}")
            st.write(f"Kaiser-Meyer-Olkin (KMO) Test: {kmo_model:.2f}")
            
            if kmo_model >= 0.6:
                n_factors = st.slider("Select Number of Factors", 1, min(data.shape[1], 10), 2)
                factor_loadings = perform_factor_analysis(data, n_factors)
                st.write("Factor Loadings:")
                st.dataframe(pd.DataFrame(factor_loadings, index=numerical_cols))
            else:
                st.warning("KMO value is too low. Factor Analysis may not be appropriate.")

            st.subheader("Interpretations")
            st.markdown("""
            **PCA**: Principal Component Analysis reduces the dimensions of the dataset while retaining the maximum variance. 
            The explained variance ratio helps in identifying the most influential components for decision-making.
            
            **Factor Analysis**: Factor loadings indicate the correlation between variables and underlying factors. 
            High loadings (>0.5) suggest significant association with that factor. 
            Use these insights to group variables or reduce noise.
            """)

st.markdown("---")
st.markdown("Developed by [Your Name]. For Agricultural Economic Analysis.")
