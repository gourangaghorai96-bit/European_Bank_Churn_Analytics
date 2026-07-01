# ============================================
# STREAMLIT DASHBOARD - COMPLETE CODE
# app.py
# ============================================

# WHY: streamlit builds our web dashboard
# WHY: pandas handles data
# WHY: plotly makes interactive charts
# WHY: pickle loads our saved ML model

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle

# ============================================
# PAGE CONFIGURATION
# ============================================

# WHY: Must be first streamlit command
# WHAT: Sets browser tab title and layout
st.set_page_config(
    page_title="European Bank Churn Analytics",
    page_icon="🏦",
    layout="wide"
)

# ============================================
# LOAD DATA AND MODEL
# ============================================

# WHY: @st.cache_data loads data only once
# WHAT: Prevents reloading every time user
#       interacts with dashboard - faster!
# INTERVIEW: Caching improves app performance

@st.cache_data
def load_data():
    # WHY: Load our banking dataset
    df = pd.read_csv('customer_churn.csv')
    return df

@st.cache_resource
def load_model():
    # WHY: Load saved Gradient Boosting model
    with open('churn_model.pkl', 'rb') as f:
        model = pickle.load(f)
    # WHY: Load saved scaler for new predictions
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

# WHY: Load everything when app starts
df = load_data()
model, scaler = load_model()

# ============================================
# SIDEBAR
# ============================================

# WHY: Sidebar has navigation and filters
# WHAT: User can switch between pages
st.sidebar.image("https://img.icons8.com/color/96/bank.png")
st.sidebar.title("🏦 Bank Churn Analytics")
st.sidebar.markdown("---")

# WHY: Radio button for page navigation
page = st.sidebar.radio(
    "Navigate To:",
    ["📊 Overview Dashboard",
     "🗺️ Geographic Analysis", 
     "👥 Customer Segments",
     "🤖 Churn Predictor"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Filters**")

# WHY: Dropdown filter for geography
geography_filter = st.sidebar.selectbox(
    "Select Country:",
    ["All", "France", "Germany", "Spain"]
)

# WHY: Dropdown filter for gender
gender_filter = st.sidebar.selectbox(
    "Select Gender:",
    ["All", "Male", "Female"]
)

# WHY: Apply filters to dataframe
# WHAT: filtered_df changes based on selection
filtered_df = df.copy()

if geography_filter != "All":
    filtered_df = filtered_df[
        filtered_df['Geography'] == geography_filter]

if gender_filter != "All":
    filtered_df = filtered_df[
        filtered_df['Gender'] == gender_filter]

# ============================================
# PAGE 1: OVERVIEW DASHBOARD
# ============================================

if page == "📊 Overview Dashboard":
    
    st.title("📊 Customer Churn Analytics Dashboard")
    st.markdown("**European Banking - Churn Pattern Analysis**")
    st.markdown("---")
    
    # WHY: KPI Cards show key numbers at top
    # WHAT: 4 columns = 4 KPI cards side by side
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total = len(filtered_df)
        st.metric("Total Customers", f"{total:,}")
    
    with col2:
        churned = filtered_df['Exited'].sum()
        st.metric("Total Churned", f"{churned:,}")
    
    with col3:
        churn_rate = filtered_df['Exited'].mean() * 100
        st.metric("Churn Rate", f"{churn_rate:.2f}%")
    
    with col4:
        retained = total - churned
        st.metric("Retained", f"{retained:,}")
    
    st.markdown("---")
    
    # WHY: Two charts side by side
    col1, col2 = st.columns(2)
    
    with col1:
        # WHY: Pie chart shows churn split
        fig1 = px.pie(
            values=filtered_df['Exited'].value_counts(),
            names=['Stayed', 'Churned'],
            title='Overall Churn Distribution',
            color_discrete_sequence=['#2ecc71', '#e74c3c']
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # WHY: Bar chart shows churn by geography
        geo_churn = filtered_df.groupby('Geography')[
            'Exited'].mean() * 100
        fig2 = px.bar(
            x=geo_churn.index,
            y=geo_churn.values,
            title='Churn Rate by Country (%)',
            color=geo_churn.values,
            color_continuous_scale='Reds',
            labels={'x': 'Country', 'y': 'Churn Rate (%)'}
        )
        st.plotly_chart(fig2, use_container_width=True)

# ============================================
# PAGE 2: GEOGRAPHIC ANALYSIS
# ============================================

elif page == "🗺️ Geographic Analysis":
    
    st.title("🗺️ Geographic Churn Analysis")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # WHY: Country wise churn rate
        geo_data = filtered_df.groupby('Geography').agg(
            Total=('Exited', 'count'),
            Churned=('Exited', 'sum')
        ).reset_index()
        geo_data['Churn Rate'] = (
            geo_data['Churned'] / geo_data['Total'] * 100
        ).round(2)
        
        fig3 = px.bar(
            geo_data,
            x='Geography',
            y='Churn Rate',
            title='Churn Rate by Geography (%)',
            color='Churn Rate',
            color_continuous_scale='Reds',
            text='Churn Rate'
        )
        fig3.update_traces(texttemplate='%{text:.1f}%')
        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        # WHY: Gender churn by country
        geo_gender = filtered_df.groupby(
            ['Geography', 'Gender'])['Exited'].mean(
            ) * 100
        geo_gender = geo_gender.reset_index()
        
        fig4 = px.bar(
            geo_gender,
            x='Geography',
            y='Exited',
            color='Gender',
            barmode='group',
            title='Churn Rate by Country and Gender (%)',
            labels={'Exited': 'Churn Rate (%)'},
            color_discrete_sequence=['#3498db', '#e74c3c']
        )
        st.plotly_chart(fig4, use_container_width=True)
    
    # WHY: Detailed table below charts
    st.markdown("### 📋 Country Summary Table")
    st.dataframe(geo_data, use_container_width=True)

# ============================================
# PAGE 3: CUSTOMER SEGMENTS
# ============================================

elif page == "👥 Customer Segments":
    
    st.title("👥 Customer Segment Analysis")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # WHY: Age group churn analysis
        # WHAT: Create age groups first
        filtered_df = filtered_df.copy()
        filtered_df['AgeGroup'] = pd.cut(
            filtered_df['Age'],
            bins=[0, 30, 45, 60, 100],
            labels=['<30', '30-45', '46-60', '60+']
        )
        
        age_churn = filtered_df.groupby(
            'AgeGroup', observed=True)[
            'Exited'].mean() * 100
        
        fig5 = px.bar(
            x=age_churn.index.astype(str),
            y=age_churn.values,
            title='Churn Rate by Age Group (%)',
            color=age_churn.values,
            color_continuous_scale='Reds',
            labels={'x': 'Age Group', 'y': 'Churn Rate (%)'}
        )
        st.plotly_chart(fig5, use_container_width=True)
    
    with col2:
        # WHY: Active vs Inactive churn
        active_churn = filtered_df.groupby(
            'IsActiveMember')['Exited'].mean() * 100
        
        fig6 = px.bar(
            x=['Inactive', 'Active'],
            y=active_churn.values,
            title='Churn Rate: Active vs Inactive (%)',
            color=active_churn.values,
            color_continuous_scale='RdYlGn_r',
            labels={'x': 'Status', 'y': 'Churn Rate (%)'}
        )
        st.plotly_chart(fig6, use_container_width=True)
    
    col3, col4 = st.columns(2)
    
    with col3:
        # WHY: Balance segment churn
        filtered_df['BalanceSeg'] = pd.cut(
            filtered_df['Balance'],
            bins=[-1, 0, 100000, 999999],
            labels=['Zero', 'Low', 'High']
        )
        
        bal_churn = filtered_df.groupby(
            'BalanceSeg', observed=True)[
            'Exited'].mean() * 100
        
        fig7 = px.bar(
            x=bal_churn.index.astype(str),
            y=bal_churn.values,
            title='Churn Rate by Balance Segment (%)',
            color=bal_churn.values,
            color_continuous_scale='Reds',
            labels={'x': 'Balance', 'y': 'Churn Rate (%)'}
        )
        st.plotly_chart(fig7, use_container_width=True)
    
    with col4:
        # WHY: Products vs churn
        prod_churn = filtered_df.groupby(
            'NumOfProducts')['Exited'].mean() * 100
        
        fig8 = px.bar(
            x=prod_churn.index,
            y=prod_churn.values,
            title='Churn Rate by Number of Products (%)',
            color=prod_churn.values,
            color_continuous_scale='Reds',
            labels={'x': 'Products', 'y': 'Churn Rate (%)'}
        )
        st.plotly_chart(fig8, use_container_width=True)

# ============================================
# PAGE 4: CHURN PREDICTOR
# ============================================

elif page == "🤖 Churn Predictor":
    
    st.title("🤖 Customer Churn Predictor")
    st.markdown("Enter customer details to predict churn!")
    st.markdown("---")
    
    # WHY: Two columns for input form
    col1, col2 = st.columns(2)
    
    with col1:
        # WHY: User inputs customer details
        credit_score = st.slider(
            "Credit Score", 300, 850, 600)
        age = st.slider("Age", 18, 92, 40)
        tenure = st.slider("Tenure (Years)", 0, 10, 5)
        balance = st.number_input(
            "Account Balance", 0, 300000, 50000)
        num_products = st.selectbox(
            "Number of Products", [1, 2, 3, 4])
    
    with col2:
        has_cr_card = st.selectbox(
            "Has Credit Card?", ["Yes", "No"])
        is_active = st.selectbox(
            "Is Active Member?", ["Yes", "No"])
        salary = st.number_input(
            "Estimated Salary", 0, 200000, 50000)
        geography = st.selectbox(
            "Country", ["France", "Germany", "Spain"])
        gender = st.selectbox(
            "Gender", ["Male", "Female"])
    
    st.markdown("---")
    
    # WHY: Predict button triggers prediction
    if st.button("🔮 Predict Churn", 
                  use_container_width=True):
        
        # WHY: Convert inputs to model format
        input_data = pd.DataFrame({
            'CreditScore': [credit_score],
            'Age': [age],
            'Tenure': [tenure],
            'Balance': [balance],
            'NumOfProducts': [num_products],
            'HasCrCard': [1 if has_cr_card=="Yes" else 0],
            'IsActiveMember': [1 if is_active=="Yes" else 0],
            'EstimatedSalary': [salary],
            'Geography_Germany': [1 if geography=="Germany" else 0],
            'Geography_Spain': [1 if geography=="Spain" else 0],
            'Gender_Male': [1 if gender=="Male" else 0]
        })
        
        # WHY: Scale input same as training data
        input_scaled = scaler.transform(input_data)
        
        # WHY: Get prediction and probability
        prediction = model.predict(input_scaled)
        probability = model.predict_proba(input_scaled)
        
        churn_prob = probability[0][1] * 100
        stay_prob = probability[0][0] * 100
        
        # WHY: Show result clearly
        if prediction[0] == 1:
            st.error(f"⚠️ HIGH RISK: Customer Likely to CHURN!")
        else:
            st.success(f"✅ LOW RISK: Customer Likely to STAY!")
        
        # WHY: Show probability meters
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Churn Probability", 
                      f"{churn_prob:.1f}%")
        with col2:
            st.metric("Stay Probability", 
                      f"{stay_prob:.1f}%")
        
        # WHY: Business recommendation
        st.markdown("### 💡 Recommended Action:")
        if churn_prob > 60:
            st.warning("""
            🚨 **URGENT ACTION NEEDED!**
            - Call customer immediately
            - Offer special retention package
            - Provide premium banking benefits
            - Assign dedicated relationship manager
            """)
        elif churn_prob > 40:
            st.info("""
            ⚠️ **MONITOR CLOSELY**
            - Send engagement email campaign
            - Offer loyalty rewards
            - Schedule customer feedback call
            """)
        else:
            st.success("""
            ✅ **CUSTOMER IS SATISFIED**
            - Continue regular engagement
            - Offer product upgrades
            - Maintain service quality
            """)