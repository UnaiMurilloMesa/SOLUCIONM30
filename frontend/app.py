"""
M-30 Digital Twin Dashboard.
Main frontend application using Streamlit.
"""
import streamlit as st
import pandas as pd
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="M-30 Digital Twin",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("Traffic Flow Optimization on Madrid's M-30")
    st.markdown("### Digital Twin & Variable Speed Limit Optimizer")
    
    # Sidebar
    st.sidebar.header("Configuration")
    selected_sensor = st.sidebar.selectbox("Select Sensor", ["PM-30-01", "PM-30-02"])
    
    # Main Layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Historical Reality")
        st.markdown("*Real-time data telemetry*")
        
        # Placeholder metrics
        st.metric(label="Current Speed", value="85 km/h", delta="-5 km/h")
        st.metric(label="Density", value="15 veh/km")
        
        # Placeholder chart
        st.info("Historical data visualization will appear here.")

    with col2:
        st.header("Optimized Simulation")
        st.markdown("*Digital Twin Prediction*")
        
        # Placeholder metrics
        st.metric(label="Optimal Speed Limit", value="90 km/h")
        st.metric(label="Projected Flow Improvement", value="+8%", delta_color="normal")
        
        # Placeholder chart
        st.success("Simulation results will appear here.")

if __name__ == "__main__":
    main()
