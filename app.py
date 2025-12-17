import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from mplsoccer import VerticalPitch

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
st.set_page_config(page_title="EPL Goal Predictor", layout="wide")

# Custom CSS to make metrics look better
st.markdown("""
<style>
    div[data-testid="metric-container"] {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
        border: 1px solid #d6d6d6;
    }
</style>
""", unsafe_allow_html=True)

# Load Model
@st.cache_resource
def load_model():
    try:
        return joblib.load('xgboost_tuned_best.pkl')
    except FileNotFoundError:
        st.error("⚠️ Model file not found. Please run 'tune_model.py' first.")
        return None

# Load Data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("Premier_League_Shots_14_24_Direct.csv")
        df['x_m'] = df['X'] * 105
        df['y_m'] = df['Y'] * 68
        df['distance'] = np.sqrt((105 - df['x_m'])**2 + (34 - df['y_m'])**2)
        df['is_long_range'] = df['distance'] > 24
        return df
    except FileNotFoundError:
        return pd.DataFrame()

model = load_model()
df_trends = load_data()

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def calc_angle(x, y):
    a = np.sqrt((105 - x)**2 + (30.34 - y)**2)
    b = np.sqrt((105 - x)**2 + (37.66 - y)**2)
    c = 7.32
    if a * b == 0: return 0
    return np.degrees(np.arccos(np.clip((a**2 + b**2 - c**2) / (2 * a * b), -1.0, 1.0)))

# ==========================================
# 3. SIDEBAR (The Control Panel)
# ==========================================
st.sidebar.header("🎯 Scenario Builder")

# Shooter Position
st.sidebar.subheader("1. Shooter Position")
x_input = st.sidebar.slider("Distance from Goal (m)", 0, 45, 28, help="0m = Goal Line, 16.5m = Penalty Box") 
y_input = st.sidebar.slider("Side Position (m)", 0, 68, 34, help="0 = Left Sideline, 34 = Center, 68 = Right Sideline")

real_x = 105 - x_input 
real_y = y_input

# Context Variables
st.sidebar.subheader("2. Match Context")
situation_map = {"Open Play": "OpenPlay", "Direct Free Kick": "DirectFreekick", "From Corner": "FromCorner", "Set Piece": "SetPiece"}
user_situation = st.sidebar.selectbox("Play Type", list(situation_map.keys()))

action_map = {"Standard Pass": "Pass", "Throughball (Counter)": "Throughball", "Cross": "Cross", "None (Individual)": "None"}
user_action = st.sidebar.selectbox("Preceding Action", list(action_map.keys()))

shot_map = {"Foot": "RightFoot", "Head": "Head"}
user_shot = st.sidebar.selectbox("Body Part", list(shot_map.keys()))

minute_input = st.sidebar.slider("Match Minute", 1, 95, 45)

# ==========================================
# 4. PREDICTION LOGIC
# ==========================================
dist = np.sqrt((105 - real_x)**2 + (34 - real_y)**2)
angle = calc_angle(real_x, real_y)
prob = 0.0

if model:
    # Build Feature Vector
    features = model.get_booster().feature_names
    input_vector = pd.DataFrame(0, index=[0], columns=features)
    
    input_vector['distance'] = dist
    input_vector['angle'] = angle
    input_vector['minute'] = minute_input
    
    # Handle One-Hot Encoding
    if f"situation_{situation_map[user_situation]}" in features:
        input_vector[f"situation_{situation_map[user_situation]}"] = 1
    if f"lastAction_{action_map[user_action]}" in features:
        input_vector[f"lastAction_{action_map[user_action]}"] = 1
    if f"shotType_{shot_map[user_shot]}" in features:
        input_vector[f"shotType_{shot_map[user_shot]}"] = 1

    # Predict
    prob = model.predict_proba(input_vector)[0][1]

# ==========================================
# 5. DASHBOARD TABS
# ==========================================
st.title("⚽ Predictive Model: Long-Range Goals")
st.markdown("### Dissertation Artifact | Premier League Analysis")

tab1, tab2, tab3 = st.tabs(["🏟️ Pitch Simulator", "📈 Trend Analysis", "🧠 Model Logic (SHAP)"])

# --- TAB 1: SIMULATOR ---
with tab1:
    col1, col2 = st.columns([1.8, 1])
    
    with col1:
        st.subheader("Interactive Pitch View")
        
        # Setup Pitch
        fig, ax = plt.subplots(figsize=(10, 7))
        pitch = VerticalPitch(pitch_type='custom', pitch_length=105, pitch_width=68, 
                              line_color='black', half=True, pitch_color='#f8f9fa', line_zorder=2)
        pitch.draw(ax=ax)
        
        # 1. Plot SHOOTER (Red)
        pitch.scatter(real_x, real_y, ax=ax, s=500, c='#ff4b4b', edgecolors='black', 
                      zorder=3, label='Shooter')
        
        # 2. Plot TRAJECTORY (Dashed Line)
        ax.plot([real_y, 34], [real_x, 105], color='red', linestyle='--', alpha=0.6, linewidth=2)
        
        # 3. Plot DEFENDERS + GK (Blue)
        # Standard 4-4-2 Block positions + GK on line
        def_x = [104.5, 98, 98, 98, 98, 90, 90, 90, 90] # GK + Back 4 + Mid 4
        def_y = [34, 18, 29, 39, 50, 20, 30, 38, 48]
        pitch.scatter(def_x, def_y, ax=ax, s=300, c='#1f77b4', edgecolors='black', 
                      alpha=0.8, zorder=2, label='Opponent')
        
        # 4. Plot TEAMMATES (Green)
        # Supporting attackers
        # 4. Plot TEAMMATES (Green)
        # Supporting attackers raw positions
        raw_att_x = [95, 95, 85, 82, 85]
        raw_att_y = [10, 58, 25, 34, 43]
        
        # Filter BOTH X and Y lists simultaneously
        att_x = []
        att_y = []
        for x, y in zip(raw_att_x, raw_att_y):
            # Only keep teammate if they are NOT standing on top of the shooter
            if abs(x - real_x) > 2: 
                att_x.append(x)
                att_y.append(y)

        if att_x: # Only plot if there are teammates left
            pitch.scatter(att_x, att_y, ax=ax, s=300, c='#2ca02c', edgecolors='black', 
                          alpha=0.8, zorder=2, label='Teammate')

        # Legend
        ax.legend(loc='lower center', ncol=3, frameon=True, fontsize=10)
        st.pyplot(fig)

    with col2:
        st.subheader("Prediction Results")
        
        # Main Probability Metric
        st.metric("Goal Probability (xG)", f"{prob*100:.2f}%")
        
        # Progress Bar for probability
        st.progress(min(prob * 5, 1.0)) # Scaled so 20% looks full-ish (since goals are rare)
        
        # Difficulty Context
        st.write("---")
        penalty_prob = 0.76
        difficulty_ratio = penalty_prob / prob if prob > 0 else 100
        st.info(f"💡 **Context:** This shot is **{difficulty_ratio:.1f}x harder** than a penalty kick.")
        
        # Technical Data Table
        st.write("---")
        st.write("**Shot Physics:**")
        col_a, col_b = st.columns(2)
        col_a.metric("Distance", f"{dist:.1f}m")
        col_b.metric("Angle", f"{angle:.1f}°")
        
        if dist > 24:
            st.warning("⚠️ **Long Range** (>24m)")
        else:
            st.success("✅ **Close Range** (<24m)")

# --- TAB 2: TRENDS ---
with tab2:
    st.subheader("Evidence: The Death of Long Shots")
    if not df_trends.empty:
        trend_stats = df_trends.groupby('season')['is_long_range'].mean() * 100
        
        fig_trend, ax_trend = plt.subplots(figsize=(10, 5))
        ax_trend.plot(trend_stats.index, trend_stats.values, marker='o', color='#ff4b4b', linewidth=3)
        ax_trend.set_ylabel("% of Total Shots (Long Range)")
        ax_trend.set_title("10-Year Decline in Long-Range Attempts")
        ax_trend.grid(True, linestyle='--', alpha=0.5)
        plt.xticks(rotation=45)
        st.pyplot(fig_trend)
        
        st.markdown("""
        **Analysis for Dissertation:**
        - **2014:** Teams took ~16-17% of shots from long range.
        - **2023:** This has dropped to ~10-11%.
        - **Conclusion:** Modern tactics (like Pep Guardiola's) actively discourage these low-probability shots shown in the Simulator.
        """)

# --- TAB 3: SHAP ---
with tab3:
    st.subheader("Model 'Brain': Why this probability?")
    if st.button("Run SHAP Explanation"):
        with st.spinner("Calculating..."):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_vector)
            
            fig_shap = plt.figure(figsize=(10, 5))
            shap.summary_plot(shap_values, input_vector, plot_type="bar", show=False, color='#1f77b4')
            st.pyplot(fig_shap)
            
            st.success("The largest bars represent the factors that pushed the probability DOWN the most (usually Distance).")