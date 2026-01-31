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

# 2-WAY SYNC LOGIC
if 'home_team' in st.session_state and "ST" in st.session_state.home_team:
    st_x, st_y = st.session_state.home_team["ST"]
    # Calculate slider equivalents
    current_dist = 105 - st_x
    current_side = st_y
else:
    # Default
    current_dist = 28
    current_side = 34

# Initialize session state for sliders if not present or sync them
if "slider_dist" not in st.session_state:
    st.session_state.slider_dist = current_dist
if "slider_side" not in st.session_state:
    st.session_state.slider_side = current_side

# If the pitch was clicked (state updated), force sliders to match
# We detect this by checking if the calculated pitch position differs from the slider state
if abs(st.session_state.slider_dist - current_dist) > 0.1:
    st.session_state.slider_dist = current_dist
if abs(st.session_state.slider_side - current_side) > 0.1:
    st.session_state.slider_side = current_side


x_input = st.sidebar.slider("Distance from Goal (m)", 0, 45, key="slider_dist", help="0m = Goal Line, 16.5m = Penalty Box") 
y_input = st.sidebar.slider("Side Position (m)", 0, 68, key="slider_side", help="0 = Left Sideline, 34 = Center, 68 = Right Sideline")

real_x = 105 - x_input 
real_y = y_input

# Update ST position from sliders if they changed (Handled by the fact that real_x drives the model, 
# AND we should update the dot if slider moves? The previous code handled Slider -> Dot via default_home usage?
# Actually, we need to update 'home_team' ST if slider moves too, to keep dots synced)
if 'home_team' in st.session_state:
     st.session_state.home_team["ST"] = (105 - x_input, y_input)



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

    # ==========================================
    # 1. STATE MANAGEMENT (11v11)
    # ==========================================
    # Initial Positions (Standard 4-3-3 Attacking vs 4-4-2 Defending)
    
    # Attacking Team (Red) - Shooter is often #9 or #10
    default_home = {
        "GK": (5, 34),
        "LB": (30, 10), "LCB": (25, 25), "RCB": (25, 43), "RB": (30, 58),
        "CDM": (45, 34), "LCM": (60, 20), "RCM": (60, 48),
        "LW": (85, 10), "ST": (95, 34), "RW": (85, 58) # ST is initially the shooter
    }

    # Defending Team (Blue)
    default_away = {
        "GK": (104.5, 34),
        "RB": (90, 10), "RCB": (98, 26), "LCB": (98, 42), "LB": (90, 58),
        "RM": (80, 15), "RCM": (85, 30), "LCM": (85, 38), "LM": (80, 53),
        "ST1": (65, 30), "ST2": (65, 38)
    }

    if 'home_team' not in st.session_state:
        st.session_state.home_team = default_home
    if 'away_team' not in st.session_state:
        st.session_state.away_team = default_away

    # ==========================================
    # 2. INTERACTIVE CONTROLS & PITCH
    # ==========================================
    from streamlit_image_coordinates import streamlit_image_coordinates
    import io
    from PIL import Image

    # --- PRESET SCENARIOS ---
    SCENARIOS = {
        "Custom Match": None,
        "Standard Start": {
            "home": default_home,
            "away": default_away
        },
        "De Bruyne vs Arsenal (Counter)": {
            "home": {
                "GK": (5, 34), "CDM": (40, 34), "LCM": (55, 20), "RCM": (55, 48),
                "LW": (85, 10), "RW": (75, 60), "ST": (75, 40) # KDB deep run
            },
            "away": {
                "GK": (104.5, 34), "RB": (90, 15), "RCB": (95, 30), "LCB": (95, 38), "LB": (90, 53),
                "CDM": (85, 34)
            }
        },
        "Deep Block (Park the Bus)": {
            "home": {
                "GK": (5, 34), "ST": (80, 34), "LW": (70, 10), "RW": (70, 58) 
            },
            "away": {
                 "GK": (104.5, 34),
                 "RB": (98, 10), "RCB": (98, 26), "CB": (98, 34), "LCB": (98, 42), "LB": (98, 58),
                 "CDM1": (92, 28), "CDM2": (92, 40), "RM": (85, 15), "LM": (85, 53), "ST": (65, 34)
            }
        }
    }

    with col2:
        st.subheader("📋 Tactics Board")

        # 1. SCENARIO SELECTOR
        selected_scenario = st.selectbox("🎬 **Choose Scenario**:", list(SCENARIOS.keys()))
        
        if selected_scenario != "Custom Match":
            # Check if we need to apply it (only if changed to avoid loop)
            # Simple way: Logic to button press or just check state
            if st.button(f"Apply '{selected_scenario}'"):
                st.session_state.home_team = SCENARIOS[selected_scenario]["home"].copy()
                st.session_state.away_team = SCENARIOS[selected_scenario]["away"].copy()
                st.rerun()

        st.write("---")
        
        team_choice = st.radio("Select Team to Edit:", ["Attacking (Red)", "Defending (Blue)"], horizontal=True)
        
        if "Attacking" in team_choice:
            active_team = st.session_state.home_team
            team_color = "Red"
            role_prefix = "Attacker"
        else:
            active_team = st.session_state.away_team
            team_color = "Blue"
            role_prefix = "Defender"

        # Filter out GK - Keeper should be fixed
        movable_players = [p for p in active_team.keys() if p != "GK"]
        selected_player = st.selectbox(f"Select {role_prefix} to Move:", movable_players)
        
        st.info(f"👉 **Action:** Click anywhere on the pitch to move **{selected_player}**.")
        
        # Show Current Coords (Read-only)
        curr_x, curr_y = active_team[selected_player]
        st.caption(f"Current Position: X={curr_x:.1f}, Y={curr_y:.1f}")

    with col1:
        st.subheader("Interactive Pitch View (Click to Move)")
        
        # 1. SETUP PITCH & PLOT PLAYERS
        fig, ax = plt.subplots(figsize=(10, 7))
        # Remove margins to make coordinate mapping easier
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        
        pitch = VerticalPitch(pitch_type='custom', pitch_length=105, pitch_width=68, 
                              line_color='black', half=False, pitch_color='#f8f9fa', line_zorder=2)
        pitch.draw(ax=ax)
        
        # Plot Home Team (Red)
        for role, (px, py) in st.session_state.home_team.items():
            pitch.scatter(px, py, ax=ax, s=400, c='#ff4b4b', edgecolors='black', zorder=3)
            ax.text(py, px, role, ha='center', va='center', color='white', fontsize=8, fontweight='bold', zorder=4)

        # Plot Away Team (Blue)
        for role, (px, py) in st.session_state.away_team.items():
            pitch.scatter(px, py, ax=ax, s=400, c='#1f77b4', edgecolors='black', zorder=3)
            ax.text(py, px, role, ha='center', va='center', color='white', fontsize=8, fontweight='bold', zorder=4)

        # --- BEST PASS LOGIC ---
        # 1. Identify Shooter (Use selected player if Attacking ST, else find 'ST')
        shooter_pos = st.session_state.home_team.get("ST", (95, 34)) # Default if missing
        
        best_pass_xg = -1
        best_teammate = None
        
        if model:
            # Check every OTHER attacking player
            for role, (tx, ty) in st.session_state.home_team.items():
                if role == "ST": continue # Don't pass to self
                
                # Calculate simple xG for this teammate position
                # Assuming similar context (Same minute, Open Play)
                t_dist = np.sqrt((105 - tx)**2 + (34 - ty)**2)
                t_angle = calc_angle(tx, ty)
                
                # Create input vector
                t_features = model.get_booster().feature_names
                t_input = pd.DataFrame(0, index=[0], columns=t_features)
                t_input['distance'] = t_dist
                t_input['angle'] = t_angle
                t_input['minute'] = minute_input
                # Hardcode some typical context for a pass receiver (Open Play, Standard Shot)
                if "situation_OpenPlay" in t_features: t_input["situation_OpenPlay"] = 1
                if "shotType_RightFoot" in t_features: t_input["shotType_RightFoot"] = 1
                
                t_prob = model.predict_proba(t_input)[0][1]
                
                if t_prob > best_pass_xg:
                    best_pass_xg = t_prob
                    best_teammate = (tx, ty)

            # Draw Line to Best Option
            if best_teammate and best_pass_xg > prob * 1.05 and best_pass_xg > 0.05: # Only suggest if 5% better and meaningful
                bx, by = best_teammate
                ax.plot([shooter_pos[1], by], [shooter_pos[0], bx], color='#2ca02c', linestyle='-', linewidth=3, alpha=0.8, zorder=1)
                ax.scatter(bx, by, s=600, facecolors='none', edgecolors='#2ca02c', linewidth=3, zorder=5)
                # Annotate
                pitch.annotate(f"Best Opt\n{best_pass_xg:.2f} xG", (bx, by-3), ax=ax, fontsize=10, color='#2ca02c', ha='center', fontweight='bold', zorder=5)
        
        # Legend (simplified)
        ax.legend([plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff4b4b', markersize=10, label='Attacking'),
                   plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4', markersize=10, label='Defending')],
                  ['Attacking', 'Defending'], loc='lower center', ncol=2, frameon=True)

        # 2. SAVE TO BUFFER
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        pil_image = Image.open(buf)

        # 3. DISPLAY & CAPTURE CLICKS
        value = streamlit_image_coordinates(pil_image, key="pitch_click", width=600)

        # 4. HANDLE CLICK EVENT
        if value:
            img_width = 600
            img_height = img_width * (pil_image.height / pil_image.width)
            click_x = value["x"]
            click_y = value["y"]
            
            rel_x_screen = click_x / img_width
            rel_y_screen = click_y / img_height
            
            new_pitch_x = 105 - (rel_y_screen * 105) 
            new_pitch_y = rel_x_screen * 68
            
            active_team[selected_player] = (new_pitch_x, new_pitch_y)
            st.rerun()

    with col2:
        st.write("---")
        st.subheader("Prediction Results")
        
        # Main Probability Metric
        st.metric("Expected Goals (xG)", f"{prob:.2f}", delta=f"{prob*100:.1f}% Chance")
        st.progress(min(prob * 5, 1.0)) 
        
        # Defense Win Probability
        defense_prob = float(1.0 - prob)
        st.metric("Defense Success (Win %)", f"{defense_prob:.2f}", delta_color="inverse")
        st.progress(defense_prob) # Full bar typically blue in default theme or just standard
        
 
        
        # Difficulty Context
        if prob > 0:
            difficulty_ratio = 0.76 / prob
            st.caption(f"Shot is **{difficulty_ratio:.1f}x harder** than a penalty.")

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