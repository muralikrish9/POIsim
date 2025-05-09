import streamlit as st
import pandas as pd
import json
import glob
import plotly.express as px
import time
from datetime import datetime
from PIL import Image
import re

# Set page config with logo
st.set_page_config(
    page_title="Jailbreak Detection Dashboard",
    page_icon="jailbreak-logo.png",
    layout="wide"
)

# Load logo
logo = Image.open('jailbreak-logo.png')

# Custom CSS for enhanced styling
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background-color: #FFF3D7;
    }
    
    /* Loading animation */
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .loading {
        display: inline-block;
        width: 50px;
        height: 50px;
        border: 3px solid #f3f3f3;
        border-radius: 50%;
        border-top: 3px solid #6C63FF;
        animation: spin 1s linear infinite;
    }
    
    /* Smooth transitions for all elements */
    * {
        transition: all 0.3s ease-in-out;
    }
    
    /* Hover effects for cards and containers */
    .stMetric:hover {
        transform: translateY(-5px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Progress bar animation */
    .stProgress > div > div {
        transition: width 0.5s ease-in-out;
    }
    
    /* Text area animations */
    .stTextArea textarea {
        transition: all 0.3s ease;
    }
    
    .stTextArea textarea:focus {
        transform: scale(1.01);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Button hover effects */
    .stButton > button {
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Tab animations */
    .stTabs [data-baseweb="tab"] {
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        transform: translateY(-2px);
    }
    
    /* Fade in animation for content */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .element-container {
        animation: fadeIn 0.5s ease-out;
    }
    
    /* Status bar and header styling */
    .stStatusWidget {
        background-color: #FFF3D7 !important;
    }
    
    .stHeader {
        background-color: #FFF3D7 !important;
    }
    
    /* Override Streamlit's default header styles */
    header[data-testid="stHeader"] {
        background-color: #FFF3D7 !important;
    }
    
    /* Toolbar styling */
    .stToolbar {
        background-color: #FFF3D7 !important;
    }
    
    /* Main content area */
    .block-container {
        background-color: #FFF3D7;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #FFF3D7;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #FFC97A;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #FFC97A;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #FFE8B8;
        padding: 1rem 0.5rem;
    }
    
    /* Sidebar content styling */
    [data-testid="stSidebar"] .sidebar-content {
        background-color: #FFE8B8;
        padding: 0 0.5rem;
    }
    
    /* Sidebar section headers */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #333333;
        border-bottom: 1px solid #FFDBA3;
        padding-bottom: 0.5rem;
        margin-top: 1rem;
        margin-bottom: 0.75rem;
        padding-left: 0.5rem;
    }
    
    /* Sidebar widgets */
    [data-testid="stSidebar"] .stNumberInput,
    [data-testid="stSidebar"] .stTextInput,
    [data-testid="stSidebar"] .stSelectbox{
        background-color: #FFDBA3;
        border-radius: 8px;
        padding: 0.75rem;
        margin: 0.5rem 0;
        padding-left: 0.5rem;
    }
    
    [data-testid="stSidebar"] .stSlider {
        background-color: #FFDBA3;
        border-radius: 8px;
        padding: 0.75rem;
        padding-left: 1rem; 
    }
    
    /* Sidebar buttons */
    [data-testid="stSidebar"] .stButton>button {
        background-color: #FFDBA3;
        color: #333333;
        border: 1px solid #FFC97A;
        padding: 0.5rem 1.25rem;
        margin: 0.5rem 0;
        margin-left: 0.5rem;
    }
    
    /* Sidebar metric cards */
    [data-testid="stSidebar"] .stMetric {
        background-color: #FFDBA3;
        border: 1px solid #FFC97A;
        padding: 1rem;
        margin: 0.75rem 0;
        padding-left: 0.5rem;
    }
    
    /* Sidebar widget labels */
    [data-testid="stSidebar"] label {
        margin-bottom: 0.5rem;
        padding-left: 0.5rem;
    }
    
    /* Main content widgets */
    .stTextInput,
    .stTextArea,
    .stSelectbox,
    .stSlider,
    .stNumberInput {
        background-color: #FFE8B8;
        border-radius: 8px;
        padding: 0.75rem;
    }
    
    /* Main content buttons */
    .stButton>button {
        background-color: #FFE8B8;
        color: #333333;
        border: 1px solid #FFDBA3;
    }
    
    /* Main content metric cards */
    .stMetric {
        background-color: #FFE8B8;
        border: 1px solid #FFDBA3;
    }
    
    /* Logo and title container */
    .logo-title-container {
        background-color: #FFF3D7;
        display: flex;
        align-items: center;
    }
    
    .logo-container {
        display: flex;
        align-items: right;
    }
    
    .title-container {
        display: flex;
        align-items: left;
        margin-top: 3rem;
    }
    
    /* Button styling */
    .stButton>button {
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        background-color: #6C63FF;
        color: white;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Metric card styling */
    .stMetric {
        background-color: #FFF9EE;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid #E9ECEF;
        transition: all 0.3s ease;
    }
    
    .stMetric:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Header styling */
    h1 {
        color: #6C63FF;
        font-weight: 700;
        margin-bottom: 1.5rem;
    }
    
    h2 {
        color: #495057;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 0.5rem 1rem;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #E9ECEF;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #6C63FF;
        color: white;
    }
    
    /* Form styling */
    .stForm {
        border: none !important;
        background-color: transparent !important;
    }
    
    /* Form container styling */
    .element-container .stForm {
        border: none !important;
        background-color: transparent !important;
    }
    
    /* Form content styling */
    .stForm > div {
        background-color: transparent !important;
    }
    
    /* Text input styling */
    .stTextInput>div>div>input,
    .stTextArea>div>div>textarea {
        border-radius: 8px;
        border: 0px solid #E9ECEF;
        padding: 0.5rem;
        background-color: #FFF9EE !important;
    }
    
    /* Number input styling */
    [data-testid="stNumberInput"] input {
        background-color: #FFF9EE !important;
        border-radius: 8px !important;
        border: 0px solid #E9ECEF !important;
        padding: 0.5rem !important;
    }
    
    /* Button styling */
    .stButton > button {
        padding: 0.25rem 0.5rem !important;
        margin: 0 !important;
        height: 2rem !important;
        min-height: 2rem !important;
        line-height: 1 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        font-size: 0.875rem !important;
        border-radius: 4px !important;
    }
    
    /* Container styling */
    .element-container {
        margin-bottom: 0 !important;
    }
    
    /* Placeholder styling */
    .placeholder {
        height: 1.75rem;
        margin-bottom: 0.5rem;
    }
    
    /* Add custom CSS for the red button */
    div[data-testid="stButton"] > button[kind="primary"] {
        background-color: #FF4B4B !important;
        border-color: #FF4B4B !important;
    }
    div[data-testid="stButton"] > button[kind="primary"]:hover {
        background-color: #FF3333 !important;
        border-color: #FF3333 !important;
    }
    div[data-testid="stButton"] > button[kind="primary"]:focus:not(:active) {
        border-color: #FF4B4B !important;
        box-shadow: 0 0 0 0.2rem rgba(255, 75, 75, 0.5) !important;
    }
</style>
""", unsafe_allow_html=True)

def load_conversation_history():
    """Load conversation history from JSON files"""
    try:
        # Find all conversation history files
        history_files = glob.glob("conversation_history.json")
        if not history_files:
            return pd.DataFrame()

        all_records = []
        for file_path in history_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    
                    # Handle empty content
                    if not content:
                        continue
                    
                    # Clean up the content
                    content = content.strip()
                    
                    # Handle common JSON format issues
                    if not content.startswith('[') and not content.startswith('{'):
                        content = '[' + content + ']'
                    elif content.startswith('{') and not content.startswith('[{'): 
                        content = '[' + content + ']'
                    
                    # Remove trailing commas
                    content = re.sub(r',\s*}', '}', content)
                    content = re.sub(r',\s*]', ']', content)
                    
                    try:
                        # Try to parse as JSON
                        data = json.loads(content)
                        
                        # Handle both single objects and arrays
                        if isinstance(data, dict):
                            data = [data]
                        elif isinstance(data, list):
                            data = data
                        else:
                            continue
                            
                        all_records.extend(data)
                        
                    except json.JSONDecodeError:
                        # If JSON parsing fails, try line by line
                        records = []
                        for line in content.split('\n'):
                            line = line.strip()
                            if not line or line in ['[', ']', '{', '}']:
                                continue
                                
                            # Clean up the line
                            line = re.sub(r',\s*$', '', line)
                            if not line.startswith('{'):
                                continue
                                
                            try:
                                record = json.loads(line)
                                records.append(record)
                            except json.JSONDecodeError:
                                continue
                                
                        all_records.extend(records)
                        
            except Exception as e:
                st.error(f"Error reading file {file_path}: {str(e)}")
                continue

        if not all_records:
            # Create a default record if no valid records found
            all_records = [{
                'text': 'No conversation history available',
                'timestamp': time.time(),
                'toxicity': {
                    'toxicity': 0.0,
                    'severe_toxicity': 0.0,
                    'obscene': 0.0,
                    'threat': 0.0,
                    'insult': 0.0,
                    'identity_attack': 0.0
                }
            }]

        # Convert to DataFrame
        df = pd.DataFrame(all_records)
        
        # Ensure required columns exist
        required_columns = ['toxicity', 'sentiment', 'text', 'timestamp']
        for col in required_columns:
            if col not in df.columns:
                if col == 'toxicity':
                    df[col] = df.apply(lambda _: {
                        'toxicity': 0.0,
                        'severe_toxicity': 0.0,
                        'obscene': 0.0,
                        'threat': 0.0,
                        'insult': 0.0,
                        'identity_attack': 0.0
                    }, axis=1)
                elif col == 'sentiment':
                    df[col] = 0.0
                elif col == 'text':
                    df[col] = 'No text available'
                elif col == 'timestamp':
                    df[col] = time.time()

        # Process toxicity scores
        try:
            df['toxicity'] = df['toxicity'].apply(lambda x: x if isinstance(x, dict) else {})
            df['toxicity_score'] = df['toxicity'].apply(lambda x: x.get('toxicity', 0.0))
            df['threat'] = df['toxicity'].apply(lambda x: x.get('threat', 0.0))
            df['insult'] = df['toxicity'].apply(lambda x: x.get('insult', 0.0))
        except Exception as e:
            st.error(f"Error processing toxicity scores: {str(e)}")
            df['toxicity_score'] = 0.0
            df['threat'] = 0.0
            df['insult'] = 0.0

        # Convert timestamp to datetime
        try:
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        except Exception as e:
            st.error(f"Error converting timestamps: {str(e)}")
            df['datetime'] = pd.Timestamp.now()

        return df

    except Exception as e:
        st.error(f"Error loading conversation history: {str(e)}")
        return pd.DataFrame()

# Load conversation history
df = load_conversation_history()

# Load all conversation history files
files = glob.glob('conversation_history.json')

records = []
for file in files:
    try:
        with open(file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            # Try to fix common JSON format issues
            if not content.startswith('[') and not content.startswith('{'):
                content = '[' + content + ']'
            elif content.startswith('{') and not content.startswith('[{'): 
                content = '[' + content + ']'
            
            try:
                data = json.loads(content)
                if isinstance(data, list):
                    records.extend(data)
                else:
                    records.append(data)
            except json.JSONDecodeError as e:
                st.error(f"Error reading {file}: Invalid JSON format")
                # Try to recover by reading line by line
                f.seek(0)
                for line in f:
                    try:
                        if line.strip():  # Skip empty lines
                            # Clean the line and ensure it's valid JSON
                            line = line.strip()
                            if line.endswith(','):
                                line = line[:-1]
                            if not line.startswith('{'):
                                continue
                            record = json.loads(line)
                            records.append(record)
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        st.error(f"Error reading {file}: {str(e)}")
        continue

# Handle case where no records were loaded
if not records:
    records = [{
        'text': 'No conversation history available',
        'timestamp': time.time(),
        'toxicity': {'toxicity': 0, 'threat': 0, 'insult': 0},
        'sentiment': 0
    }]

# Convert to DataFrame
df = pd.DataFrame(records)

# Ensure required columns exist
required_columns = ['toxicity', 'sentiment', 'text', 'timestamp']
for col in required_columns:
    if col not in df.columns:
        if col == 'toxicity':
            df[col] = df.apply(lambda x: {'toxicity': 0, 'threat': 0, 'insult': 0}, axis=1)
        elif col == 'sentiment':
            df[col] = 0
        elif col == 'text':
            df[col] = 'No text available'
        elif col == 'timestamp':
            df[col] = time.time()

# Expand Detoxify toxicity dictionary
try:
    toxicity_scores = pd.json_normalize(df['toxicity'])
    df = df.drop(columns=['toxicity'])
    df = df.join(toxicity_scores)
except Exception as e:
    st.error(f"Error processing toxicity scores: {str(e)}")
    # Add default toxicity columns if processing fails
    df['toxicity'] = 0
    df['threat'] = 0
    df['insult'] = 0

# Convert timestamp to datetime
df['datetime'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
df['datetime'] = df['datetime'].fillna(pd.Timestamp.now())

# --- 1. Handle Reset BEFORE Widgets ---
if "reset_triggered" not in st.session_state:
    st.session_state.reset_triggered = False

# Reset logic before rendering anything
if st.session_state.reset_triggered:
    st.session_state.clear()
    st.session_state.reset_triggered = False
    st.rerun()

### FUNCTIONS -------------------------------------------

# --- Classify jailbreak based on toxicity scores ---
def classify_jailbreak(row):
    try:
        if row['toxicity'] > 0.7 or row['insult'] > 0.6 or row['threat'] > 0.5:
            return True
        else:
            return False
    except:
        return False

df['is_jailbreak'] = df.apply(classify_jailbreak, axis=1)

# --- Add Sentiment Category ---
def classify_sentiment(score):
    if score <= -0.3:
        return 'Negative'
    elif score >= 0.3:
        return 'Positive'
    else:
        return 'Neutral'

df['sentiment_category'] = df['sentiment'].apply(classify_sentiment)

# --- Highlight Jailbreak Rows ---
def highlight_jailbreaks(row):
    color = 'background-color: rgba(255, 0, 0, 0.2)' if row['is_jailbreak'] else ''
    return [color] * len(row)

# --- Add Status Badge Column ---
def badge_label(is_jailbreak):
    if is_jailbreak:
        return "🛑 JAILBREAK"
    else:
        return "✅ SAFE"

df['status_badge'] = df['is_jailbreak'].apply(badge_label)

# --- Dynamic Coloring for Detoxify Scores ---
def color_detoxify(val):
    """
    Color detoxify scores:
    Green for low (safe)
    Yellow for medium
    Red for high toxicity
    """
    if isinstance(val, (float, int)):
        # Clamp value between 0 and 1
        v = max(0, min(val, 1))
        if v < 0.3:
            color = 'background-color: #d4f8e8'  # Light green
        elif v < 0.7:
            color = 'background-color: #fff3cd'  # Light yellow
        else:
            color = 'background-color: #f8d7da'  # Light red
        return color
    else:
        return ''


# Default values
DEFAULT_TOXICITY = 0.0
DEFAULT_THREAT = 0.0
DEFAULT_INSULT = 0.0


# --------------------------------------------------------

# Title with logo
st.markdown('<div class="logo-title-container" style="margin-bottom: 0.5rem; margin-top: 0.2rem;">', unsafe_allow_html=True)
col1, col2 = st.columns([1, 4])
with col1:
    st.markdown('<div class="logo-container" style="margin-top: 0.2rem;">', unsafe_allow_html=True)
    st.image(logo, width=120)
    st.markdown('</div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="title-container" style="margin-top: 0.7rem;">', unsafe_allow_html=True)
    st.markdown('<h1 style="font-size:1.7rem; font-weight:700; color:#2a2a3b; margin-bottom:0.2rem; font-family:Segoe UI,Arial,sans-serif; letter-spacing:-1px;">Jailbreak Detection Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# --- Glassmorphism style for all major boxes ---
glass_style = "background: rgba(220,230,255,0.22); border-radius: 18px; box-shadow: 0 4px 24px rgba(60,60,120,0.10); padding: 1.1rem 1.1rem 0.9rem 1.1rem; margin-bottom: 1.1rem; border: 1.5px solid rgba(255,255,255,0.25); backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px);"

### SIDEBAR ----------------------------------------------

# Sidebar filters
st.sidebar.header("Filters")
show_only_jailbreaks = st.sidebar.checkbox("Show only Jailbreaks", value=False)
search_prompt = st.sidebar.text_input("Search Prompt", value=st.session_state.get('search_prompt', ""), key='search_prompt')

# Sidebar: Pagination Controls
st.sidebar.header("Pagination Settings")
rows_per_page = st.sidebar.number_input('Rows per page', min_value=5, max_value=100, value=20, step=5)

# Sidebar: Detoxify Score Filters with Session State
st.sidebar.header("Detoxify Score Filters")

# Initialize sliders with dynamic default if missing
if 'toxicity_slider' not in st.session_state:
    st.session_state['toxicity_slider'] = DEFAULT_TOXICITY
if 'threat_slider' not in st.session_state:
    st.session_state['threat_slider'] = DEFAULT_THREAT
if 'insult_slider' not in st.session_state:
    st.session_state['insult_slider'] = DEFAULT_INSULT

# Create sliders that always use the current session state value
min_toxicity = st.sidebar.slider(
    "Minimum Toxicity", 0.0, 1.0, key="toxicity_slider"
)
min_threat = st.sidebar.slider(
    "Minimum Threat", 0.0, 1.0, key="threat_slider"
)
min_insult = st.sidebar.slider(
    "Minimum Insult", 0.0, 1.0, key="insult_slider"
)

# Time Filter
time_filter = st.sidebar.selectbox(
    "Show prompts from...",
    ("All Time", "Last 24 Hours", "Last 7 Days"),
    index=0,  # Default index
    key='time_filter'
)

# Refresh Section
st.sidebar.header("Refresh Data")
if st.sidebar.button("🔄 Refresh Now"):
    st.experimental_rerun()

refresh_rate = st.sidebar.number_input('Auto-refresh every X seconds (0 = off)', min_value=0, max_value=600, value=0)
if refresh_rate > 0:
    time.sleep(refresh_rate)
    st.experimental_rerun()

# --- 2. Reset Button that sets the flag ---
if st.sidebar.button("🔄 Reset All Filters"):
    st.session_state.reset_triggered = True
    st.rerun()

# --------------------------------------------------------


# Apply filters
filtered_df = df.copy()
if show_only_jailbreaks:
    filtered_df = filtered_df[filtered_df['is_jailbreak'] == True]
if search_prompt:
    filtered_df = filtered_df[filtered_df['text'].str.contains(search_prompt, case=False, na=False)]

# Apply Detoxify filters
filtered_df = filtered_df[
    (filtered_df['toxicity'] >= min_toxicity) &
    (filtered_df['threat'] >= min_threat) &
    (filtered_df['insult'] >= min_insult)
]

# Apply time filtering
now = datetime.now()

if time_filter == "Last 24 Hours":
    filtered_df = filtered_df[filtered_df['datetime'] >= now - pd.Timedelta(days=1)]
elif time_filter == "Last 7 Days":
    filtered_df = filtered_df[filtered_df['datetime'] >= now - pd.Timedelta(days=7)]
# (Else: All Time → do nothing)

# --- Summary Statistics ---
total_prompts = len(filtered_df)
total_jailbreaks = filtered_df['is_jailbreak'].sum()
percent_jailbreaks = (total_jailbreaks / total_prompts * 100) if total_prompts > 0 else 0

# Avg Toxicity
avg_toxicity = filtered_df['toxicity'].mean()

if filtered_df.empty:
    st.warning("⚠️ No data available with the current filters. Try adjusting your Detoxify thresholds, time filters, or search terms.")
    st.stop()

# --------------------------------------------------------

### --- TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["Analyze Prompt", "Browse", "Charts", "⚙️ Settings"])

# --- Prompt Analysis Tab ---
with tab1:
    st.header("Analyze New Prompt")

    # Create a form for prompt input
    with st.form("analyze_prompt_form"):
        prompt = st.text_area("Enter your prompt to analyze:", height=100)
        submitted = st.form_submit_button("Analyze Prompt")
        
        if submitted and prompt:
            # Show loading animation
            with st.spinner('Loading models and analyzing prompt...'):
                # Initialize detector
                from classifier.jailbreak_detector import JailbreakDetector
                detector = JailbreakDetector()
                
                # Analyze the prompt
                result = detector.predict(prompt)
            
            # Display results
            st.subheader("Analysis Results")

            # --- Model Response Box ---
            model_response_html = result.model_response.replace('\n', '<br>')
            st.markdown(f"""
            <div style='{glass_style}'>
                <h4 style='margin-top:0; margin-bottom: 0.4rem; color: #5A4FFF; font-weight: 700; font-size: 1.08rem; font-family: Segoe UI, Arial, sans-serif;'>Model Response</h4>
                <div style='color: #232336; font-size: 1.01rem; font-family: Segoe UI, Arial, sans-serif; line-height: 1.6;'>
                    {model_response_html}</div>
            </div>
            """, unsafe_allow_html=True)

            # --- Metrics Section ---
            st.markdown("<div style='margin-bottom: 2rem;'></div>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)

            # Toxicity score with progress bar
            toxicity_score = result.details['toxicity_score']
            toxicity_color = "#FF4B4B" if toxicity_score > 0.7 else "#FFA500" if toxicity_score > 0.3 else "#00CC00"
            col1.markdown(f"**Toxicity Score**")
            col1.progress(toxicity_score, text=f"{toxicity_score:.2%}")
            col1.markdown(f"<span style='color: {toxicity_color}; font-weight:600;'>Risk Level: {'High' if toxicity_score > 0.7 else 'Medium' if toxicity_score > 0.3 else 'Low'}</span>", unsafe_allow_html=True)

            # Jailbreak status with explanation
            jailbreak_status = "🛑 JAILBREAK" if result.score > 0.7 else "✅ SAFE"
            col2.markdown(f"**Status**")
            col2.markdown(f"<h2 style='text-align: center; color: {'#FF4B4B' if result.score > 0.7 else '#00CC00'}; font-weight: 700;'> {jailbreak_status} </h2>", unsafe_allow_html=True)
            col2.markdown(f"<span style='color: {'#FF4B4B' if result.score > 0.7 else '#00CC00'}; font-weight:600;'>This prompt is {'likely a jailbreak attempt' if result.score > 0.7 else 'considered safe'}</span>", unsafe_allow_html=True)

            # Sentiment with color coding
            sentiment_score = result.details['sentiment_score']
            sentiment_color = "#FF4B4B" if sentiment_score < -0.3 else "#00CC00" if sentiment_score > 0.3 else "#808080"
            sentiment_text = "Negative" if sentiment_score < -0.3 else "Positive" if sentiment_score > 0.3 else "Neutral"
            col3.markdown(f"**Sentiment**")
            col3.markdown(f"<h2 style='text-align: center; color: {sentiment_color}; font-weight: 700;'> {sentiment_text} </h2>", unsafe_allow_html=True)
            col3.markdown(f"<span style='color: {sentiment_color}; font-weight:600;'>Score: {sentiment_score:.2f}</span>", unsafe_allow_html=True)

            # --- Prompt Text Box ---
            prompt_html = prompt.replace('\n', '<br>')
            st.markdown(f"""
            <div style='{glass_style}'>
                <h4 style='margin-top:0; margin-bottom: 0.4rem; color: #5A4FFF; font-weight: 700; font-size: 1.08rem; font-family: Segoe UI, Arial, sans-serif;'>Prompt Text</h4>
                <div style='color: #232336; font-size: 1.01rem; font-family: Segoe UI, Arial, sans-serif; line-height: 1.6;'>
                    {prompt_html}</div>
            </div>
            """, unsafe_allow_html=True)

            # --- Analysis Explanation (glass style) ---
            explanation = result.explanation
            explanation_parts = explanation.split('\n\n')
            part0 = explanation_parts[0] if len(explanation_parts) > 0 else ""
            part1 = explanation_parts[1] if len(explanation_parts) > 1 else ""
            st.markdown(
                f"""
                <div style='{glass_style}'>
                    <h4 style='margin-top:0; margin-bottom: 0.4rem; color: #5A4FFF; font-weight: 700; font-size: 1.08rem; font-family: Segoe UI, Arial, sans-serif;'>Analysis Explanation</h4>
                    <div style='color: #232336; font-size: 1.01rem; font-family: Segoe UI, Arial, sans-serif; line-height: 1.6;'>
                        {part0}<br>{part1}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # --- Reduce vertical spacing between sections ---
            st.markdown("<div style='margin-bottom: 0.7rem;'></div>", unsafe_allow_html=True)

            st.divider()
            st.markdown("**Toxicity Analysis**")
            col1, col2, col3 = st.columns(3)
            toxicity_scores = result.details
            with col1:
                toxicity = toxicity_scores.get('toxicity', 0)
                st.markdown("**Toxicity**")
                st.progress(toxicity, text=f"{toxicity:.2%}")
            with col2:
                severe_toxicity = toxicity_scores.get('severe_toxicity', 0)
                st.markdown("**Severe Toxicity**")
                st.progress(severe_toxicity, text=f"{severe_toxicity:.2%}")
            with col3:
                obscene = toxicity_scores.get('obscene', 0)
                st.markdown("**Obscene**")
                st.progress(obscene, text=f"{obscene:.2%}")
            col1, col2, col3 = st.columns(3)
            with col1:
                threat = toxicity_scores.get('threat', 0)
                st.markdown("**Threat**")
                st.progress(threat, text=f"{threat:.2%}")
            with col2:
                insult = toxicity_scores.get('insult', 0)
                st.markdown("**Insult**")
                st.progress(insult, text=f"{insult:.2%}")
            with col3:
                identity_attack = toxicity_scores.get('identity_attack', 0)
                st.markdown("**Identity Attack**")
                st.progress(identity_attack, text=f"{identity_attack:.2%}")
            if len(explanation_parts) > 2:
                part2 = explanation_parts[2]
                st.markdown(part2)
            st.divider()

# --- Browse Tab ---
with tab2:
    st.header("Browse Results")

    num_pages = (len(filtered_df) - 1) // rows_per_page + 1

    if num_pages >= 1:
        page = st.number_input('Page', min_value=1, max_value=num_pages, step=1, key='page_input')
        start_idx = (page - 1) * rows_per_page
        end_idx = start_idx + rows_per_page

        # Full paginated DataFrame
        paginated_df = filtered_df.iloc[start_idx:end_idx]

        # Create a clean copy just for display
        display_df = paginated_df.copy()

        # Ensure required columns exist
        required_display_columns = ['status_badge', 'text', 'toxicity', 'threat', 'insult', 'sentiment', 'datetime']
        for col in required_display_columns:
            if col not in display_df.columns:
                if col == 'status_badge':
                    display_df[col] = '✅ SAFE'
                elif col == 'text':
                    display_df[col] = 'No text available'
                elif col in ['toxicity', 'threat', 'insult']:
                    display_df[col] = 0
                elif col == 'sentiment':
                    display_df[col] = 0
                elif col == 'datetime':
                    display_df[col] = pd.Timestamp.now()

        # Drop unwanted columns
        columns_to_hide = ['is_user', 'timestamp', 'entities']
        columns_to_show = [col for col in display_df.columns if col not in columns_to_hide]
        display_df = display_df[columns_to_show]

        # Move status_badge to front
        columns_order = ['status_badge'] + [col for col in display_df.columns if col != 'status_badge']
        display_df = display_df[columns_order]

        # Apply row highlight
        styled_df = display_df.style.apply(highlight_jailbreaks, axis=1)

        # Apply color to detoxify columns using applymap
        detoxify_columns = ['toxicity', 'threat', 'insult']
        for col in detoxify_columns:
            if col in display_df.columns:
                styled_df = styled_df.applymap(color_detoxify, subset=[col])

        # Initialize selection state if not exists
        if "selected_row" not in st.session_state:
            st.session_state.selected_row = None

        # Create a container for the table and buttons
        container = st.container()
        
        # Display the dataframe with updated styling
        try:
            st.dataframe(
                styled_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    'status_badge': st.column_config.TextColumn(
                        'Status',
                        help='Jailbreak status of the prompt'
                    ),
                    'toxicity': st.column_config.ProgressColumn(
                        'Toxicity',
                        help='Toxicity score',
                        format='%.2f',
                        min_value=0,
                        max_value=1
                    ),
                    'threat': st.column_config.ProgressColumn(
                        'Threat',
                        help='Threat score',
                        format='%.2f',
                        min_value=0,
                        max_value=1
                    ),
                    'insult': st.column_config.ProgressColumn(
                        'Insult',
                        help='Insult score',
                        format='%.2f',
                        min_value=0,
                        max_value=1
                    )
                }
            )
        except Exception as e:
            st.error(f"Error displaying dataframe: {str(e)}")
            # Fallback to basic display
            st.dataframe(display_df, use_container_width=True)

        # Add subheader and dropdown for review selection
        st.subheader("Review Analysis")
        
        # Create a dropdown with unique prompt options
        prompt_options = []
        prompt_indices = []
        seen_prompts = set()
        
        for idx, row in paginated_df.iterrows():
            prompt_text = row['text']
            if prompt_text not in seen_prompts:
                seen_prompts.add(prompt_text)
                prompt_options.append(prompt_text)
                prompt_indices.append(idx)
        
        selected_prompt = st.selectbox(
            "Select a prompt to review:",
            options=prompt_options,
            index=None,
            placeholder="Choose a prompt to review..."
        )
        
        # Update selected row based on dropdown selection
        if selected_prompt:
            prompt_index = prompt_options.index(selected_prompt)
            st.session_state.selected_row = start_idx + prompt_indices[prompt_index]

        # Display detailed analysis for selected row
        if st.session_state.selected_row is not None:
            try:
                selected_data = filtered_df.iloc[st.session_state.selected_row]
                
                with st.expander("Collpase/Expand", expanded=True):
                    # Create columns for metrics
                    col1, col2, col3 = st.columns(3)
                    
                    # Toxicity score with progress bar
                    toxicity_score = selected_data['toxicity']
                    toxicity_color = "red" if toxicity_score > 0.7 else "orange" if toxicity_score > 0.3 else "green"
                    col1.markdown(f"**Toxicity Score**")
                    col1.progress(toxicity_score, text=f"{toxicity_score:.2%}")
                    col1.markdown(f"<span style='color: {toxicity_color}'>Risk Level: {'High' if toxicity_score > 0.7 else 'Medium' if toxicity_score > 0.3 else 'Low'}</span>", unsafe_allow_html=True)
                    
                    # Jailbreak status with explanation
                    jailbreak_status = "🛑 JAILBREAK" if selected_data['is_jailbreak'] else "✅ SAFE"
                    col2.markdown(f"**Status**")
                    col2.markdown(f"<h2 style='text-align: center; color: {'red' if selected_data['is_jailbreak'] else 'green'}'> {jailbreak_status} </h2>", unsafe_allow_html=True)
                    col2.markdown(f"<span style='color: {'red' if selected_data['is_jailbreak'] else 'green'}'>This prompt is {'likely a jailbreak attempt' if selected_data['is_jailbreak'] else 'considered safe'}</span>", unsafe_allow_html=True)
                    
                    # Sentiment with color coding
                    sentiment_score = selected_data['sentiment']
                    sentiment_color = "red" if sentiment_score < -0.3 else "green" if sentiment_score > 0.3 else "gray"
                    sentiment_text = "Negative" if sentiment_score < -0.3 else "Positive" if sentiment_score > 0.3 else "Neutral"
                    col3.markdown(f"**Sentiment**")
                    col3.markdown(f"<h2 style='text-align: center; color: {sentiment_color}'> {sentiment_text} </h2>", unsafe_allow_html=True)
                    col3.markdown(f"<span style='color: {sentiment_color}'>Score: {sentiment_score:.2f}</span>", unsafe_allow_html=True)
                    
                    st.divider()
                    st.markdown("**Prompt Text**")
                    st.text_area("", selected_data['text'], height=75, disabled=True)
                    
                    st.divider()
                    st.markdown("**Analysis Explanation**")
                    if 'analysis' in selected_data and selected_data['analysis']:
                        # Split the explanation into parts
                        explanation_parts = selected_data['analysis']['explanation'].split('\n\n')
                        
                        # Display the analysis part
                        st.markdown(explanation_parts[0])
                        
                        # Display the response analysis part
                        st.markdown(explanation_parts[1])
                        
                        st.divider()
                        st.markdown("**Toxicity Analysis**")
                        col1, col2, col3 = st.columns(3)
                        
                        # Get toxicity scores from the analysis details
                        toxicity_scores = selected_data['analysis']['details']
                        
                        # First row of metrics
                        with col1:
                            toxicity = toxicity_scores.get('toxicity', 0)
                            st.markdown("**Toxicity**")
                            st.progress(toxicity, text=f"{toxicity:.2%}")
                            
                        with col2:
                            severe_toxicity = toxicity_scores.get('severe_toxicity', 0)
                            st.markdown("**Severe Toxicity**")
                            st.progress(severe_toxicity, text=f"{severe_toxicity:.2%}")
                            
                        with col3:
                            obscene = toxicity_scores.get('obscene', 0)
                            st.markdown("**Obscene**")
                            st.progress(obscene, text=f"{obscene:.2%}")
                        
                        # Second row of metrics
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            threat = toxicity_scores.get('threat', 0)
                            st.markdown("**Threat**")
                            st.progress(threat, text=f"{threat:.2%}")
                            
                        with col2:
                            insult = toxicity_scores.get('insult', 0)
                            st.markdown("**Insult**")
                            st.progress(insult, text=f"{insult:.2%}")
                            
                        with col3:
                            identity_attack = toxicity_scores.get('identity_attack', 0)
                            st.markdown("**Identity Attack**")
                            st.progress(identity_attack, text=f"{identity_attack:.2%}")
                        
                        # Display the final assessment if it exists
                        if len(explanation_parts) > 2:
                            part2 = explanation_parts[2]
                            st.markdown(part2)
                    else:
                        st.warning("No analysis explanation available for this prompt.")
            except IndexError:
                st.session_state.selected_row = None
                st.warning("Selected row is no longer available. Please select a different row.")

    st.divider()

    ### --- Summary Panel ---
    st.markdown("## Summary Panel")
    
    col1, col2, col3 = st.columns(3)
    
    col1.metric("Total Prompts", total_prompts)
    col2.metric("% Jailbreaks", f"{percent_jailbreaks:.2f}%")
    col3.metric("Avg Toxicity", f"{avg_toxicity:.2f}")

# --- Charts Tab ---
with tab3:
    st.header("Analysis Charts")

    st.divider()

    # Safe vs Jailbreak Chart
    try:
        if 'is_jailbreak' in df.columns:
            counts = df['is_jailbreak'].value_counts()
            fig = px.bar(
                x=counts.index,
                y=counts.values,
                title='Safe vs Jailbreak'
            )
            fig.update_layout(
                paper_bgcolor='#FFF3D7',
                plot_bgcolor='#FFF3D7',
                font_color='black'
            )
            st.plotly_chart(fig)
        else:
            st.warning("No jailbreak data available for charting.")
    except Exception as e:
        st.error(f"Error generating jailbreak chart: {str(e)}")

    st.divider()

    # --- Sentiment Distribution ---
    try:
        if 'sentiment_category' in df.columns:
            sentiment_chart_type = st.selectbox(
                "Select Sentiment Chart Type",
                ("Bar Chart", "Pie Chart")
            )

            sentiment_counts = df['sentiment_category'].value_counts()

            if sentiment_chart_type == "Bar Chart":
                fig = px.bar(
                    x=sentiment_counts.index,
                    y=sentiment_counts.values,
                    title='Sentiment Split'
                )
                fig.update_layout(
                    paper_bgcolor='#FFF3D7',
                    plot_bgcolor='#FFF3D7',
                    font_color='black'
                )
                st.plotly_chart(fig)
            elif sentiment_chart_type == "Pie Chart":
                fig = px.pie(
                    names=sentiment_counts.index,
                    values=sentiment_counts.values,
                    title='Sentiment Split'
                )
                fig.update_layout(
                    paper_bgcolor='#FFF3D7',
                    plot_bgcolor='#FFF3D7',
                    font_color='black'
                )
                st.plotly_chart(fig)
        else:
            st.warning("No sentiment data available for charting.")
    except Exception as e:
        st.error(f"Error generating sentiment chart: {str(e)}")

    st.divider()

    # Detoxify Score Distribution
    try:
        detoxify_columns = ['toxicity', 'threat', 'insult']
        available_columns = [col for col in detoxify_columns if col in df.columns]
        
        if available_columns:
            fig = px.line(df[available_columns], title='Detoxify Score Distributions')
            fig.update_layout(
                paper_bgcolor='#FFF3D7',
                plot_bgcolor='#FFF3D7',
                font_color='black'
            )
            st.plotly_chart(fig)
        else:
            st.warning("No Detoxify score data available for charting.")
    except Exception as e:
        st.error(f"Error generating Detoxify score chart: {str(e)}")

    st.divider()

    # --- Moving Average of Toxicity over Time ---
    try:
        if not filtered_df.empty and 'toxicity' in filtered_df.columns and 'datetime' in filtered_df.columns:
            scores_df = filtered_df[['datetime', 'toxicity']].copy()
            scores_df = scores_df.sort_values('datetime')
            scores_df['toxicity_ma'] = scores_df['toxicity'].rolling(window=10, min_periods=1).mean()

            fig = px.line(scores_df, x='datetime', y='toxicity_ma', title='Toxicity Moving Average')
            fig.update_layout(
                paper_bgcolor='#FFF3D7',
                plot_bgcolor='#FFF3D7',
                font_color='black'
            )
            st.plotly_chart(fig)
        else:
            st.warning("No toxicity data available for the current filter selection.")
    except Exception as e:
        st.error(f"Error generating toxicity moving average chart: {str(e)}")

    st.divider()

    ### --- Summary Panel ---
    st.markdown("## Summary Panel")
    
    try:
        col1, col2, col3 = st.columns(3)
        
        total_prompts = len(filtered_df)
        total_jailbreaks = filtered_df['is_jailbreak'].sum() if 'is_jailbreak' in filtered_df.columns else 0
        percent_jailbreaks = (total_jailbreaks / total_prompts * 100) if total_prompts > 0 else 0
        avg_toxicity = filtered_df['toxicity'].mean() if 'toxicity' in filtered_df.columns else 0
        
        col1.metric("Total Prompts", total_prompts)
        col2.metric("% Jailbreaks", f"{percent_jailbreaks:.2f}%")
        col3.metric("Avg Toxicity", f"{avg_toxicity:.2f}")
    except Exception as e:
        st.error(f"Error generating summary panel: {str(e)}")

# --- Settings Tab ---
with tab4:
    st.header("⚙️ Settings")
    st.write("Adjust filters, refresh rates, and preferences using the sidebar.")

    # Create a row for the buttons with custom spacing
    col1, col2 = st.columns([0.6, 0.4])
    
    with col1:
        # Download button
        st.download_button(
            label="Download Filtered Results as CSV",
            data=filtered_df.to_csv(index=False).encode('utf-8'),
            file_name='filtered_jailbreaks.csv',
            mime='text/csv',
            key='download_button_filtered'
        )

    with col2:
        # Clear Memory button with red styling
        if st.button("⛔ Clear Memory", help="Click to clear all conversation history", 
                    type="primary", 
                    use_container_width=True):
            try:
                import subprocess
                result = subprocess.run(["python", "clear_history.py"], 
                                     capture_output=True, 
                                     text=True, 
                                     check=True)
                st.success("Memory cleared successfully!")
                # Force a rerun to refresh the data
                st.rerun()
            except subprocess.CalledProcessError as e:
                st.error(f"Error clearing memory: {e.stderr}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")


