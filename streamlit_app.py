import streamlit as st
import pandas as pd
import json
import glob
import plotly.express as px
import time
import datetime

# Load all conversation history files
files = glob.glob('conversation_backups/conversation_history_*.json')

records = []
for file in files:
    with open(file, 'r') as f:
        data = json.load(f)
        if isinstance(data, list):
            records.extend(data)
        else:
            records.append(data)

# Convert to DataFrame
df = pd.DataFrame(records)

# Expand Detoxify toxicity dictionary
toxicity_scores = pd.json_normalize(df['toxicity'])
df = df.drop(columns=['toxicity'])
df = df.join(toxicity_scores)

# Convert timestamp to datetime
df['datetime'] = df['timestamp'].apply(lambda x: datetime.datetime.fromtimestamp(x))

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
def classify_jailbreak(toxicity_scores):
    try:
        if toxicity_scores['toxicity'] > 0.7 or toxicity_scores['insult'] > 0.6 or toxicity_scores['threat'] > 0.5:
            return True
        else:
            return False
    except:
        return False

df['is_jailbreak'] = df['toxicity'].apply(classify_jailbreak)

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
        if val < 0.3:
            color = 'background-color: rgba(0, 255, 0, 0.2)'  # Green
        elif val < 0.7:
            color = 'background-color: rgba(255, 255, 0, 0.3)'  # Yellow
        else:
            color = 'background-color: rgba(255, 0, 0, 0.3)'    # Red
        return color
    else:
        return ''


# Default values
DEFAULT_TOXICITY = 0.0
DEFAULT_THREAT = 0.0
DEFAULT_INSULT = 0.0


# --------------------------------------------------------

st.title("🚨 Jailbreak Detection Dashboard")

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
    "Minimum Toxicity", 0.0, 1.0, value=st.session_state['toxicity_slider'], key="toxicity_slider"
)
min_threat = st.sidebar.slider(
    "Minimum Threat", 0.0, 1.0, value=st.session_state['threat_slider'], key="threat_slider"
)
min_insult = st.sidebar.slider(
    "Minimum Insult", 0.0, 1.0, value=st.session_state['insult_slider'], key="insult_slider"
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
now = datetime.datetime.now()

if time_filter == "Last 24 Hours":
    filtered_df = filtered_df[filtered_df['datetime'] >= now - datetime.timedelta(days=1)]
elif time_filter == "Last 7 Days":
    filtered_df = filtered_df[filtered_df['datetime'] >= now - datetime.timedelta(days=7)]
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

### --- Summary Panel ---
st.markdown("## 📊 Summary Panel")

col1, col2, col3 = st.columns(3)

col1.metric("Total Prompts", total_prompts)
col2.metric("% Jailbreaks", f"{percent_jailbreaks:.2f}%")
col3.metric("Avg Toxicity", f"{avg_toxicity:.2f}")

# --------------------------------------------------------


### --- TABS ---
tab1, tab2, tab3 = st.tabs(["📄 Browse", "📊 Charts", "⚙️ Settings"])

# --- Pagination ---
with tab1:
    st.header("📄 Browse Results")

    num_pages = (len(filtered_df) - 1) // rows_per_page + 1

    if num_pages >= 1:
        page = st.number_input('Page', min_value=1, max_value=num_pages, step=1, key='page_input')
        start_idx = (page - 1) * rows_per_page
        end_idx = start_idx + rows_per_page

        # Full paginated DataFrame
        paginated_df = filtered_df.iloc[start_idx:end_idx]

        # Create a clean copy just for display
        display_df = paginated_df.copy()

        # Drop unwanted columns
        columns_to_hide = ['is_user', 'timestamp', 'entities']
        columns_to_show = [col for col in display_df.columns if col not in columns_to_hide]
        display_df = display_df[columns_to_show]

        # Move status_badge to front
        columns_order = ['status_badge'] + [col for col in display_df.columns if col != 'status_badge']
        display_df = display_df[columns_order]

        # Apply row highlight
        styled_df = display_df.style.apply(highlight_jailbreaks, axis=1)

        # Apply color to detoxify columns
        detoxify_columns = ['toxicity', 'threat', 'insult']
        styled_df = styled_df.applymap(color_detoxify, subset=detoxify_columns)

        st.dataframe(styled_df, use_container_width=True, hide_index=True)

with tab2:
    st.header("📊 Analysis Charts")

    # Safe vs Jailbreak Chart
    st.subheader("Safe vs Jailbreak")
    counts = df['is_jailbreak'].value_counts()
    st.bar_chart(counts)

    # --- Sentiment Distribution ---
    st.subheader("📊 Sentiment Distribution")

    sentiment_chart_type = st.selectbox(
        "Select Sentiment Chart Type",
        ("Bar Chart", "Pie Chart")
    )

    sentiment_counts = df['sentiment_category'].value_counts()

    if sentiment_chart_type == "Bar Chart":
        st.bar_chart(sentiment_counts)
    elif sentiment_chart_type == "Pie Chart":
        fig = px.pie(
            names=sentiment_counts.index,
            values=sentiment_counts.values,
            title='Sentiment Split'
        )
        st.plotly_chart(fig)

    # Detoxify Score Distribution
    st.subheader("Detoxify Score Distributions")
    if 'toxicity' in df.columns:
        scores_df = pd.json_normalize(df['toxicity'])
        st.line_chart(scores_df)

   # --- Moving Average of Toxicity over Time ---
    if not filtered_df.empty:
        scores_df = pd.json_normalize(filtered_df['toxicity'])

        # Only proceed if toxicity exists
        if 'toxicity' in scores_df.columns:
            scores_df['datetime'] = filtered_df['datetime'].values
            scores_df = scores_df.sort_values('datetime')
            scores_df['toxicity_ma'] = scores_df['toxicity'].rolling(window=10, min_periods=1).mean()

            st.subheader("📈 Toxicity Moving Average Over Time")
            fig = px.line(scores_df, x='datetime', y='toxicity_ma', title='Toxicity Moving Average')
            st.plotly_chart(fig)
        else:
            st.warning("No toxicity data available for the current filter selection.")
    else:
        st.warning("No data available to plot toxicity moving average.")


with tab3:
    st.header("⚙️ Settings")
    st.write("Adjust filters, refresh rates, and preferences using the sidebar.")

    st.download_button(
        label="Download Filtered Results as CSV",
        data=filtered_df.to_csv(index=False).encode('utf-8'),
        file_name='filtered_jailbreaks.csv',
        mime='text/csv',
        key='download_button_filtered'
    )


