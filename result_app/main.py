import streamlit as st
from pathlib import Path
import re

# Set page config
st.set_page_config(
    page_title="Master Thesis Presentation",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Force white background and dark text
st.markdown("""
    <style>
        .stApp {
            background-color: white;
            color: black !important;
        }
        .stMarkdown, .stText, h1, h2, h3, h4, h5, h6, p, .stTextInput > div > div > input {
            color: black !important;
        }
        .stSelectbox > div > div > div {
            background-color: white;
            color: black !important;
        }
        .stRadio > div {
            background-color: white;
            color: black !important;
        }
        .stCheckbox > label > div {
            background-color: white !important;
        }
        .stSlider > div > div {
            background-color: white !important;
        }
        .stPlotlyChart, .stDataFrame {
            background-color: white !important;
        }
    </style>
    """, unsafe_allow_html=True)

# Rest of your code remains the same
def extract_numbers_from_dirname(dirname):
    match = re.search(r'run_qubits_(\d+)_depth_(\d+)_iteration_(\d+)', str(dirname))
    if match:
        n_qubits = int(match.group(1))
        depth = int(match.group(2))
        iteration = int(match.group(3))
        return n_qubits, depth, iteration
    return None, None, None

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Home", 
    "Generated Model", 
    "VQE Results", 
    "Derivative Prices", 
    "LP VQE Results", 
    "Average Results by Qubits and Reps", 
    "Performance Metrics Summary",
    "LP Average Results by Qubits and Reps",
    "LP D_min, VQE Price, D_max Comparison"
])

# Directory selection
st.sidebar.title("Select Directory")
model_base_path = Path("model/test_2024_09_06")
run_dirs = [d for d in model_base_path.iterdir() if d.is_dir()]

run_dirs_with_numbers = [(d, *extract_numbers_from_dirname(d)) for d in run_dirs if extract_numbers_from_dirname(d) is not None]
sorted_run_dirs = sorted(run_dirs_with_numbers, key=lambda x: (x[1], x[2], x[3]))  

sorted_run_dir_strs = [f"Qubits: {d[1]}, Depth: {d[2]}, Iteration: {d[3]}" for d in sorted_run_dirs]
selected_run_dir = st.sidebar.selectbox("Select the run directory", sorted_run_dir_strs)

selected_run_dir_path = dict(zip(sorted_run_dir_strs, [str(d[0]) for d in sorted_run_dirs]))[selected_run_dir]

def display_home():
    st.title("Welcome to the Master Thesis Presentation")
    st.write("""
        This is the home page for the Master Thesis Presentation.
        Navigate using the sidebar to explore different sections.
    """)

if page == "Home":
    display_home()
elif page == "Generated Model":
    import _pages.page1 as page1
    page1.main(selected_run_dir_path)
elif page == "VQE Results":
    import _pages.page2 as page2
    page2.main(selected_run_dir_path)
elif page == "Derivative Prices":
    import _pages.page3 as page3
    page3.main(selected_run_dir_path)
elif page == "LP VQE Results":
    import _pages.page4 as page4
    page4.main(selected_run_dir_path)
elif page == "Average Results by Qubits and Reps":
    import _pages.page5 as page5
    page5.main()
elif page == "Performance Metrics Summary":
    import _pages.page6 as page6
    page6.main()
elif page == "LP Average Results by Qubits and Reps":
    import _pages.page7 as page7
    page7.main()
elif page == "LP D_min, VQE Price, D_max Comparison":
    import _pages.page8 as page8
    page8.main()