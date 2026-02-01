import streamlit as st
from dashboard_functions import *
import sys

results_folder = sys.argv[1] + "/"

subdivision = get_subdivisions(results_folder)

selected_folder = dict()

st.set_page_config(layout="wide")

for subdivision_name, cases in subdivision.items():
    selected_folder[subdivision_name] = st.sidebar.multiselect(
        subdivision_name + " case", cases, cases
    )

for case, shown_run in selected_folder.items():
    show_valid_tables(results_folder, shown_run)

    display_training_scores(results_folder, shown_run)

    display_validation_trees(results_folder, shown_run)

    display_power_valid(results_folder, shown_run)

    display_power_train(results_folder, shown_run)
