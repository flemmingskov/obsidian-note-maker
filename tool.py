'''
File: tool.py
Author: Flemming Skov 
Main streamlit/python file
Start app from a terminal window typing: "streamlit run 'path_to_tool_folder'/tool.py
Latest version: March 6 2022
'''

# Import libraries
import streamlit as st
import os, time, datetime, sqlite3, platform, logging, itertools, re, math, sys, random
import copy
import pandas as pd
import numpy as np
from nltk.stem import WordNetLemmatizer
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
from nltk.stem import WordNetLemmatizer
from adjustText import adjust_text
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from matplotlib import ticker

wnl = WordNetLemmatizer()
exec(open("./apps/function_library.py").read())

def main():
    exec(open("./apps/function_library.py").read())    

    st.sidebar.title("Tools for sciencescaping")
    st.sidebar.info("Version 0.99 - May 11, 2021")
    
    # Menu layout
    page = st.sidebar.selectbox("Choose a tool page: ",
             ["Home",
                    "1 - import raw data",
                    "2 - create keyword vocabulary",
                    "3 - extract keywords from title",
                    "4 - to obsidian",
                    "5 - build co-occurrence matrix",
                    "6 - create network graph",
                    "7 - adjust graph layout",
                    "8 - prepare data for mapping",
                    "9 - create maps"])

    if page == "Home":
        exec(open("./apps/tool0_home.py").read())

    elif page == "1 - import raw data":
        exec(open("./apps/tool1_import.py").read())  

    elif page == "2 - create keyword vocabulary":
        exec(open("./apps/tool2_keyword_vocabulary.py").read())
          
    elif page == "3 - extract keywords from title":
        exec(open("./apps/tool3_keyword_extraction.py").read())

    elif page == "4 - to obsidian":
        exec(open("./apps/tool4_obsidian.py").read())

    elif page == "5 - build co-occurrence matrix":
        exec(open("./apps/tool4_prep_network.py").read())

    elif page == "6 - create network graph":
        exec(open("./apps/tool5_graph.py").read()) 
          
    elif page == "7 - adjust graph layout":
        exec(open("./apps/tool6_adjust_layout.py").read()) 

    elif page == "8 - prepare data for mapping":
        exec(open("./apps/tool7_prepare_map_db.py").read())

    elif page == "9 - create maps":
        exec(open("./apps/map1_distribution.py").read())

       
    st.sidebar.markdown("   ")
    #st.sidebar.image("scimap.png", caption="Science map")
    st.sidebar.title("GitLab - code")
    st.sidebar.info(
        '''
        This a develooping project and you are very welcome to *contribute* your 
        comments or questions to
        [GitLab](https://gitlab.com/flemmingskov/plosone_science_map). 
        ''')
    st.sidebar.title("About")
    st.sidebar.info(
        """
        This app is maintained by Flemming Skov - 
        [AU Pure](https://pure.au.dk/portal/da/persons/flemming-skov(d16e357d-aa51-4bd3-ae16-9059110a3fe8).html).   
        """)
   
if __name__ == "__main__":
    main()
