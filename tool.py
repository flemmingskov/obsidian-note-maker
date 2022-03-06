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
import copy, zipfile
import pandas as pd
import numpy as np
#import altair as alt
#import seaborn as sns
#import matplotlib.pyplot as plt
#from matplotlib import ticker
#plt.style.use('seaborn-whitegrid')
#from adjustText import adjust_text
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
wnl = WordNetLemmatizer()


time_stamp = datetime.datetime.now().strftime('(%Y,%b,%d)')  # time stamp

exec(open("./function_library.py").read())    

st.sidebar.title("Obsidian Note Maker")
st.sidebar.info("Version 0.01 - March 6, 2022")

# Menu layout
page = st.sidebar.selectbox("Choose a tool page: ",
            ["Home",
                "1 - import raw data",
                "2 - create keyword vocabulary",
                "3 - extract keywords from title",
                "4 - to obsidian"])

# CREATE SIDEBAR AND CONTENTS
st.sidebar.markdown("   ")
st.sidebar.title("GitHub code")
st.sidebar.info(
    '''
    This a develooping project and you are very welcome to *contribute* your 
    comments or questions to
    [GitHub](https://github.com/flemmingskov/obsidian-note-maker). 
    ''')
st.sidebar.title("Author")
st.sidebar.info(
    """
    This app is created and maintained by Flemming Skov - 
    [AU Pure](https://pure.au.dk/portal/da/persons/flemming-skov(d16e357d-aa51-4bd3-ae16-9059110a3fe8).html).

    Email: fs@ecos.au.dk   
    """)

# CREATE TITLE AND FIRST SECTION
st.title('A note maker for Obsidian')
st.info('From Web of Science export files to Obsidian .md files')

# Expander for About & help ....
with st.expander("About & help  ..."):
    st.markdown(
        f"""       
        Use this tool to **import data from Web of Science**. When a set of records has been found in Web of Science, use the 'Export ...' 
        function and select 'Other File Formats'. It is  possible to export up to 1000 records a time and the process may have to be repeated. Use the 'records from .. to ..'
        to select records. Use 'Full Record' in 'Record Content' and File Format 'Tab-delimited'. Copy export .txt files
        to 'Data' folder in the chosen workspace.

        * Import files using the ....  upload thing
        * Any number of .txt files in the 'Data' folder will be imported when pressing 'Run'
        * All processed obsidian note files will be exportet as a .zip  
    """)

# UPLOAD RAW DATA 
# def append_member(zip_file, member):
#     with zipfile.ZipFile(zip_file, mode="a") as archive:
#         archive.write(member)

uploaded_files = st.file_uploader("Choose a file", accept_multiple_files=True)
# for uploaded_file in uploaded_files:
#     append_member('MyZip', uploaded_file.name)
#     bytes_data = uploaded_file.read()
#     st.write("File uploaded: ", uploaded_file.name)
#      #st.write(str(len(bytes_data)))
#      #st.write(bytes_data)

        
new_df = pd.DataFrame()

# step 1 - open and import file content(s) to new dataframe
#os.chdir(my_workspace+'data/')
for uploaded_file in uploaded_files:

    print(' .. importing: ')
    columnames = []
    for i in range(0, 68):
        columnames.append(str(i))
   
    wosIn_df = pd.read_csv(uploaded_file, names=columnames,
                        index_col=False,
                        delimiter='\t',
                        skiprows=1)
    
    # NOTE - a problem encountered, where the numbering of last three lines had changed
    #wosIn_df = wosIn_df[wosIn_df.columns[[1, 8, 19, 20, 21, 22, 24, 25,
    #                                    27, 32, 9, 44, 58, 59, 61, 54, 33, 34]]] 
    # 
    # The following works from February 2022 Web of Science (new fields added)                   
    wosIn_df = wosIn_df[wosIn_df.columns[[1, 8, 19, 20, 21, 22, 25, 26,
                                        28, 33, 9, 45, 59, 61, 63, 55, 34, 35]]]

    wosIn_df.replace(r'\s+', np.nan, regex=True).replace('', np.nan)
    wosIn_df = wosIn_df.fillna('')
    wosIn_df[100] = time_stamp    # New column with time stamp (today)
    wosIn_df.columns = ['authors', 'title', 'kw1', 'kw2', 'abstr',
                        'inst', 'email', 'autID', 'funding', 'cites',
                        'journal', 'year', 'wos_sub_cat1',
                        'wos_sub_cat2', 'wosid', 'doi', 'usc1', 'usc2', 'time_stamp']
    new_df = pd.concat([new_df, wosIn_df])

new_df = new_df.drop_duplicates(subset='wosid', keep='last')

st.dataframe(new_df)

#st.dataframe(new_df)