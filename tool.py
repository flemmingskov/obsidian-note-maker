'''
File: tool.py
Author: Flemming Skov 
Main streamlit/python file
Start app from a terminal window typing: "streamlit run 'path_to_tool_folder'/tool.py
Latest version: March 8 2022
'''

# Import libraries
import streamlit as st
import os, time, datetime, sqlite3, platform, logging, itertools, re, math, sys, random
import copy, zipfile
import pandas as pd
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
wnl = WordNetLemmatizer()

time_stamp = datetime.datetime.now().strftime('(%Y,%b,%d)')  # time stamp

exec(open("./function_library.py").read())    

st.sidebar.title("Obsidian Note Maker")
st.sidebar.info("Version 0.01 - March 6, 2022")

# Menu layout
# page = st.sidebar.selectbox("Choose a tool page: ",
#             ["Home",
#                 "1 - import raw data",
#                 "2 - create keyword vocabulary",
#                 "3 - extract keywords from title",
#                 "4 - to obsidian"])

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

# various fixed settings 
(label_name, faculty_name, discipline_name, unit_name, person_name) \
        = ('biol', 'Sciences', 'Biology', '', '')

# CREATE TITLE AND FIRST SECTION
st.title('A note maker for Obsidian')

# Expander for About & help ....
with st.expander("Instructions"):
    st.subheader('From Web of Science export files to Obsidian .md files')
    st.markdown(
        f"""       
        Use this tool to import data as **Web of Science tab delimited files** and convert them to **Obsidian .md files**. 
        When a set of records has been chosen in Web of Science, use the 'Export' function and select 'Tab delimited file'.
        It is possible to export up to 1000 records a time and the process may have to be repeated. Use option 'records from .. to ..'
        to select records. Use 'Full Record' in 'Record Content' and File Format 'Tab-delimited'. 

        * **Drag** export .txt files from your download folder to the 'Drag and drop files here' below.
        * Press '**Run**' to import data
        * Press '**Export**' to process Obsidian note files as a .zip  
    """)

#st.markdown('___')

st.markdown(' ')
st.subheader('I - Import .txt files')
st.write('  ')
    
# I - open and import file content(s) to dataframe wosIn_df
uploaded_files = st.file_uploader("", accept_multiple_files=True)

new_df = pd.DataFrame()
wosIn_df = pd.DataFrame()
for uploaded_file in uploaded_files:

    print(' .. importing: ')
    columnames = []
    for i in range(0, 69):
        columnames.append(str(i))
  
    wosIn_df = pd.read_csv(uploaded_file, names=columnames,
                        index_col=False,
                        delimiter='\t',
                        skiprows=1)
    
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
    new_df[["cites", "year", "usc1", "usc2"]] = new_df[["cites", "year", "usc1", "usc2"]].apply(pd.to_numeric, \
                        errors='coerce')
    new_df.loc[new_df['year'] < 1966, 'year'] = np.nan
    new_df = new_df.dropna(subset=['year', 'wosid'])

    new_df = new_df.reset_index(drop=True)
    new_df['year'] = new_df.year.astype(int)
    new_df['cites'] = new_df.cites.astype(int)

# II - linking unique categories to WoS IDs        
wos_list, year_list, cat_list, cite_list, faculty_list, discipline_list, unit_lilst, \
    person_list, label_list, email_list, funding_list = ([] for i in range(11))

for index, row in new_df.iterrows():
    wosid = row[14]
    yr = row[11]
    cat1list = row[12] #specific categories
    cat2list = row[13]
    cites = row[9]

    for cat in cat1list.split(";"):
        if cat != "":
            cat = cat.lstrip()
            wos_list.append(wosid)
            year_list.append(yr)
            cat_list.append(cat)
            cite_list.append(cites)
            faculty_list.append(faculty_name)
            discipline_list.append(discipline_name)
            unit_lilst.append(unit_name)
            person_list.append(person_name)
            label_list.append(label_name)

links_df = pd.DataFrame({'wosid': wos_list, 'year': year_list,
                    'category': cat_list, 'cites': cite_list,
                    'faculty': faculty_list,
                    'discipline': discipline_list, 'unit': unit_lilst,
                    'person': person_list, 'label': label_list})
links_df = links_df[1:]
links_df = links_df[['wosid', 'year', 'cites', 'category', 'faculty',
                    'discipline', 'unit', 'person', 'label']]
dep_links_df = links_df
dep_links_df = dep_links_df.drop_duplicates(keep='last')

#ß st.dataframe(new_df)


# III -  creating a keyword vocabulary
st.markdown(' ')
st.subheader('II - List of unique keywords')
st.write('  ')
run_script =  st.button('Create list of keywords')

if run_script:
    searchCriteria = """(links_df.category != 'dummyCriteria')"""   ## If necessary to select subset
    subset = (dep_links_df.loc[eval(searchCriteria), ['wosid']]
                    .drop_duplicates(['wosid']))
    dataIn = (pd.merge(subset, new_df, on='wosid', how='inner')[['authors', 'title', 
                    'kw1', 'kw2', 'abstr', 'inst', 'email', 'autID', 'funding', 
                    'cites', 'journal', 'year', 'wos_sub_cat1', 'wos_sub_cat2', 
                    'wosid', 'time_stamp']]  )
                    
    # st.dataframe(dataIn)

    synonyms = import_synonyms()
    (wosid_list, year_list, orig_list, clean_list, comb_list, keyword_list) = \
        ([], [], [], [], [], [])
    
    for index, row in dataIn.iterrows():  # Iterate over all rows in dataframe
        concat_clean = []

        if row[2] != '':
            keyword_list = row[2] + ';' +  row[3]
        else:
            keyword_list = row[3]     

        for kw in keyword_list.split(";"):  # iterate over all keywords in one article
            cleaned_keyword = clean_keyword(kw)
            wosid_list.append(row[14])
            year_list.append(row[11])
            orig_list.append(kw)
            clean_list.append(cleaned_keyword)
            concat_clean.append(cleaned_keyword)
        comb_list.append(list(itertools.combinations(concat_clean, 2)))                

    raw_keyword_list = ((pd.DataFrame({'keyword': clean_list, 'orgKw': orig_list,
                'year': year_list, 'wosid': wosid_list})[:])
                [['wosid', 'year', 'keyword', 'orgKw']] )

    keyword_count = pd.DataFrame(raw_keyword_list['keyword'].value_counts())
    keyword_count['label'] = keyword_count.index   # create new column and copy index to it
    keyword_count.columns = ["keyword_count", "keyword"]

    ## LIST of keywords to erase from vocabulary
    for delete_kw in ['neanderthal']:
    # for delete_kw in ['dna-barcoding', 'environmental-dna', 'metabarcoding', 'edna', 'dna', 'barcoding',
    #             'extracellular-dna', 'dna-metabarcoding', 'dna-barcodes', 'barcode']:
        keyword_count = keyword_count[keyword_count.keyword != delete_kw]
    #   keyword_count.drop(keyword_count.loc[keyword_count['keyword']==delete_kw].index, inplace=True)
        
    keyword_count = keyword_count.reset_index(drop=True)
    total_count = len(keyword_count)
    ones = total_count - len(keyword_count[keyword_count['keyword_count'] == 1])
    twos = total_count - len(keyword_count[keyword_count['keyword_count'] <= 2])

    st.markdown(
    f"""Total number of keywords is {total_count}. Number of keywords with two or 
    more occurrences: {ones}. Number of keywords with three or 
    more occurrences: {twos}""") 

    st.dataframe(keyword_count)

#             keyword_count.to_sql('vocabulary', conn, if_exists='replace')
#             conn.close()     

#     except Exception as e:
#         print("Program Error: ")
#         print(e)

#     finally:
#         print('... step 2 - vocabulary of keywords created')
