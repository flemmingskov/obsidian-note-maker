'''
File: tool.py
Author: Flemming Skov 
Start app from a terminal window typing: "streamlit run 'path_to_tool_folder'/tool.py
Latest version: March 14 2022
'''

# IMPORT LIBRARIES
##################
import streamlit as st
import pandas as pd
import numpy as np
import datetime
import platform
import itertools
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
wnl = WordNetLemmatizer()
# import os, time, sqlite3, logging, re, math, sys, random, copy, zipfile


# SETTING VARIABLES
###################
time_stamp = datetime.datetime.now().strftime('(%Y,%b,%d)')
current_time = datetime.datetime.now().strftime("%H:%M:%S")
imported_keywords_chosen = []
private_keywords_chosen = []


# DEFINE FUNCTIONS
#################
def get_platform():
    if (platform.system() == 'Darwin'):
#        logging.info(' running MacOS ...')
        mainwork = '/Users/flemmingskov/Desktop/'
    else:
 #       logging.info(' running Windows 10 ...')
        mainwork = 'C:/Users/au3406/iCloudDrive/Desktop/'
    return mainwork

def clean_keyword (old_keyword):
    composite_keyword = ''
    old_keyword = old_keyword.replace("/", "_")
    keyword = old_keyword.replace("-", " ")     # separating keywords joined by '-' in order to lemmatize and rejoin
    for keyword_part in keyword.split(" "):
        if keyword_part != "":
            keyword_part = wnl.lemmatize(keyword_part.lower())
            #if (synonyms.get(keyword_part, "none")) != 'none':   # check if part of keyword is a synonym
            #    keyword_part = synonyms.get(keyword_part)              
            if composite_keyword != "":
                composite_keyword = composite_keyword + "-" + keyword_part
            if composite_keyword == "":
                composite_keyword = composite_keyword + keyword_part   
            #if (synonyms.get(composite_keyword, "none")) != 'none':   # check if the new composite keyword is a synonym
            #    composite_keyword = synonyms.get(composite_keyword)
    if composite_keyword != '':
        return composite_keyword

def adjacent_pairs(seq):
    pairList = []
    it = iter(seq)
    prev = next(it)
    for item in it:
        pairList.append(prev + '-' + item)
        prev = item
    return pairList

def extract_keywords_from_text (myText, keyword_list):
    myText = myText.replace("-", " ").replace(".", " ").replace(",", " ").lower()
    text_tokens = word_tokenize(myText)
    tokens_without_sw = [word for word in text_tokens if not word in set(stopwords.words('english'))]
    all_combinations = adjacent_pairs(tokens_without_sw) + tokens_without_sw
    for keyword_raw in all_combinations:
        cleaned_keyword = clean_keyword(keyword_raw)
    all_combinations_clean = [word for word in all_combinations if word in keyword_list]
    all_combinations_clean = ';'.join([str(elem) for elem in list(set(all_combinations_clean))])
    return all_combinations_clean 



# SIDEBAR LAYOUT
################
st.sidebar.title("Obsidian Note Maker")
st.sidebar.info("Version 0.02 - March 14, 2022")
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

# MAIN PAGE LAYOUT
##################
st.title('Convert Web of Science data to Obsidian notes')

# EXPANDER for About & help
###########################
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

# EXPANDER for Settings
with st.expander("Settings and controls ..."):
    st.write("Extract keywords from:")
    run_keywords = st.checkbox('Web of Science', value=True, key='runKeywords')
    run_title = st.checkbox('Title', value=False, key='runTitle')
    run_abstract = st.checkbox('Abstract', value=False, key='runKAbstract')
    number_keywords_to_include = st.slider('Number of keywords to include (sorted by occurrences)', 0, 250, 50)


# UPLOAD FILES
##############
st.write(' - - -')
st.subheader('-> upload WoS  files')
st.markdown('Files should be __tab delimited__ with option __full record__ \
    using the __Export__ button in Web of Science')
    
uploaded_wos_files = st.file_uploader("", accept_multiple_files=True)

if not uploaded_wos_files:
  st.warning('Please, upload one ore more Web of Science export files')
  st.stop()


# initializing data frames
raw_wos_data_df = pd.DataFrame()
processed_wos_data_df = pd.DataFrame()

for uploaded_file in uploaded_wos_files:
    print(' .. executed ' + str(current_time))
    columnames = []
    for i in range(0, 69):
        columnames.append(str(i))
  
    raw_wos_data_df = pd.read_csv(uploaded_file, names=columnames,
                        index_col=False,
                        delimiter='\t',
                        skiprows=1)
    
    # The following works from February 2022 Web of Science (new fields added)                   
    raw_wos_data_df = raw_wos_data_df[raw_wos_data_df.columns[[1, 8, 19, 20, 21, 22, 25, 26,
                                        28, 33, 9, 45, 59, 61, 63, 55, 34, 35]]]
         
    raw_wos_data_df.replace(r'\s+', np.nan, regex=True).replace('', np.nan)
    raw_wos_data_df = raw_wos_data_df.fillna('')
    raw_wos_data_df[100] = time_stamp    # New column with time stamp (today)
    raw_wos_data_df.columns = ['authors', 'title', 'kw1', 'kw2', 'abstr',
                        'inst', 'email', 'autID', 'funding', 'cites',
                        'journal', 'year', 'wos_sub_cat1',
                        'wos_sub_cat2', 'wosid', 'doi', 'usc1', 'usc2', 'time_stamp']
    processed_wos_data_df = pd.concat([processed_wos_data_df, raw_wos_data_df])

    processed_wos_data_df = processed_wos_data_df.drop_duplicates(subset='wosid', keep='last')
    processed_wos_data_df[["cites", "year", "usc1", "usc2"]] = processed_wos_data_df[["cites", "year", "usc1", "usc2"]].apply(pd.to_numeric, \
                        errors='coerce')
    processed_wos_data_df.loc[processed_wos_data_df['year'] < 1966, 'year'] = np.nan
    processed_wos_data_df = processed_wos_data_df.dropna(subset=['year', 'wosid'])
    processed_wos_data_df = processed_wos_data_df.reset_index(drop=True)
    processed_wos_data_df['year'] = processed_wos_data_df.year.astype(int)
    processed_wos_data_df['cites'] = processed_wos_data_df.cites.astype(int)

with st.expander("Show records imported"):
    st.dataframe(processed_wos_data_df)

# linking unique categories to WoS IDs        
wos_list, year_list, cat_list, cite_list, person_list, email_list, funding_list = ([] for i in range(7))

for index, row in processed_wos_data_df.iterrows():
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

links_to_categories_df = pd.DataFrame({'wosid': wos_list, 'year': year_list,
                    'category': cat_list, 'cites': cite_list})
links_to_categories_df = links_to_categories_df[1:]
links_to_categories_df = links_to_categories_df[['wosid', 'year', 'cites', 'category']]
links_to_categories_df = links_to_categories_df.drop_duplicates(keep='last')


# SHOW KEYWORDS FROM UPLOADED FILES
###################################
st.write(' - - -')
st.subheader('-> keywords in uploaded files')

search_criteria = """(links_to_categories_df.category != 'dummyCriteria')"""
selected_records = (links_to_categories_df.loc[eval(search_criteria), ['wosid']]
                .drop_duplicates(['wosid']))
extract_kewords_from_df = (pd.merge(selected_records, processed_wos_data_df, on='wosid', how='inner')[['authors', 'title', 
                'kw1', 'kw2', 'abstr', 'inst', 'email', 'autID', 'funding', 
                'cites', 'journal', 'year', 'wos_sub_cat1', 'wos_sub_cat2', 
                'wosid', 'time_stamp']]  )

(wosid_list, year_list, orig_list, clean_list, comb_list, keyword_list) = \
    ([], [], [], [], [], [])

for index, row in extract_kewords_from_df.iterrows():  # Iterate over all rows in dataframe
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
keyword_count = keyword_count.reset_index(drop=True)
total_count = len(keyword_count)
ones = total_count - len(keyword_count[keyword_count['keyword_count'] == 1])
twos = total_count - len(keyword_count[keyword_count['keyword_count'] <= 2])

st.markdown(
f"""Total number of keywords is {total_count}. Number of keywords with two or 
more occurrences: {ones}. Number of keywords with three or 
more occurrences: {twos}""") 

with st.expander("Show all keywords from uploaded files"):
    st.dataframe(keyword_count)
    
imported_keyword_series = keyword_count['keyword'].squeeze()
imported_keywords_default = imported_keyword_series[0:number_keywords_to_include]
imported_keywords_options = imported_keyword_series[0:500]

imported_keywords_chosen = st.multiselect(
    'Add or delete keywords from list',
    imported_keywords_options,
    imported_keywords_default)


# SHOW PRIVATE KEYWORDS
#######################
st.write(' - - -')
st.subheader('-> private keywords')
st.markdown('Optionally expand the search with a list of __private keywords__')

with st.expander('Show multiselect', expanded=True):
    uploaded_keywords = st.file_uploader("", accept_multiple_files=False)
    if not uploaded_keywords:
        final_list_of_keywords = []
        st.warning('No private keywords uploaded')
    else:
        private_keywords = pd.read_csv(uploaded_keywords).squeeze()
        private_keywords_chosen = st.multiselect(
        'Add or delete keywords from list',
        private_keywords,
        private_keywords)
    final_list_of_keywords = list(set(imported_keywords_chosen + private_keywords_chosen))
    final_list_of_keywords.sort()


# SHOW LIST OF COMBINED KEYWORDS FOR FINAL EDITING
##################################################
st.write(' - - -')
st.subheader('-> combined keywords')
st.markdown('Final list of combined WoS and private keywords - sorted alphabetically')
final_keywords_chosen = st.multiselect(
    'Add or delete keywords from list',
    final_list_of_keywords,
    final_list_of_keywords)


# EXTRACTING KEYWORDS FROM TITLE (and perhabs? abstract)
##################################################
st.write(' - - -')
st.subheader('-> extract keywords from title and abstract')
st.markdown('   ')

keywords_approved = final_keywords_chosen

# set to supress warning of chained assigment - not recommended, but looks nicer in output ;)                
pd.set_option("mode.chained_assignment", None)

run_script =  st.button('Extract keywords from imported files')
if run_script:
    with st.spinner('Processing and extracting keywords ...'):
        clean_list = []
        processed_wos_data_df['kw1_clean'] = ''
        processed_wos_data_df['kw2_clean'] = ''
        processed_wos_data_df['kw_title'] = ''
        processed_wos_data_df['kw_abst'] = ''

        for index, row in processed_wos_data_df.iterrows():  # Iterate over all rows in dataframe

            # cleaning regular keywords
            if run_keywords:
                current_record = row[14]
                keyword_list = row[2]
                concat_clean = ''
                for kw in keyword_list.split(";"):
                    cleaned_keyword = clean_keyword(kw)
                    if concat_clean == '':
                        concat_clean = cleaned_keyword
                    else:
                        concat_clean = concat_clean + ';' + cleaned_keyword
                processed_wos_data_df.loc[index,['kw1_clean']] = concat_clean  

                ## keywords plus from Web of Science
                keyword_list = row[3]
                concat_clean = ''
                for kw in keyword_list.split(";"):
                    cleaned_keyword = clean_keyword(kw)
                    if concat_clean == '':
                        concat_clean = cleaned_keyword
                    else:
                        concat_clean = concat_clean + ';' + cleaned_keyword
                processed_wos_data_df.loc[index,['kw2_clean']] = concat_clean

            # extracting keywords from title
            if run_title:
                title_text = row[1]
                keywords_in_title = extract_keywords_from_text(title_text, keywords_approved) 
                processed_wos_data_df.loc[index,['kw_title']] = keywords_in_title

            # extracting keywords from abstract
            if run_abstract:
                abstract_text = row[4]
                if abstract_text != '':
                    keywords_in_abstract = extract_keywords_from_text(abstract_text, keywords_approved)
                    processed_wos_data_df.loc[index,['kw_abst']] = keywords_in_abstract
                else:
                    processed_wos_data_df.loc[index,['kw_abst']] = ''

    # dataOut = processed_wos_data_df[['wosid', 'title', 'kw1_clean', 'kw2_clean', 'kw_title', 'kw_abst', 'year']]
    # dataOut[["year"]] = dataOut[["year"]].apply(pd.to_numeric)
    # dataOut = dataOut[dataOut['year'].notna()]

    with st.expander("Show records processed"):
        st.dataframe(processed_wos_data_df)