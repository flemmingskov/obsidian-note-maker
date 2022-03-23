'''
File: tool.py
Author: Flemming Skov 
Start app from a terminal window typing: "streamlit run 'path_to_tool_folder'/tool.py
Latest version: March 23 2022
Goal: experimenting with zipping
'''

# IMPORT LIBRARIES
##################
import streamlit as st
import pandas as pd
import numpy as np
import datetime
import platform
import itertools
import re
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
empty_line = ("\n\n")


# DEFINE FUNCTIONS
#################
def get_platform():
    if (platform.system() == 'Darwin'):
        mainwork = '/Users/flemmingskov/Desktop/'
    else:
        mainwork = 'C:/Users/au3406/iCloudDrive/Desktop/'
    return mainwork

def clean_keyword (old_keyword):
    composite_keyword = ''
    old_keyword = old_keyword.replace("/", "_")
    keyword = old_keyword.replace("-", " ")     # separating keywords joined by '-' in order to lemmatize and rejoin
    for keyword_part in keyword.split(" "):
        if keyword_part != "":
            keyword_part = wnl.lemmatize(keyword_part.lower())       
            if composite_keyword != "":
                composite_keyword = composite_keyword + "-" + keyword_part
            if composite_keyword == "":
                composite_keyword = composite_keyword + keyword_part   
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
st.sidebar.info("Version 0.88 - March 20, 2022")
st.sidebar.markdown("   ")

# EXPANDER for Settings
st.sidebar.header('Options')
with st.sidebar.expander("Show options"):
    st.write("Extract keywords from:")
    run_keywords = st.checkbox('Web of Science', value=True, key='runKeywords')
    run_title = st.checkbox('Title', value=False, key='runTitle')
    run_abstract = st.checkbox('Abstract', value=False, key='runKAbstract')
    number_keywords_to_include = st.slider('Number of keywords to include (sorted by occurrences)', 0, 250, 50)

st.sidebar.header("GitHub code")
st.sidebar.info(
    '''
    This a develooping project and you are very welcome to *contribute* your 
    comments or questions to
    [GitHub](https://github.com/flemmingskov/obsidian-note-maker). 
    ''')
st.sidebar.header("Author")
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
    st.subheader('From Web of Science records to Obsidian .md files')
    st.markdown(
        f"""       
        Use this tool to import data as **Web of Science tab delimited files** and convert them to **Obsidian .md files**. 
        
        __Export from Web of Science__:
        
        When a set of records has been chosen in Web of Science, use the 'Export' function and select 'Tab delimited file'.
        It is possible to export up to 1000 records a time and the process may have to be repeated. Use option 'records from .. to ..'
        to select records. Use 'Full Record' in 'Record Content' and File Format 'Tab-delimited'.

        __Three steps to export notes__: 
        * **UPLOAD FILES** export .txt files from your download folder to the 'Drag and drop files here' below.
        * **MANAGE KEYWORDS** select which keywords to use
        * **PREPARE NOTES FOR OBSIDIAN** to process Obsidian note files as a .zip  
    """)

# UPLOAD FILES
##############
st.write(' - - -')
st.header(' UPLOAD FILES')
st.markdown('Files should be __tab delimited__ with option __full record__ \
    using the __Export__ button in Web of Science')
    
uploaded_wos_files = st.file_uploader("", accept_multiple_files=True)

if not uploaded_wos_files:
  st.warning('Please, upload one ore more Web of Science export files')
  st.stop()

# initializing data frames
wos_data_df = pd.DataFrame()
#wos_data_df = pd.DataFrame()

for uploaded_file in uploaded_wos_files:
    print(' .. executed ' + str(current_time))
    columnames = []
    for i in range(0, 69):
        columnames.append(str(i))
  
    raw_data_df = pd.read_csv(uploaded_file, names=columnames,
                        index_col=False,
                        delimiter='\t',
                        skiprows=1)
    
    # The following works from February 2022 Web of Science (new fields added)                   
    raw_data_df = raw_data_df[raw_data_df.columns[[1, 8, 19, 20, 21, 22, 25, 26,
                                        28, 33, 9, 45, 59, 61, 63, 55, 34, 35]]]
         
    raw_data_df.replace(r'\s+', np.nan, regex=True).replace('', np.nan)
    raw_data_df = raw_data_df.fillna('')
    raw_data_df[100] = time_stamp    # New column with time stamp (today)
    raw_data_df.columns = ['authors', 'title', 'kw1', 'kw2', 'abstr',
                        'inst', 'email', 'autID', 'funding', 'cites',
                        'journal', 'year', 'wos_sub_cat1',
                        'wos_sub_cat2', 'wosid', 'doi', 'usc1', 'usc2', 'time_stamp']
    wos_data_df = pd.concat([wos_data_df, raw_data_df])

    wos_data_df = wos_data_df.drop_duplicates(subset='wosid', keep='last')
    wos_data_df[["cites", "year", "usc1", "usc2"]] = wos_data_df[["cites", "year", "usc1", "usc2"]].apply(pd.to_numeric, \
                        errors='coerce')
    wos_data_df.loc[wos_data_df['year'] < 1966, 'year'] = np.nan
    wos_data_df = wos_data_df.dropna(subset=['year', 'wosid'])
    wos_data_df = wos_data_df.reset_index(drop=True)
    wos_data_df['year'] = wos_data_df.year.astype(int)
    wos_data_df['cites'] = wos_data_df.cites.astype(int)

with st.expander("Show records imported"):
    st.dataframe(wos_data_df)

# linking unique categories to WoS IDs        
wos_list, year_list, cat_list, cite_list, person_list, email_list, funding_list = ([] for i in range(7))

for index, row in wos_data_df.iterrows():
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
st.header("MANAGE KEYWORDS")
st.subheader('1 - keywords from uploaded files')

search_criteria = """(links_to_categories_df.category != 'dummyCriteria')"""
selected_records = (links_to_categories_df.loc[eval(search_criteria), ['wosid']]
                .drop_duplicates(['wosid']))
extract_kewords_from_df = (pd.merge(selected_records, wos_data_df, on='wosid', how='inner')[['authors', 'title', 
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
st.subheader('2 - add private keywords (optional')
st.markdown('Optionally expand the search with a list of __private keywords__')

with st.expander('Upload file manage selections', expanded=False):
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


# EXTRACTING KEYWORDS FROM TITLE (and perhaps? abstract)
##################################################
st.write(' - - -')
st.header('PREPARE NOTES FOR OBSIDIAN')

with st.expander("Show list of selected keywords"):
    st.dataframe(final_list_of_keywords)

# set to supress warning of chained assigment - not recommended, but looks nicer in output ;)                
pd.set_option("mode.chained_assignment", None)

#run_script =  st.button('Extract keywords from title/abstract')
keyword_extracted_wos_data_df = wos_data_df.copy(deep=True)

##st.header('test')
#st.dataframe(keyword_extracted_wos_data_df)

#with st.expander("Extract keywords and create Obsidian notes"):
update_df = st.checkbox("Update data frame with keywords")
if update_df:
    with st.spinner('Processing and extracting keywords ...'):

        clean_list = []
        keyword_extracted_wos_data_df['kw1_clean'] = ''
        keyword_extracted_wos_data_df['kw2_clean'] = ''
        keyword_extracted_wos_data_df['kw_title'] = ''
        keyword_extracted_wos_data_df['kw_abst'] = ''

        for index, row in keyword_extracted_wos_data_df.iterrows():  # Iterate over all rows in dataframe

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
                keyword_extracted_wos_data_df.loc[index,['kw1_clean']] = concat_clean  

                ## keywords plus from Web of Science
                keyword_list = row[3]
                concat_clean = ''
                for kw in keyword_list.split(";"):
                    cleaned_keyword = clean_keyword(kw)
                    if concat_clean == '':
                        concat_clean = cleaned_keyword
                    else:
                        concat_clean = concat_clean + ';' + cleaned_keyword
                keyword_extracted_wos_data_df.loc[index,['kw2_clean']] = concat_clean

            # extracting keywords from title
            if run_title:
                title_text = row[1]
                keywords_in_title = extract_keywords_from_text(title_text, final_list_of_keywords) 
                keyword_extracted_wos_data_df.loc[index,['kw_title']] = keywords_in_title

            # extracting keywords from abstract
            if run_abstract:
                abstract_text = row[4]
                if abstract_text != '':
                    keywords_in_abstract = extract_keywords_from_text(abstract_text, final_list_of_keywords)
                    keyword_extracted_wos_data_df.loc[index,['kw_abst']] = keywords_in_abstract
                else:
                    keyword_extracted_wos_data_df.loc[index,['kw_abst']] = ''

    for col in keyword_extracted_wos_data_df.columns:
        print(col) 

    st.dataframe(keyword_extracted_wos_data_df)
    wos_data_for_obsidian = keyword_extracted_wos_data_df


    # OBSIDIAN EXPORT SCRIPT
    run_script =  st.button('Export to Obsidian')

    if run_script:
        keyword_list = final_list_of_keywords
        for col in wos_data_for_obsidian.columns:
            print(col)

        papers_in = wos_data_for_obsidian[['wosid', 'authors', 'title', 'abstr', 'year', 'journal', 'cites', 'wos_sub_cat1', 'doi', 'usc1', 'usc2', 'kw1_clean', 'kw2_clean', 'kw_title', 'kw_abst']].fillna('')
        papers_in['title'] = papers_in.title.astype(str)
        papers_in['year'] = papers_in.year.astype(int)
        papers_in['usc1'] = papers_in.usc1.astype(int)
        papers_in['usc2'] = papers_in.usc2.astype(int)

        # CREATE OBSIDIAN NOTES FOR KEYWORDS
        MDspace = get_platform() + 'testtest/'
        for keyword_item in keyword_list:
            keyword_item = keyword_item.replace("/", "-")
            md_file = open(MDspace + '% ' + str(keyword_item)+'.md',"w")

            md_file.write('#### ' + keyword_item + empty_line + '- - -' + empty_line)
            md_file.write('#keyword' + empty_line + '- - -' + empty_line)       
            md_file.write('\n\n' + '##### Notes: ' + empty_line)
            md_file.close()
            
        # CREATE OBSIDIAN NOTES FOR PAPERS WITH LINKS TO KEYWORDS AND SUBJECTS
        category_list = []

        for index, row in papers_in.iterrows():
            all_keywords_in_paper_list = []
            all_keywords_in_paper = ()
            wosid = row[0]
            wosid = wosid[4:]  
            authors = row[1]
            title_author = authors.split(',')[0]

            title_paper = row[2]
            split_title = title_paper.split(' ')[0:4]
            short_title = ' '.join(split_title)
            
            abstract = row[3]
            year = str(int(row[4]))
            
            note_title_unclean = title_author + ' ' + year + ' -  ' + short_title
            note_title = re.sub('[^a-zA-Z0-9 \n\.]', '', note_title_unclean)
        
            journal = row[5]
            journal = journal.title()
            cites = int(row[6])
            recent = int(row[9])
            historic = int(row[10])
            
            wos_categories = row[7]
            doi = row[8]

            author_keywords = row[11].split(";")
            plus_keywords = row[12].split(";")
            title_keywords = row[13].split(";")
            abstract_keywords = row[14].split(";")
            all_keywords_in_paper_set = set(list(itertools.chain(author_keywords,plus_keywords,title_keywords)))

            all_keywords_in_paper = list(all_keywords_in_paper_set)

            all_keywords_in_paper = list(filter(None, all_keywords_in_paper))
            all_keywords_in_paper = sorted(all_keywords_in_paper, key=str.lower)

            md_file = open(MDspace + note_title +'.md',"w")
            
            md_file.write('__' + title_paper + '__' + empty_line)
            md_file.write('' + journal +  empty_line)
            md_file.write('_' + authors +  '_' + empty_line)
            md_file.write('' + 'Data: [year:: ' + year + ']  [cites:: ' + str(cites) + '] [recent:: ' + str(recent) + '] [historic:: ' + str(historic) + ']')
            md_file.write(empty_line + '- - -' + empty_line)
            md_file.write('#paper   Web of Science id: ' + wosid + '     ')
            md_file.write('[Google Scholar ](https://scholar.google.dk/scholar?q=' + doi + ')' + empty_line)
            md_file.write(empty_line + '- - -' + empty_line)

            md_file.write(abstract)
            md_file.write(empty_line)
            md_file.write('- - -' + '\n')

            md_file.write('_[rating:: 0] (scale: 0-10)_ \n')
            md_file.write('- [ ] checked \n' + '- - -' + '\n')
            
            md_file.write('*WoS categories:*' + ': \n')
            paper_categories = wos_categories.split(";")
            for category in paper_categories:
                categoryLstrp = category.lstrip()
                md_file.write('[[%% ' + categoryLstrp + ']]' + '  ')
                category_list.append(categoryLstrp)
            
            md_file.write(empty_line + '*Keyword links:*' + '\n')
            add_keywords = []
            for kw in all_keywords_in_paper:
                if kw in keyword_list:
                    md_file.write('[[% ' + kw + ']]' + '\n') # + empty_line)
                else:
                    add_keywords.append(kw)
            md_file.write(empty_line +  'Additional keywords: ' + empty_line)
            add_keywords_nicelist = ('\n'.join(add_keywords))
            md_file.write(str(add_keywords_nicelist))        
            md_file.close()
            
        # CREATE NOTES FOR SUBJECT CATEGORIES
        category_list = list(set(category_list))
        category_list.sort()

        for category_item in category_list:
            md_file = open(MDspace + '%% ' + str(category_item)+'.md',"w")

            md_file.write('#### ' + category_item + empty_line + '- - -' + empty_line)
            md_file.write('#subject' + empty_line + '- - -' + empty_line)       
            md_file.write('\n\n' + '##### Characteristic keywords (most frequent first): ' + empty_line)

            # selResKey = data_keywords[data_keywords[column_identifier] == category_item]
            # selResKey = selResKey.dropna()
            # selResKey.sort_values(by='year', ascending=0)

            # totList = selResKey['keyword'].value_counts()
            # totList = pd.DataFrame(totList).reset_index()

            # countKeywords = 0
            # for index, row in totList.iterrows():
            #     countKeywords = countKeywords + 1
            #     if (int(row[1]) < 2) or (countKeywords > 25):
            #         break
            #     md_file.write(str(row[1]) + ' - [[' + str(row[0]) + ']] ' + ' \n')

            md_file.close()

        print('... obsidian mark-down files created')