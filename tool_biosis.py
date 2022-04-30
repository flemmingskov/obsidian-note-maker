'''
File: tool.py
Author: Flemming Skov 
Start app from a terminal window typing: "streamlit run 'path_to_tool_folder'/tool.py
Latest version: April 30 2022
Current goal: Alternative import of data (for BIOSIS)
'''

# IMPORT LIBRARIES
##################
import streamlit as st
import pandas as pd
import numpy as np
import datetime
import platform
import itertools
import zipfile
import re
import os
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


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
        desktop = os.path.join(os.path.join(os.path.expanduser('~')), 'Downloads/')
    else:
        desktop = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Downloads/')
    return desktop

def clean_keyword (old_keyword):
    composite_keyword = ''
    old_keyword = old_keyword.replace("/", "_")
    keyword = old_keyword.replace("-", " ")
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
st.sidebar.title("BIOSIS importer")
st.sidebar.info("Version 0.1 - April 30, 2022")
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
st.title('Convert BIOSIS id to doi')

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
##############

genre = st.radio("Type of input: ",('WoS', 'BIOSIS'))

#st.write(' - - -')
st.header(' UPLOAD FILES')
st.markdown('Files should be __tab delimited__ with option __full record__ \
    using the __Export__ button in Web of Science')
    
uploaded_wos_files = st.file_uploader("", accept_multiple_files=True)

if not uploaded_wos_files:
  st.warning('Please, upload one ore more Web of Science export files')
  st.stop()

for uploaded_file in uploaded_wos_files:
    columnames = []
    for i in range(0, 70):
        columnames.append(str(i))
    raw_data_df = pd.read_csv(uploaded_file, names=columnames,
                        index_col=False,
                        delimiter='\t',
                        skiprows=1)

wos_data_df = pd.DataFrame()

if genre == 'BIOSIS':
    ##### New code!
    raw_bios_df = raw_data_df
    raw_bios_df = raw_bios_df[raw_bios_df.columns[[46]]]
    raw_bios_df.replace(r'\s+', np.nan, regex=True).replace('', np.nan)
    raw_bios_df = raw_bios_df.fillna('')
    #st.write(raw_bios_df)
    
if genre == 'WoS':
    # The following works from February 2022 Web of Science (new fields added)                   
    raw_data_df = raw_data_df[raw_data_df.columns[[1, 8, 19, 20, 21, 22, 25, 26,
                                        20, 33, 9, 45, 59, 61, 63, 55, 34, 35]]]
         
    raw_data_df.replace(r'\s+', np.nan, regex=True).replace('', np.nan)
    raw_data_df = raw_data_df.fillna('')
    raw_data_df[100] = time_stamp    # New column with time stamp (today)
    raw_data_df.columns = ['authors', 'title', 'kw1', 'kw2', 'abstr',
                        'inst', 'email', 'autID', 'funding', 'cites',
                        'journal', 'year', 'wos_sub_cat1',
                        'wos_sub_cat2', 'wosid', 'doi', 'usc1', 'usc2', 'time_stamp']
    wos_data_df = pd.concat([wos_data_df, raw_data_df])

    wos_data_df = wos_data_df.drop_duplicates(subset='wosid', keep='last')
    wos_data_df[["cites", "year", "usc1", "usc2"]] = wos_data_df[["cites", "year", "usc1",
                        "usc2"]].apply(pd.to_numeric, errors='coerce')
    wos_data_df.loc[wos_data_df['year'] < 1966, 'year'] = np.nan
    wos_data_df = wos_data_df.dropna(subset=['year', 'wosid'])
    wos_data_df = wos_data_df.reset_index(drop=True)
    wos_data_df['year'] = wos_data_df.year.astype(int)
    wos_data_df['cites'] = wos_data_df.cites.astype(int)

    print('... files uploaded successfully')
    with st.expander("Show records imported"):
        st.dataframe(wos_data_df)

    # linking unique categories to WoS IDs        
    wos_list, year_list, cat_list, cite_list, person_list, email_list, funding_list = \
        ([] for i in range(7))

    #for index, row in wos_data_df.iterrows():
    for row in wos_data_df.itertuples(index=False):
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


    # MANAGE KEYWORDS
    #################
    ################# 

    st.write(' - - -')
    st.header("MANAGE KEYWORDS")

    # 1 - import and show keywords from uploaded files
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

    for row in extract_kewords_from_df.itertuples(index=False):
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


    # 2 - import and show predefined keywords
    st.subheader('2 - add predefined keywords (optional)')
    st.markdown('Optionally expand the search with a list of __predefined keywords__')

    with st.expander('Upload file and manage keywords', expanded=False):
        uploaded_keywords = st.file_uploader("", accept_multiple_files=False)
        if not uploaded_keywords:
            final_list_of_keywords = []
            st.warning('No  keywords uploaded')
        else:
            predifined_keywords = pd.read_csv(uploaded_keywords).squeeze()
            private_keywords_chosen = st.multiselect(
            'Add or delete keywords from list',
            predifined_keywords,
            predifined_keywords)
        final_list_of_keywords = list(set(imported_keywords_chosen + private_keywords_chosen))
        final_list_of_keywords.sort()

    # 3 - final list of keywords for checking and exporting
    st.subheader('3 - combined list of keywords')
    st.markdown('Check the final list of keywors and export list for later use')

    with st.expander("Show combined list"):
        st.dataframe(final_list_of_keywords)

        keyword_contents = ''
        for keyword_n in final_list_of_keywords:
            keyword_contents += (keyword_n + '\n')

        st.download_button('Download list of keywords for later use', keyword_contents, file_name='Keyword list')



    # PREPARE NOTES FOR OBSIDIAN
    ############################ 3
    ############################ 3

    st.write(' - - -')
    st.header('PREPARE NOTES FOR OBSIDIAN')

    # set to supress warning of chained assigment - not recommended, but looks nicer in output ;)                
    pd.set_option("mode.chained_assignment", None)

    keyword_extracted_wos_data_df = wos_data_df.copy(deep=True)

    zip_name = st.text_input('Name of .zip file:', 'ObsidianNotes') +'.zip'
    st.caption('Name of the .zip archive that will contain your notes. Streamlit will save the file in your current workspace')
    run_to_obsidian =  st.button('Export Obsidian notes to Zip archive file')

    if run_to_obsidian:
        with st.spinner('Processing and extracting keywords ...'):
            clean_list = []
            keyword_extracted_wos_data_df['kw1_clean'] = ''
            keyword_extracted_wos_data_df['kw2_clean'] = ''
            keyword_extracted_wos_data_df['kw_title'] = ''
            keyword_extracted_wos_data_df['kw_abst'] = ''

            for index, row in keyword_extracted_wos_data_df.iterrows():
            #for row in keyword_extracted_wos_data_df.itertuples(index=False):

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


        # OBSIDIAN EXPORT SCRIPT
        my_desktop = get_platform()
        zf = zipfile.ZipFile(my_desktop + zip_name, "a", compression=zipfile.ZIP_DEFLATED)
        keyword_list = final_list_of_keywords

        papers_in = keyword_extracted_wos_data_df[['wosid', 'authors', 'title', 'abstr', 'year', 'journal', 'cites', 'wos_sub_cat1', 'doi', 'usc1', 'usc2', 'kw1_clean', 'kw2_clean', 'kw_title', 'kw_abst']].fillna('')
        papers_in['title'] = papers_in.title.astype(str)
        papers_in['year'] = papers_in.year.astype(int)
        papers_in['usc1'] = papers_in.usc1.astype(int)
        papers_in['usc2'] = papers_in.usc2.astype(int)

        # CREATE OBSIDIAN NOTES FOR KEYWORDS
        for keyword_item in keyword_list:
            str_content_md = '' 
            keyword_item = '% ' + keyword_item.replace("/", "-")
            str_content_md += ('#### ' + keyword_item + empty_line + '- - -' + empty_line)
            str_content_md += ('#keyword' + empty_line + '- - -' + empty_line)       
            str_content_md += ('\n\n' + '##### Notes: ' + empty_line)
            zf.writestr(str(keyword_item + '.md'), str_content_md)
            
        # CREATE OBSIDIAN NOTES FOR PAPERS WITH LINKS TO KEYWORDS AND SUBJECTS
        category_list = []

        for row in papers_in.itertuples(index=False):
            str_content_md = ''

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
            note_title += '.md'
        
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
            all_keywords_in_paper_set = set(list(itertools.chain(author_keywords,plus_keywords,title_keywords, abstract_keywords)))

            all_keywords_in_paper = list(all_keywords_in_paper_set)

            all_keywords_in_paper = list(filter(None, all_keywords_in_paper))
            all_keywords_in_paper = sorted(all_keywords_in_paper, key=str.lower)
        
            str_content_md += ('__' + title_paper + '__' + empty_line)
            str_content_md += ('' + journal +  empty_line)
            str_content_md += ('_' + authors +  '_' + empty_line)
            str_content_md += ('' + 'Data: [year:: ' + year + ']  [cites:: ' + str(cites) + '] [recent:: ' + str(recent) + '] [historic:: ' + str(historic) + ']')
            str_content_md += (empty_line + '- - -' + empty_line)
            str_content_md += ('#paper   Web of Science id: ' + wosid + '     ')
            str_content_md += ('[Google Scholar ](https://scholar.google.dk/scholar?q=' + doi + ')' + empty_line)
            str_content_md += ('- - -' + empty_line)

            str_content_md += '> [!abstract]' + '\n'
            str_content_md += (abstract)
            str_content_md += (empty_line)
            str_content_md += ('- - -' + empty_line)

            str_content_md += ('_[rating:: 0] (scale: 0-10)_' + empty_line)
            str_content_md += ('- [ ] checked' + empty_line + '- - -' + empty_line)
            
            str_content_md += ('*WoS categories:*' + ':' + empty_line)
            paper_categories = wos_categories.split(";")
            for category in paper_categories:
                categoryLstrp = category.lstrip()
                str_content_md += ('[[%% ' + categoryLstrp + ']]' + '\n')
                category_list.append(categoryLstrp)
            
            str_content_md += (empty_line + '*Keyword links:*' + empty_line)
            add_keywords = []
            for kw in all_keywords_in_paper:
                if kw in keyword_list:
                    str_content_md += ('[[% ' + kw + ']]' + '\n') # + empty_line)
                else:
                    add_keywords.append(kw)
            str_content_md += (empty_line +  'Additional keywords: ' + empty_line)
            add_keywords_nicelist = ('\n'.join(add_keywords))
            str_content_md += (str(add_keywords_nicelist))  

            zf.writestr(note_title, str_content_md)      

        # CREATE NOTES FOR SUBJECT CATEGORIES
        category_list = list(set(category_list))
        category_list.sort()

        for category_item in category_list:
            str_content_md = ''
            category_item = '%% ' + str(category_item)

            str_content_md += ('#### ' + category_item + empty_line + '- - -' + empty_line)
            str_content_md += ('#subject' + empty_line + '- - -' + empty_line)       
            str_content_md += ('\n\n' + '##### Characteristic keywords (most frequent first): ' + empty_line)

            zf.writestr(category_item + '.md', str_content_md)


        zf.close()
        print('... obsidian mark-down files created in zip archive')

        st.success('Notes exported succesfully. Check the exported data here:')
        st.dataframe(keyword_extracted_wos_data_df)




# PREPARE LIST OF DOI's from BIOSIS data
############################ 3
############################ 3

st.write(' - - -')
st.header('EXPORT DOIs only')

# set to supress warning of chained assigment - not recommended, but looks nicer in output ;)                
pd.set_option("mode.chained_assignment", None)

doi_list = ''
doi_counter = 0

#papers_in = keyword_extracted_wos_data_df[['wosid', 'authors', 'title', 'abstr', 'year', 'journal', 'cites', 'wos_sub_cat1', 'doi', 'usc1', 'usc2', 'kw1_clean', 'kw2_clean', 'kw_title', 'kw_abst']].fillna('')

for row in raw_bios_df.itertuples(index=False):
    doi = row[0]
    if doi:
        doi_list += ('OR DO = ' + doi  + "\n")
        doi_counter += 1

st.download_button('Download list of dois for later use', doi_list, file_name='doiList')


print('... dois extracted to txt-file')

st.success('DOIS created succesfully. Number of doi-lines: ' + str(doi_counter))

st.write(doi_list)
#st.write(keyword_extracted_wos_data_df)



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
            #     str_content_md += (str(row[1]) + ' - [[' + str(row[0]) + ']] ' + ' \n')

            # md_file.close()