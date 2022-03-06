# FUNCTION LIBRARY
# This file contains a few functions that are used from individual scripts in the tool

def get_platform():
    if (platform.system() == 'Darwin'):
#        logging.info(' running Mac OS ...')
        mainwork = '/Users/flemmingskov/Desktop/'
    else:
 #       logging.info(' running Windows 10 ...')
        mainwork = 'C:/Users/au3406/iCloudDrive/Desktop/'
    return mainwork

def map_bearing (x, y, center_x, center_y):
    angle = math.degrees(math.atan2(y - center_y, x - center_x))
    bearing = (90 - angle) % 360
    return bearing

def import_synonyms():
    xlDic = pd.ExcelFile(get_platform()+'/git/scimap/aliasDic.xlsx')
    dict_df = xlDic.parse('keys')
    synonyms = {}
    for index, row in dict_df.iterrows():
        keyw = str(row[0])
        keywa = str(row[1])
        synonyms[keyw] = keywa
    return synonyms
synonyms = import_synonyms()    

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

def clean_keyword (old_keyword):
    composite_keyword = ''
    keyword = old_keyword.replace("-", " ")     # separating keywords joined by '-' in order to lemmatize and rejoin
    for keyword_part in keyword.split(" "):
        if keyword_part != "":
            keyword_part = wnl.lemmatize(keyword_part.lower())
            if (synonyms.get(keyword_part, "none")) != 'none':   # check if part of keyword is a synonym
                keyword_part = synonyms.get(keyword_part)              
            if composite_keyword != "":
                composite_keyword = composite_keyword + "-" + keyword_part
            if composite_keyword == "":
                composite_keyword = composite_keyword + keyword_part   
            if (synonyms.get(composite_keyword, "none")) != 'none':   # check if the new composite keyword is a synonym
                composite_keyword = synonyms.get(composite_keyword)
    if composite_keyword != '':
        return composite_keyword    

def convert_str_to_list (string): 
    li = list(string.split(";"))
    return li

def clean_str (string):
    result = str(string).replace("'", '')
    return result