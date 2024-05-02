from flask import Flask, request , render_template

app = Flask(__name__)

@app.route('/')
def func1():
    return render_template('home.html')


@app.route('/result')
def fun2():

    query = request.args.get('string')

    import zipfile
    import io
    import sqlite3
    import pandas as pd
    import numpy as np
    import re
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer


    conn = sqlite3.connect('eng_subtitles_database.db')
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    print(cursor.fetchall())

    df = pd.read_sql_query("""SELECT * FROM zipfiles""", conn)

    df = df[:10]

    def func1(a):
        with io.BytesIO(a) as f:
            with zipfile.ZipFile(f, 'r') as zip_file:
                subtitle_content = zip_file.read(zip_file.namelist()[0])
        srt_content = subtitle_content.decode('latin-1').split('\r\n')
        temp = ''
        for i in srt_content:
            if '-->' not in i and i != '' and not i.isdigit():
                temp+=i
        temp = temp.replace('</i>', '').replace('<i>', '').replace("\\" ,  "")
        return temp

    df.content = df.content.apply(func1)


    def pre_processing(raw_text):
        #removing special characters
        temp = re.sub('[^a-zA-Z]',' ',raw_text)

        temp = temp.lower()
        
        #tokenizing 
        tokens = temp.split()
        
        #removing stop words
        tokens1 = [i for i in tokens if i not in stopwords.words('english')]
        
        #steming 
        stem = PorterStemmer()
        tokens2 = [stem.stem(i) for i in tokens1]
                
        
        t_range = [i for i in range(0,int(len(tokens2)),768) ]

        if len(t_range) == 1:
            return ' '.join(tokens2)
        else :
            ch_array = []

            for i in range(int(len(t_range)-1)):
                ch_array.append(tokens2[t_range[i]:t_range[i+1]])

            ch_array1 = [] 

            for i in ch_array:
                ch_array1.append(' '.join(i))

            return ch_array1
        
    df.content = df.content.apply(pre_processing)

    df = df.explode('content').reset_index(drop = True)

    from sentence_transformers import SentenceTransformer, util

    model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')

    df['embeddings'] = df.content.apply(lambda a : model.encode(a))

    df['c_sim'] = df.embeddings.apply(lambda a : np.array(util.cos_sim(model.encode(pre_processing(query)),a))[0][0])

    movie = ''
    for i in df[df.c_sim == df.c_sim.agg('max')]['name']:
        movie+= i
    movie = movie.replace('.',' ')

    return render_template('result.html', Movie = movie)


if __name__ == '__main__':
    app.run(debug = True)




'''import zipfile
import io
import sqlite3
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


conn = sqlite3.connect('eng_subtitles_database.db')
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
print(cursor.fetchall())

df = pd.read_sql_query("""SELECT * FROM zipfiles""", conn)

df = df[:10]

def func1(a):
    with io.BytesIO(a) as f:
        with zipfile.ZipFile(f, 'r') as zip_file:
            subtitle_content = zip_file.read(zip_file.namelist()[0])
    srt_content = subtitle_content.decode('latin-1').split('\r\n')
    temp = ''
    for i in srt_content:
        if '-->' not in i and i != '' and not i.isdigit():
            temp+=i
    temp = temp.replace('</i>', '').replace('<i>', '').replace("\\" ,  "")
    return temp

df.content = df.content.apply(func1)


def pre_processing(raw_text):
    #removing special characters
    temp = re.sub('[^a-zA-Z]',' ',raw_text)

    temp = temp.lower()
    
    #tokenizing 
    tokens = temp.split()
    
    #removing stop words
    tokens1 = [i for i in tokens if i not in stopwords.words('english')]
    
    #steming 
    stem = PorterStemmer()
    tokens2 = [stem.stem(i) for i in tokens1]
               
    
    t_range = [i for i in range(0,int(len(tokens2)),768) ]

    if len(t_range) == 1:
        return ' '.join(tokens2)
    else :
        ch_array = []

        for i in range(int(len(t_range)-1)):
            ch_array.append(tokens2[t_range[i]:t_range[i+1]])

        ch_array1 = [] 

        for i in ch_array:
            ch_array1.append(' '.join(i))

        return ch_array1
    
df.content = df.content.apply(pre_processing)

df = df.explode('content').reset_index(drop = True)

from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')

df['embeddings'] = df.content.apply(lambda a : model.encode(a))

df['c_sim'] = df.embeddings.apply(lambda a : np.array(util.cos_sim(model.encode(pre_processing(query)),a))[0][0])'''