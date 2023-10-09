import streamlit as st
import pandas as pd
import nltk
from nltk import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import swifter
import string
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import re
import requests
from bs4 import BeautifulSoup
import csv
import numpy as np
import string

Home, Learn, Proses, Jml_Term, TermFrequency, Logaritmic, Binary, tfidf, topik = st.tabs(['Home', 'Learn Data', 'Preprocessing','Total Term', 'Term Frequency', 'Logaritmic Frequency','Binary Frequency','TF-IDF','Topik Modelling'])

with Home :
   
    
    st.write("""
        Nama   : Alvina Maharani\n
        NIM    : 200411100029 \n
        Kelas  : PPW 
        """)
   

    
with Learn :

    st.header("Proses Crawling Data")
    st.write("Mengambil data dari website")
    import streamlit as st


    # Fungsi untuk mendapatkan data dari halaman detail
    def get_data(url):
        page = requests.get(url)
        if page.status_code == 200:
            page_html = page.content
            page_soup = BeautifulSoup(page_html, 'html.parser')
            data = []

            container_header = page_soup.find('div', {'style': 'float:left; width:540px;'})
            judul = container_header.find('a', class_='title').text.strip().replace('\r\n', '')
            person = container_header.find_all('span')
            abstrak = page_soup.find('p', {'align': 'justify'}).text.strip().replace('\r\n', '')
            data.append(judul)
            for i in person:
                split_text = i.text.strip().split(':')
                data.append(split_text[1])
            data.append(abstrak)
            return data

    # Fungsi untuk melakukan crawling halaman
    def crawl_pta_trunojoyo(url):
        response = requests.get(url)

        if response.status_code == 200:
            html = response.content
            soup = BeautifulSoup(html, 'html.parser')
            next_button = soup.find_all('a', class_='pag_button')
            for i in next_button:
                if i.text.strip() == '>':
                    next_page_link = i.get('href')
            data = []

            content_page = soup.find('ul', class_='items')
            link_content = content_page.find_all('li')
            for detail in link_content:
                link_view = detail.find('a', class_='gray button').get('href')
                data.append(get_data(link_view))
            return data, next_page_link

    # Fungsi utama Streamlit
    def main():
        st.title="Crawler PTA Trunojoyo"
        url = st.text_input("Masukkan URL PTA Trunojoyo:")
        num_page = st.number_input("Masukkan jumlah halaman yang akan di-crawl:", min_value=1, value=1)

        if st.button("Crawl"):
            next_page_link = url
            final_crawl = []
            for i in range(num_page):
                hasil_crawl, next_page_link = crawl_pta_trunojoyo(next_page_link)
                final_crawl += hasil_crawl

            # Menampilkan hasil dalam bentuk tabel di Streamlit
            st.write(f'Data Berhasil Di Crawl Dengan Jumlah {len(final_crawl)} data')
            st.write(final_crawl)

            # Menyimpan data ke dalam file CSV
            with open('pta_trunojoyo.csv', 'w', newline='', encoding='utf-8') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(['Judul', 'Penulis', 'Dosen Pembimbing I', 'Dosen Pembimbing II', 'Abstrak'])
                csvwriter.writerows(final_crawl)

    if __name__ == "__main__":
        main()
    
    st.write("Tampilan data hasil crawling")
    # URL menuju file CSV di GitHub
    url = 'https://raw.githubusercontent.com/alvina-maharani/ppw/main/pta_trunojoyo.csv'

    # Membaca data dari URL
    df = pd.read_csv(url)
    df
    


with Proses :

    st.header("Proses Prepocessing Data")
    st.write("1. Menghapus tanda baca")
    df['Abstrak'] = df['Abstrak'].replace(np.nan, '')
    import re
    def cleaning(data):
        data = re.sub(r'#[A-Za-z0-9]+',' ',data)
        data = re.sub(r"http\S+",' ',data)
        data = re.sub(r'[0-9]+',' ',data)
        data = re.sub(r"[-()\"#/@;:<>{}'+=~|.!?,_]", " ", data)
        data = data.strip(' ')
        data = data.strip("\n")
        return data

    df["cleaning"]= df["Abstrak"].apply(cleaning)
    df

    st.write("2. Case Folding")
    df['casefolding'] = [entry.lower() for entry in df['cleaning']]
    df

    st.write("3. Tokenisasi")
    def word_tokenize_wrapper(text):
        return word_tokenize(text)

    df['Tokenizing'] = df['casefolding'].str.lower().apply(word_tokenize_wrapper)
    df

    st.write("4. Stopword")
    # Mengambil daftar stopword dari NLTK
    daftar_stopword_nltk = stopwords.words('indonesian')

    # Membaca daftar stopword dari file di GitHub
    import requests

    url = "https://raw.githubusercontent.com/alvina-maharani/ppw/main/daftar_stopword.txt"
    response = requests.get(url)

    if response.status_code == 200:
        daftar_stopword_manual = response.text.split('\n')
    else:
        daftar_stopword_manual = []

    # Menggabungkan kedua daftar stopword
    daftar_stopword = daftar_stopword_manual + daftar_stopword_nltk

    def stopwordText(words):
        return [word for word in words if word not in daftar_stopword]
    
    df['Stopword Removal'] = df['Tokenizing'].apply(stopwordText)
    df

    st.write("4. Stemming")
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    def stemmed_wrapper(term):
        return stemmer.stem(term)

    term_dict = {}

    for document in df['Stopword Removal']:
        for term in document:
            if term not in term_dict:
                term_dict[term] = ' '

    for term in term_dict:
        term_dict[term] = stemmed_wrapper(term)
        print(term,":" ,term_dict[term])
        
    def stemmingText(document):
        return [term_dict[term] for term in document]

    df['Stemming'] = df['Stopword Removal'].swifter.apply(stemmingText)
    df

    st.header("Tampilan data setelah preprocessing")
    # URL menuju file Excel di GitHub
    url = 'https://raw.githubusercontent.com/alvina-maharani/ppw/main/HasilPreposPTA.xlsx'

    # Membaca data dari URL
    corpus = pd.read_excel(url)
    corpus

    def remove_punct(text):
        text = "".join([char for char in text if char not in string.punctuation])
        return text
    corpus["Clean"] = corpus["Stemming"].apply(lambda x: remove_punct(x))
    corpus

with Jml_Term :

    st.header("Jumlah Term")
    data_kata = corpus["Stemming"]
    data = pd.DataFrame(data_kata)
    data["Jumlah Kata"] = data["Stemming"].apply(lambda x: len(x))
    jumlah_kata = data["Jumlah Kata"].sum()
    st.write(f"Jumlah Term: {jumlah_kata}")

with TermFrequency :
    
    st.header("Tampilan Term Frequency")
    
    vectorizer = CountVectorizer()
    count_matrix = vectorizer.fit_transform(corpus['Clean'])
    count_array = count_matrix.toarray()
    corpus_vsm = pd.DataFrame(data=count_array,columns = vectorizer.vocabulary_.keys())
    new_column = corpus['Clean']
    corpus_vsm.insert(0, 'Teks', new_column)
    corpus_vsm

with Logaritmic :

    st.header("Tampilan Logaritmic Frequency")
    ectorizer = CountVectorizer()
    count_matrix = vectorizer.fit_transform(corpus['Clean'])
    log_count_matrix = np.log1p(count_matrix.toarray())
    log_vsm_corpus = pd.DataFrame(data=log_count_matrix, columns=vectorizer.get_feature_names_out())
    new_kolom = corpus['Clean']
    log_vsm_corpus.insert(0, 'Teks', new_kolom)
    log_vsm_corpus

with Binary :

    st.header("Tampilan Binary Frequency")
    vectorizer = CountVectorizer(binary=True)
    X = vectorizer.fit_transform(corpus['Clean'])
    feature_names = vectorizer.get_feature_names_out()
    corpus_vsm_binary = pd.DataFrame(data=X.toarray(), columns=feature_names)
    kolom_baru = corpus['Clean']
    corpus_vsm_binary.insert(0, 'Teks', kolom_baru)
    corpus_vsm_binary

with tfidf :

    st.header("TF-IDF")
    vectorizer = TfidfVectorizer()
    bobot = vectorizer.fit_transform(corpus["Stemming"])
    st.write("Bobot : ")
    bobot

    corpus_bobot = pd.DataFrame(bobot.todense().T,
                        index =vectorizer.get_feature_names_out(),
                        columns=[f'D{i+1}' for i in range(len(corpus["Stemming"]))])
    corpus_bobot


with topik :

    st.header ("Topik Modelling LDA")

    url = 'https://raw.githubusercontent.com/alvina-maharani/ppw/main/TermFrequensi.csv'
    data = pd.read_csv(url)
    data

    X = data.drop('Teks', axis=1)
    X

    from sklearn.decomposition import LatentDirichletAllocation

    st.write("Proporsi Topik dalam Dokumen")
    topik = 3
    alpha = 0.1 #distribusi topik dalam dokumen
    beta = 0.2 #distribusi kata dalam topik

    #membuat model LDA
    lda = LatentDirichletAllocation(n_components=topik, doc_topic_prior=alpha, topic_word_prior=beta)
    proporsi_topik_dokumen = lda.fit_transform(X)

    dokumen = data['Teks']
    output_proporsi_TD = pd.DataFrame(proporsi_topik_dokumen, columns=['Topik 1', 'Topik 2', 'Topik 3'])
    output_proporsi_TD.insert(0,'Teks', dokumen)
    output_proporsi_TD


    st.write("Distribusi Kata pada Topik")
    distribusi_kata_topik = pd.DataFrame(lda.components_)
    distribusi_kata_topik















    
