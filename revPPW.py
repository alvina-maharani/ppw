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

 

    # Fungsi untuk mengekstrak data dari halaman web
    def get_data(url):
        page = requests.get(url)
        if page.status_code == 200:
            page_html = page.content
            page_soup = BeautifulSoup(page_html, 'html.parser')
            data = []

            container_header = page_soup.find('div', {'style': 'float:left; width:540px;'})
            
            if container_header:
                judul = container_header.find('a', class_='title')
                if judul:
                    judul = judul.text.strip().replace('\r\n', '')
                    data.append(judul)
                
                person = container_header.find_all('span')
                for i in person:
                    split_text = i.text.strip().split(':')
                    if len(split_text) == 2:
                        data.append(split_text[1])
            
            abstrak = page_soup.find('p', {'align': 'justify'})
            if abstrak:
                abstrak = abstrak.text.strip().replace('\r\n', '')
                data.append(abstrak)
            
            return data

    # Fungsi untuk melakukan crawling halaman
    def crawl_pta_trunojoyo(url):
        response = requests.get(url)
        if response.status_code == 200:
            html = response.content
            soup = BeautifulSoup(html, 'html.parser')
            next_button = soup.find_all('a', class_='pag_button')
            
            next_page_link = None
            for i in next_button:
                if i.text.strip() == '>':
                    next_page_link = i.get('href')
                    break
            
            data = []
            content_page = soup.find('ul', class_='items')
            
            if content_page:
                link_content = content_page.find_all('li')
                for detail in link_content:
                    link_view = detail.find('a', class_='gray button')
                    if link_view:
                        link_view = link_view.get('href')
                        data.append(get_data(link_view))
            
            return data, next_page_link

    def main():
        global df
        st.title("Crawler PTA Trunojoyo")
        
        # Kotak pilihan untuk memilih tipe URL
        url_type = st.selectbox("Pilih jenis URL:", ("URL Teknik Industri", "URL Teknik Informatika", "URL Manajemen Informatika", "URL Teknik Multimedia dan Jaringan", "URL Mekatronika", "URL Teknik Elektro", "URL Sistem Informasi", "URL Teknik Mesin", "URL Teknik Mekatronika", "URL Lainnya"))
        
        # Kode URL default yang sesuai
        default_urls = {
            "URL Teknik Industri": "https://pta.trunojoyo.ac.id/c_search/byprod/9",
            "URL Teknik Informatika": "https://pta.trunojoyo.ac.id/c_search/byprod/10",
            "URL Manajemen Informatika": "https://pta.trunojoyo.ac.id/c_search/byprod/11",
            "URL Teknik Multimedia dan Jaringan": "https://pta.trunojoyo.ac.id/c_search/byprod/19",
            "URL Mekatronika": "https://pta.trunojoyo.ac.id/c_search/byprod/20",
            "URL Teknik Elektro": "https://pta.trunojoyo.ac.id/c_search/byprod/23",
            "URL Sistem Informasi": "https://pta.trunojoyo.ac.id/c_search/byprod/31",
            "URL Teknik Mesin": "https://pta.trunojoyo.ac.id/c_search/byprod/32",
            "URL Teknik Mekatronika": "https://pta.trunojoyo.ac.id/c_search/byprod/33",
            "URL Lainnya": "https://pta.trunojoyo.ac.id/"
        }
        
        default_url = default_urls.get(url_type, "https://pta.trunojoyo.ac.id/")
        
        url = st.text_input("Masukkan URL:", value=default_url)
        num_page = st.number_input("Masukkan jumlah halaman yang akan di-crawl:", min_value=1, value=1)

        if st.button("Crawl"):
            next_page_link = url
            final_crawl = []
            for i in range(num_page):
                hasil_crawl, next_page_link = crawl_pta_trunojoyo(next_page_link)
                final_crawl.extend(hasil_crawl)
        
        

            # Menampilkan hasil dalam bentuk tabel di Streamlit
            st.write(f'Data Berhasil Di Crawl Dengan Jumlah {len(final_crawl)} data')
            df = pd.DataFrame(final_crawl,  columns=['Judul', 'Penulis', 'Dosen Pembimbing I', 'Dosen Pembimbing II', 'Abstrak'])  # Ubah data menjadi DataFrame
            st.dataframe(df)  

            # Menyimpan data ke dalam file CSV
            with open('pta_trunojoyo.csv', 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['Judul', 'Penulis', 'Dosen Pembimbing I', 'Dosen Pembimbing II', 'Abstrak']
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(fieldnames)
                csvwriter.writerows(final_crawl)

    if __name__ == "__main__":
        main()
    df = pd.read_csv('pta_trunojoyo.csv')
      

with Proses :
    import numpy as np
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
    daftar_stopword_nltk = stopwords.words('indonesian')
    import requests

    def stopwordText(words):
        return [word for word in words if word not in daftar_stopword_nltk]
    
    df['Stopword Removal'] = df['Tokenizing'].apply(stopwordText)
    df

    stopword_manual = st.text_area("Masukkan kata pisahkan dengan koma")
    
    if st.button("Stopword") :
        data_stopword1 = re.sub(r",",' ',stopword_manual)
        data_split = data_stopword1.split()
        data_stopword = daftar_stopword_nltk + data_split
        def stopwordbaru(words):
            return [word for word in words if word not in data_stopword]
        df['Stopword Removal'] = df['Stopword Removal'].apply(stopwordbaru)
        df
        
    
   
   
   



with Jml_Term :

    st.header("Jumlah Term")
    data_kata = df["Stopword Removal"]
    data = pd.DataFrame(data_kata)
    data["Jumlah Kata"] = data["Stopword Removal"].apply(lambda x: len(x))
    jumlah_kata = data["Jumlah Kata"].sum()
    st.write(f"Jumlah Term: {jumlah_kata}")


with TermFrequency :
    
    st.header("Tampilan Term Frequency")
    # Gabungkan list teks menjadi satu string
    df['Stopword Removal'] = df['Stopword Removal'].apply(lambda x: ' '.join(x) if isinstance(x, list) else x)
    vectorizer = CountVectorizer()
    count_matrix = vectorizer.fit_transform(df['Stopword Removal'])
    count_array = count_matrix.toarray()
    corpus_vsm = pd.DataFrame(data=count_array,columns = vectorizer.vocabulary_.keys())
    new_column = df['Stopword Removal']
    corpus_vsm.insert(0, 'Teks', new_column)
    corpus_vsm


with Logaritmic :

    st.header("Tampilan Logaritmic Frequency")
    ectorizer = CountVectorizer()
    count_matrix = vectorizer.fit_transform(df['Stopword Removal'])
    log_count_matrix = np.log1p(count_matrix.toarray())
    log_vsm_corpus = pd.DataFrame(data=log_count_matrix, columns=vectorizer.get_feature_names_out())
    new_kolom = df['Stopword Removal']
    log_vsm_corpus.insert(0, 'Teks', new_kolom)
    log_vsm_corpus

with Binary :

    st.header("Tampilan Binary Frequency")
    vectorizer = CountVectorizer(binary=True)
    X = vectorizer.fit_transform(df['Stopword Removal'])
    feature_names = vectorizer.get_feature_names_out()
    corpus_vsm_binary = pd.DataFrame(data=X.toarray(), columns=feature_names)
    kolom_baru = df['Stopword Removal']
    corpus_vsm_binary.insert(0, 'Teks', kolom_baru)
    corpus_vsm_binary

with tfidf :

    st.header("TF-IDF")
    vectorizer = TfidfVectorizer()
    bobot = vectorizer.fit_transform(df["Stopword Removal"])
    st.write("Bobot : ")
   

    corpus_bobot = pd.DataFrame(bobot.todense().T,
                        index =vectorizer.get_feature_names_out(),
                        columns=[f'D{i+1}' for i in range(len(df["Stopword Removal"]))])
    corpus_bobot

    st.write("Clustering : ")
    


with topik :

    st.header ("Topik Modelling LDA")
    data = pd.DataFrame(corpus_vsm)

    X = data.drop('Teks', axis=1)
    

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
