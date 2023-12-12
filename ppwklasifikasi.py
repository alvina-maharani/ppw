import streamlit as st
import pandas as pd
import re
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import swifter
import string
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn import svm
import joblib

Klasifikasi, Preprocessing,  Akurasi = st.tabs(["Klasifikasi Berita","Pre-Processing", "Akurasi"])

with Klasifikasi:
    def cleaningulasan(ulasan):
        ulasan = re.sub(r'@[A-Za-a0-9]+',' ',ulasan)
        ulasan = re.sub(r'#[A-Za-z0-9]+',' ',ulasan)
        ulasan = re.sub(r"http\S+",' ',ulasan)
        ulasan = re.sub(r'[0-9]+',' ',ulasan)
        ulasan = re.sub(r"[-()\"#/@;:<>{}'+=~|.!?,_]", " ", ulasan)
        ulasan = ulasan.strip(' ')
        ulasan = ulasan.strip("\n")
        return ulasan

    def casefoldingText(ulasan):
        ulasan = ulasan.lower()
        return ulasan
    
    nltk.download('punkt')

    def word_tokenize_wrapper(text):
        return word_tokenize(text)
    
    nltk.download('stopwords')


    daftar_stopword = stopwords.words('indonesian')
    # Masukan Kata dalam Stopwors Secara Manula
    # Tambahakan Data Stopwords Manual
    # tambahan_stopword = str(input("Masukkan Kata Yang ingin di stopword: "))
    # tambahan_stopword = re.sub(r",", ' ', tambahan_stopword)
    # tambahan_stopword = tambahan_stopword.split()
    # daftar_stopword.extend(tambahan_stopword)
    daftar_stopword = set(daftar_stopword)

    def stopwordText(words):
        return [word for word in words if word not in daftar_stopword]
    

    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    def stemmed_wrapper(term):
        return stemmer.stem(term)
    
    def stemmingText(document):
        return [term_dict[term] for term in document]
    

    df = pd.read_excel("Hasil-Prepos-fix.xlsx")


    def labels(rate):
        if rate == "Makanan":
            return 1
        elif rate == "Kesehatan":
            return 2
        elif rate == "Olahraga":
            return 3
        else:
            return "Non Kategori"

    df["Kategori"] = df["Label"].apply(labels)
    # df.tail(5)

    def remove_punct(text):
        text = "".join([char for char in text if char not in string.punctuation])
        return text
    df["Clean"] = df["Stemming"].apply(lambda x: remove_punct(x))
    del df["Unnamed: 0"]
    # st.write(df.head(5))

    X = df['Clean']
    Y = df['Kategori']

    x_train, x_test, y_train, y_test = model_selection.train_test_split(X.str.lower(), Y, test_size=0.2)

    vectorizer = TfidfVectorizer()
    x_train = vectorizer.fit_transform(x_train)
    x_test = vectorizer.transform(x_test)
    Encoder = LabelEncoder()
    y_train = Encoder.fit_transform(y_train)
    y_test = Encoder.fit_transform(y_test)


    SVM = svm.SVC(kernel='sigmoid', C=1.0, gamma='scale') #Jika dengan Kernel Linear
    SVM = SVM.fit(x_train,y_train)

    # acc_score = cross_val_score(SVM, x_train, y_train, cv=5, scoring='accuracy')
    # pre_score = cross_val_score(SVM, x_train, y_train, cv=5, scoring='precision_macro')
    # rec_score = cross_val_score(SVM, x_train, y_train, cv=5, scoring='recall_macro')
    # f_score = cross_val_score(SVM, x_train, y_train, cv=5, scoring='f1_macro')

    # joblib_models = joblib.load("model_sigmoid.unknown")

    def prediksi(text):
        # vectorizer = TfidfVectorizer()
        tfidf_vektor = vectorizer.transform([text])
        pred = SVM.predict(tfidf_vektor)
        if pred == 0:
            hate = "Makanan"
        elif pred == 1:
            hate = "Kesehatan"
        elif pred == 2:
            hate = "Olahraga"
        else:
            hate = "Error"
        return hate
    

    masukkan_kalimat = st.text_input("Masukkan Data")
    if st.button("Proses"):
        masukkan_kalimat1 = cleaningulasan(masukkan_kalimat)
        masukkan_kalimat1 = casefoldingText(masukkan_kalimat1)
        masukkan_kalimat1 = word_tokenize_wrapper(masukkan_kalimat1)
        masukkan_kalimat1 = stopwordText(masukkan_kalimat1)
        masukkan_kalimat1 = " ".join(masukkan_kalimat1)
        masukkan_kalimat1 = stemmed_wrapper(masukkan_kalimat1)


        st.write(masukkan_kalimat1)
        st.write("Sentimen Review :",prediksi(masukkan_kalimat1))

with Preprocessing:
    st.write(f"Data Inputan: {masukkan_kalimat}")
    masukkan_kalimat = cleaningulasan(masukkan_kalimat)
    st.write(f"Data Cleaning: {masukkan_kalimat}")
    masukkan_kalimat = casefoldingText(masukkan_kalimat)
    st.write(f"Data Case Folding: {masukkan_kalimat}")
    masukkan_kalimat = word_tokenize_wrapper(masukkan_kalimat)
    st.write(f"Data Tokenisasi: {masukkan_kalimat}")
    masukkan_kalimat = stopwordText(masukkan_kalimat)
    st.write(f"Data Stopwords: {masukkan_kalimat}")
    masukkan_kalimat = " ".join(masukkan_kalimat)
    masukkan_kalimat = stemmed_wrapper(masukkan_kalimat)
    st.write(f"Data Stemming: {masukkan_kalimat}")


    st.write(masukkan_kalimat)

with Akurasi:
    data_akurasi = pd.read_excel("Akurasi.xlsx")
    del data_akurasi["Unnamed: 0"]
    data_akurasi