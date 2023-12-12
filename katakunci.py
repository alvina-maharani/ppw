import streamlit as st

data, word_degree, word_freq, keyword = st.tabs(['Data','Word Degree', 'Word Frequency', 'Keyword'])

with data:
    import streamlit as st
    import pandas as pd
    from io import BytesIO

    # Input teks dari pengguna
    input_text = st.text_area('Masukkan teks:', '''
    Data 1
    Data 2
    Data 3
    ''')

    # Mengeklik tombol untuk membuat Excel
    if st.button('Simpan Data'):
        # Membuat DataFrame dengan nama kolom 'Konten'
        df = pd.DataFrame({'Konten': input_text.split('\n')})

        
        st.session_state.df = df
        
        # Membuat buffer BytesIO untuk menyimpan file Excel
        excel_buffer = BytesIO()

        # Menulis DataFrame ke file Excel
        df.to_excel(excel_buffer, index=False, sheet_name='Sheet1')

        # Menyimpan file Excel sebagai BytesIO
        st.success('Data berhasil disimpan!!')
    
    import pandas as pd
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer

    nltk.download('punkt')
    nltk.download('stopwords')

    # Fungsi preprocessing
    def is_punct(word):
        return len(word) == 1 and word.isascii() and not word.isalnum()

    def is_numeric(word):
        return word.isnumeric()

    def is_stopword(word):
        return word.lower() in stopwords.words('indonesian')

    def is_valid_candidate(word):
        return not is_punct(word) and not is_numeric(word) and not is_stopword(word)
    

with word_degree:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.stem import PorterStemmer

    # Fungsi ekstraksi kata kunci
    def extract_keywords(text, num_keywords=5):
        sentences = sent_tokenize(text)
        candidate_keywords = []

        for sentence in sentences:
            words = word_tokenize(sentence)
            stemmer = PorterStemmer()

            for word in words:
                if is_valid_candidate(word):
                    candidate_keywords.append(stemmer.stem(word.lower()))

        word_frequency = nltk.FreqDist(candidate_keywords)
        word_degree = {}

        for word in candidate_keywords:
            word_degree[word] = 0

        for sentence in sentences:
            words = word_tokenize(sentence)
            for i in range(len(words) - 1):
                if is_valid_candidate(words[i]) and is_valid_candidate(words[i + 1]):
                    word_degree[stemmer.stem(words[i].lower())] += 1
                    word_degree[stemmer.stem(words[i + 1].lower())] += 1

        word_scores = {}
        for word in candidate_keywords:
            word_scores[word] = {
                'word_degree': word_degree[word],
                'word_frequency': word_frequency[word],
                'total_score': word_degree[word] + word_frequency[word]
            }

        sorted_keywords = sorted(word_scores.items(), key=lambda x: x[1]['total_score'], reverse=True)
        top_keywords = [keyword for keyword, score in sorted_keywords[:num_keywords]]

        return {
            'keywords': top_keywords,
            'word_degree': word_degree,
            'word_frequency': word_frequency
        }
    
    result = df['Konten'].apply(extract_keywords)
    df['Word_Degree'] = result.apply(lambda x: x['word_degree'])
    st.write(df['Word_Degree'])

with word_freq:
    df['Word_Frequency'] = result.apply(lambda x: x['word_frequency'])
    st.write(df['Word_Frequency'])

with keyword:
    df['Keywords'] = result.apply(lambda x: x['keywords'])
    st.write(df['Keywords'])
