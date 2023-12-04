import streamlit as st
import pandas as pd
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize


Home, Learn, Tokenisasi, Kalimat, Jml_Kata, tfidf, Kosinus, Grafik, Graph, Closenes, Page_Rank, Eigenvector, Eigenvalue = st.tabs(['Home','Learn Data', 'Tokenisasi', 'Total Kalimat', 'Total kata', 'TF-IDF', 'Cosine Similarity', 'Graph Kosinus(Bulatan 2 desimal)', 'Graph Kosinus(Threshold:0.6)', 'Closeness Centrality','Page Rank','Eigenvector Centrality','Eigenvalue Centrality'])

with Home :
   
 
    st.write("""
        Nama   : Alvina Maharani\n
        NIM    : 200411100029 \n
        Kelas  : PPW \n
        Topik  : Text Summarization
        """)
    
with Learn :
    st.header("Tampilan Data")
    url = 'https://raw.githubusercontent.com/alvina-maharani/ppw/main/cobappwbuaru.xlsx'
    df = pd.read_excel(url)
    df

with Tokenisasi :
    st.header("Data Di Tokenisasi Menjadi Per Kalimat")
    import nltk
    nltk.download('punkt')
    from nltk.tokenize import sent_tokenize

    df['Konten'] = df['Konten'].apply(lambda x: sent_tokenize(str(x)))
    df

with Kalimat :
    st.header("Mengekstrak kalimat-kalimat dari DataFrame")
    kalimat = df['Konten'].explode().reset_index(drop=True)
    df_tampilan = pd.DataFrame({ 'kalimat': kalimat})
    df_tampilan

with Jml_Kata:
    st.header("Melihat jumlah kata per kalimat")
    df_tampilan['jumlah_kata'] = df_tampilan['kalimat'].apply(lambda x: len(x.split()))
    df_tampilan

with tfidf:
    import math
    from sklearn.feature_extraction.text import TfidfVectorizer

    df_tampilan = pd.DataFrame(df_tampilan)
    # Hitung TF untuk setiap kata dalam setiap kalimat
    tf_values = []
    for index, row in df_tampilan.iterrows():
        tf_dict = {}
        for word in row['kalimat'].split():
            tf_dict[word] = row['kalimat'].split().count(word) / row['jumlah_kata']
        tf_values.append(tf_dict)

    # Menghitung IDF untuk setiap kata
    idf_values = {}
    total_documents = len(df_tampilan)
    for tf in tf_values:
        for word in tf:
            if word in idf_values:
                idf_values[word] += 1
            else:
                idf_values[word] = 1

    # Menghitung TF-IDF
    tfidf_values = []
    for tf in tf_values:
        tfidf_dict = {}
        for word, tf_value in tf.items():
            tfidf_dict[word] = tf_value * math.log(total_documents / (1 + idf_values[word]))
        tfidf_values.append(tfidf_dict)

    # Konversi list of dicts ke dalam DataFrame
    df_tfidf = pd.DataFrame(tfidf_values)
    df_tfidf.fillna(0, inplace=True)

    # Tampilkan hasil TF-IDF DataFrame
    df_tfidf

with Kosinus :
    st.header('Cosine Similarity')
    from collections import Counter
    from sklearn.metrics.pairwise import cosine_similarity

    df_tf_idf = pd.DataFrame(df_tfidf)
    df_tf_idf = df_tf_idf.fillna(0)

    tfidf_matrix = df_tf_idf.to_numpy()

    # Menghitung kesamaan kosinus
    similarity_matrix = cosine_similarity(tfidf_matrix)
    df_tf_idf = pd.DataFrame(similarity_matrix)

    kalimat = ["Kalimat " + str(i) for i in range(1, len(similarity_matrix) + 1)]
    df_tf_idf = df_tf_idf.set_axis(kalimat, axis=0)
    df_tf_idf = df_tf_idf.set_axis(kalimat, axis=1)

    df_tf_idf

with Grafik:
    import streamlit as st
    import networkx as nx
    import pandas as pd
    import matplotlib.pyplot as plt


# Create a graph
    G = nx.Graph()
    for i in range(len(similarity_matrix)):
        for j in range(len(similarity_matrix[0])):
            if i != j:
                similarity = round(similarity_matrix[i][j], 2)
                G.add_edge(i, j, weight=similarity)

    # Streamlit app
    st.title("Graph Visualization with Cosine Similarity")

    # Draw the graph
    fig, ax = plt.subplots(figsize=(8, 6))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, font_weight='bold')
    edge_labels = nx.get_edge_attributes(G, 'weight')
    edge_labels = {(i, j): f"{weight:.2f}" for (i, j), weight in edge_labels.items()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.title("Visualisasi Graph dengan Kesamaan Kosinus (Bulatan 2 Desimal)")
    st.pyplot(fig)

with Graph:
    import streamlit as st
    import networkx as nx
    import pandas as pd
    import matplotlib.pyplot as plt

    G = nx.Graph()
    threshold = 0.06  # Threshold untuk menyambungkan node

    # Tambahkan semua node ke grafik
    G.add_nodes_from(range(len(similarity_matrix)))

    # Tambahkan edge antara node yang nilainya melebihi threshold
    for i in range(len(similarity_matrix)):
        for j in range(len(similarity_matrix[0])):
            if i != j and similarity_matrix[i][j] > threshold:
                similarity = round(similarity_matrix[i][j], 2)  # Bulatkan nilai ke 2 angka dibelakang koma
                G.add_edge(i, j, weight=similarity)

    # Streamlit app
    st.title(f"Graph Visualization with Cosine Similarity (Threshold: {threshold})")

    # Draw the graph
    fig, ax = plt.subplots(figsize=(8, 6))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, font_weight='bold')
    edge_labels = nx.get_edge_attributes(G, 'weight')
    edge_labels = {(i, j): f"{weight:.2f}" for (i, j), weight in edge_labels.items()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    # Tambahkan label pada node yang tidak terhubung
    isolated_nodes = list(nx.isolates(G))
    if isolated_nodes:
        pos_extra = {node: (pos[node][0], pos[node][1] + 0.1) for node in isolated_nodes}

    plt.title(f"Visualisasi Graph dengan Kesamaan Kosinus (Threshold: {threshold})")
    st.pyplot(fig)

with Closenes:
    # Menghitung closeness centrality dari graph
    closeness = nx.closeness_centrality(G)

    # Menampilkan closeness centrality
    st.write("Closeness Centrality:")
    for node, closeness_value in closeness.items():
        st.write(f"Node {node}: {closeness_value}")
    
    sorted_pagerank = sorted(closeness.items(), key=lambda x: x[1], reverse=True)

    st.write("Top 3 sentences based on closeness centrality:")
    for node, rank in sorted_pagerank[:3]:
        st.write(df_tampilan['kalimat'].iloc[node])

    st.write("=============================")
    st.write("Top 3 node based on closeness centrality:")
    for node, rank in sorted_pagerank[:3]:
        st.write(f"Node {node} dengan PageRank {rank:.4f}")


with Page_Rank:
    # Hitung PageRank
    pagerank = nx.pagerank(G)

    # Menampilkan Closeness Centrality
    st.write("PageRank:")
    for node, rank in pagerank.items():
        st.write(f"Node {node}: {rank}")

    # Menampilkan 3 kalimat dengan PageRank tertinggi
    sorted_pagerank = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)

    st.write("=============================")
    st.write("Top 3 sentences based on PageRank:")
    for node, rank in sorted_pagerank[:3]:
        st.write(df_tampilan['kalimat'].iloc[node])

    st.write("=============================")
    st.write("Top 3 node based on PageRank:")
    for node, rank in sorted_pagerank[:3]:
        st.write(f"Node {node} dengan PageRank {rank:.4f}")

with Eigenvector:
    # Hitung Eigenvector Centrality
    eigenvector = nx.eigenvector_centrality(G)

    # Hitung Eigenvalue Centrality
    eigenvalue = nx.eigenvector_centrality_numpy(G)

    # Menampilkan Eigenvector Centrality
    st.write("Eigenvector Centrality:")
    for node, eigenvector_value in eigenvector.items():
        st.write(f"Node {node}: {eigenvector_value:.4f}")

    # Menampilkan 3 kalimat dengan PageRank tertinggi
    sorted_pagerank = sorted(eigenvector.items(), key=lambda x: x[1], reverse=True)

    st.write("=============================")
    st.write("Top 3 sentences based on Eigenvector Centrality:")
    for node, rank in sorted_pagerank[:3]:
        st.write(df_tampilan['kalimat'].iloc[node])

    st.write("=============================")
    st.write("Top 3 node based on Eigenvector Centrality:")
    for node, rank in sorted_pagerank[:3]:
        st.write(f"Node {node} dengan PageRank {rank:.4f}")

with Eigenvalue:
    # Menampilkan Eigenvalue Centrality
    st.write("Eigenvalue Centrality:")
    for node, eigenvalue_value in eigenvalue.items():
        st.write(f"Node {node}: {eigenvalue_value:.4f}")

    # Menampilkan 3 kalimat dengan PageRank tertinggi
    sorted_pagerank = sorted(eigenvalue.items(), key=lambda x: x[1], reverse=True)

    st.write("=============================")
    st.write("Top 3 sentences based on Eigenvalue Centrality:")
    for node, rank in sorted_pagerank[:3]:
        st.write(df_tampilan['kalimat'].iloc[node])

    st.write("=============================")
    st.write("Top 3 node based on Eigenvalue Centrality:")
    for node, rank in sorted_pagerank[:3]:
        st.write(f"Node {node} dengan PageRank {rank:.4f}")


    



    






