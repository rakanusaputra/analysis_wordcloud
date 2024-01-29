# Import library Streamlit
import streamlit as st
from wordcloud import WordCloud  # Mengimport WordCloud
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax

# Fungsi untuk melakukan analisis sentimen
def analisis_sentimen(teks, model, tokenizer):
    # Tokenisasi teks
    tokens = tokenizer(teks, return_tensors='pt', truncation=True, padding=True)

    # Prediksi sentimen menggunakan model
    with torch.no_grad():
        hasil = model(**tokens)

    # Hitung probabilitas distribusi softmax
    probabilitas = softmax(hasil.logits, dim=1)

    # Ambil label dengan probabilitas tertinggi
    kelas_terprediksi = torch.argmax(probabilitas, dim=1).item()

    return kelas_terprediksi, probabilitas[0][kelas_terprediksi].item()

def main():
    # Menambahkan judul untuk aplikasi
    st.title("Analisis Sentimen dan WordCloud dengan Streamlit")

    # Menambahkan input untuk mengunggah file teks
    file = st.file_uploader("Unggah file teks", type=["txt"])

    # Memproses file yang diunggah (jika ada)
    if file is not None:
        # Membaca isi file
        teks = file.read().decode("utf-8")
        # Menampilkan teks yang diunggah
        st.header("Teks yang Diunggah:")
        st.write(teks)

        # Load model dan tokenizer
        nama_model = "indobenchmark/indobert-base-p1"
        model = AutoModelForSequenceClassification.from_pretrained(nama_model)
        tokenizer = AutoTokenizer.from_pretrained(nama_model)

        # Analisis sentimen
        kelas_terprediksi, kepercayaan = analisis_sentimen(teks, model, tokenizer)

        # Tampilkan hasil analisis sentimen
        label_sentimen = "Positif" if kelas_terprediksi == 1 else "Negatif"
        st.header("Hasil Analisis Sentimen:")
        st.write(f"Sentimen: {label_sentimen} dengan tingkat kepercayaan: {kepercayaan}")

        # Membuat dan menampilkan WordCloud
        tag_cloud(teks)  # Memanggil fungsi tag_cloud

# Generate a word cloud
def tag_cloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    # Display the generated word cloud using matplotlib
    st.header("WordCloud")
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

if __name__ == "__main__":
    main()
