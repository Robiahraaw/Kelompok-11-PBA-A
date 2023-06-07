import streamlit as st

st.title('Deteksi Emosi Kalimat')

sentence = st.text_input("Masukkan kalimat")
deteksi = st.button ("Deteksi Kalimat")

if deteksi :
    import nltk
    nltk.download('punkt')
    from nltk.tokenize import word_tokenize
    from nltk.stem import PorterStemmer
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

    # Inisialisasi normalizer
    def normalize_word(word):
        # Tambahkan aturan normalisasi sesuai kebutuhan
        # Misalnya menghapus tanda baca dan karakter khusus
        return word.lower().strip()

    # Inisialisasi tokenizer
    def tokenize_sentence(sentence):
        # Gunakan tokenizer sesuai kebutuhan
        return word_tokenize(sentence)

    # Inisialisasi stemmer
    def stem_word(word):
        # Menggunakan Sastrawi Stemmer untuk bahasa Indonesia
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        return stemmer.stem(word)

    # Contoh penggunaan
    normalized_tokens = [normalize_word(token) for token in tokenize_sentence(sentence)]
    stemmed_tokens = [stem_word(token) for token in normalized_tokens]

    from sklearn.feature_extraction.text import TfidfVectorizer

    # Definisikan kamus emosi
    emotion_dict = {
        'Sedih': ['hancur','sepi','iba','larat','pilu','mati','haru','trenyuh','pesimis','sengsara','sedih', 'kesal', 'frustasi', 'kecewa','depresi','duka','masygul','melankolis','merana','muram','murung','susah','sesal','pegal','gelisah','getir','hampa','pahit','resah','patah','putus','lemas','lesu','duka','hina','malang','prihatin','derita','nestapa','nista','capek','lelah','penat','belasungkawa','dukacita','kabung','sungkawa','tragis','bencana','karam','cengeng','cemberut','kecut','masam','nista','hancur','hina','nista','rendah','cela','galau','duka','nestapa','tragedi','celaka','musibah','gagal','lapar','miskin','sakit','ajal','pergi','maut','hampa','kalah','kandas','apes','rugi','sial','melarat','sengsara', 'sukar', 'lesu','muram','sendu','tangis','muram','pahit','sayu','sendu','suram','susah','sengsara','ratap','rintih','sedu','bimbang','pilu','murung','gundah','risau','eluh','rengek','mewek'],
        'bahagia': ['bahagia', 'senang', 'gembira', 'ceria','bangga','cerah','girang','hura-hura','ria','suka','sukacita','aman','enak','lega','lincah','makmur','meriah','nyaman','puas','ria','selesa','semarak','tenang','tenteram'],
        'marah': ['marah', 'geram', 'emosi', 'jengkel','benci','kesal','rewel','dongkol','mangkel','uring-uringan','mangkel','gerundel','gerutu','omel','bengis','gemas','dendam','dengki','benci','geram','murka','amuk','angkara','ganas','ribut','rusuh','maki','rutuk','cemooh','cerca','cibir','damprat','ejek','hardik','umpat','bengis','gertak','bentak','hardik','labrak'],
        'takut': ['takut', 'ketakutan', 'cemas', 'khawatir','bimbang','gamang','gelisah','gentar','waswas','curiga','gemetar','gugup','khawatir','ngeri','panik','resah','risau','seram','keji','seram','segan','momok','debar','teror','gertak'],
        'jijik': ['jijik', 'muak', 'mual', 'bau','cemong','comot','kotor','jorok','kumuh','cemar','jelek','usang','cabul','malu'],
        'kaget': ['kaget', 'kejut', 'heran', 'terpesona','kagum','takjub','pukau','pesona','heran','peranjat']
    }

    # Ekstraksi teks dari kolom yang sesuai
    texts = stemmed_tokens

    # Inisialisasi vektorisasi TF-IDF
    vectorizer = TfidfVectorizer()

    # Melakukan vektorisasi pada teks dokumen
    tfidf_matrix = vectorizer.fit_transform(texts)

    # Mendapatkan daftar fitur (kata-kata) dari vektorisasi
    features = vectorizer.get_feature_names_out()

    # Mendapatkan matriks bobot TF-IDF
    tfidf_weights = tfidf_matrix.toarray()

    # Mendefinisikan fungsi untuk mendapatkan emosi dari kata berdasarkan bobot
    def get_emotion_from_word(word):
        for emotion, words in emotion_dict.items():
            if word in words:
               return emotion
        return None

    # Mendapatkan emosi dari setiap dokumen berdasarkan bobot TF-IDF
    document_emotions = []

    for i, document_weights in enumerate(tfidf_weights):
        document_emotion = None  # Menyimpan emosi dari dokumen
        max_weight = 0  # Menyimpan bobot maksimum

        for j, weight in enumerate(document_weights):
            if weight > max_weight:
                word = features[j]
                emotion = get_emotion_from_word(word)
                if emotion:
                   document_emotion = emotion
                   max_weight = weight

        document_emotions.append(document_emotion)

    from collections import Counter

    # Menghitung kemunculan setiap emosi yang tidak None
    emotion_counts = Counter(emotion for emotion in document_emotions if emotion is not None)

    # Mendapatkan emosi dengan kemunculan terbanyak
    ##most_common_emotion = emotion_counts.most_common(1)[0][0]
    if len(emotion_counts) > 0:
        most_common_emotion = emotion_counts.most_common(1)[0][0]
    else:
        most_common_emotion = "Tidak ada"

    st.success (f'Emosi yang terdeteksi : {most_common_emotion}')