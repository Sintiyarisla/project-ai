import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
import glob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import streamlit as st


# Daftar rekomendasi bahan yang umum
common_ingredients = [
        # Rempah dan bumbu
        "cabai", "cabai hijau", "cabai merah", "cabai rawit", "bawang merah",
        "bawang putih", "bawang daun", "bawang bombay", "kunyit", "jahe",
        "lengkuas", "serai", "daun salam", "daun pandan", "daun jeruk",
        "daun bawang", "daun kemangi", "ketumbar", "lada", "merica", "kemiri",
        "gula", "gula merah", "garam", "kaldu", "penyedap rasa", "terasi",
        "saus sambal", "saus tiram", "cabai gendot", "kecap",

        # Daun dan sayuran
        "bayam", "sawi", "sawi hijau", "sawi putih", "kol", "kubis", "brokoli",
        "kembang kol", "wortel", "kentang", "timun", "petai", "buncis",
        "kacang panjang", "terong", "lobak", "kangkung", "selada", "tomat",
        "tomat ceri", "jamur", "labu siam", "jagung", "daun singkong",
        "daun melinjo", "daun katuk", "pare", "daun kunyit", "daun pegagan",
        "daun seledri", "daun kari", "daun mint", "daun pisang", "jeruk nipis",
        "jeruk limau", "jeruk purut",

        # Protein
        "ayam", "ayam kampung", "daging ayam", "daging sapi", "daging kambing",
        "ikan", "mujair", "gurami", "lele", "udang", "cumi", "kepiting", "tahu",
        "tongkol", "tempe", "telur", "hati ayam", "ati ampela", "sosis", "bakso",

        # Bahan cair
        "minyak", "minyak goreng", "santan", "santan instan", "air kelapa",
        "minyak wijen", "susu cair", "susu kental manis", "krimer",
        "krim kental manis", "cuka", "air asam jawa", "kecap asin", "kecap manis",
        "saus teriyaki",

        # Tepung dan bahan tambahan
        "tepung terigu", "tepung beras", "tepung tapioka", "tepung jagung",
        "tepung panir", "tepung serbaguna", "mentega", "margarine", "keju",
        "cokelat", "mayones", "selai kacang", "coklat bubuk", "vanili", "kornet",
        "baking powder",

        # Jenis nasi dan mie
        "beras", "nasi", "nasi putih", "nasi goreng", "nasi kunyit", "mie instan",
        "mie telur", "bihun", "kwetiau", "spaghetti",

        # Penyedap rasa dan bumbu instan
        "masako", "royco", "kaldu bubuk", "kaldu jamur", "saori", "sasa",
        "ajinomoto", "maggie", "seledri", "terigu", "kulit lumpia", "tepung roti",

        # Ikan
        "bandeng", "patin", "tongkol", "cakalang", "kembung", "tenggiri",
        "bawal", "nila", "salmon", "tuna", "dori", "kakap merah", "kakap putih",
        "teri", "makarel", "sarden", "baronang", "belut", "pari", "gabus",
        "betutu", "kerapu", "sidat", "peda", "layur", "belanak", "cucut",
        "barakuda", "selar", "julung-julung", "bandeng presto", "kepe-kepe",

        # Lain-lain
        "kelapa parut", "kelapa muda", "air kelapa", "gula pasir", "gula jawa",
        "madu", "tempe", "kacang tanah", "kacang mede", "ketan", "beras ketan",
        "susu bubuk", "kerupuk", "tepung ayam serbaguna", "tulang kambing",
        "tulang sumsum kambing", "tulang iga kambing", "kaki kambing",
        "otak kambing", "jeroan kambing", "paha kambing",
        "perutan kambing", "iga kambing", "hati kambing", "daging giling kambing",
        "tusuk sate kambing", "kepala kambing", "tulang sapi",
        "tulang sumsum sapi", "tulang iga sapi", "kornet sapi", "daging iris sapi",
        "kaki sapi", "otak sapi", "jeroan sapi", "paha sapi",
        "perutan sapi", "iga sapi", "hati sapi", "daging giling sapi",
        "tusuk sate sapi", "kepala sapi", "susu kambing", "susu sapi",
        "daging asap", "jantung sapi", "kikil sapi", "kulit sapi",
        "daging sirloin sapi"
    ]


# Download stopwords (run this only once if not already downloaded)
nltk.download('stopwords')

# 1. Load Multiple Datasets
file_paths = glob.glob('filter_data_*.csv')
data_list = [pd.read_csv(file_path) for file_path in file_paths]
data = pd.concat(data_list, ignore_index=True)

# Filter out rows where 'Main Ingredients' is NaN
data = data.dropna(subset=['Main Ingredients'])

# 2. Preprocessing: Stop Words Removal
stop_words = set(stopwords.words('indonesian'))  # Indonesian stopwords from NLTK

def remove_stopwords(text):
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

def filter_recommendations(input_vector, ingredient_vectors, top_indices):
    matches = []
    for idx in top_indices:
        # Hitung kecocokan berdasarkan bahan yang ada
        common_terms = (input_vector.toarray() * ingredient_vectors[idx].toarray()).sum()
        matches.append((common_terms, idx))
    # Urutkan berdasarkan kecocokan bahan (common_terms)
    matches.sort(reverse=True, key=lambda x: x[0])
    return [idx for _, idx in matches]


#urut data yg paling mirip
def exact_match_bonus(input_ingredients, recommendations):
    input_set = set(input_ingredients.split(', '))
    recommendations['Exact Match'] = recommendations['Main Ingredients'].apply(
        lambda x: len(input_set.intersection(set(x.split(', '))))
    )
    return recommendations.sort_values(by=['Exact Match', 'Title'], ascending=False)

#Jika resep memiliki bahan yang tidak dimiliki user, kurangi skor
def penalize_missing_ingredients(input_ingredients, recommendations):
    input_set = set(input_ingredients.split(', '))
    # Hitung penalti berdasarkan bahan yang tidak ada dalam resep
    recommendations['Penalty'] = recommendations['Main Ingredients'].apply(
        lambda x: len(set(x.split(', ')) - input_set)  # Menghitung bahan yang hilang
    )
    # Urutkan berdasarkan Exact Match terlebih dahulu, lalu penalti (semakin sedikit penalti semakin baik)
    return recommendations.sort_values(by=['Exact Match', 'Penalty', 'Title'], ascending=[False, True, True])



# Apply stopword removal
data['Cleaned Ingredients'] = data['Main Ingredients'].apply(remove_stopwords)

# 3. Vectorize Cleaned Ingredients using TfidfVectorizer from sklearn
tfidf = TfidfVectorizer()
ingredient_vectors = tfidf.fit_transform(data['Cleaned Ingredients'])

# 4. Manually Implement Cosine Similarity
def cosine_similarity(vec1, vec2):
    vec1_dense = vec1.toarray()
    vec2_dense = vec2.toarray()
    
    dot_product = np.dot(vec1_dense, vec2_dense.T)[0][0]
    norm_vec1 = np.linalg.norm(vec1_dense)
    norm_vec2 = np.linalg.norm(vec2_dense)
    
    return dot_product / (norm_vec1 * norm_vec2) if norm_vec1 and norm_vec2 else 0

def combine_similarity_and_match(similarity, exact_match, penalty):
    # Mengubah bobot
    return similarity * 0.5 + exact_match * 0.3 - penalty * 0.2

# 5. Manually Implement KNN
def recommend_recipes(input_ingredients, n_recommendations=100, prioritize_full_match=False):
    # Preprocess input dan transform menggunakan TF-IDF
    cleaned_input = remove_stopwords(input_ingredients)
    input_vector = tfidf.transform([cleaned_input])

    #hitung cosine similarity
    similarities = []
    for idx in range(ingredient_vectors.shape[0]):
        sim = cosine_similarity(input_vector, ingredient_vectors[idx])
        similarities.append((sim, idx))

    # urutkan berdasarkan cosine similarity
    similarities.sort(reverse=True, key=lambda x: x[0])

    # Ambil semua indeks (tidak dibatasi oleh n_recommendations di sini)
    top_indices = [idx for _, idx in similarities]

    # Ambil semua rekomendasi berdasarkan indeks
    recommendations = data.iloc[top_indices]

    # Terapkan urutan berdasarkan Exact Match dan Penalty
    recommendations = exact_match_bonus(input_ingredients, recommendations)
    recommendations = penalize_missing_ingredients(input_ingredients, recommendations)

    # Menambahkan nilai cosine similarity dengan cara yang aman
    def get_cosine_similarity(idx):
        # Cari nilai similarity untuk indeks yang diberikan
        sim_value = next((sim for sim, ix in similarities if ix == idx), None)
        return sim_value if sim_value is not None else 0  # Default ke 0 jika tidak ditemukan

    # Tambahkan kolom Cosine Similarity
    recommendations['Cosine Similarity'] = recommendations.index.map(get_cosine_similarity)

    # Tambahkan skor komposit untuk setiap rekomendasi
    recommendations['Combined Score'] = recommendations.apply(
        lambda row: combine_similarity_and_match(
            similarity=row['Cosine Similarity'],  # Gunakan similarity yang telah disimpan
            exact_match=row['Exact Match'],
            penalty=row['Penalty']
        ),
        axis=1
    )


    recommendations = recommendations.sort_values(
        by=['Exact Match', 'Combined Score'], ascending=[False, False]
    )

    # Hapus duplikasi berdasarkan judul
    recommendations = recommendations.drop_duplicates(subset='Title')

    # Pangkas hingga n_recommendations setelah semua pengurutan
    recommendations = recommendations.head(n_recommendations)

    # Tampilkan hasil debugging
    # st.write("Debugging Output:")
    # st.write(recommendations[['Title', 'Exact Match', 'Penalty', 'Combined Score']])

    return recommendations[['Title', 'Ingredients', 'Steps', 'Main Ingredients']]


# 6. Ground Truth Creation
def generate_ground_truth(input_ingredients, data):
    input_set = set(input_ingredients.split(', '))
    data['Ideal Match'] = data['Main Ingredients'].apply(
        lambda x: len(input_set.intersection(set(x.split(', ')))) > 0
    )
    return data

# 7. Evaluate the Recommendations (Accuracy)
def evaluate_recommendations(input_ingredients, recommendations, data):
    # Generate ground truth based on input_ingredients
    ground_truth = generate_ground_truth(input_ingredients, data)
    
    # Compare ground truth with the recommended recipes
    recommended_titles = recommendations['Title'].tolist()
    ground_truth_values = ground_truth[ground_truth['Title'].isin(recommended_titles)]['Ideal Match'].tolist()
    
    # Calculate accuracy (Proportion of True matches)
    accuracy = accuracy_score([True] * len(ground_truth_values), ground_truth_values)
    return accuracy

#fungsi memberikan koma pada input user
def format_input(ingredients):
    ingredients = ingredients.strip()
    if "," in ingredients:
        ingredients = ", ".join([item.strip() for item in ingredients.split(",")])
    else:
        ingredients = ", ".join(ingredients.split())
    return ingredients


st.set_page_config(
    page_title="Rekomendasi Resep Masakan",
    layout="wide", 
    initial_sidebar_state="expanded"  
)

# Tambah gambar header
st.image(
    "assets/image_head.jpg", 
    caption="Selamat Datang di Aplikasi Rekomendasi Resep Masakan",
    use_container_width=True
)

# Streamlit Setup
st.title("🍳Rekomendasi Resep Masakan")
st.markdown("Masukkan bahan makanan yang kamu miliki. Pilih bahan dari daftar rekomendasi di bawah ini, lalu tambahkan bahan tambahan secara manual jika ada yang belum tercantum. Buat hidangan favoritmu jadi lebih spesial! ✨")
     

# Pilihan bahan tambahan
selected_ingredients = st.multiselect(
    "📝Pilih bahan dari rekomendasi:", 
    options=sorted(common_ingredients),
    default=[],
    help="Klik untuk menambahkan bahan ke daftar."
)

# Input bahan dari pengguna
user_ingredients = st.text_input(
    "➕ Tambahkan bahan yang kurang (pisahkan dengan koma):", 
    placeholder="contoh: gurami, bawang merah, bawang putih, kemiri"
)

# Format input bahan agar konsisten
user_ingredients = format_input(user_ingredients)

# Cek apakah input manual sudah ada koma di akhir
if user_ingredients and not user_ingredients.endswith(','):
    # Jika tidak ada koma di akhir, tambahkan koma
    user_ingredients += ','

# Gabungkan input manual dan pilihan
if selected_ingredients:
    # Tambahkan pilihan ke input jika belum ada
    if user_ingredients:
        # Jika input manual ada, pastikan ada koma sebelum input pilihan
        user_ingredients += ' ' + ', '.join(selected_ingredients)
    else:
        # Jika input manual kosong, langsung tambahkan pilihan
        user_ingredients = ', '.join(selected_ingredients)


# Tampilkan bahan yang akan digunakan
st.markdown("### *Bahan yang digunakan:*")
user_ingredients = user_ingredients.rstrip(', ')
st.info(user_ingredients)

n_recommendations = st.slider(
    "Jumlah rekomendasi yang diinginkan:",
    min_value=1,
    max_value=50,
    value=10
)

if st.button("🔍Dapatkan Rekomendasi"):
    # Generate recommendations
    recommendations = recommend_recipes(user_ingredients, n_recommendations)
    
    accuracy = evaluate_recommendations(user_ingredients, recommendations, data)
    # print(f"Akurasi rekomendasi: {accuracy * 100:.2f}%")  # Tampilkan di terminal
    # st.metric(label="🎯 Akurasi Rekomendasi", value=f"{accuracy * 100:.2f}%")
    # Tampilkan hasil rekomendasi
    st.subheader("Hasil Rekomendasi:")
    for idx, row in recommendations.iterrows():
        title = row['Title']
        if ' ala ' in title.lower():
            title = title.split(' ala ')[0]
        title = title.title()
        with st.expander(f"🍽 {title}"):
            st.markdown(f"<h3 style='font-size: 26px; font-weight: bold; color: #4CAF50; text-align: center;'>{title}</h3>", unsafe_allow_html=True)
                
            # Bahan-Bahan (Ingredients)
            st.write("### Bahan-Bahan:")
            ingredients_list = row['Ingredients'].split('--')  # Pisahkan berdasarkan '--'
            for ingredient in ingredients_list:
                ingredient = ingredient.strip()  # Hapus spasi di awal dan akhir
                if ingredient:  # Hanya tampilkan jika ada isinya
                    st.markdown(f"- {ingredient}")  # Menampilkan dengan tanda minus untuk unordered list

            # Langkah-Langkah (Steps)
            st.write("### Langkah-Langkah:")
            steps_list = row['Steps'].split('--')  # Pisahkan berdasarkan '--'
            for i, step in enumerate(steps_list, 1):
                step = step.strip()  # Hapus spasi di awal dan akhir
                if step:  # Hanya tampilkan jika ada isinya
                    st.markdown(f"{i}. {step}")  # Menampilkan dengan angka untuk ordered list

    # Simpan rekomendasi ke file CSV
    if st.button("💾Simpan Rekomendasi ke CSV"):
        output_file = "resep_rekomendasi.csv"
        recommendations.to_csv(output_file, index=False)
        st.success(f"Rekomendasi disimpan ke {output_file}")
