import re
import pandas as pd
from joblib import Parallel, delayed

# 2. Kamus Kata Tidak Baku ke Baku
custom_dictionary = {
    # Cabai
    "cabe": "cabai",
    "cabe ijo": "cabai hijau",
    "cabe merah": "cabai merah",
    "cabe rawit": "cabai rawit",
    "cabe hijau": "cabai hijau",

    # Bawang
    "bwg": "bawang",
    "bamer": "bawang merah",
    "baput": "bawang putih",
    "bawang merah besar": "bawang merah",
    "bawang ijo": "bawang daun",
    "bawang bombai": "bawang bombay",

    # Jeruk
    "jeruk nipis": "jeruk nipis",
    "jeruk limo": "jeruk limau",
    "jeruk purut": "jeruk purut",
    "limao":"limau",

    # Daun dan sayuran
    "kemangi": "kemangi",
    "daun pandan": "pandan",
    "daun salam": "daun salam",
    "daun jeruk": "daun jeruk",
    "daun bawang": "bawang daun",
    "sereh": "serai",
    "kol": "kubis",
    "kobis": "kubis",
    "kubis putih": "kubis",
    "kubis hijau": "kubis",
    "selada": "selada",
    "bayam": "bayam",
    "mentimun": "timun",
    "pete":"petai",

    # Daging dan telur
    "daging ayam": "ayam",
    "ayampotong":"ayam potong",
    "ati": "hati",
    "daging cincang": "daging cincang",
    "ayam kampung": "ayam kampung",
    "hintalu jaruk": "telur asin",
    "telor":"telur",
    "mujaer":"mujair",
    "grameh":"gurami",
    "gurameh":"gurami",
    "gurame":"gurami",
    "gram sapi" : "gram daging sapi",
    "gram kambing" : "gram daging kambing",
    "gram paha kambing" : "gram daging paha kambing",
    "1 ekor kambing" : "daging kambing",
    "kg kambing" : "kilogram daging kambing",
    "bahan campuran Kambing" : "daging kambing",
    "balungan" : "tulang",
    "gram paha sapi" : "gram daging paha sapi",
    "1 ekor sapi" : "daging sapi",
    "kg sapi" : "kilogram daging sapi",
    "smoke beef" : "daging asap",
    "daging Se'i Sapi" : "daging iris sapi",


    # Minyak
    "minyak goreng": "minyak",
    "mentega": "mentega",
    "margarine": "mentega",
    "minyak sayur": "minyak",

    # Bumbu dan rempah
    "kunir": "kunyit",
    "jahe merah": "jahe",
    "laos": "lengkuas",
    "lengkuas": "lengkuas",
    "ketumbar bubuk": "ketumbar",
    "merica bubuk": "merica",
    "marica": "merica",
    "kaldu bubuk": "kaldu",
    "masako": "kaldu ayam",
    "royco": "kaldu ayam",
    "gula pasir": "gula",
    "gula jawa": "gula merah",
    "garem": "garam",
    "penyedap": "penyedap rasa",
    "lada hitam": "lada",
    "miri":"kemiri",
    "kamiri":"kemiri",
    "mrica":"merica",

    # Saus
    "saos": "saus",
    "saos sambal": "saus sambal",
    "saos tomat": "saus tomat",
    "saos tiram": "saus tiram",
    "saos teriyaki": "saus teriyaki",
    "kecap manis": "kecap manis",
    "kecap asin": "kecap asin",
    "saus BBQ": "saus barbekyu",
    "barbecue": "barbekyu",

    # Santan dan susu
    "santan kara": "santan instan",
    "susu cair": "susu cair",
    "susu bubuk": "susu bubuk",
    "krimer kental manis": "susu kental manis",
    "krim kental manis": "susu kental manis",
    "skm":"susu kental manis",

    # Tepung
    "tepung serbaguna": "tepung serbaguna",
    "tepung beras": "tepung beras",
    "tepung tapioka": "tapioka",
    "tepung terigu": "terigu",
    "tepung jagung": "tepung panir",

    # Minuman dan bahan tambahan
    "air kelapa": "air kelapa",
    "kelapa parut": "kelapa parut",

    # Lain-lain
    "gula putih": "gula",
    "madu alami": "madu",
    "keju parut": "keju",
    "tomat": "tomat",
    "susu kental manis": "susu kental manis",
    "krim kental manis": "susu kental manis",
    "kaldu ayam": "kaldu ayam",
    "santan instan": "santan instan",
    "santapan": "makanan",
    "mentega putih": "mentega",
    "margarine cair": "mentega",
    "suir":"suwir",
    "gule":"gulai",
    "tempeh":"tempe",
    "sambel":"sambal",
    "sardines":"sarden",

    # Sayuran lain
    "wortel": "wortel",
    "mentimun": "timun",
    "kacang panjang": "kacang panjang",
    "kacang hijau": "kacang hijau",
    "buncis": "buncis",
    "terong": "terong",
    "lobak": "lobak",
    "kentang": "kentang",
    "jamur": "jamur",
    "tomat ceri": "tomat ceri",

    # Jenis mie
    "mie telur": "mie telur",
    "mie instan": "mie instan",
    "spaghetti": "spaghetti",
    "bihun": "bihun",

    # Jenis nasi
    "nasi putih": "nasi",
    "nasi goreng": "nasi goreng",
    "nasi kunir": "nasi kunyit",

    # Singkatan
    "tbsp": "sendok makan",
    "sdm": "sendok makan",
    "sdt": "sendok teh",
    "tsp": "sendok teh",
    "ml": "mililiter",
    "l": "liter",
    "g": "gram",
    "kg": "kilogram",
    "kilo" : "kilogram",
    "oz": "ons",
    "lb": "pound",
    "cup": "cangkir",
    "can": "kaleng",
    "pkg": "pak",
    "btl": "botol",
    "gls": "gelas",
    "stgh": "setengah",
    "uk": "ukuran",
    "bh": "buah",
    "gr": "gram",
    "ltr": "liter",
    "btr": "butir",
    "btg": "batang",
    "sdm": "sendok makan",
    "sdkt": "sedikit",
    "sckpnya": "secukupnya",
    "dgng" : "daging",
    "tulng" : "tulang",
    "kurleb" : "kurang lebih",
    "bwg" : "bawang",
    "brp" : "berapa"
}

master_ingredients = [
    # Rempah dan bumbu
    "cabai", "cabai hijau", "cabai merah", "cabai rawit", "bawang merah",
    "bawang putih", "bawang daun", "bawang bombay", "kunyit", "jahe",
    "lengkuas", "serai", "daun salam", "daun pandan", "daun jeruk",
    "daun bawang", "daun kemangi", "ketumbar", "lada", "merica", "kemiri",
    "gula", "gula merah", "garam", "kaldu", "kaldu ayam", "penyedap rasa",
    "terasi", "saus sambal", "saus tiram", "cabai gendot", "kecap",

    # Daun dan sayuran
    "bayam", "sawi", "sawi hijau", "sawi putih", "kol", "kubis", "kubis putih",
    "kubis hijau", "brokoli", "kembang kol", "wortel", "kentang", "timun",
    "petai", "buncis", "kacang panjang", "kacang hijau", "terong", "lobak",
    "kangkung", "selada", "tomat", "tomat ceri", "jamur", "labu siam", "jagung",
    "daun singkong", "daun melinjo", "daun katuk", "pare","daun salam", "daun pandan",
    "daun jeruk", "daun bawang", "daun kemangi","kemangin","daun singkong", "daun melinjo",
    "daun katuk", "daun kunyit", "daun pegagan","daun sawi", "daun bayam", "daun pakis",
    "daun seledri", "daun kari", "daun mint", "daun pisang","jeruk nipis","jeruk limau",
    "jeruk purut",

    # Protein
    "ayam", "ayam kampung", "daging ayam", "daging sapi", "daging kambing",
    "ikan", "mujair", "gurami", "lele", "udang", "cumi", "kepiting", "tahu", "tongkol",
    "tempe", "telur", "hati ayam", "ati ampela", "kaldu ayam", "sosis", "bakso", "udang",

    # Bahan cair
    "minyak", "minyak goreng", "santan", "santan instan", "air kelapa", "minyak wijen",
    "susu cair", "susu kental manis", "krimer", "krim kental manis",
    "cuka", "air asam jawa", "kecap asin", "kecap manis", "saus teriyaki",

    # Tepung dan bahan tambahan
    "tepung terigu", "tepung beras", "tepung tapioka", "tepung jagung",
    "tepung panir", "tepung serbaguna", "mentega", "margarine", "keju",
    "cokelat", "mayones", "selai kacang", "coklat bubuk", "vanili", "kornet", "baking powder",

    # Jenis nasi dan mie
    "beras", "nasi", "nasi putih", "nasi goreng", "nasi kunyit", "mie instan",
    "mie telur", "bihun", "kwetiau", "spaghetti",

    # Penyedap rasa dan bumbu instan
    "masako", "royco", "kaldu bubuk", "kaldu jamur", "saori", "sasa",
    "ajinomoto", "maggie", "seledri", "terigu","kulit lumpia", "tepung roti",

     # Ikan
    "bandeng", "lele", "patin", "tongkol",
    "cakalang", "kembung", "tenggiri", "bawal", "nila", "mujair",
    "salmon", "tuna", "dori", "kakap merah", "kakap putih", "teri",
    "makarel", "sarden", "baronang", "belut", "pari",
    "gabus", "betutu", "kerapu", "sidat",
    "kuwae", "tengiri", "tongkol putih", "peda", "layur", "belanak",
    "kakap hitam", "cucut", "barakuda", "selar", "asin", "julung-julung",
    "bandeng presto", "kepe-kepe", "gembung"

    # Lain-lain
    "kelapa parut", "kelapa muda", "air kelapa", "daun pisang", "gula pasir",
    "gula jawa", "madu", "tempe", "kacang tanah", "kacang mede", "ketan",
    "beras ketan", "susu bubuk", "hintalu jaruk", "kerupuk","tepung ayam serbaguna",
    "daun salam", "daun pandan", "daun jeruk", "daun bawang", "daun kemangi","kemangin",
    "daun singkong", "daun melinjo", "daun katuk", "daun kunyit", "daun pegagan",
    "daun sawi", "daun bayam", "daun pakis", "daun seledri", "daun kari",
    "daun mint", "daun pisang", "tulang kambing", "tulang sumsum kambing", "tulang iga kambing",
    "kaki kambing", "otak kambing", "ati kambing", "jeroan kambing", "paha kambing", "perutan kambing",
    "iga kambing", "hati kambing", "daging giling kambing", "tusuk sate kambing", "kepala kambing",
    "tulang sapi", "tulang sumsum sapi", "tulang iga sapi", "kornet sapi", "daging iris sapi",
    "kaki sapi", "otak sapi", "ati sapi", "jeroan sapi", "paha sapi", "perutan sapi",
    "iga sapi", "hati sapi", "daging giling sapi", "tusuk sate sapi", "kepala sapi", "susu kambing",
    "susu sapi", "daging asap", "jantung sapi", "kikil sapi", "kulit sapi", "daging sirloin sapi"
]



def normalize_text(text):
    if not isinstance(text, str):
        return text
    text = text.lower()
    # Ganti kata tidak baku menggunakan kamus
    for non_baku, baku in custom_dictionary.items():
        text = re.sub(rf'\b{non_baku}\b', baku, text)
    return text

# Fungsi Ekstraksi dan Filter Ingredients
def extract_main_ingredients(ingredients):
    # Normalisasi teks terlebih dahulu
    ingredients = normalize_text(ingredients)

    # Pisahkan berdasarkan delimiter '--' atau ','
    parts = re.split(r'--|,', ingredients)

    # List untuk menyimpan bahan yang ditemukan
    filtered_ingredients = []
    for part in parts:
        # Hilangkan angka dan karakter khusus
        part = re.sub(r'\d+', '', part)  # Hapus angka
        part = re.sub(r'[^a-z\s]', '', part).strip()  # Hapus karakter non-alfabet
        part = part.lower()  # Pastikan huruf kecil

        # Cek apakah frasa (bahan) ada di master_ingredients
        for master_item in master_ingredients:
            if master_item in part and master_item not in filtered_ingredients:
                filtered_ingredients.append(master_item)

    # Gabungkan bahan yang sudah difilter dengan koma
    return ', '.join(filtered_ingredients)

# Proses setiap file
input_files = [
    "filter_data_sapi.csv",
    "filter_data_tahu.csv",
    "filter_data_telur.csv",
    "filter_data_tempe.csv",
    "filter_data_udang.csv",
    "filter_data_ayam.csv",
    "filter_data_ikan.csv",
    "filter_data_kambing.csv"
]

for input_file in input_files:
    # Baca file CSV
    data = pd.read_csv(input_file)

    # Drop rows with missing ingredients
    data = data.dropna(subset=['Title', 'Ingredients', 'Steps'])

    # Terapkan normalisasi pada kolom 'Ingredients' dan 'Steps'
    data['Title'] = data['Title'].apply(normalize_text)
    data['Ingredients'] = data['Ingredients'].apply(normalize_text)
    data['Steps'] = data['Steps'].apply(normalize_text)

    # Terapkan fungsi ekstraksi bahan utama
    data['Main Ingredients'] = data['Ingredients'].apply(extract_main_ingredients)

    # Simpan hasil ke file output
    output_file = input_file.replace("filter_data", "filter_data")
    data.to_csv(output_file, index=False)
    print(f"File berhasil disimpan: {output_file}")
