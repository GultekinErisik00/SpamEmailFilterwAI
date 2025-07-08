import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Gerekli nltk dosyalarını indir (ilk çalıştırmada bir kez yeterli)
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Hazırlık
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# 🔧 Temizleme fonksiyonu
def clean_email(text):
    text = text.lower()  # 1. Küçük harfe dönüştürme
    text = re.sub(r'[^a-z\s]', '', text)  # 2. Noktalama işaretlerini kaldır
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]  # 3+4. Stop-word çıkar + lemmatization
    return " ".join(tokens)
