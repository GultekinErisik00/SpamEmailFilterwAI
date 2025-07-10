# Yapay Zeka Destekli Spam E-Posta Sınıflandırma Sistemi

Bu projenin 1. yonteminde, **TF-IDF**, **LSA** ve **Word2Vec** tabanlı **hibrit öznitelik çıkarımı** kullanarak spam e-postaları sınıflandırmak için geliştirilmiştir. Üç farklı makine öğrenmesi modeliyle test edilmiştir:

- **Support Vector Machine (SVM)**
- **Random Forest**
- **MLPClassifier (Yapay Sinir Ağı)**

Ayrıca hiperparametre optimizasyonu için **GridSearchCV** ve **RandomizedSearchCV** kullanılmıştır.

---

Kullanılan Veri Setleri

Projede iki farklı veri seti kullanılmıştır:

1. **spam_modified.csv** (~500KB)  
   → Hiperparametre seçimi için **GridSearchCV** uygulanmıştır.

2. **enron_spam_data_modified.csv** (~50MB)  
   → Veri boyutu büyük olduğu için **RandomizedSearchCV** tercih edilmiştir.

Her veri setinde:

- `text`: E-posta içeriği
- `label`: Spam (`1`) veya ham (`0`) etiketleri yer almaktadır.

---

## Özellik (Feature) Çıkarımı

`extract_features()` fonksiyonu ile aşağıdaki adımlar gerçekleştirilir:

1. **Ön İşleme (Preprocessing)**
   - Küçük harfe çevirme, noktalama işaretlerini temizleme, stop-word filtreleme vb.

2. **TF-IDF + LSA (Latent Semantic Analysis)**
   - TF-IDF ile 5000 kelimeye kadar vektör çıkarılır.
   - LSA ile bu vektörler 100 boyuta indirgenir.

3. **Word2Vec + KMeans**
   - Word2Vec modeli kelimeler arası anlamsal benzerlikleri öğrenir.
   - KMeans ile 50 kümeye ayrılır.
   - Her metin, bu kümelerdeki kelime sıklıklarına göre 50 boyutlu bir vektöre dönüştürülür.

4. **Sonuç**
   - Toplamda her e-posta için 150 öznitelikten oluşan bir vektör elde edilir (100 + 50).

---

## Model Eğitimi ve Değerlendirme

Modeller %80 eğitim / %20 test ayrımı ile değerlendirilir.

### Kullanılan Modeller:
- `SVC(probability=True)`
- `RandomForestClassifier()`
- `MLPClassifier()`

### Değerlendirme Metrikleri:
- Doğruluk (Accuracy)
- F1 Skoru
- ROC AUC
- Classification Report

---

## 🔍 Hiperparametre Optimizasyonu

| Model | Dataset 1 (GridSearch) | Dataset 2 (RandomizedSearch) |
|-------|-------------------------|-------------------------------|
| SVM   | C, kernel, gamma        | Daha az parametmer kullanıldı             |
| RF    | n_estimators, max_depth | Daha az parametmer kullanıldı             |
| MLP   | hidden_layer_sizes, learning_rate_init, max_iter | Aynı |

---

## Proje Nasıl Çalıştırılır?

Dosyalar yüklendikten sonra method1.ipynb dosyası açılarak dosya çalıştırılır ve sonuclar en alttaki kod bloğunda gözlemlenir
