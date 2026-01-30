#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd

#Gerçek haberler #10 Adet
real_news = [
    {"title": "Merkez Bankası Faiz Kararını Açıkladı", 
     "text": "Merkez Bankası, politika faizini 250 baz puan artırarak %47,5 seviyesine çıkardı."},
    {"title": "YÖK’ten Yeni Açıklama: Bazı Bölümler Tamamen Online Olacak", 
     "text": "YÖK, uzaktan eğitime dair yeni düzenlemelerin hayata geçirileceğini duyurdu."},
    {"title": "İstanbul’da Toplu Taşıma Ücretlerine Zam Geldi", 
     "text": "İETT tarafından yapılan açıklamaya göre yeni tarifeler 1 Nisan'dan itibaren geçerli olacak."},
    {"title": "2025 Asgari Ücret Zammı Resmi Gazete’de Yayınlandı",
     "text": "Yeni asgari ücret 22.104 TL olarak belirlendi ve yürürlüğe girdi."},
    {"title": "2025-YKS Başvuruları Başladı", 
     "text": "ÖSYM, YKS başvurularının 1 Nisan’a kadar süreceğini açıkladı."},
    {"title": "Sağlık Bakanı: Yeni Aile Hekimliği Modeline Geçiliyor", 
     "text": "Pilot uygulama 5 ilde başladı, sistemin yıl sonuna kadar yaygınlaşması hedefleniyor."},
    {"title": "TÜİK Verilerine Göre Enflasyon Şubat’ta Yavaşladı", 
     "text": "TÜFE, geçen aya göre %2.3 artış gösterdi."},
    {"title": "Milli Eğitim Bakanlığı'ndan Yeni Ders: Dijital Okuryazarlık",
     "text": "2025-2026 eğitim yılında liselere yeni bir ders daha geliyor."},
    {"title": "Köprü ve Otoyol Geçiş Ücretlerine Zam Geldi", 
     "text": "Yeni tarifelere göre 15 Temmuz Şehitler Köprüsü geçişi 39 TL oldu."},
    {"title": "Cumhurbaşkanı, Yapay Zeka Strateji Belgesi’ni İmzaladı", 
     "text": "Belge ile yapay zeka yatırımları ve eğitimi teşvik edilecek."}
]

#Sahte haberler #10 Adet 
fake_news = [
    {"title": "Medipol Üniversitesi, Yapay Zeka ile Notları Otomatik Verecek", 
     "text": "Rektörlük, öğrenci sayısı arttığı için yapay zeka destekli notlama sistemine geçtiğini duyurdu."},
    {"title": "YÖK, Üniversite Girişini NFT ile Kayıtlı Hale Getirecek", 
     "text": "NFT sahipleri doğrudan kontenjan kazanacak. Pilot uygulama başlıyor."},
    {"title": "Elektrikli Scooter’lara Ruhsat ve Plaka Zorunluluğu Geldi", 
     "text": "Yeni yönetmelikle tüm scooter’lara vergi ve plaka zorunluluğu geliyor."},
    {"title": "Twitter, Türkiye'de XTL Adlı Yeni Kripto Para ile Ödeme Alacak", 
     "text": "Kullanıcılar reklam ve mavi tik ücretlerini XTL ile ödeyebilecek."},
    {"title": "İstanbul’da Yağmur Nedeniyle Evden Çıkmak Yasaklandı", 
     "text": "Valilik, şiddetli yağış nedeniyle vatandaşlara dışarı çıkmama uyarısında bulundu."},
    {"title": "YKS 2025'te Test Yerine Instagram Hikayesiyle Yapılacak",
     "text": "ÖSYM, gençlerin alışkanlıklarına uyum sağlamak için sistemi değiştirdi."},
    {"title": "TOGG, Uçan Araç Modelini Tanıttı: 2026'da Gökyüzünde", 
     "text": "TOGG'un ilk prototipi, 2 saat havada kalabiliyor."},
    {"title": "Whatsapp, Okul Gruplarına Otomatik Sessize Alma Özelliği Getiriyor", 
     "text": "Yeni özellikle sabah 08:00-18:00 arası okul grupları susturulacak."},
    {"title": "Kredi Kartı Limiti Üniversite Notuna Göre Belirlenecek", 
     "text": "Bankalar, kredi skoru yerine transkript isteyecek."},
    {"title": "Yapay Zeka, TBMM'de Milletvekillerine Yerine Oy Kullandı", 
     "text": "Yeni sistem pilot olarak denendi, milletvekilleri duruma itiraz etti."}
]



# In[3]:


# 1-DataFrame'e çevir
real_df = pd.DataFrame(real_news)
fake_df = pd.DataFrame(fake_news)

# 2-Etiketleri ekle
real_df["label"] = 1
fake_df["label"] = 0

# 3-Birleştir ve karıştır
df = pd.concat([real_df, fake_df]).sample(frac=1).reset_index(drop=True)

# 4-Başlık + içerik birleştir
df["text"] = df["title"] + " " + df["text"]



# In[4]:


import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-ZçğıöşüÇĞİÖŞÜ\s]", "", text)  # Türkçe karakterler korunsun
    text = re.sub(r"\s+", " ", text)
    return text

df["text"] = df["text"].apply(clean_text)



# In[5]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Girdi ve etiket ekleme
X = df["text"]
y = df["label"]

# TF-IDF vektörleştirme
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_vect = vectorizer.fit_transform(X)

# Eğitim ve test verisi ayırma
X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size=0.1, random_state=42)

# Model Kısmı
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(X_train, y_train)

# Test Kısmı
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Doğruluk Oranı: %{accuracy * 100:.2f}")


# In[6]:


def predict_news(text):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    result = model.predict(vec)[0]
    return "Gerçek Haber" if result == 1 else "Sahte Haber"


# In[7]:


# Örnek deneme 1
print(predict_news("Türkiye, Avrupa Birliği'ne vizesiz giriş sağlayacak"))


# In[8]:


# Örnek deneme 2
print(predict_news("YÖK, 2025 Bahar Döneminde Hibrit Eğitim Modeline Geçileceğini Açıkladı."))


# In[9]:


# Örnek deneme 3
print(predict_news("Aile ve Sosyal Politikalar Bakanlığı 20 Bin Yeni Atama Yapacak"))


# In[15]:


# Örnek deneme 4
print(predict_news("ulaş oruspuçocuğu mu "))


# In[37]:


import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Sahte", "Gerçek"],
            yticklabels=["Sahte", "Gerçek"])
plt.xlabel("Tahmin")
plt.ylabel("Gerçek")
plt.title("Confusion Matrix")
plt.show()


# In[ ]:




