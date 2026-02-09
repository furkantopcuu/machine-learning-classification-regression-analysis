# machine-learning-classification-regression-analysis
Dry Bean ve Algerian Forest Fires veri setleri üzerinde SVM ve XGBoost algoritmaları ile kapsamlı sınıflandırma ve regresyon analizi.
# Multi-Dataset Machine Learning Analysis: Classification & Regression

Bu proje, yapılandırılmış veri setleri üzerinde modern makine öğrenmesi algoritmalarının performansını ölçmek ve karşılaştırmak amacıyla geliştirilmiştir. Çalışma kapsamında hem **Sınıflandırma (Classification)** hem de **Regresyon (Regression)** disiplinleri, uçtan uca bir veri bilimi hattı (pipeline) ile ele alınmıştır.

## 📌 Proje Özeti

Proje, iki farklı karmaşıklıktaki veri seti üzerinde yürütülen kapsamlı bir analizdir:
1.  **Dry Bean Dataset:** 7 farklı fasulye türünün morfolojik özellikleri üzerinden sınıflandırılması.
2.  **Algerian Forest Fires Dataset:** Meteorolojik veriler kullanılarak Yangın Hava İndeksi (FWI) tahmini.

Analiz sürecinde **Support Vector Machines (SVM)** ve **XGBoost** algoritmaları kullanılmış; model başarısı Çapraz Doğrulama (Cross-Validation) ve çeşitli performans metrikleri ile onaylanmıştır.

## 🛠 Kullanılan Teknolojiler

* **Dil:** Python 3.x
* **Kütüphaneler:** * `Scikit-learn`: SVM modelleri, Ölçeklendirme (StandardScaler) ve Metrikler.
    * `XGBoost`: Gradient Boosting tabanlı yüksek performanslı sınıflandırma ve regresyon.
    * `Pandas` & `NumPy`: Veri manipülasyonu ve matris işlemleri.
    * `Matplotlib` & `Seaborn`: Hata matrisleri ve regresyon grafiklerinin görselleştirilmesi.
    * `Tabulate`: Sonuçların tablo formatında raporlanması.

## 📊 Karşılaştırmalı Model Performansları

| Veri Seti | Model | Ana Metrik | İkincil Metrik |
| :--- | :--- | :--- | :--- |
| **Dry Bean (Sınıflandırma)** | **SVM** | **%92.84 (Accuracy)** | **%92.85 (F1-Score)** |
| Dry Bean (Sınıflandırma) | XGBoost | %92.40 (Accuracy) | %92.40 (F1-Score) |
| **Algerian Forest (Regresyon)** | **XGBoost** | **0.71 (MAE)** | **%23.82 (SMAPE)** |
| Algerian Forest (Regresyon) | SVR | 1.58 (MAE) | %48.31 (SMAPE) |



## 🚀 Öne Çıkan Analiz Adımları

* **Veri Ön İşleme:** Eksik verilerin yönetimi, `LabelEncoding` ile kategorik dönüşüm ve `StandardScaler` ile özellik normalizasyonu.
* **Sınıflandırma Analizi:** Çok sınıflı problemlerde modelin ayırıcılığının **Confusion Matrix** ile görselleştirilmesi.
* **Regresyon Analizi:** Tahmin edilen ve gerçek değerlerin karşılaştırılması, modelin hata payının (MAE) minimize edilmesi.

## 📂 Kurulum ve Çalıştırma

1. Depoyu klonlayın:
   ```bash
   git clone [https://github.com/kullaniciadi/machine-learning-analysis.git](https://github.com/kullaniciadi/machine-learning-analysis.git)
   pip install xgboost scikit-learn pandas matplotlib seaborn tabulate
