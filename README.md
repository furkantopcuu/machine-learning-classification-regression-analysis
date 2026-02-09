# Comprehensive Machine Learning Analysis: Classification & Regression

Bu proje; veri madenciliği, istatistiksel analiz ve modern makine öğrenmesi algoritmalarının (SVM & XGBoost) gerçek dünya senaryoları üzerindeki etkinliğini kanıtlamak amacıyla geliştirilmiştir. Proje, yapılandırılmış verilerde sınıflandırma ve regresyon problemlerine uçtan uca bir çözüm sunar.

## 🎯 Neden Bu Projeyi Geliştirdik?
Bu çalışmanın temel amacı, farklı veri dağılımlarına sahip iki problem türünde (Sınıflandırma ve Regresyon), doğrusal olmayan modeller (SVM) ile gradyan artırma tabanlı modellerin (XGBoost) performansını kıyaslamaktır. Bu sayede hangi algoritmanın hangi veri yapısında daha verimli çalıştığı deneysel olarak gözlemlenmiştir.

---

## 🛠 Ne Kullandık ve Neden Kullandık?

### 1. Algoritmalar
* **XGBoost (Extreme Gradient Boosting):** * *Neden:* Hem sınıflandırma hem regresyon görevlerinde hızı ve yüksek tahmin başarısı nedeniyle seçilmiştir. Özellikle karmaşık veri setlerinde aşırı öğrenmeyi (overfitting) engelleyen düzenleme (regularization) parametreleri sunduğu için tercih edilmiştir.
* **SVM (Support Vector Machines):**
    * *Neden:* Yüksek boyutlu verilerde ve sınıfların net ayrılması gereken durumlarda (Dry Bean gibi) etkili olduğu için seçilmiştir. `RBF` çekirdeği sayesinde doğrusal olmayan ilişkileri yakalama gücünden yararlanılmıştır.

### 2. Kütüphaneler ve Araçlar
* **Scikit-Learn:** Veri ölçeklendirme (`StandardScaler`), model değerlendirme metrikleri ve SVM implementasyonu için endüstri standardı olduğu için kullanılmıştır.
* **Pandas & NumPy:** Büyük veri setlerinin (13k+ satır) hızlı manipülasyonu ve matris işlemleri için tercih edilmiştir.
* **Matplotlib & Seaborn:** Veri dağılımlarını ve model başarılarını (Confusion Matrix) görselleştirerek analizi somutlaştırmak için kullanılmıştır.
* **Tabulate:** Analiz sonuçlarını karmaşık loglar yerine, okunabilir ve profesyonel tablolar halinde sunmak için tercih edilmiştir.

---

## 📊 Karşılaştırmalı Analiz Sonuçları

Yapılan testler sonucunda elde edilen metrikler şöyledir:

| Görev Türü | Model | Metrik 1 (Başarı) | Metrik 2 (Hata/Hassasiyet) | Gerekçe |
| :--- | :--- | :--- | :--- | :--- |
| **Sınıflandırma** | **SVM** | **%92.84 (Accuracy)** | **%92.85 (F1-Score)** | Morfolojik özellikler arasındaki marjı en iyi SVM yakaladı. |
| **Sınıflandırma** | XGBoost | %92.40 (Accuracy) | %92.40 (F1-Score) | Yakın performans sergiledi ancak eğitim süresi daha kısaydı. |
| **Regresyon** | **XGBoost** | **0.71 (MAE)** | **%23.82 (SMAPE)** | Karmaşık hava durumu verilerinde SVR'ı ikiye katlayan doğruluk sağladı. |
| **Regresyon** | SVR | 1.58 (MAE) | %48.31 (SMAPE) | Regresyonda gürültülü veriye karşı daha duyarlı kaldı. |



---

## ⚙️ Kurulum ve Çalıştırma
1.Depoyu klonlayın:
```bash
git clone [https://github.com/kullaniciadi/machine-learning-analysis.git](https://github.com/kullaniciadi/machine-learning-analysis.git)

2.Gereksinimlerin Yüklenmesi
```bash
pip install xgboost scikit-learn pandas numpy matplotlib seaborn tabulate openpyxl
