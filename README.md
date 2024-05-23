# AttackDetection_Quantum_ML

Bu proejede CICIDS-2017 dataseti üzerinde Quantum Support-Vector-Machine Algoritması kullanılarak, saldırı tespiti gerçekleştirilmiştir.

Proje için gerekli kütüphaneler aşağıdaki gibidir:

    from google.colab import drive
    import numpy as np
    import seaborn as sns
    import pandas as pd
    from sklearn.decomposition import PCA
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.metrics import classification_report, confusion_matrix
    import time
    from qiskit import BasicAer
    from qiskit.circuit.library import ZZFeatureMap
    from qiskit.aqua import QuantumInstance, aqua_globals
    from qiskit.aqua.algorithms import QSVM
    from qiskit.aqua.utils import split_dataset_to_data_and_labels, map_label_to_class_name
    from qiskit.aqua.algorithms import SklearnSVM

**1. Veri Ön işleme:**
- Veri, FTP-Patator.csv dosyasından yüklenir ve ön işleme adımları uygulanır. Etiketler "BENIGN" ve "SALDIRI" olarak etiketlenir ve makine öğrenimi algoritması için uygun hale getirilir.
- Veriler, standart ölçeklendirme ve PCA (Principal Component Analysis) kullanılarak ön işlenir. Ardından min-max normalizasyonu yapılır.

**2. QSVM Algoritması ile Eğitim ve Test**
- QSVM algoritması için gerekli olan eğitim ve test verileri hazırlanır ve QSVM modeli eğitilir. Ardından, model test verisi üzerinde değerlendirilir.

**3.SklearnSVM Algoritması ile Eğitim ve Test**
- Aynı veri seti, geleneksel SklearnSVM (Support Vector Machine) algoritması ile de eğitilir ve test edilir.

**4.Performans Değerlendirmesi ve Sonuçlar**
- QSVM ve SklearnSVM algoritmalarının test verileri üzerindeki performansları değerlendirilir ve karşılaştırılır. Her iki algoritmanın doğruluk (accuracy) değerleri hesaplanır ve sonuçlar raporlanır.


### SVM (Support Vector Machine)

#### Mantık:
SVM, bir sınıflandırma ve regresyon modelleme algoritmasıdır. Temel prensibi, veri noktalarını sınıflandırmak için bir karar sınırı oluşturmaktır. Bu sınır, veri noktalarını en iyi şekilde ayıran ve maksimum marjin ile iki sınıf arasındaki mesafeyi en büyük hale getiren bir hiperdüzlemdir.

#### Matematiksel Formül:
![Ekran Resmi 2024-05-23 17 52 32](https://github.com/handecavsi/AttackDetection_Quantum_ML/assets/34586454/c14439e5-71ff-43b7-b079-de1f8e8d7c23)

![SVM](https://github.com/handecavsi/AttackDetection_Quantum_ML/assets/34586454/a6400924-5f93-44f4-bc75-67356054991e)

### QSVM (Quantum Support Vector Machine)

#### Mantık:
QSVM, SVM'nin kuantum versiyonudur. Klasik SVM'in aksine, QSVM kuantum devreleri kullanarak verileri sınıflandırır. Veriler kuantum devrelerine giriş olarak verilir ve kuantum devreleri, verileri temsil eden kuantum durumlarını işler. Sonuç olarak, QSVM klasik SVM'den daha karmaşık ve daha güçlü sınıflandırma yeteneklerine sahip olabilir.

#### Matematiksel Formül:
![Ekran Resmi 2024-05-23 17 52 32](https://github.com/handecavsi/AttackDetection_Quantum_ML/assets/34586454/a3bdc4d6-6584-4068-aa96-2966155e3b94)

![QSVM](https://github.com/handecavsi/AttackDetection_Quantum_ML/assets/34586454/665eebee-0f5c-42f1-90d3-f4539e517628)


# Kuantum Devreleri:
Kuantum devreleri, kuantum bilgisayarlar veya kuantum hesaplama için temel yapı taşlarıdır. Bu devreler, kuantum kapıları ve kubitler gibi kuantum bileşenlerinden oluşur. Kuantum devreleri, klasik bilgisayarların işlemesiyle benzer matematiksel ve mantıksal işlemleri gerçekleştirebilir, ancak kuantum mekaniği prensiplerine dayanır. Kuantum devreleri, belirli kuantum algoritmalarını ve problemlerin çözümünü hedefleyen kuantum hesaplama alanında kullanılır.

**QSVM'nin Avantajları:**
1. Paralel İşleme Yeteneği: QSVM, kuantum devrelerinin paralel işleme yeteneklerinden faydalanır. Kuantum paralelizmi, QSVM'nin aynı anda çok sayıda olası çözümü değerlendirerek sınıflandırma yapmasına izin verir.

2. Hafıza ve Boyut İyileştirmesi: QSVM, kuantum durumları üzerinde çalışır ve bu durumlar çok daha karmaşık bilgiyi kodlayabilir. Bu, büyük veri setlerini işlemek ve daha karmaşık sınıflandırma problemlerini çözmek için potansiyel bir avantaj sağlar.

3. Doğrusal Olmayan Karar Sınırları: Klasik SVM, doğrusal karar sınırları oluştururken QSVM, doğrusal olmayan karar sınırları oluşturabilir. Bu, QSVM'nin daha karmaşık veri yapılarını daha iyi modelleyebileceği anlamına gelir.

4. Ölçeklenebilirlik Potansiyeli: QSVM, kuantum paralelizmi ve kuantum süperpozisyon özellikleri sayesinde bazı problemlerde klasik algoritmalardan daha iyi ölçeklenebilirlik sağlayabilir.

5. Veri Hedefleri İçin Optimal Yaklaşım: QSVM, belirli veri kümesi yapıları için daha uygun bir sınıflandırma yaklaşımı sunabilir. Özellikle, kuantum devrelerinin özelliğinden faydalanarak bazı veri yapıları için daha etkili sınıflandırma sağlayabilir.

Bu avantajlar, QSVM'nin bazı sınıflandırma problemlerinde standart SVM'den daha etkili olmasını sağlar. Ancak, uygulamanın belirli gereksinimlerine ve veri yapısına bağlı olarak, QSVM'nin avantajları ve dezavantajları dikkate alınmalıdır.

Quantum SVM hakkında daha çok bilgi edinmek isterseniz yayınımı inceleyebilirsiniz: 

- *Quantum Deep Learning* [link](https://drive.google.com/file/d/1M9Me9yu4bleYUVq0hr1iK7tU2Ghsd69O/view)
- _Cite this: Çavşi Zaim H., Yilmaz M., Yolaçan E.N., “Quantum Deep Learning”, Yorumlanabilir ve Açıklanabilir Yapay Zeka ve Güncel Konular, Yapay Zeka ve Büyük Veri Kitap Serisi 4, Nobel Akademik Yayıncılık Eğitim Danışmanlık, 2022._







