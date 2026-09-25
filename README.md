# AU-AIR Veri Seti ile İHA Görüntülerinde Ağaç Tabanlı Modellerle Nesne Sınıflandırma

Bu proje, AU-AIR veri setindeki insansız hava aracı (İHA/Drone) görüntüleri üzerinde yer alan nesneleri, ağaç tabanlı makine öğrenmesi ve derin öğrenme tekniklerini harmanlayarak sınıflandırmayı amaçlamaktadır. Görüntülerdeki nesneler önceden eğitilmiş derin öğrenme modelleri kullanılarak vektör uzayına dönüştürülmüş ve Karar Ağacı (Decision Tree), Rastgele Orman (Random Forest) ve Gradient Boosting algoritmaları ile sınıflandırılmıştır.

**AU-AIR Veri Seti Görselleri:** https://drive.google.com/open?id=1pJ3xfKtHiTdysX5G3dxqKTdGESOBYCxJ

**AU-AIR Veri Seti Etiketleri:** https://drive.google.com/open?id=1boGF0L6olGe_Nu7rd1R8N7YmQErCb0xA

## 🏗️ Proje Mimarisi ve İş Akışı

Sistem temel olarak dört aşamadan oluşmaktadır:

1. **Veri Ön İşleme ve Optimizasyon:** `annotations.json` dosyasındaki koordinatlar kullanılarak her bir nesne resimden tek tek kırpılmıştır. Öznitelik çıkarma döngüsünü hızlandırmak adına kırpılan resimler ilk çalıştırmada diske `.npy` dosyası olarak kaydedilir ve sonraki çalıştırmalarda doğrudan hafızaya yüklenerek muazzam bir zaman tasarrufu sağlanır.
2. **Öznitelik Çıkarımı (Feature Extraction):** Kırpılan resimler `img2vec_pytorch` kütüphanesi yardımıyla ResNet18 (512 boyut) ve ResNet50 (2048 boyut) modellerine beslenerek matematiksel öznitelik vektörleri elde edilmiştir.
3. **Boyut İndirgeme (PCA):** Yüksek boyutlu vektörler, donanımsal işlem yükünü hafifletmek amacıyla PCA kullanılarak 10, 20 ve 30 boyuta düşürülmüş; ayrıca performans kıyası yapabilmek adına boyut azaltmanın uygulanmadığı (None) ham uzay durumları da teste dahil edilmiştir.
4. **Model Eğitimi ve Sınıflandırma:** Veri setindeki aşırı sınıf dengesizliğini hafifletmek amacıyla Karar Ağacı ve Rastgele Orman algoritmalarında `class_weight='balanced'` parametresi aktif hale getirilmiştir. Projede her model için kapsamlı hiperparametre arama (Grid Search / Fine-tuning) süreçleri yürütülmüştür.

## 📊 Veri Seti Dağılımı

Modelin aynı resimdeki nesneleri hem eğitim hem de test süreçlerinde görerek ezberlemesini (overfitting) önlemek amacıyla resim bazlı bölünme uygulanmıştır:
*   **Eğitim (Train):** 92.144 nesne.
*   **Geçerleme (Validation):** 20.042 nesne.
*   **Test:** 19.652 nesne.

Veri seti toplam 8 sınıftan oluşmaktadır (Human, Car, Truck, Van, Motorbike, Bicycle, Bus, Trailer). Eğitim setinin %77.7'lik çok büyük bir kısmı "Car" (Araba) sınıfından oluştuğu için veri setinde aşırı derecede belirgin bir sınıf dengesizliği (class imbalance) mevcuttur.

## 🏆 Deneysel Sonuçlar ve En İyi Model

Üç algoritmanın tamamı da en yüksek geçerleme (validation) skorunu daha derin bir mimari sunan ResNet50 öznitelikleriyle yakalamıştır. İnce ayar (fine-tuning) süreçleri sonrasında elde edilen nihai test sonuçları şu şekildedir:

*   **Decision Tree:** PCA uygulanmadan (None) eğitilen ve maksimum derinlik (max_depth) sınırı 30'a esnetilen model, test setinde **%76.00** genel doğruluğa (accuracy) ulaşmıştır.
*   **Gradient Boosting:** Donanımı kilitlemesini engellemek için PCA boyutu 30'a indirgenen ve ağaç sayısı 25'te tutulan model, %99'luk sıkıştırma oranına rağmen **%85.00** test doğruluğunu muhafaza etmiştir.
*   **En İyi Model (Random Forest):** PCA uygulanmadan ham uzayda (None) 200 ağaç (n_estimators) ve 25 derinlik (max_depth) ile eğitilen Rastgele Orman modeli, nihai test setinde **%86.00** genel doğruluk oranına ve 0.51 Macro F1-Skoruna ulaşarak projenin en dengeli ve kararlı algoritması olmuştur. Bu model ayrıca `class_weight` desteğiyle Sınıf 4 (Motorbike) gibi azınlık sınıflarda başarılı tahminler yakalamıştır.

## ⚙️ Kurulum ve Çalıştırma

**Gereksinimler:**
Projenin çalışması için `numpy`, `Pillow`, `scikit-learn`, ve `img2vec_pytorch` kütüphanelerinin yüklü olması gerekmektedir.

1. Depoyu bilgisayarınıza klonlayın.
2. Gerekli bağımlılıkları sisteminize kurun.
3. `Assignment2.py` dosyasını çalıştırarak öznitelik çıkarma ve model eğitim sürecini başlatın.

> **⚠️ Önemli Not (Donanım ve Veri Boyutu):**
> Vektör çıkarımı sonucunda oluşan `.npy` uzantılı matris dosyaları GitHub boyut sınırlarını aştığı için bu depoya yüklenmemiştir. Kod ilk çalıştırıldığında bu `.npy` dosyalarını kendi bilgisayarınızda otomatik olarak üretecek ve diskinize kaydedecektir. Ayrıca Gradient Boosting algoritması ham öznitelik uzayında CPU'yu aşırı zorlayabileceği için kod içerisinde PCA kullanımı ile sınırlandırılmıştır.
> İndirilen dosyaların isimleri koddakinden farklı olabilir. O yüzden kodda dosya isimlerini düzeltmeyi unutmayın.
