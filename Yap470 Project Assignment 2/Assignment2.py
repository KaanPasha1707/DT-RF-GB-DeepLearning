import os, numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from img2vec_pytorch import Img2Vec
import json, random
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier 
from collections import Counter

with open("C:/Users/USER/OneDrive/Belgeler/auair2019annotations/annotations.json", "r") as file:
    tümEtiketler = json.load(file)

etiketHaritası = {}
for e in tümEtiketler['annotations']:
    resimAd = e['image_name'] 
    bbox = e.get('bbox',[])   

    if len(bbox) > 0:
        etiketHaritası[resimAd] = bbox

imageKlasör = "C:/Users/USER/OneDrive/Belgeler/auair2019data (1)/images"     
tümResimler = os.listdir(imageKlasör)

nesneliResimler = []
for n in tümResimler:
    if n in etiketHaritası:
        nesneliResimler.append(n)

print(f"toplam {len(tümResimler)} resimden {len(nesneliResimler)} tanesinde nesne bulundu")   

random.seed(3)
random.shuffle(nesneliResimler)

toplamResimSayısı = len(nesneliResimler)
trainDataSınır = int(toplamResimSayısı * 0.7)
validationDataSınır = int(toplamResimSayısı * 0.85)

trainResimleri = nesneliResimler[:trainDataSınır]
validationResimleri = nesneliResimler[trainDataSınır:validationDataSınır]
testResimleri = nesneliResimler[validationDataSınır:]

print(f"Resim dağılımı -> Train: {len(trainResimleri)}, Validation: {len(validationResimleri)}, Test: {len(testResimleri)}")

def vektörDöndürme(resimListesi, img2vecModel):
    vektörler = []
    etiketler = []
    count = 0

    for isim in resimListesi:
        yol = os.path.join(imageKlasör, isim)
        mevcutBboxlar = etiketHaritası.get(isim, [])

        try:
            tamResim = Image.open(yol).convert('RGB')
            for box in mevcutBboxlar:
                sınıf = box['class']
                top = int(box['top'])
                left =  int(box['left'])
                width = int(box['width'])
                height = int(box['height'])

                kırpılmışResim = tamResim.crop((left, top, left + width, top + height))
                vektör = img2vecModel.get_vec(kırpılmışResim)

                vektörler.append(vektör.flatten())
                etiketler.append(sınıf)
            count += 1
            if count % 500 == 0:
               print(f"ilerleme: {count}")
        except Exception as e:
            continue

    return np.array(vektörler), np.array(etiketler)

modeller = ['resnet18','resnet50']
for m in modeller:
    if os.path.exists(f"{m}TrainResim.npy"):
       print(f"{m} modelin önceden kaydedilmiş nesne vektörleri bulunuyor.")
       pass
    else:
        print(f"\n{m} modeli kullanılarak nesne tabanlı öznitelik vektörleri çıkarılıyor.")
        img2vec = Img2Vec(model=m, cuda=False)

        TrainResim, TrainEtiket = vektörDöndürme(trainResimleri,img2vec)
        ValidationResim, ValidationEtiket = vektörDöndürme(validationResimleri,img2vec)
        TestResim, TestEtiket = vektörDöndürme(testResimleri,img2vec)
        
        np.save(f"{m}TrainResim.npy",TrainResim)
        np.save(f"{m}TrainEtiket.npy",TrainEtiket)
        np.save(f"{m}ValidationResim.npy",ValidationResim)
        np.save(f"{m}ValidationEtiket.npy",ValidationEtiket)
        np.save(f"{m}TestResim.npy",TestResim)
        np.save(f"{m}TestEtiket.npy",TestEtiket)
        print(f"\n{m} modelin tüm alt veri setleri başarıyla '.npy' olarak diske kaydedildi.")

pcaBoyutları = [None,10,20,30]

enİyiDT = {'skor': -1, 'konfigürasyon': {}}
enİyiRF = {'skor': -1, 'konfigürasyon': {}}
enİyiGB = {'skor': -1, 'konfigürasyon': {}}

decisionTreeÖrnekleri = [
    {'criterion': 'entropy', 'max_depth': 15, 'min_samples_split': 8},
    {'criterion': 'entropy', 'max_depth': 25, 'min_samples_split': 12},
    {'criterion': 'entropy', 'max_depth': 30, 'min_samples_split': 10},
    {'criterion': 'entropy', 'max_depth': 20, 'min_samples_split': 10}
]

randomForestÖrnekleri = [
    {'n_estimators': 150, 'max_depth': 20, 'max_features': 'sqrt'},
    {'n_estimators': 200, 'max_depth': 20, 'max_features': 'sqrt'},
    {'n_estimators': 200, 'max_depth': 25, 'max_features': 'sqrt'},
    {'n_estimators': 250, 'max_depth': 20, 'max_features': 'sqrt'}
]

gradientBoostingÖrnekleri = [
    {'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 20},
    {'learning_rate': 0.12, 'max_depth': 5, 'n_estimators': 25},
    {'learning_rate': 0.08, 'max_depth': 6, 'n_estimators': 20},
    {'learning_rate': 0.01, 'max_depth': 4, 'n_estimators': 30}
]

for m in modeller:
    TrainResim = np.load(f"{m}TrainResim.npy", mmap_mode='r')
    TrainEtiket = np.load(f"{m}TrainEtiket.npy", mmap_mode='r')
    ValidationResim = np.load(f"{m}ValidationResim.npy", mmap_mode='r')
    ValidationEtiket = np.load(f"{m}ValidationEtiket.npy", mmap_mode='r')
       
    print(f"\n>>> {m.upper()} Eğitim veri setindeki nesne sayısı: {len(TrainResim)}, Dağılım: {Counter(TrainEtiket)}")
    print(f">>> {m.upper()} Validation veri setindeki nesne sayısı: {len(ValidationResim)}, Dağılım: {Counter(ValidationEtiket)}")

    for pca in pcaBoyutları:
          
        if pca is not None:
            p = PCA(n_components=pca, random_state=3)
            train = p.fit_transform(TrainResim)
            val = p.transform(ValidationResim)
        else:
            p = None
            train = TrainResim
            val = ValidationResim  
  
        for dt in decisionTreeÖrnekleri:
            print(f"--- {m} modeli için {pca} PCA değerinde Decision Tree Eğitiliyor ---")
            model = DecisionTreeClassifier(
                criterion = dt['criterion'],
                max_depth = dt['max_depth'],
                min_samples_split = dt['min_samples_split'],
                class_weight='balanced',
                random_state=3 
            )
            model.fit(train,TrainEtiket)
            dtValidation = model.predict(val)
            skor = accuracy_score(ValidationEtiket, dtValidation)

            if skor > enİyiDT['skor']:
                enİyiDT['skor'] = skor
                enİyiDT['konfigürasyon'] = {
                    'model': 'Decision Tree', 'öznitelik': m, 'pca boyutu': pca,
                    'parametreler': dt, 'model nesnesi': model, 'pca nesnesi': p
                }

        for rf in randomForestÖrnekleri:
            print(f"--- {m} modeli için {pca} PCA değerinde Random Forest Eğitiliyor ---")
            model = RandomForestClassifier(
                n_estimators=rf['n_estimators'],
                max_depth=rf['max_depth'],
                max_features=rf['max_features'],
                class_weight='balanced',
                random_state=3,
                n_jobs=-1
            )
            model.fit(train,TrainEtiket)
            rfValidation = model.predict(val)
            skor = accuracy_score(ValidationEtiket, rfValidation)

            if skor > enİyiRF['skor']:
                enİyiRF['skor'] = skor
                enİyiRF['konfigürasyon'] = {
                    'model': 'Random Forest', 'öznitelik': m, 'pca boyutu': pca,
                    'parametreler': rf, 'model nesnesi': model, 'pca nesnesi': p
                }

        for gb in gradientBoostingÖrnekleri:

            if pca is None:
                continue

            print(f"--- {m} modeli için {pca} PCA değerinde Gradient Boosting Eğitiliyor ---")
            model = GradientBoostingClassifier(
                learning_rate = gb['learning_rate'],
                max_depth=gb['max_depth'],
                n_estimators = gb['n_estimators'],
                random_state=3, 
                verbose=1
            )        
            model.fit(train,TrainEtiket)
            gbValidation = model.predict(val)
            skor = accuracy_score(ValidationEtiket, gbValidation)

            if skor > enİyiGB['skor']:
                enİyiGB['skor'] = skor
                enİyiGB['konfigürasyon'] = {
                    'model': 'Gradient Boosting', 'öznitelik': m, 'pca boyutu': pca,
                    'parametreler': gb, 'model nesnesi': model, 'pca nesnesi': p
                }

algoritmalar = {
    'Decision Tree': enİyiDT,
    'Random Forest': enİyiRF,
    'Gradient Boosting': enİyiGB
}

print("--- Her model için en iyi validation skorları ---")

for model, veri in algoritmalar.items():
    konfig = veri['konfigürasyon']
    print(f"\n>>> {model} Modeli:")
    print(f"En İyi Validation Doğruluğu: {veri['skor']:.4f}")
    print(f"En İyi Öznitelik Çıkarıcı: {konfig['öznitelik']}")
    print(f"En İyi PCA Boyutu: {konfig['pca boyutu']}")
    print(f"En İyi Hiperparametreler: {konfig['parametreler']}")

    TestResim = np.load(f"{konfig['öznitelik']}TestResim.npy")
    TestEtiket = np.load(f"{konfig['öznitelik']}TestEtiket.npy")

    if konfig['pca boyutu'] is not None:  
       test = konfig['pca nesnesi'].transform(TestResim)
    else:
       test = TestResim

    testTahmin = konfig['model nesnesi'].predict(test)

    print(f"\n --- {model} Test Seti Sonuçları ---")
    print(classification_report(TestEtiket, testTahmin))
    print(f" --- {model} Confusion Matrix ---")
    print(confusion_matrix(TestEtiket, testTahmin))
