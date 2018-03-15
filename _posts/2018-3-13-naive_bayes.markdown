---
layout: post
comments: true
title: Naive Bayes Sınıflandırma Algoritması(Teorik)
excerpt: "Naive Bayes algoritmasını teorik olarak inceleyip, örnek üzerinde göstereceğiz."
date: 2018-03-13  19:00:00
published: true
---

Naive-Bayes, basit ve aynı zamanda etkili bir **sınıflandırma** algoritmasıdır. Bayes teoremine dayanmaktadır. Algoritmadaki ‘Naive’ kısmı veri kümesindeki özniteliklerin(features) birbirinden bağımsız olduğu varsayımından gelir. Yani veri kümesinde bir özniteliğin varlığı diğer özniteliklerden herhangi birine bağlı değildir.Gerçek hayatta ise bu varsayım pek doğru olmaz. Bayes kısmı ise İngiliz matematikçi *Thomas Bayes’ten* gelmektedir.

Naive-Bayes sınıflandırma algoritması bir mesajın spam olup olmadığının kontrolü, bir makalenin içeriğinin hangi konuda(teknoloji, spor, politik) olduğunun sınıflandırılması veya yüz tanıma gibi alanlarda kullanılır. Bu algoritmanın nasıl çalıştığını anlayabilmek için önce Bayes teoremine bakalım.Bayes teoremi, koşulsal olasılık(conditional probability) üzerine kurulu bir teoremdir.


<img src=/_assets/naive_bayes_images/kume_gosterim.png>

<p>
P(A|B) : B nin olasılığı verildiğinde A olayının olma olasılığı
P(B|A) : A nın olasılığı verildiğinde B olayının olma olasılığı
P(A) : A olayının olma olasılığı
P(B) : B olayının olma olasılığı
</p>
Bu terimleri daha iyi anlayabilmek için bir örnek verelim. Şimdi bir popülasyonun içerisinde bir kanser türünün bulunma olasılığı %1 olsun. Kanser olan bir hastanın yaptırdığı bir testte %90 olasılıkla sonucun pozitif olduğu gözlemlenmiş. Kanser olmayan bir hastanın yaptırdığı testte ise %90 olasılıkla sonucun negatif olduğu gözlemlenmiş.Verilen bu bilgiler ışığında test sonucu pozitif olan bir hastanın kanser hastası olma olasılığı nedir?

<img src="images/kume_gosterim.png">


Bu diyagramda küçük daire kanser olma olasılığını, büyük daire test sonucunun pozitif olma ihtimalini, dikdörtgen ise bütün popülasyonu gösteriyor. Şimdi verilen bilgileri olasılıksal olarak gösterelim.
Bize verilen olasıklar :

<img src="images/ornek_verilenler1.png" width="512px" height="64px"> <br>

Bulmamız gerekenleri daha resmi bir şekilde ifade edersek  :


<img src="images/ornek_verilenler2.png" width="512px" height="64px">

</p>
Not: <img src="images/neg_sign.png" align=left width="32px" height="32px"> İsareti  mantıksal değili anlamına gelmektedir.
Aslında yukarıda yapmış olduğumuz işlemler ile sonucu bulduğumuz düşünülebilir. Ancak farketmiş olabileceğiniz üzere bir problemimiz var. Olasılıklar toplamı 1 değil. Bunu sağlayabilmek için normalleştirme işlemini yapmamız gerekiyor. Normalleştirme yukarıda hesapladığımız olasılıkların toplanması ve toplamın ayrı ayrı bu olasılıklara bölünmesi işlemidir.
Normalleştirme değeri = 0.009 + 0.099 = 0.108 = P(Pozitif)
</p>
<br>
<img src="images/ornek_hesaplamalar.png" align=left width="512px" height="64px">

Probleme geri dönersek bizden test sonucu pozitif olan bir hastanın kanser olma olasılığını bulmamız istenmişti. Sonuçlara baktığımızda %8.3 olasılıkla test sonucu pozitif olan bir hasta kanser olabilir. Bunu sınıflandırma problemi üzerinden yorumlarsak kanser olmama olasılığı, olma olasılığına göre çok daha fazla çıktı. Yani Naive Bayes sınıflandırmasına göre bu hasta kanser değildir.
Sonuçlar
Bu yazıda Naive Bayes sınıflandırma algoritmasına bir giriş yaptık ve temelinde duran Bayes teoremini inceledik.

Faydalı Linkler
https://tr.wikipedia.org/wiki/Naive_Bayes_s%C4%B1n%C4%B1fland%C4%B1r%C4%B1c%C4%B1

Bazı İngilizce kavramların Türkçe karşılıklarını sayın Ethem Alpaydın hocamızın kitabında bulunan sözlükten aldım.
https://www.cmpe.boun.edu.tr/~ethem/i2ml2e_tr/i2tr_sozluk.pdf

