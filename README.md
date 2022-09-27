# Gold Price Prediction using Time Series <br> Denny Nur Ramadhan

## Project Overview

Secara historis, emas telah digunakan sebagai bentuk mata uang di berbagai belahan dunia termasuk Amerika Serikat. Di masa sekarang, logam mulia seperti emas dipegang dengan bank sentral dari semua negara untuk menjamin pembayaran kembali utang luar negeri, dan juga untuk mengendalikan inflasi yang mengakibatkan mencerminkan kekuatan finansial negara. Baru-baru ini, negara-negara berkembang di dunia, seperti Cina, Rusia, dan India telah menjadi pembeli besar emas, sedangkan Amerika Serikat, SoUSA, Afrika Selatan, dan Australia termasuk di antara penjual besar emas. Pada submission ini, akan membahas model machine learning untuk memprediksi harga emas menggunakan data yang didapatkan dari [kaggle](https://www.kaggle.com/).

## Business Understanding
### Problem Statements
1. Bagaimana cara mengetahui fitur yang berpengaruh terhadap pergerakan harga emas ?
2. Bagaimana cara membuat model machine learning untuk memprediksi emas menggunakan data uji yang ada?

### Goals
1. Mengetahui fitur yang mempengaruhi harga saham emas.
2. Membuat model machine learning untuk memprediksi saham emas.

### Solution statements
Karena dataset terkait hanya berisi tentang data tanggal dan harga, maka solusi yang tepat untuk masalah ini adalah dengan menggunakan pendekatan Time Series. Untuk memahami fitur yang memiliki pengaruh terhadap harga saham, Heatplot digunakan agar pemetaan korelasi antar kolom menjadi lebih mudah dimengerti.

Untuk model machine learning menggunakan layer LSTM (Long Short Term Memory) dalam model. LSTM adalah jenis jaringan saraf berulang yang memiliki kemampuan untuk mengingat atau melupakan output dari data yang melalui arsitekturnya. Ini dilakukan tanpa mengubah konteks dari data yang ada. Dengan pendekatan ini, LSTM mampu mengatasi masalah RNN, yang mana RNN tidak mampu memprediksi kata yang disimpan dalam memori jangka panjang. Dengan bertambahnya panjang celah, RNN tidak memberikan kinerja yang efisien. Berbeda dengan LSTM yang dapat secara default menyimpan informasi. Dengan kinerja seperti ini, LSTM cocok untuk digunakan dalam proses analisa dan prediksi data deret waktu. Beberapa keuntungan untuk menggunakan LSTM untuk kasus Time Series adalah:

1. Tidak ada prasyarat tertentu dalam implementasi model
2. Dapat mengatur parameter tuning secara kustom agar menyesuaikan bentuk data.
3. Cocok untuk digunakan di dataset yang banyak
4. Dapat bekerja dengan baik untuk neural network dengan fungsi non-linear

Menurut [referensi](https://www.springml.com/blog/time-series-forecasting-arima-vs-lstm/) **LSTM (Long Short Term Memory).** baik digunakan pada kasus Time Series

Sedangkan [optimizer](https://deepdatascience.wordpress.com/2016/11/18/which-lstm-optimizer-to-use/) menggunakan **Adam Optimizer** karena secara keseluruhan optimizer ini lebih bagus dibandingkan dengan optimizer lain.


## Data Understanding
Pada submission ini, dataset diambil dari [Kaggle](https://www.kaggle.com) yang bernama **[Gold Pice Prediction Dataset](https://www.kaggle.com/datasets/sid321axn/gold-price-prediction-dataset)**. Berikut adalah daftar kolom di file CSV yang tersedia:

  * Date - Tanggal trading emas (datatype : string object)
  * Open - Harga ketika pertama kali diumumkan di tanggal tersebut (datatype : float64)
  * High - Harga tertinggi di tanggal tersebut (datatype : float64)
  * Low -  Harga terendah di tanggal tersebut (datatype : float64)
  * Close - Harga emas ketika diakhir period (datatype : float64)
  * Adj Close - Close value setelah mempertimbangkan dividen dan stock split (datatype : float64)
  * Volume - Jumlah transaksi emas di tanggal tersebut (datatype : float64)
  
<br>

![Image data Overview](https://github.com/denuradhan/GoldPricePrediction/blob/main/assets/data_overview.png?raw=true)


Dari data tersebut, terlihat bahwa rata-rata harga saham TLKM disajikan sangat lengkap mulai dari Harga Open sampai Adj Close nya periode 2004 sampai 2020. Disertai dengan informasi penting lainnya, seperti harga saham tertinggi dan terendah dalam durasi tersebut.

![Grafik Saham TLKM 2004 - 2021](https://github.com/denuradhan/GoldPricePrediction/blob/main/assets/data_understanding.png?raw=true)

Dari grafik tersebut, dapat diambil kesimpulan bahwa harga saham TLKM mengalami perubahan secara signifikan dalam durasi tersebut. Pada tahun 2004 sampai 2008 merupakan lonjakan harga saham TLKM pertama, kemudian terjadi peningkatan yang sangat drastis di tahun 2012 - Q1 2018. Kemudian mengalami penurunan yang cukup signifikan di akhir tahun 2019 sampai 2 Oktober 2020 karena pengaruh COVID-19.

Sebelum melanjutkan ke tahap preparation, kita perlu untuk melihat korelasi antar fitur yang mempengaruhi pergerakan saham berdasarkan dataset yang ada. Saya mencoba menyajikan korelasi tersebut dalam bentuk heatmap agar dapat dipetakan dengan jelas. 

![Heatmap korelasi antar fitur](https://github.com/denuradhan/GoldPricePrediction/blob/main/assets/korelasi_antar_fitur.png?raw=true)

## Data Preparation
Dalam tahap ini, saya menyiapkan dataframe yang telah menyimpan data dari CSV tersebut untuk dilakukan beberapa pengecekan, pertama kita perlu memeriksa adanya null values. Ini perlu dilakukan untuk menjaga akurasi dari prediksi model yang akan kita lakukan di proses pelatihan data. 

Berikut hasil cek data null oleh library **pandas** : <br>

![Check null values](https://github.com/denuradhan/GoldPricePrediction/blob/main/assets/missing_value.png?raw=true)

Dari 3980 data, terdapat 3944 data yang tidak ada null valuesnya, ini artinya ada beberapa data yang null. Untuk mengatasinya, kita bisa menghapus row yang null dengan **dropna()** dari library **pandas**


```
df_new = df.dropna(how='any',axis=0) 
df_new
```


Kemudian, kita cek juga untuk duplikasi data. Berikut hasil cek duplikasi data oleh library **pandas** : <br>

![Check duplicate values](https://github.com/denuradhan/GoldPricePrediction/blob/main/assets/check_duplicate_value.png?raw=true)

Langkah berikutnya adalah reduksi dimensi dengan menggunakan PCA. Dari Heatmap korelasi di bagian Data Understanding, dapat kita simpulkan bahwa kolom yang mempunyai korelasi rendah adalah kolom 'Volume'. Setelah menghapus kolom tersebut, tersisa kolom 'Low', 'Open','High','Close', dan 'Adj Close'. Untuk meningkatkan efisiensi pelatihan model dengan cara meminimalisasi fitur yang digunakan tanpa menghapus informasi yang ada didalamnya

![Reduksi Dimensi PCA](https://github.com/denuradhan/GoldPricePrediction/blob/main/assets/PCA_dimensional_reduction.png?raw=true)

Selain pengecekan data dan pembagian dataset ke data latih dan data uji, kita juga perlu untuk mengatur skala data. Hal ini perlu dilakukan agar skor MAE kita tidak menjadi terlalu besar, jika hal ini terjadi, akan mengakibatkan prediksi kita sangat buruk. 

Sebagai rangkuman, langkah yang telah dilakukan pada tahap ini adalah: 

1. Penghapusan missing values 
2. Penghapusan duplikat data
3. Reduksi Dimensi dengan PCA
4. Train Test Split dengan ratio 80% data latih dan 20% data uji.
5. Penskalaan data latih dan data uji dengan MinMax Scaler untuk mencegah data leakage

## Modeling

Dari banyaknya opsi penggunaan model yang ada untuk kasus Time Series, saya mencoba mengimplementasikan LSTM dalam pembuatan model. 

Dalam prosesnya, kita telah mengetahui bahwa LSTM ini merupakan perbaikan dari RNN Tradisional dimana LSTM mampu menyimpan nilai yang penting dan menghapus nilai yang tidak penting dalam jangka waktu yang lama secara default.

Semakin kompleks sebuah model ML, maka kemungkinan model tersebut mengalami overfitting pun semakin tinggi. Walaupun secara arsitektur sudah cocok dengan data, menggunakan loss function yang tepat, dan metrik yang sesuai, masih ada kemungkinan overfitting. Oleh karena itu, selain LSTM saya juga menggunakan Dropout layer untuk mencegah terjadinya overfitting selama proses pelatihan data. Simpelnya, dropout layer yang berperan sebagai perantara hidden layer dan output layer ini dimatikan secara bergantian selama proses pelatihan data berlangsung. Secara keseluruhan, alur dari arsitektur model ini adalah LSTM layer sebagai input layer, kemudian melewati dropout layer dengan value sebesar 0.5 untuk meningkatkan variasi output, barulah menggunakan Dense Layer dengan 1 unit perceptron sebagai output layernya.

## Model Evaluation

Berikut visualisasi untuk nilai MAE dan loss value di tahap pelatihan dan pengujian <br><br>
![Model Evaluation Result](https://github.com/denuradhan/GoldPricePrediction/blob/main/assets/model_evaluation.png?raw=true)

Berikut visualisasi untuk prediksi data latih harga saham TLKM dibandingkan dengan data aslinya dalam periode 15 Desember 2011 - 21 Desember 2012 (80% dataset) <br><br>
![Prediction Result](https://github.com/denuradhan/GoldPricePrediction/blob/main/assets/model_prediction.png?raw=true)


## Evaluation

- ***Mean Absolute Error*** <br><br>
![MAE Formula](https://github.com/denuradhan/GoldPricePrediction/blob/main/assets/MAE_Formula.png?raw=true)

Metrik ini digunakan untuk mengetahui kesalahan model atau memberitahu seberapa besar error model yang sudah di latih kepada data yang akan diuji.

- ***Mean Squared Error***:  <br><br>
![MSE Formula](https://github.com/denuradhan/GoldPricePrediction/blob/main/assets/MSE.png?raw=true)<br>
Fungsi loss yang paling sederhana dan sering digunakan untuk kasus regresi

<br>

## Penutup
Demikian laporan dan metrik dari implementasi Machine Learning untuk analisis harga emas. 

Terimakasih.
