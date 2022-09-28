# Gold Price Prediction using _Time Series_ <br> Denny Nur Ramadhan

## Project Overview

Secara historis, emas telah digunakan sebagai bentuk mata uang di berbagai belahan dunia termasuk Amerika Serikat. Di masa sekarang, logam mulia seperti emas dipegang dengan bank sentral dari semua negara untuk menjamin pembayaran kembali utang luar negeri, dan juga untuk mengendalikan inflasi yang mengakibatkan mencerminkan kekuatan finansial negara. Baru-baru ini, negara-negara berkembang di dunia, seperti Cina, Rusia, dan India telah menjadi pembeli besar emas, sedangkan Amerika Serikat, SoUSA, Afrika Selatan, dan Australia termasuk di antara penjual besar emas. Pada submission ini, akan membahas model _machine learning_ untuk memprediksi harga emas menggunakan data yang didapatkan dari [kaggle](https://www.kaggle.com/).

## Business Understanding
### Problem Statements
1. Bagaimana cara mengetahui fitur yang berpengaruh terhadap pergerakan harga emas ?
2. Bagaimana cara membuat model _machine learning_ untuk memprediksi emas menggunakan data uji yang ada?

### Goals
1. Mengetahui fitur yang mempengaruhi harga saham emas.
2. Membuat model _machine learning_ untuk memprediksi saham emas.

### Solution statements
Karena dataset terkait hanya berisi tentang data tanggal dan harga, maka solusi yang tepat untuk masalah ini adalah dengan menggunakan pendekatan _Time Series_. Untuk memahami fitur yang memiliki pengaruh terhadap harga emas, Heatplot digunakan agar pemetaan korelasi antar kolom menjadi lebih mudah dimengerti.

Untuk model _machine learning_ menggunakan layer LSTM (_Long Short Term Memory_) dalam model. LSTM adalah jenis jaringan saraf berulang yang memiliki kemampuan untuk mengingat atau melupakan output dari data yang melalui arsitekturnya [1]. Ini dilakukan tanpa mengubah konteks dari data yang ada. Dengan pendekatan ini, LSTM mampu mengatasi masalah RNN (_Recurrent Neural Network_) [2], yang mana RNN tidak mampu memprediksi kata yang disimpan dalam memori jangka panjang. Dengan bertambahnya panjang celah, RNN tidak memberikan kinerja yang efisien. Berbeda dengan LSTM yang dapat secara default menyimpan informasi. Dengan kinerja seperti ini, LSTM cocok untuk digunakan dalam proses analisa dan prediksi data deret waktu. Beberapa keuntungan untuk menggunakan LSTM untuk kasus _Time Series_ adalah:

1. Tidak ada prasyarat tertentu dalam implementasi model
2. Dapat mengatur parameter tuning secara kustom agar menyesuaikan bentuk data.
3. Cocok untuk digunakan di dataset yang banyak
4. Dapat bekerja dengan baik untuk neural network dengan fungsi non-linear


## Data Understanding
Pada submission ini, dataset diambil dari [Kaggle](https://www.kaggle.com) yang bernama **[Gold Pice Prediction Dataset](https://www.kaggle.com/datasets/sid321axn/gold-price-prediction-dataset)**. Terdapat 1718 data yang ada pada dataset. Bila diperhatikan terdapat banyak kolom yang tersedia pada dataset yang telah dicantumkan. Tetapi untuk mempercepat proses pelatihan model Berikut adalah daftar 6 kolom di file CSV yang digunakan sebagai dataset pada proyek ini :

  * Date - Tanggal trading emas (datatype : string object)
  * Open - Harga ketika pertama kali diumumkan di tanggal tersebut (datatype : float64)
  * High - Harga tertinggi di tanggal tersebut (datatype : float64)
  * Low -  Harga terendah di tanggal tersebut (datatype : float64)
  * Close - Harga emas ketika diakhir period (datatype : float64)
  * Adj Close - Close value setelah mempertimbangkan dividen dan stock split (datatype : float64)
  * Volume - Jumlah transaksi emas di tanggal tersebut (datatype : float64)

<br>

![Image data Overview](https://github.com/denuradhan/GoldPricePrediction/blob/main/assets/data_overview.png?raw=true)<br>
Gambar 1. menampilkan informasi tentang DataFrame 

Dari Gambar 1. terlihat bahwa rata-rata harga emas disajikan sangat lengkap mulai dari Harga Open sampai Adj Close nya periode 2012 sampai 2019. Disertai dengan informasi penting lainnya, seperti harga tertinggi dan terendah dalam durasi tersebut.

![Grafik Saham emas 2012 - 2019](https://github.com/denuradhan/GoldPricePrediction/blob/main/assets/data_understanding.png?raw=true)<br>
Gambar 2. Grafik harga emas berdasarkan harga awal, akhir dan tertinggi

Dari Gambar 2. dapat diambil kesimpulan bahwa harga emas mengalami perubahan secara signifikan dalam durasi tersebut. Pada tahun 2012 merupakan lonjakan harga saham emas pertama, kemudian terjadi penurunan yang sangat drastis di tahun 2013.

Sebelum melanjutkan ke tahap preparation, kita perlu untuk melihat korelasi antar fitur yang mempengaruhi pergerakan saham berdasarkan dataset yang ada. Saya mencoba menyajikan korelasi tersebut dalam bentuk heatmap agar dapat dipetakan dengan jelas. 

![Heatmap korelasi antar fitur](https://github.com/denuradhan/GoldPricePrediction/blob/main/assets/korelasi_antar_fitur.png?raw=true)<br>
Gambar 3. Heatmap Korelasi Antar Fitur

## Data Preparation
Dalam tahap ini, saya menyiapkan dataframe yang telah menyimpan data dari CSV tersebut untuk dilakukan beberapa pengecekan, pertama kita perlu memeriksa adanya null values. Ini perlu dilakukan untuk menjaga akurasi dari prediksi model yang akan kita lakukan di proses pelatihan data. 

Berikut hasil cek data null oleh library **pandas** : <br>

![Check null values](https://github.com/denuradhan/GoldPricePrediction/blob/main/assets/missing_value.png?raw=true)<br>
Gambar 4. Mengecek Data yang kosong

Dari 1718 data, tidak ada data dengan nilai null pada valuesnya, ini artinya ada beberapa data yang null. Apabila terdapat null value bisa menghapus row yang null dengan **dropna()** dari library **pandas**

Kemudian, kita cek juga untuk duplikasi data. Berikut hasil cek duplikasi data oleh library **pandas** : <br>

![Check duplicate values](https://github.com/denuradhan/GoldPricePrediction/blob/main/assets/check_duplicate_value.png?raw=true)<br>
Gambar 5. Mengecek Data yang kembar (duplikasi)

Langkah berikutnya adalah reduksi dimensi dengan menggunakan PCA. Dari _Heatmap_ korelasi di bagian _Data Understanding_, dapat kita simpulkan bahwa kolom yang mempunyai korelasi rendah adalah kolom `'Volume'`. Setelah menghapus kolom tersebut, tersisa kolom ```'Low'```, `'Open'`,`'High'`,`'Close'`, dan `'Adj Close'`. Untuk meningkatkan efisiensi pelatihan model dengan cara meminimalisasi fitur yang digunakan tanpa menghapus informasi yang ada didalamnya

![Reduksi Dimensi PCA](https://github.com/denuradhan/GoldPricePrediction/blob/main/assets/PCA_dimensional_reduction.png?raw=true) <br>
Gambar 6. Reduksi Dimensi PCA

Selain pengecekan data dan pembagian dataset ke data latih dan data uji, kita juga perlu untuk mengatur skala data. Hal ini perlu dilakukan agar skor MAE kita tidak menjadi terlalu besar, jika hal ini terjadi, akan mengakibatkan prediksi kita sangat buruk. 

Sebagai rangkuman, langkah yang telah dilakukan pada tahap ini adalah: 

1. Penghapusan missing values 
2. Penghapusan duplikat data
3. Reduksi Dimensi dengan PCA
4. membagi data latih dan daja uji dengan ratio 80% data latih dan 20% data uji.
5. Penskalaan data latih dan data uji dengan _MinMax Scaler_ untuk mencegah data leakage

## Modeling

Dari banyaknya opsi penggunaan model yang ada untuk kasus _Time Series_, saya mencoba mengimplementasikan LSTM dalam pembuatan model. 

Dalam prosesnya, kita telah mengetahui bahwa LSTM ini merupakan perbaikan dari RNN Tradisional dimana LSTM mampu menyimpan nilai yang penting dan menghapus nilai yang tidak penting dalam jangka waktu yang lama secara default.

Semakin kompleks sebuah model ML, maka kemungkinan model tersebut mengalami _overfitting_ pun semakin tinggi. Walaupun secara arsitektur sudah cocok dengan data, menggunakan loss function yang tepat, dan metrik yang sesuai, masih ada kemungkinan _overfitting_. Oleh karena itu, selain LSTM saya juga menggunakan _dropout layer_ untuk mencegah terjadinya _overfitting_ selama proses pelatihan data [3]. Simpelnya, _dropout layer_ yang berperan sebagai perantara _hidden layer_ dan _output layer_ ini dimatikan secara bergantian selama proses pelatihan data berlangsung. Secara keseluruhan, alur dari arsitektur model ini adalah LSTM layer sebagai input layer, kemudian melewati _dropout layer_ dengan value sebesar 0.5 untuk meningkatkan variasi output, barulah menggunakan _Dense Layer_ dengan 1 unit perceptron sebagai _output layer_ nya. Algoritma optimisasi yang digunakan adalah Adam. Adam adalah algoritma pengoptimalan yang dapat digunakan sebagai ganti dari prosedur _stochastic gradient descent_ klasik untuk memperbarui _weight network_ secara iteratif berdasarkan data training [4].

## Model Evaluation

Berikut visualisasi untuk nilai MAE dan _loss value_ di tahap pelatihan dan pengujian <br><br>
![Model Evaluation Result](https://github.com/denuradhan/GoldPricePrediction/blob/main/assets/model_evaluation.png?raw=true)<br>
Gambar 7. Hasil Evaluasi Model

Berikut visualisasi untuk prediksi data latih harga emas dibandingkan dengan data aslinya dalam periode 15 Desember 2011 - 21 Desember 2012 (80% dataset) <br><br>
![Prediction Result](https://github.com/denuradhan/GoldPricePrediction/blob/main/assets/model_prediction.png?raw=true)<br>
Gambar 8. Hasil prediksi


## Evaluation

- ***Mean Absolute Error*** <br><br>
![MAE Formula](https://github.com/denuradhan/GoldPricePrediction/blob/main/assets/MAE_Formula.png?raw=true)<br>
Gambar 9. Formula MAE

Metrik ini digunakan untuk mengetahui kesalahan model atau memberitahu seberapa besar error model yang sudah di latih kepada data yang akan diuji.

- ***Mean Squared Error***:  <br><br>
![MSE Formula](https://github.com/denuradhan/GoldPricePrediction/blob/main/assets/MSE.png?raw=true)<br>
Gambar 10. Formula MSE

Fungsi loss yang paling sederhana dan sering digunakan untuk kasus regresi.

<br>

## Penutup
Demikian laporan dan metrik dari implementasi _machine learning_ untuk analisis harga emas. 

Terimakasih.

## Refensi

[1]	N. Cowan, “What are the differences between long-term, short-term, and working memory? Nelson,” NIH Public Access, vol. 6123, no. 07, 2009, doi: 10.1016/S0079-6123(07)00020-9.What.

[2]	H. P. Nguyen, J. Liu, and E. Zio, “A long-term prediction approach based on long short-term memory neural networks with automatic parameter optimization by Tree-structured Parzen Estimator and applied to time-series data of NPP steam generators,” Applied Soft Computing Journal, vol. 89, 2020, doi: 10.1016/j.asoc.2020.106116.

[3]	J. Liang and R. Liu, “Stacked denoising autoencoder and dropout together to prevent _overfitting_ in deep neural network,” 2016. doi: 10.1109/CISP.2015.7407967.

[4]	D. Wang, D. Tan, and L. Liu, “Particle swarm optimization algorithm: an overview,” Soft comput, vol. 22, no. 2, 2018, doi: 10.1007/s00500-016-2474-6.

 
