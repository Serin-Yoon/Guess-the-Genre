# Guess the Genre 🎞️

Anyone may have misunderstood the movie genre only by looking at movie posters.
We wondered if machine learning and deep learning could accurately predict the genre of the movie only by looking at the poster.
Therefore, using machine learning and deep learning, respectively, we created models that predict the movie genre based on posters.


## Dataset
- [Kaggle Movie Genre from its Poster](https://www.kaggle.com/datasets/neha1703/movie-genre-from-its-poster)
- **Before**: IMDB ID / IMDB link / Title / IMDB Score / Genre / Poster
→ Delete a row if Poster is empty or invalid
- **After**: Title / Genre / Poster (`{IMDB_ID}.jpg`)

### Deep Learning
| Genre | # | Genre | # | Genre | # | Genre | # |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| **Action** | 4,705 | **Adult** | 8 | **Adventure** | 3,399 | **Animation** | 1,558 |
| **Biography** | 1,775 | **Comedy** | 11,193 | **Crime** | 4,593 | **Documentary**| 3,371 |
| **Drama** | 3,371 | **Family** | 17,654 | **Fantasy** | 1,789 | **Film-Noir** | 318 |
| **Game-Show** | 1 | **History** | 1,280 | **Horror** | 3,544 | **Music** | 1,133 |
| **Musical** | 710 | **Mystery** | 2,092 | **News** | 77 | **Reality-TV** | 2 |
| **Romance** | 5,379 | **Sci-Fi** | 1,787 | **Short** | 851 | **Sport** | 629 |
| **Talk-Show** | 6 | **Thriller** | 4,251 | **War** | 1,027 | **Western** | 722 |

### Machine Learning
| Genre | # | Genre | # | Genre | # | Genre | # |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| **Animation** | 679 | **Comedy** | 11,193 | **Family** | 1,035 | **Romance** | 2,926 |

- Due to the limiation of implementing a multi-label, we chose 1 genre for each movie (ex. Toystory: Animation, Adventure, Comedy, ... → Animation)
- Then, we reorganized the dataset by selecting 4 genres that show the biggest difference between feature vectors among 28 genres.


## Model

### Machine Learning: KNN
<img width="460" alt="image" src="https://github.com/Serin-Yoon/Guess-the-Genre/assets/53158200/8f01e25d-2166-4c78-b11a-c2d4035c333e">

### Machine Learning: Ensemble (KNN + SVM)
<img width="350" alt="image" src="https://github.com/Serin-Yoon/Guess-the-Genre/assets/53158200/a4f4cbe5-243b-4992-8dfa-077670defd22">

### Deep Learning: Custom Model
<img width="1676" alt="image" src="https://github.com/Serin-Yoon/Guess-the-Genre/assets/53158200/7db4b73b-1752-4fd3-9b22-33f02054ad2a">

### Deep Learning: Fine-tuned Pre-trained VGG-16
<img width="900" alt="image" src="https://github.com/Serin-Yoon/Guess-the-Genre/assets/53158200/676c263f-6204-4491-8def-63abaa3848ce">


## Inference
### Machine Learning: KNN
<img width="1230" alt="image" src="https://github.com/Serin-Yoon/Guess-the-Genre/assets/53158200/72217ac4-27c1-4259-aafb-736fc616f0a3">
<img width="1230" alt="image" src="https://github.com/Serin-Yoon/Guess-the-Genre/assets/53158200/264651e1-a53a-40e1-a065-1348c179f2bd">

### Machine Learning: Ensemble
<img width="1230" alt="image" src="https://github.com/Serin-Yoon/Guess-the-Genre/assets/53158200/09ec67ed-9748-4d26-9391-fd9308343a49">
<img width="1230" alt="image" src="https://github.com/Serin-Yoon/Guess-the-Genre/assets/53158200/90b161d2-aa2b-4792-a81d-1cba649ab811">

### Deep Learning: Custom Model
<img width="1230" alt="image" src="https://github.com/Serin-Yoon/Guess-the-Genre/assets/53158200/309bc8b5-2052-4a80-8643-69427c610ec5">
<img width="1230" alt="image" src="https://github.com/Serin-Yoon/Guess-the-Genre/assets/53158200/425d7cb4-a83e-4dcb-a208-e1c2692ebdb9">

### Deep Learning: Fine-tuned Pre-trained VGG-16
<img width="1230" alt="image" src="https://github.com/Serin-Yoon/Guess-the-Genre/assets/53158200/ff51d4ac-d480-45e9-a9c3-4a750e7ae822">
<img width="1230" alt="image" src="https://github.com/Serin-Yoon/Guess-the-Genre/assets/53158200/88babb4a-0e5e-4077-881f-0413417c0911">


## Comparison

| Model | Train | Parameter | Structure Complexity | Dimension | K | Distance Metric | Memory | Accuracy |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| **KNN** | 14.1s | 28,000 | O(N) | 256 | 8 | Euclidean distance | 111,376 MB | 70.92% |
| **Ensemble** | 15.8s | 56,003 | O(N^3) | 256 | 8 | Euclidean distance | 154,544 MB | 71.69% |

| Model | Learning Rate | Epoch | Optimizer | Train | Parameter | Layer | Memory | Accuracy |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| **Custom Model** | 0.0001 | 150 | Adam | 6h 33m 10s | 505,660 | 13 | 141,856 MB | 77.28% |
| **VGG-16** | 0.0001 | 50 | RMSprop | 2h 37m 11s | 27,327,324 | 5 | 571,264 MB | 82.22% |


- The accuracy of machine learning and deep learning are not dramatically different, but the performance of deep learning is much better. This is because machine learning currently uses 4 out of 28 labels, and only 1 label is offered as an inference value, and deep learning uses all 28 labels and is implemented as a multi-label.
- In machine learning, there is a limit to accurately predicting genres with only histograms, so we thought about various methods such as poster composition, number of people, and text. However, the extraction process was often more difficult than the machine learning algorithm itself, and even if the extraction was made, the accuracy did not change significantly. However, it was good that machine learning ended in seconds without turning it for hours like deep learning after storing it in memory.
- Deep learning required much more datasets than machine learning. It also took a lot of time even to use a local GPU, and memory was used much more. Also, the process of increasing accuracy by changing the parameters of the model was difficult. However, the pre-processing process was not difficult because deep learning did not require feature extraction. Since the output structure of the model can be set directly, it is very suitable for datasets that can have multi-label, such as movie posters.


## Further Work

- Currently, the number of datasets for each label is different, and if the number of datasets for each label is unified and learned, both machine learning and deep learning are expected to have higher accuracy.
- Machine learning infers the majority of posters as comedy. Likewise, deep learning infers the majority of posters as drama. It is expected that different results will come out if they are trained except for the comedy/drama dataset.


## Contributor
- [Serin Yoon](https://github.com/serin-yoon)
- [Dongmin Son](https://github.com/dmson1218)


## Reference
https://github.com/d-misra/Multi-label-movie-poster-genre-classification 
