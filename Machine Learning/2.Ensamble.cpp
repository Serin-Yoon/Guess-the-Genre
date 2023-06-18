#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>

using namespace std;
using namespace cv;
using namespace cv::ml;

// 함수 원형 선언
Mat calcColorHistogram(Mat image);
vector<float> calcHistogramFeature(Mat hist);
int convertLabelToInt(const string& label);

unordered_map<string, int> labelMap = {
    {"Comedy", 0},
    {"Animation", 1},
    {"Romance", 2},
    {"Family", 3}
};

unordered_map<int, string> revMap = {
    {0, "Comedy"},
    {1, "Animation"},
    {2, "Romance"},
    {3, "Family"}
};

int main() {
    string str_buf;
    string sec_buf;
    fstream fs;

    fs.open("/Users/dmson1218/Documents/MovieGenre2.csv");

    vector<pair<string, string>> dataset;

    while (getline(fs, str_buf, '\n')) {
        int delimiter = str_buf.find(',');

        string posterName = str_buf.substr(0, delimiter);
        string posterGenre = str_buf.substr(delimiter + 1);
        posterGenre.erase(remove(posterGenre.begin(), posterGenre.end(), '\r'), posterGenre.end());

        pair<string, string> tempset = make_pair(posterName, posterGenre);
        dataset.push_back(tempset);
    }

    fs.close();
    dataset.erase(dataset.begin());

    vector<pair<string, string>> trainset = vector<pair<string,string>>(dataset.begin() + 1833, dataset.end() - 1);
    vector<pair<string, string>> testset = vector<pair<string,string>>(dataset.begin(), dataset.begin() + 1);

    // KNN 모델 생성
    Ptr<KNearest> knn = KNearest::create();
    knn->setAlgorithmType(KNearest::BRUTE_FORCE);
    knn->setDefaultK(8);

    // SVM 모델 생성
    Ptr<SVM> svm = SVM::create();
    svm->setType(SVM::C_SVC);
    svm->setKernel(SVM::LINEAR);

    // 각 이미지에 대한 특징 벡터 계산 및 훈련 데이터셋 구성
    Mat trainData, trainLabels;
    for (auto data : trainset) {
        // 이미지 파일 로드
        Mat image = imread("/Users/dmson1218/Documents/" + data.first);

        // 이미지의 색상 분포도 계산
        Mat colorHist = calcColorHistogram(image);

        // 계산된 특징 벡터 추출
        vector<float> feature = calcHistogramFeature(colorHist);

        // 훈련 데이터셋에 추가
        Mat featureMat(feature, CV_32F);
        trainData.push_back(featureMat.t());
        int label = convertLabelToInt(data.second);
        trainLabels.push_back(label);
    }

    // KNN 모델 훈련
    knn->train(trainData, ROW_SAMPLE, trainLabels);

    // SVM 모델 훈련
    svm->train(trainData, ROW_SAMPLE, trainLabels);

    int result = 0;
    int count = 0;
    // 앙상블 모델을 이용하여 이미지를 분류
    for (auto data : testset) {
        // 이미지 파일 로드
        Mat image = imread("/Users/dmson1218/Documents/" + data.first);

        // 이미지의 색상 분포도 계산
        Mat colorHist = calcColorHistogram(image);

        // 계산된 특징 벡터 추출
        vector<float> feature = calcHistogramFeature(colorHist);

        // 입력 데이터를 특징 벡터로 변환
        Mat featureMat(feature, CV_32F);
        featureMat = featureMat.t();

        // KNN 모델을 이용하여 입력 데이터를 분류
        Mat neighborResponses;
        float knnPrediction = knn->findNearest(featureMat, knn->getDefaultK(), noArray(), neighborResponses);
        int knnresult = 0;
        for (int i = 0; i < 8; i++) {
            if (neighborResponses.at<float>(0, i) == knnPrediction) knnresult++;
        }

        // SVM 모델을 이용하여 입력 데이터를 분류
        float svmPrediction = svm->predict(featureMat);

        // 앙상블 모델의 예측 결과 결합
        float ensemblePrediction;
        // KNN 결과의 확률이 80 이상인 경우 KNN 결과를 사용하고, 그렇지 않은 경우 SVM 결과를 사용
        if (knnresult * 100 / 8 >= 80.0) {
            ensemblePrediction = knnPrediction;
        } else {
            ensemblePrediction = svmPrediction;
        }

        // 예측 결과 출력
        if (ensemblePrediction == labelMap[data.second]) result++;
        count++;
        cout << endl << "============= Prediction =============" << endl << endl;
        cout << "Title: 세얼간이" << endl;
        cout << "Model: EnSemble (KNN (ksize: 8) + SVM)" << endl << endl;
        cout << "- Classified as " << revMap[ensemblePrediction] << endl << endl;
        cout << "- KNN result: " << revMap[knnPrediction] << " " << knnresult * 100 / 8 << "%" << endl;
        cout << "- SVM result: " << revMap[svmPrediction] << endl << endl;
        cout << "======================================" << endl << endl;

        imshow("Poster", image);
        waitKey(0);
    }

    return 0;
}


// 이미지의 색상 분포도 계산
Mat calcColorHistogram(Mat image) {
    // RGB 이미지를 YUV 색 공간으로 변환
    Mat yuvImage;
    cvtColor(image, yuvImage, COLOR_BGR2YUV);

    // YUV 이미지에서 Y, U, V 채널로 분리
    vector<Mat> yuvChannels;
    split(yuvImage, yuvChannels);

    // Y, U, V 채널 각각의 히스토그램 계산
    int histSize = 256; // 히스토그램 구간 수
    float range[] = {0, 256}; // 히스토그램 범위
    const float* histRange = {range};
    bool uniform = true, accumulate = false;
    Mat yHist, uHist, vHist;
    calcHist(&yuvChannels[0], 1, 0, Mat(), yHist, 1, &histSize, &histRange, uniform, accumulate);
    calcHist(&yuvChannels[1], 1, 0, Mat(), uHist, 1, &histSize, &histRange, uniform, accumulate);
    calcHist(&yuvChannels[2], 1, 0, Mat(), vHist, 1, &histSize, &histRange, uniform, accumulate);

    // 계산된 히스토그램을 하나의 행렬로 합치기
    Mat hist;
    int histCount = 3;
    int channels[] = {0};
    int dims = 1;
    const int histSizeArray[] = {histSize};
    const float* histRanges[] = {histRange};
    bool histUniform = true, histAccumulate = false;
    calcHist(&yuvImage, 1, channels, Mat(), hist, dims, histSizeArray, histRanges, histUniform, histAccumulate);
    hist = hist.reshape(1, 1);

    return hist;
}

// 계산된 히스토그램에서 특징 벡터 추출
vector<float> calcHistogramFeature(Mat hist) {
    vector<float> feature;

    for (int i = 0; i < hist.cols; i++) {
        feature.push_back(hist.at<float>(0, i));
    }

    return feature;
}

// 범주형 데이터를 정수로 변환하는 함수
int convertLabelToInt(const string& label) {
    // 장르에 대한 매핑 정보 정의
    return labelMap[label];
}
