#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
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

int main() {
    string str_buf;
    string sec_buf;
    fstream fs;

    fs.open("/Users/dmson1218/Documents/MovieGenre2.csv");

    vector<pair<string, string>> dataset;

    while (!fs.eof()) {
        getline(fs, str_buf, '\n');
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

    // 각 이미지에 대한 특징 벡터 계산 및 훈련 데이터셋 구성
    Mat trainData, trainLabels;
    for (auto data : trainset) {
        // 이미지 파일 로드
        Mat image = imread("/Users/dmson1218/Documents/" + data.first);

        if (image.empty()) {
            cout << "Failed to load image: " << data.first << endl;
            continue;
        }

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

    // KNN 모델 생성
    Ptr<KNearest> knn = KNearest::create();
    knn->setIsClassifier(true);
    knn->setAlgorithmType(KNearest::BRUTE_FORCE);

    // KNN 모델 훈련
    knn->train(trainData, ROW_SAMPLE, trainLabels);

    // KNN 모델을 이용하여 이미지를 분류
    for (auto data : testset) {
        // 이미지 파일 로드
        Mat image = imread(data.first);

        if (image.empty()) {
            cout << "Failed to load image: " << data.first << endl;
            continue;
        }

        // 이미지의 색상 분포도 계산
        Mat colorHist = calcColorHistogram(image);

        // 계산된 특징 벡터 추출
        vector<float> feature = calcHistogramFeature(colorHist);

        // 입력 데이터를 특징 벡터로 변환
        Mat featureMat(feature, CV_32F);
        featureMat = featureMat.t();

        // KNN 모델을 이용하여 입력 데이터를 분류
        Mat neighborResponses;
        Mat results;
        Mat dists;
        int k = 8;
        knn->findNearest(featureMat, k, results, neighborResponses, dists);

        // 예측 결과 출력
        int prediction = static_cast<int>(results.at<float>(0, 0));
        string genre = "";
        for (auto pair : labelMap) {
            if (pair.second == prediction) {
                genre = pair.first;
                break;
            }
        }
        
        cout << endl << "============= Prediction =============" << endl << endl;
        cout << "Title: 웰컴투동막골" << endl;
        cout << "Model: KNN (ksize: " << k << ")" << endl << endl;
        cout << "- Classified as " << genre << endl << endl;
        cout << "======================================" << endl << endl;

        imshow("Poster", image);
        waitKey(0);
    }

    // KNN 모델 평가
    Mat testData, testLabels;
    for (auto data : testset) {
        // 이미지 파일 로드
        Mat image = imread(data.first);

        if (image.empty()) {
            cout << "Failed to load image: " << data.first << endl;
            continue;
        }

        // 이미지의 색상 분포도 계산
        Mat colorHist = calcColorHistogram(image);

        // 계산된 특징 벡터 추출
        vector<float> feature = calcHistogramFeature(colorHist);

        // 입력 데이터를 특징 벡터로 변환
        Mat featureMat(feature, CV_32FC1);

        // 테스트 데이터셋에 추가
        testData.push_back(featureMat.t());
        int label = convertLabelToInt(data.second);
        testLabels.push_back(label);
    }

    // KNN 모델을 이용하여 테스트 데이터를 분류
    Mat predictions;
    Mat neighborResponses;
    Mat dists;
    int k = 8;
    knn->findNearest(testData, k, predictions, neighborResponses, dists);

    int correctCount = 0;
    for (int i = 0; i < predictions.rows; i++) {
        float predictedLabel = predictions.at<float>(i, 0);
        float trueLabel = testLabels.at<float>(i, 0);

        // 정확도 계산
        if (predictedLabel == trueLabel)
            correctCount++;
    }

    int totalCount = predictions.rows;

    // 정확도 출력
    float accuracy = (float)correctCount / totalCount * 100;
    //cout << "Accuracy: " << accuracy << "%" << endl;

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

    // 계산된 히스토그램을 하나의 행렬로 병합
    Mat hist;
    hconcat(yHist, uHist, hist);
    hconcat(hist, vHist, hist);

    return hist;
}

// 히스토그램을 특징 벡터로 변환
vector<float> calcHistogramFeature(Mat hist) {
    vector<float> feature;
    feature.reserve(hist.total());
    for (int i = 0; i < hist.rows; i++) {
        for (int j = 0; j < hist.cols; j++) {
            feature.push_back(hist.at<float>(i, j));
        }
    }
    return feature;
}

// 레이블을 정수로 변환
int convertLabelToInt(const string& label) {
    if (labelMap.find(label) != labelMap.end())
        return labelMap[label];
    return -1;
}
