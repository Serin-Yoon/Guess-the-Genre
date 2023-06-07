#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;
using namespace cv::dnn;

using namespace std;
using namespace cv;

int main() {
    Net net = readNet("Frozen_VGG.pb"); // "Frozen_Custom"

    if (net.empty()) {
        cerr << "Network load failed!" << endl;
        return -1;
    }

    string modelName = "Fine-Tuned Pre-trained VGG-16"; // "Custom Model"
    string title = "Starwars"; // "The Secret Life of Pets";
    string imgURL = "TestData/" + title + ".jpg";

    Mat img = imread(imgURL, IMREAD_COLOR);

    if (img.empty()) {
        cerr << "Image load failed!" << endl;
        return -1;
    }

    string genre[28] = {
        "Action", "Adult", "Adventure", "Animation", "Biography", "Comedy", "Crime",
        "Documentary", "Drama", "Family", "Fantasy", "Film-Noir", "Game-Show",
        "History", "Horror", "Music", "Musical", "Mystery", "News", "Reality-TV",
        "Romance", "Sci-Fi", "Short", "Sport", "Talk-Show", "Thriller", "War", "Western"
    };

    namedWindow("Guess The Genre", WINDOW_AUTOSIZE);
    resizeWindow("Guess The Genre", 21, 30);
    imshow("Guess The Genre", img);

    while (true) {
        int c = waitKey(0);
        if (c == 27) break;
        if (c == ' ') {
            /* Inference */
            Mat inputBlob = blobFromImage(img, 1/255.f, Size(200, 150));
            net.setInput(inputBlob);
            Mat probs = net.forward(); // Inference

            /* Check Result */
            Mat probMat(probs.size[2], probs.size[3], CV_32F, probs.ptr<float>());

            cout << "\n============= Prediction =============" << endl << endl;
            cout << " Title: " << title << endl;
            cout << " Model: " << modelName << endl << endl;

            if (modelName == "Custom Model") {
                for (int i = 0; i < 28; i++) {
                    float prob = probs.at<float>(i);

                    if (prob > 0.2) {
                        cout << " - " << genre[i] << ": " << fixed << setprecision(2) << probs.at<float>(i) * 100 << "%" << endl;
                    }
                }
            }

            if (modelName == "Fine-Tuned Pre-trained VGG-16") {
                for (int i = 0; i < 28; i++) {
                    float prob = probs.at<float>(i);

                    // prob > 0.0001
                    if (prob > 0.0001) {
                        cout << " - " << genre[i] << ": " << fixed << setprecision(2) << probs.at<float>(i) * 100 << "%" << endl;
                    }
                }
            }
            cout << "\n======================================" << endl;
        }
    }

    // Memory Usage
    struct rusage r_usage;
    getrusage(RUSAGE_SELF, &r_usage);
    cout << "Maximum resident set size (Memory): " << r_usage.ru_maxrss << " KB" << endl;

    return 0;
}