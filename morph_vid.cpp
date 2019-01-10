#include <iostream>
#include <vector>
#include <cmath>
#include <ctime>
#include "opencv2/opencv.hpp"
#include "dlib/opencv/cv_image.h"
#include "dlib/image_processing.h"
#include "dlib/image_processing/frontal_face_detector.h"

using namespace cv;
using std::cout;
using std::cin;
using std::endl;
using std::vector;

typedef vector<Vec2d> fdots;

class lPair {
public:
    Vec2d P_, Q_, P, Q;
    lPair(Vec2d p_, Vec2d q_, Vec2d p, Vec2d q) : P_(p_), Q_(q_), P(p), Q(q) {}
};

fdots face_dots(Mat img, char *path) {
    using namespace dlib;
    static frontal_face_detector detector = get_frontal_face_detector();
    static shape_predictor predictor;
    static bool pred_init = false;
    if (!pred_init) {
        try {
            deserialize(path) >> predictor;
        } catch (std::exception &e) {
            cout << e.what() << endl;
        }
        pred_init = true;
    }

    // convert color Mat to grayscale array2d
    cv::cvtColor(img, img, CV_BGR2GRAY, 1);
    array2d<unsigned char> dimg;
    assign_image(dimg, cv_image<unsigned char>(img));

    std::vector<dlib::rectangle> dets = detector(dimg);
    if (!dets.size()) {
        return std::vector<Vec2d>(1);
    }
    std::vector<std::vector<cv::Vec2d>> faces;
    for (int i = 0; i < dets.size(); ++i) {
        full_object_detection shape = predictor(dimg, dets[i]);
        std::vector<cv::Vec2d> pt;
        for (int j = 0; j < 68; j++)
            pt.push_back(cv::Vec2d(shape.part(j).x(),shape.part(j).y()));
        // add dots 69~72 for 4 corners
        pt.push_back(cv::Vec2d(0         ,0         )); // ul
        pt.push_back(cv::Vec2d(img.cols-1,0         )); // ur
        pt.push_back(cv::Vec2d(img.cols-1,img.rows-1)); // dr
        pt.push_back(cv::Vec2d(0         ,img.rows-1)); // dl
        faces.push_back(pt);
    }

    return faces[0];
}

vector<Vec2s> llist;

void init_llist(char *path) {
    std::fstream ls(path);
    char dummy;
    while (!ls.eof()) {
        short p, q;
        ls >> p >> dummy >> q;
        if (p>72 || q>72)
            continue;
        llist.push_back(Vec2s(p,q));
    }
    ls.close();
}

vector<lPair> gen_lines(fdots from, fdots to) {
    vector<lPair> lines;
    for (Vec2s lmap : llist) {
        short p = lmap[0]-1, q = lmap[1]-1;
        lines.push_back(lPair(from[p],from[q],to[p],to[q]));
    }
    return lines;
}

Vec4b get_pixel(double y, double x, Mat src) {
    //cout << y << " " << x << endl;
    if (x<0 or y<0 or x>src.size[1]-1 or y>src.size[0]-1) // out of frame
        return Vec4b(255,255,255, 0);
    double xt = x-(int)x, yt = y-(int)y;
    //cout << xt << " " << yt << endl;
    int u = (int)floor(y);
    int d = (int)ceil(y);
    int l = (int)floor(x);
    int r = (int)ceil(x);
    //cout << u << " " << d << " " << l << " " << r << endl;
    Vec4d ul = src.at<Vec4b>(u,l);
    Vec4d ur = src.at<Vec4b>(u,r);
    Vec4d dl = src.at<Vec4b>(d,l);
    Vec4d dr = src.at<Vec4b>(d,r);
    Vec4d bgra(0,0,0,0);
    bgra += ul * xt     * yt;
    bgra += ur * (1-xt) * yt;
    bgra += dl * xt     * (1-yt);
    bgra += dr * (1-xt) * (1-yt);
    return (Vec4b)bgra;
}

double mag(Vec2d x) {
    return sqrt(x.dot(x));
}
double dist(double u, double v, Vec2d X, Vec2d P, Vec2d Q) {
    if (u < 0)
        return mag(X-P);
    if (u > 1)
        return mag(X-Q);
    if (v < 0)
        return -v;
    return v;
}

void morph(Mat &dest, Mat src, vector<lPair> pairs) {
    clock_t t1 = clock();
    double a = 0.00000001, b = 2.0, p = 0;
    for (int y = 0; y < dest.size[0]; y++) {
        for (int x = 0; x < dest.size[1]; x++) {
            Vec2d X(x,y), X_s(0,0);
            double weights = 0;
            for (lPair pair : pairs){
                Vec2d PQ = pair.Q - pair.P;
                Vec2d P_Q_ = pair.Q_ - pair.P_;
                Vec2d PX = X-pair.P;
                double u = PX.dot(PQ) / PQ.dot(PQ);
                double v = PX.dot(Vec2d(-PQ[1],PQ[0])) / sqrt(PQ.dot(PQ));
                Vec2d X_ = (pair.P_*(1-u) + pair.Q_*u) + v*Vec2d(-P_Q_[1],P_Q_[0])/sqrt(P_Q_.dot(P_Q_));
                double weight = pow(pow(PQ.dot(PQ),p/2)/(a+dist(u,v,X,pair.P,pair.Q)),b);
                X_s += (X_-X) * weight;
                weights += weight;
            }
            X_s = X + X_s/weights;
            dest.at<Vec4b>(y,x) = get_pixel(X_s[1], X_s[0], src);
        }
    }
    t1 = clock() - t1;
    cout<<(((float)t1)/CLOCKS_PER_SEC)<<" s elapsed."<<endl;
}

// https://www.learnopencv.com/alpha-blending-using-opencv-cpp-python/
void alphaBlend(Mat& foreground, Mat& background, Mat& outImage) {
    // Find number of pixels. 
    int numberOfPixels = foreground.rows * foreground.cols * foreground.channels();

    Mat alpha(foreground.rows, foreground.cols, CV_8UC1);
    int from_to[] = {3,0};
    mixChannels(&foreground, 1, &alpha, 1, from_to, 1);
    alpha.convertTo(alpha, CV_32FC1);

    // Get floating point pointers to the data matrices
    float* fptr = reinterpret_cast<float*>(foreground.data);
    float* bptr = reinterpret_cast<float*>(background.data);
    float* aptr = reinterpret_cast<float*>(alpha.data);
    float* outImagePtr = reinterpret_cast<float*>(outImage.data);

    // Loop over all pixesl ONCE
    for( 
    int i = 0; 
    i < numberOfPixels; 
    i++, outImagePtr++, fptr++, aptr++, bptr++
    ) {
        *outImagePtr = (*fptr)*(*aptr) + (*bptr)*(1 - *aptr);
    }
}

int main(int argc, char *argv[]) {
    if (argc < 6) {
        cout << "Usage: morphvid <source_img_path> <source_video_path> <destination_path> <shape_predictor_68_face_landmarks.dat_path> <face_lines.csv_path>\n";
        return -1;
    }
    // load video
    VideoCapture cap(argv[2]);
    if (!cap.isOpened()) {
        cout<<"Invalid video."<<endl;
        return -1;
    }
    cout<<"Video loaded."<<endl;
    // load source as BGRA
    unsigned int length = (int)cap.get(CV_CAP_PROP_FRAME_HEIGHT) / 2,
                  width = (int)cap.get(CV_CAP_PROP_FRAME_WIDTH)  / 2;
    Mat source1 = imread(argv[1], IMREAD_UNCHANGED); // need to support alpha
    Mat source2;
    Mat source2_a;
    Mat source1l = source1.clone();
    cout<<"Images loaded."<<endl;
    // calc face lines
    fdots face1 = face_dots(source1, argv[4]);
    if (face1.size() < 72)
        return -1;
    cout<<"Face in source is found."<<endl;
    // prepare line list
    init_llist(argv[5]);
    for (Vec2s lmap : llist) {
        short p = lmap[0]-1, q = lmap[1]-1;
        line(source1l, Point(face1[p][0],face1[p][1]), Point(face1[q][0],face1[q][1]), Scalar(255,0,0));
    }
    imshow("Source", source1l);
    waitKey(1);
    // calculate destination
    Mat desti_a(length, width, CV_8UC4);
    Mat desti  (length, width, CV_8UC3);
    VideoWriter dest(argv[3], CV_FOURCC('X','V','I','D'), cap.get(CV_CAP_PROP_FPS), Size(width,length));
    if (!dest.isOpened()) {
        cout<<"Writing video failed."<<endl;
        return -1;
    }
    for (int i=0; i<cap.get(CV_CAP_PROP_FPS)*60; i++) {
        cap >> source2;
        if (source2.empty())
            break;
        resize(source2, source2, Size(width,length), 0, 0, INTER_AREA);
        fdots face2 = face_dots(source2, 0);
        if (face2.size() < 72) {
            dest << source2;
            continue;
        }
        morph(desti_a, source1, gen_lines(face1, face2));
        cvtColor(source2, source2_a, CV_BGR2BGRA, 4);
        //alphaBlend(desti_a, source2_a, desti_a);
        cvtColor(desti_a, desti, CV_BGRA2BGR, 3);
        imshow("Video", source2);
        imshow("Result", desti);
        dest << desti;
        waitKey(10);
    }
    // show images
    return 0;
}