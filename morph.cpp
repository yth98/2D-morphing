#include <iostream>
#include <vector>
#include <cmath>
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
    if (!dets.size())
        return std::vector<Vec2d>(68);
    std::vector<std::vector<cv::Vec2d>> faces;
    for (int i = 0; i < dets.size(); ++i) {
        full_object_detection shape = predictor(dimg, dets[i]);
        std::vector<cv::Vec2d> pt;
        for (int j = 0; j < 68; j++)
            pt.push_back(cv::Vec2d(shape.part(j).x(),shape.part(j).y()));
        faces.push_back(pt);
    }

    return faces[0];
}

vector<lPair> gen_lines(fdots from, fdots to, char *path) {
    std::fstream llist(path);
    char dummy;
    vector<lPair> lines;
    while (!llist.eof()) {
        int p, q;
        llist >> p >> dummy >> q;
        lines.push_back(lPair(from[p-1],from[q-1],to[p-1],to[q-1]));
    }
    llist.close();
    return lines;
}

Vec4b get_pixel(double y, double x, Mat src) {
    //cout << y << " " << x << endl;
    if (x<0 or y<0 or x>src.size[1]-1 or y>src.size[0]-1) // out of frame
        return Vec4b(0,0,0,0);
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
int cross_direction(Vec2d x, Vec2d y) {
    double t = x[0]*y[1]-x[1]*y[0];
    return (t > 0) ? 1 : ((t < 0) ? -1 : 0);
}
double projection_factor(Vec2d x, Vec2d y) {
    return x.dot(y) / x.dot(x);
}

void morph(Mat &dest, Mat src, vector<lPair> pairs) {
    double a = 0.01, b = 0.5, p = 1;
    pairs.push_back(lPair(Vec2d(0,0),Vec2d(0,300),Vec2d(0,0),Vec2d(180,240)));
    pairs.push_back(lPair(Vec2d(100,0),Vec2d(400,0),Vec2d(90,10),Vec2d(350,90)));
    for (int y = 0; y < dest.size[0]; y++) {
        for (int x = 0; x < dest.size[1]; x++) {
            Vec2d X(x,y), X_s(0,0);
            double weights = 0;
            for (lPair pair : pairs){
                Vec2d PQ = pair.Q - pair.P;
                Vec2d P_Q_ = pair.Q_ - pair.P_;
                Vec2d XP = X-pair.P;
                double u = projection_factor(PQ, XP);
                Vec2d v = XP-u*PQ;
                Vec2d v_ = Vec2d(-P_Q_[1],P_Q_[0])/mag(PQ) * mag(v)*cross_direction(PQ,XP);
                Vec2d X_ = (pair.P_*(1-u) + pair.Q_*u) + v_;
                double weight = pow(pow(mag(PQ),p)/(a+mag(v)),b);
                X_s += (X_-X) * weight;
                weights += weight;
            }
            X_s = X + X_s/weights;
            dest.at<Vec4b>(y,x) = get_pixel(X_s[1], X_s[0], src);
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc < 6) {
        cout << "Usage: morph <source1_path> <source2_path> <destination_path> <shape_predictor_68_face_landmarks.dat_path> <face_lines.csv_path>\n";
        return -1;
    }
    // load source as BGRA
    unsigned int length = 480, width = 720;
    Mat source1 = imread(argv[1], IMREAD_COLOR);
    Mat source2 = imread(argv[2], IMREAD_COLOR);
    resize(source1, source1, Size(width,length), 0, 0, INTER_AREA);
    resize(source2, source2, Size(width,length), 0, 0, INTER_AREA);
    Mat source1_a(length, width, CV_8UC4);
    Mat source2_a(length, width, CV_8UC4);
    cvtColor(source1, source1_a, CV_BGR2BGRA, 4);
    cvtColor(source2, source2_a, CV_BGR2BGRA, 4);
    // calc face lines
    double perc = 0.45;
    fdots face1 = face_dots(source1, argv[4]);
    fdots face2 = face_dots(source2, 0);
    fdots facmid;
    for (int i = 0; i < 68; i++)
        facmid.push_back(face1[i]*(1-perc) + face2[i]*perc);
    // calculate destination
    Mat dest1(length, width, CV_8UC4);
    Mat dest2(length, width, CV_8UC4);
    Mat destination(length, width, CV_8UC4);
    morph(dest1, source1_a, gen_lines(face1, facmid, argv[5]));
    morph(dest2, source2_a, gen_lines(face2, facmid, argv[5]));
    addWeighted(dest1,(1-perc),dest2,perc,0,destination);
    // show images
    imshow("Source 1", source1);
    imshow("Source 2", source2);
    imshow("Result", destination);
    waitKey();
    imwrite(argv[3], destination);
    cout<<"Done."<<endl;
    return 0;
}