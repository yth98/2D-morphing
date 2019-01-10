#include <iostream>
#include <vector>
#include <cmath>
#include <sys/stat.h> // works under linux
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

fdots face_dots(Mat img, char *path, int m) {
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
        cout<<"Face not found in image!"<<endl;
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
        // manual specified line arrangement
        switch(m) {
            case 0:
            pt.push_back(cv::Vec2d(223,165));//hair
            pt.push_back(cv::Vec2d(262,117));
            pt.push_back(cv::Vec2d(301,111));
            pt.push_back(cv::Vec2d(345,121));
            pt.push_back(cv::Vec2d(389,112));
            pt.push_back(cv::Vec2d(416,153));
            pt.push_back(cv::Vec2d(455,207));
            pt.push_back(cv::Vec2d(512,198));
            pt.push_back(cv::Vec2d(502,150));
            pt.push_back(cv::Vec2d(458, 93));
            pt.push_back(cv::Vec2d(399, 46));//83
            pt.push_back(cv::Vec2d(361, 31));
            pt.push_back(cv::Vec2d(257, 45));
            pt.push_back(cv::Vec2d(197, 86));
            pt.push_back(cv::Vec2d(166,120));
            pt.push_back(cv::Vec2d(154,179));
            pt.push_back(cv::Vec2d(212,430));//neck
            pt.push_back(cv::Vec2d(352,571));
            pt.push_back(cv::Vec2d(418,453));
            pt.push_back(cv::Vec2d(106,460));
            pt.push_back(cv::Vec2d(536,415));
            pt.push_back(cv::Vec2d(  5,541));
            pt.push_back(cv::Vec2d(638,462));
            pt.push_back(cv::Vec2d(174,214));//ear
            pt.push_back(cv::Vec2d(164,235));
            pt.push_back(cv::Vec2d(179,312));
            pt.push_back(cv::Vec2d(479,242));
            pt.push_back(cv::Vec2d(468,249));
            pt.push_back(cv::Vec2d(468,262));
            break;
            case 1:
            pt.push_back(cv::Vec2d(281,145));//hair
            pt.push_back(cv::Vec2d(306,118));
            pt.push_back(cv::Vec2d(344,106));
            pt.push_back(cv::Vec2d(378,121));
            pt.push_back(cv::Vec2d(403,118));
            pt.push_back(cv::Vec2d(425,132));
            pt.push_back(cv::Vec2d(447,183));
            pt.push_back(cv::Vec2d(481,187));
            pt.push_back(cv::Vec2d(482,150));
            pt.push_back(cv::Vec2d(457,102));
            pt.push_back(cv::Vec2d(428, 73));//83
            pt.push_back(cv::Vec2d(387, 54));
            pt.push_back(cv::Vec2d(315, 54));
            pt.push_back(cv::Vec2d(265, 98));
            pt.push_back(cv::Vec2d(249,122));
            pt.push_back(cv::Vec2d(240,160));
            pt.push_back(cv::Vec2d(264,306));//neck
            pt.push_back(cv::Vec2d(347,383));
            pt.push_back(cv::Vec2d(398,341));
            pt.push_back(cv::Vec2d( 67,423));
            pt.push_back(cv::Vec2d(550,422));
            pt.push_back(cv::Vec2d(  3,599));
            pt.push_back(cv::Vec2d(639,524));
            pt.push_back(cv::Vec2d(247,167));//ear
            pt.push_back(cv::Vec2d(242,184));
            pt.push_back(cv::Vec2d(252,244));
            pt.push_back(cv::Vec2d(464,205));
            pt.push_back(cv::Vec2d(452,249));
            pt.push_back(cv::Vec2d(452,262));
        }
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
        if (p>101|| q>101)
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
}

int main(int argc, char *argv[]) {
    if (argc < 6) {
        cout << "Usage: morph <source1_path> <source2_path> <destination_folder> <shape_predictor_68_face_landmarks.dat_path> <face_lines.csv_path>\n";
        return -1;
    }
    // load source as BGRA
    unsigned int length = 640, width = 640;
    Mat source1 = imread(argv[1], IMREAD_COLOR);
    Mat source2 = imread(argv[2], IMREAD_COLOR);
    resize(source1, source1, Size(width,length), 0, 0, INTER_AREA);
    resize(source2, source2, Size(width,length), 0, 0, INTER_AREA);
    Mat source1_a(length, width, CV_8UC4);
    Mat source2_a(length, width, CV_8UC4);
    cvtColor(source1, source1_a, CV_BGR2BGRA, 4);
    cvtColor(source2, source2_a, CV_BGR2BGRA, 4);
    cout<<"Images loaded."<<endl;
    // calc face lines
    fdots face1 = face_dots(source1, argv[4], 0);
    fdots face2 = face_dots(source2, 0,       1);
    if (face1.size() < 72 || face2.size() < 72)
        return -1;
    cout<<"Faces found."<<endl;
    fdots facmid;
    Mat dest1(length, width, CV_8UC4);
    Mat dest2(length, width, CV_8UC4);
    mkdir(argv[3],S_IRWXU|S_IRGRP|S_IXGRP|S_IROTH|S_IXOTH);
    init_llist(argv[5]);
    for (int i = 0; i <= 20; i++) {
        double alpha = 1.0*i/20;
        facmid.clear();
        for (int j = 0; j < 72; j++)
            facmid.push_back(face1[j]*(1-alpha) + face2[j]*alpha);
        // calculate destination
        Mat destination(length, width, CV_8UC4);
        morph(dest1, source1_a, gen_lines(face1, facmid));
        morph(dest2, source2_a, gen_lines(face2, facmid));
        addWeighted(dest1,(1-alpha),dest2,alpha,0,destination);
        char loc[100];
        sprintf(loc, "%s/%d.png", argv[3], i);
        imwrite(loc, destination);
        cout<<(i+1)<<" of 21 done, saved at "<<loc<<endl;
    }
    cout<<"Done."<<endl;
    return 0;
}