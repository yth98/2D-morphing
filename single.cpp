#include <iostream>
#include <vector>
#include <cmath>
#include "opencv2/opencv.hpp"

using namespace cv;
using std::cout;
using std::cin;
using std::endl;
using std::vector;

class lPair {
public:
    Vec2d P_, Q_, P, Q;
    lPair(Vec2d p_, Vec2d q_, Vec2d p, Vec2d q) : P_(p_), Q_(q_), P(p), Q(q) {}
};

vector<lPair> pairs;

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

void morph(Mat &dest, Mat src) {
    pairs.push_back(lPair(Vec2d(0,0),Vec2d(0,1),Vec2d(0,0),Vec2d(0.2,0.8)));
    for (lPair pair : pairs){
        Vec2d PQ = pair.Q - pair.P;
        Vec2d P_Q_ = pair.Q_ - pair.P_;
        for (int y = 0; y < dest.size[0]; y++) {
            for (int x = 0; x < dest.size[1]; x++) {
                Vec2d X(x,y), PX = X-pair.P;
                double u = PX.dot(PQ) / PQ.dot(PQ);
                double v = PX.dot(Vec2d(-PQ[1],PQ[0])) / sqrt(PQ.dot(PQ));
                Vec2d X_ = (pair.P_*(1-u) + pair.Q_*u) + v*Vec2d(-P_Q_[1],P_Q_[0])/sqrt(P_Q_.dot(P_Q_));
                dest.at<Vec4b>(y,x) = get_pixel(X_[1], X_[0], src);
            }
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        cout << "Usage: single <source_path> <destination_path>\n";
        return -1;
    }
    // load source as BGRA
    Mat source = imread(argv[1], IMREAD_COLOR);
    Mat source_a(source.size[0], source.size[1], CV_8UC4);
    cvtColor(source, source_a, CV_BGR2BGRA, 4);
    // calculate destination
    Mat destination(source.size[0], source.size[1], CV_8UC4);
    morph(destination, source_a);
    imshow("Result", destination);
    waitKey();
    imwrite(argv[2], destination);
    cout<<"Done."<<endl;
    return 0;
}