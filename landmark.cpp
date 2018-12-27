#include <iostream>
#include "dlib/gui_widgets.h"
#include "dlib/image_io.h"
#include "dlib/image_processing/frontal_face_detector.h"

using namespace dlib;
using std::cout;
using std::cin;
using std::endl;

int main(int argc, char *argv[]) {
    if (argc < 2) {
        cout << "Usage: landmark <image_path>\n";
        return -1;
    }


    array2d<unsigned char> img;
    load_image(img, argv[1]);

    frontal_face_detector detector = get_frontal_face_detector();
    std::vector<rectangle> dets = detector(img);
    cout << "Number of faces detected: " << dets.size() << endl;

    image_window win;
    win.clear_overlay();
    win.set_image(img);
    win.add_overlay(dets, rgb_pixel(255,0,0));

    cout << "Hit enter..." << endl;
    cin.get();

    return 0;
}