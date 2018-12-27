#include <iostream>
#include "dlib/gui_widgets.h"
#include "dlib/image_io.h"
#include "dlib/image_processing.h"
#include "dlib/image_processing/frontal_face_detector.h"
#include "dlib/image_processing/render_face_detections.h"

using namespace dlib;
using std::cout;
using std::cin;
using std::endl;

int main(int argc, char *argv[]) {
    if (argc < 3) {
        cout << "Usage: landmark <image_path> <shape_predictor_68_face_landmarks.dat_path>\n";
        return -1;
    }

    frontal_face_detector detector = get_frontal_face_detector();
    shape_predictor predictor;
    deserialize(argv[2]) >> predictor;

    array2d<unsigned char> img;
    load_image(img, argv[1]);

    std::vector<rectangle> dets = detector(img);
    cout << "Number of faces detected: " << dets.size() << endl;

    std::vector<full_object_detection> faces;
    for (unsigned int i = 0; i < dets.size(); ++i) {
        full_object_detection shape = predictor(img, dets[i]);
        cout << "number of parts: "<< shape.num_parts() << endl;
        faces.push_back(shape);
    }

    image_window win;
    win.clear_overlay();
    win.set_image(img);
    win.add_overlay(dets, rgb_pixel(255,0,0));
    win.add_overlay(render_face_detections(faces));

    cout << "Hit enter..." << endl;
    cin.get();

    return 0;
}