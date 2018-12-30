g++ -o single single.cpp `pkg-config opencv --cflags --libs`
g++ -o multi multi.cpp `pkg-config opencv --cflags --libs`
#g++ -o landmark landmark.cpp `pkg-config dlib-1 --cflags --libs` -lpthread -lX11 -ljpeg
g++ -o morph morph.cpp `pkg-config --cflags --libs opencv` -ldlib -lpthread -lX11 -ljpeg
g++ -o morphing morph_ing.cpp `pkg-config --cflags --libs opencv` -ldlib -lpthread -lX11 -ljpeg
#./landmark ./img/person_1/image004.jpg python/shape_predictor_68_face_landmarks.dat
./morph "./img/person_1/Ming Ouhyoung.png" ./img/person_1/unnamed.jpg ./morph.png python/shape_predictor_68_face_landmarks.dat face_lines.csv
./morphing "./img/person_1/Ming Ouhyoung.png" ./img/person_1/unnamed.jpg ./trans2/ python/shape_predictor_68_face_landmarks.dat face_lines.csv