#g++ -o single single.cpp `pkg-config opencv --cflags --libs`
#g++ -o multi multi.cpp `pkg-config opencv --cflags --libs`
g++ -o landmark landmark.cpp `pkg-config dlib-1 --cflags --libs` -lpthread -lX11 -ljpeg
./landmark ./img/person_1/image004.jpg