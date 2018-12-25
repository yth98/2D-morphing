g++ -o single single.cpp `pkg-config opencv --cflags --libs`
./single img/people_bunch.jpg ./output.jpg