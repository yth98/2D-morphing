g++ -o multi multi.cpp `pkg-config opencv --cflags --libs`
./multi img/people_bunch.jpg ./output.jpg