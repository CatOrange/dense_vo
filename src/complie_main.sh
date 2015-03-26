rm main
g++ main_linux.cpp -fopenmp -Wall -std=c++11 -Ofast -pipe -march=native -mtune=native -I `pkg-config --libs --cflags opencv` -o main
./main
