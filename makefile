main: main.cpp layer.o
	g++ -o main main.cpp layer.o -Wall

layer.o: layer.h layer.cpp
	g++ -c layer.cpp -Wall
clean:
	rm main *.o