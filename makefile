main: main.cpp NeuralNetwork.h NeuralNetwork.o layer.h layer.o
	g++ -o main main.cpp NeuralNetwork.o layer.o -Wall

NeuralNetwork.o: NeuralNetwork.h NeuralNetwork.cpp
	g++ -c NeuralNetwork.cpp -Wall

layer.o: layer.h layer.cpp
	g++ -c layer.cpp -Wall

clean:
	rm main *.o