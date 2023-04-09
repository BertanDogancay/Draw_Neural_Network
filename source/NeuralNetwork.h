#pragma once
#include "Matrix.h"
#include <cstdlib>

inline float Sigmoid(float x) {
	return 1.0 / (1 + exp(-x));
}

inline float DSigmoid(float x) {
	return x * (1 - x);
}

class NeuralNetwork {
public:
	std::vector<uint32_t> _topology; //number of neurons at each layer
	std::vector<Matrix> _weightMatrices;
	std::vector<Matrix> _valueMatrices;
	std::vector<Matrix> _biasMatrices;
	float _learningRate;
public:
	NeuralNetwork(std::vector<uint32_t> topology, float learningRate = 0.1f);
	bool feedForward(std::vector<float> input);
	bool backPropagate(std::vector<float> targetOutput);
	std::vector<float> getPrediction();
};