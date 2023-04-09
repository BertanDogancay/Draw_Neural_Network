#include <cstdlib>
#include <iostream>

#include "NeuralNetwork.h"
#include "Matrix.h"

NeuralNetwork::NeuralNetwork(std::vector<uint32_t> topology, float learningRate)
	:_topology(topology),
	_weightMatrices({}),
	_valueMatrices({}),
	_biasMatrices({}),
	_learningRate(learningRate)
{
	for (uint32_t i = 0; i < topology.size() - 1; i++) {
		Matrix weightMatrix(topology[i], topology[i + 1]);
		weightMatrix = weightMatrix.applyFunction([](const float& f) {
			return (float)rand() / RAND_MAX;
			});
		_weightMatrices.push_back(weightMatrix);

		Matrix biasMatrix(1, topology[i + 1]);
		biasMatrix = biasMatrix.applyFunction([](const float&) {
			return (float)rand() / RAND_MAX;
			});
		_biasMatrices.push_back(biasMatrix);
	}
	_valueMatrices.resize(topology.size());
}

bool NeuralNetwork::feedForward(std::vector<float> input) {
	if (input.size() != _topology[0]) {
		std::cout << "error: wrong number of inputs. requires 2" << std::endl;
		return false;
	}

	Matrix values(1, input.size());
	for (uint32_t i = 0; i < input.size(); i++) {
		values._vals[i] = input[i];
	}

	for (uint32_t i = 0; i < _weightMatrices.size(); i++) {
		_valueMatrices[i] = values;
		values = values.multiply(_weightMatrices[i]);
		values = values.add(_biasMatrices[i]);
		values = values.applyFunction(Sigmoid);
	}

	_valueMatrices[_weightMatrices.size()] = values;
	return true;
}

bool NeuralNetwork::backPropagate(std::vector<float> targetOutput) {
	if (targetOutput.size() != _topology.back()) {
		std::cout << "error: wrong number of outputs. requires 1" << std::endl;
		return false;
	}

	Matrix errors(1, targetOutput.size());
	errors._vals = targetOutput;

	Matrix sub = _valueMatrices.back().negative();
	errors = errors.add(sub);

	for (int i = _weightMatrices.size() - 1; i >= 0; i--) {
		Matrix trans = _weightMatrices[i].transpose();
		Matrix prevErrors = errors.multiply(trans);
		Matrix dOutput = _valueMatrices[i + 1].applyFunction(DSigmoid);
		Matrix gradients = errors.multiplyElements(dOutput);
		gradients = gradients.multiplyScaler(_learningRate);
		Matrix weightGradients = _valueMatrices[i].transpose().multiply(gradients);
		_weightMatrices[i] = _weightMatrices[i].add(weightGradients);
		_biasMatrices[i] = _biasMatrices[i].add(gradients);
		errors = prevErrors;
	}
	true;
}

std::vector<float> NeuralNetwork::getPrediction() {
	return _valueMatrices.back()._vals;
}