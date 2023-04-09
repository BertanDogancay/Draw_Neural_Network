#pragma once
#include <vector>
#include <cmath>
#include <cassert>
#include <functional>

class Matrix {
public:
	uint32_t _rows;
	uint32_t _cols;
	std::vector<float> _vals;
public:
	Matrix();
	Matrix(uint32_t rows, uint32_t cols);
	Matrix applyFunction(std::function<float(const float&)> func);
	float& at(uint32_t row, uint32_t col);
	Matrix multiply(Matrix& target);
	Matrix multiplyScaler(float sVal);
	Matrix multiplyElements(Matrix& target);
	Matrix add(Matrix& target);
	Matrix addScaler(float sVal);
	Matrix negative();
	Matrix transpose();
};
