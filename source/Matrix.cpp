#include <vector>
#include <cmath>
#include <cassert>
#include <functional>

#include "Matrix.h"

Matrix::Matrix()
	:_rows(0),
	_cols(0),
	_vals({})
{

}

Matrix::Matrix(uint32_t rows, uint32_t cols)
	:_rows(rows),
	_cols(cols),
	_vals({})
{
	_vals.resize(rows * cols, 0.0f);
}

Matrix Matrix::applyFunction(std::function<float(const float&)> func) {
	Matrix output(_rows, _cols);
	for (uint32_t i = 0; i < output._rows; i++) {
		for (uint32_t j = 0; j < output._cols; j++) {
			output.at(i, j) = func(at(i, j));
		}
	}
	return output;
}

float& Matrix::at(uint32_t row, uint32_t col) {
	return _vals[row * _cols + col];
}

Matrix Matrix::multiply(Matrix& target) {
	assert(_cols == target._rows);
	Matrix output(_rows, target._cols);
	for (uint32_t i = 0; i < output._rows; i++) {
		for (uint32_t j = 0; j < output._cols; j++) {
			float result = 0.0f;
			for (uint32_t k = 0; k < _cols; k++) {
				result += at(i, k) * target.at(k, j);
			}
			output.at(i, j) = result;
		}
	}
	return output;
}

Matrix Matrix::multiplyScaler(float sVal) {
	Matrix output(_rows, _cols);
	for (uint32_t i = 0; i < output._rows; i++) {
		for (uint32_t j = 0; j < output._cols; j++) {
			output.at(i, j) = at(i, j) * sVal;
		}
	}
	return output;
}

Matrix Matrix::multiplyElements(Matrix& target) {
	assert(_rows = target._rows && _cols == target._cols);
	Matrix output(_rows, _cols);
	for (uint32_t i = 0; i < output._rows; i++) {
		for (uint32_t j = 0; j < output._cols; j++) {
			output.at(i, j) = at(i, j) * target.at(i, j);
		}
	}
	return output;
}

Matrix Matrix::add(Matrix& target) {
	assert(_rows == target._rows && _cols == target._cols);
	Matrix output(_rows, _cols);
	for (uint32_t i = 0; i < output._rows; i++) {
		for (uint32_t j = 0; j < output._cols; j++) {
			output.at(i, j) = at(i, j) + target.at(i, j);
		}
	}
	return output;
}

Matrix Matrix::addScaler(float sVal) {
	Matrix output(_rows, _cols);
	for (uint32_t i = 0; i < output._rows; i++) {
		for (uint32_t j = 0; j < output._cols; j++) {
			output.at(i, j) = at(i, j) + sVal;
		}
	}
	return output;
}

Matrix Matrix::negative() {
	Matrix output(_rows, _cols);
	for (uint32_t i = 0; i < output._rows; i++) {
		for (uint32_t j = 0; j < output._cols; j++) {
			output.at(i, j) = -at(i, j);
		}
	}
	return output;
}

Matrix Matrix::transpose() {
	Matrix output(_cols, _rows);
	for (uint32_t i = 0; i < _rows; i++) {
		for (uint32_t j = 0; j < _cols; j++) {
			output.at(j, i) = at(i, j);
		}
	}
	return output;
}