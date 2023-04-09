#include <iostream>
#include <sstream>
#include <chrono>
#include <thread>
#include <string>
#include <algorithm>
#include <SFML/Graphics.hpp>

#include "NeuralNetwork.h"

int main() {
	sf::RenderWindow window(sf::VideoMode(2000, 1000), "Frame");

	std::vector<uint32_t> topology = { 4, 8, 1 };
	NeuralNetwork nn(topology, 0.1);

	std::vector<std::vector<float>> targetInputs{
		{0.0f, 0.0f, 0.0f, 0.0f},
		{0.0f, 0.0f, 0.0f, 1.0f},
		{0.0f, 0.0f, 1.0f, 0.0f},
		{0.0f, 0.0f, 1.0f, 1.0f},
		{0.0f, 1.0f, 0.0f, 0.0f},
		{0.0f, 1.0f, 0.0f, 1.0f},
		{0.0f, 1.0f, 1.0f, 0.0f},
		{0.0f, 1.0f, 1.0f, 1.0f},
		{1.0f, 0.0f, 0.0f, 0.0f},
		{1.0f, 0.0f, 0.0f, 1.0f},
		{1.0f, 0.0f, 1.0f, 0.0f},
		{1.0f, 0.0f, 1.0f, 1.0f},
		{1.0f, 1.0f, 0.0f, 0.0f},
		{1.0f, 1.0f, 0.0f, 1.0f},
		{1.0f, 1.0f, 1.0f, 0.0f},
		{1.0f, 1.0f, 1.0f, 1.0f}
	};

	std::vector<std::vector<float>> targetOutputs{
		{0.0f},
		{1.0f},
		{1.0f},
		{0.0f},
		{0.0f},
		{1.0f},
		{1.0f},
		{0.0f},
		{1.0f},
		{0.0f},
		{0.0f},
		{1.0f},
		{1.0f},
		{0.0f},
		{0.0f},
		{1.0f},
	};
	
	std::vector<std::vector<sf::CircleShape>> circles;
	std::vector<std::vector<sf::Vertex>> lines;
	std::vector<sf::Vertex> line;
	std::vector<sf::Text> numbers;
	const float radius = 30.f;
	float weightMedian = 0.0f;
	float totalWeight = 0.0f;
	float minValue = 0.0f;
	float maxValue = 0.0f;
	float alpha;
	int step = 0;

	circles.resize(topology.size());

	for (int i = 0; i <= circles.size() - 1; i++) {
		for (int j = 1; j <= topology[i]; j++) {
			sf::CircleShape circle(radius);
			circle.setFillColor(sf::Color::Black);
			circle.setPosition(i * (window.getSize().x / topology.size()) + (window.getSize().x / (2 * topology.size())) - radius, window.getSize().y - j *
				(window.getSize().y / topology[i]) + (window.getSize().y / (2 * topology[i])) - radius);
			circles[i].push_back(circle);
		}
	}

	for (int i = 0; i < topology.size() -1; i++) {
		for (int j = 0; j < topology[i]; j++) {
			for (int k = 0; k < topology[i + 1]; k++) {
				line.push_back(sf::Vertex(circles[i][j].getPosition() + sf::Vector2f(radius, radius), sf::Color::Red));
				line.push_back(sf::Vertex(circles[i + 1][k].getPosition() + sf::Vector2f(radius, radius), sf::Color::Red));
				lines.push_back(line);
				line.clear();
			}
		}
	}

	sf::Font font;
	if (!font.loadFromFile("arial.ttf")) {
		return 1;
	}

	for (int i = 0; i < circles.size(); i++) {
		for (int j = 0; j < circles[i].size(); j++) {
			sf::Text text;
			std::ostringstream ss;
			ss << 1.0f;
			text.setString(ss.str());
			text.setFont(font);
			text.setCharacterSize(24);
			text.setFillColor(sf::Color::White);
			text.setPosition(circles[i][j].getPosition() + sf::Vector2f(radius - text.getLocalBounds().width / 2.f,
				radius - text.getLocalBounds().height / 2.f));
			numbers.push_back(text);
		}
	}
	
	while (window.isOpen()) {
		sf::Event event;
		totalWeight = 0.0f;

		while (window.pollEvent(event)) {
			if (event.type == sf::Event::Closed)
				window.close();
		}
		window.clear();

		uint32_t index = rand() % targetInputs.size();
		nn.feedForward(targetInputs[index]);
		nn.backPropagate(targetOutputs[index]);
		nn.feedForward(targetInputs[index]);

		for (uint32_t i = 0; i < nn._valueMatrices.size(); i++) {
			Matrix matrix(nn._valueMatrices[i]._rows, nn._valueMatrices[i]._cols);
			matrix = nn._valueMatrices[i];
			for (uint32_t j = 0; j < matrix._cols; j++) {
				std::ostringstream ss;
				ss << matrix.at(0, j);
				if (i == 0) 
					numbers[j].setString(ss.str());
				else {
					uint32_t sum = 0;
					for (uint32_t k = 0; k < i; k++) {
						sum += topology[k];
					}
					numbers[sum + j].setString(ss.str());
				}
			}
		}
		
		for (uint32_t i = 0; i < nn._weightMatrices.size(); i++) {
			Matrix matrix(nn._weightMatrices[i]._rows, nn._weightMatrices[i]._cols);
			matrix = nn._weightMatrices[i];
			for (uint32_t j = 0; j < matrix._rows; j++) {
				for (uint32_t k = 0; k < matrix._cols; k++) {
					maxValue = 2.0f * weightMedian;
					minValue = -1.0f;
					alpha = (matrix.at(j, k) - minValue) / (maxValue - minValue);
					totalWeight += matrix.at(j, k);
					sf::Color lineColor;
					if (matrix.at(j, k) > weightMedian)
						lineColor = sf::Color(237, 116, 9, 255 * alpha);
					else
						lineColor = sf::Color(13, 162, 255, 255 * alpha);
					if (i == 0) {
						lines[j * matrix._cols + k][0].color = lineColor;
						lines[j * matrix._cols + k][1].color = lineColor;
					}
					else {
						uint32_t sum = 0;
						for (uint32_t x = 0; x < i; x++) {
							sum += nn._weightMatrices[x]._rows * nn._weightMatrices[x]._cols;
						}
						lines[j * matrix._cols + k + sum][0].color = lineColor;
						lines[j * matrix._cols + k + sum][1].color = lineColor;
					}
				}
			}
		}

		weightMedian = totalWeight / (float)lines.size();

		sf::Text text("Step: " + std::to_string(step), font, 35);
		text.setPosition(sf::Vector2f(window.getSize().x - 300,  50));
		window.draw(text);

		for (const auto& circle : circles) {
			for (const auto& c : circle) {
				window.draw(c);
			}
		}

		for (const auto& number : numbers) {
			window.draw(number);
		}

		for (int i = 0; i < lines.size(); i++) {
			window.draw(&lines[i][0], lines.size(), sf::Lines);
		}

		window.display();
		step++;
	}

	return 0;
}