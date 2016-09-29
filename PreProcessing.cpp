#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <ctime>
#include <vector>
#include <string>
#include <cstring>
#include <cmath>
#include <algorithm>

#include "util.h"

int main(int argc, char* argv[])
{
	/* read input data from command line */
	std::ifstream data(argv[1], std::ios::in);

	std::vector<std::vector<float> > train;
	
	std::string line;

	std::getline(data, line);

	std::vector<std::string> label;

	Split(line, " ", &label);

	label.pop_back();	

	while(std::getline(data, line)) {
		std::vector<std::string> ret;
		Split(line, " ", &ret);
		std::vector<float> rowData;
		for(size_t id = 0; id < ret.size(); id++) {
			float tmp;
			sscanf((ret[id]).c_str(), "%f", &tmp);
			rowData.push_back(tmp);
		}
		train.push_back(rowData);
	}

	data.close();

#ifdef DEBUG
	std::cerr << "all of training data:\n";

	for(size_t row = 0; row < label.size(); row++) {
		std::cerr << label[row] << " ";
	}
	std::cerr << "\n";

	for(size_t row = 0; row < 5; row++) {
		for(size_t col = 0; col < train[row].size(); col++) {
			std::cerr << train[row][col] << " ";
		} 
		std::cerr << "\n";
	}
#endif

	/* select train set */
	int numTrain;
	int numTree;	

	sscanf(argv[2], "%d", &numTrain);
	sscanf(argv[3], "%d", &numTree);

	int M = label.size();
	int N = train.size();

	int numFeatures = 2 * static_cast<int>(std::sqrt(M));

	for(int s = 0; s < numTree; s++) {
		int* A = new int[M];
		for(int i = 0; i < M; i++) {
			A[i] = i;
		}

		sleep(5);	
		srand(unsigned(time(0)));
	
		RandomPerm(A, numFeatures, M);
	
		std::cerr << "Select feature labels:" << "\n";
		for(int i = 0; i < numFeatures; i++) {
			std::cerr << label[A[i]] << " ";
		} 
		std::cerr << "\n";
	
		std::vector<std::vector<float> > trainSet;
	
		std::cerr << "Generating Train Set\n";
	
		for(int i = 0; i < numTrain; i++) {
			
			std::vector<float> trainData;
	
		//	do {
		//		srand((unsigned)time(NULL));
				int dataIdx = rand() % N;
		
				for(int j = 0; j < numFeatures; j++) {
					trainData.push_back(train[dataIdx][A[j]]);
				}
		
				trainData.push_back(train[dataIdx][M]);
		//	} while(*std::max_element(trainData.begin(), trainData.end()-1) > 0);
	
			trainSet.push_back(trainData);
		}
	
	
		std::cerr << "End generating train set\n";
	
#ifdef DEBUG
		std::cerr << "train set 1:\n";
	
		for(size_t i = 0; i < numFeatures; i++) {
			std::cerr << label[A[i]] << " ";
		}
		std::cerr << "\n";
	
		for(size_t row = 0; row < trainSet.size(); row++) {
			for(size_t col = 0; col < trainSet[row].size(); col++) {
				std::cerr << trainSet[row][col] << " ";
			} 
			std::cerr << "\n";
		}
#endif

		char fileName[20];
		sprintf(fileName, "train_set_%d", s);

		std::ofstream trainOut(fileName, std::ios::out);

		for(int i = 0; i < numFeatures; i++) {
			trainOut << "c" << label[A[i]] << " ";
		};
		trainOut << "label" << "\n";
		

		for(size_t row = 0; row < trainSet.size(); row++) {
			for(size_t col = 0; col < trainSet[row].size() - 1; col++) {
				trainOut << trainSet[row][col] << " ";
			}
			trainOut << trainSet[row][numFeatures] << "\n";
		}
		trainOut << std::flush;
		trainOut.close();	

		delete[] A;
	}

	return 0;
}
