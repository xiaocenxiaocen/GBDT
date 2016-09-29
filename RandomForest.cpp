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
#include "DecisionTree.h"

class RandomForest {
private:
	friend class DecisionTree;
	int k; /* number of trees */
	int numClass; 
	DecisionTree* forest;
	std::vector<std::string> label;
	std::vector<std::vector<int> > labelList;

public:
	RandomForest(): k(0), forest(NULL) {}
	RandomForest(const std::vector<std::string>& _label, const int _n) {
		k = 0;
		forest = NULL;
		label = _label;
		numClass = _n;
	}
	~RandomForest();
	void CreateForest(const std::vector<std::string> fileList);
	void Predictor(const std::vector<std::vector<float> >& test, std::vector<int>& result);	
};

RandomForest::~RandomForest()
{
	if(k==1) delete forest;
	else delete[] forest;

}

void RandomForest::CreateForest(const std::vector<std::string> fileList)
{
	k = fileList.size();

	if(k==1) forest = new DecisionTree;
	else forest = new DecisionTree[k];
	
	labelList = std::vector<std::vector<int> >(k);

	for(size_t i_file = 0; i_file < k; i_file++) {
		forest[i_file].DataParser(fileList[i_file]);

		for(size_t i = 0; i < forest[i_file].label.size(); i++) {
			int tmp;

			std::vector<std::string>::iterator ite = std::find(label.begin(), label.end(), forest[i_file].label[i]);
			
			tmp = static_cast<int>(ite - label.begin());

			labelList[i_file].push_back(tmp);
		}

		forest[i_file].TreeGrowth();

		forest[i_file].Prune();
		
		forest[i_file].Print();		
	}

	return;
}

void RandomForest::Predictor(const std::vector<std::vector<float> >& test, std::vector<int>& result)
{
	std::vector<std::vector<int> > result_trees(test.size(), std::vector<int>(numClass, 0));
 
	for(size_t i = 0; i < k; i++) {
		std::vector<std::vector<float> > test_tmp;
		std::vector<int> classIdx;
		for(size_t j = 0; j < test.size(); j++) {
			std::vector<float> tmp;
			for(size_t kk = 0; kk < labelList[i].size(); kk++) {
				tmp.push_back(test[j][labelList[i][kk]]);
			}
			test_tmp.push_back(tmp);
		}
		forest[i].Predictor(test_tmp, classIdx);

		for(size_t j = 0; j < test_tmp.size(); j++) {
			result_trees[j][classIdx[j]]++;
		}
	}

	result.resize(test.size());
	for(size_t j = 0; j < test.size(); j++) {
		result[j] = std::max_element(result_trees[j].begin(), result_trees[j].end()) - result_trees[j].begin();
	}
}

int main(int argc, char* argv[])
{
	std::ifstream input(argv[1], std::ios::in);
	std::ifstream label(argv[2], std::ios::in);

	std::string line;
	std::vector<std::string> labelList;

	std::getline(label, line);

	Split(line, " ", &labelList);

	std::vector<std::string> fileList;

	while(std::getline(input, line)) {
		fileList.push_back(line);		
	}

	int num;
	sscanf(argv[3], "%d", &num);

	RandomForest forest = RandomForest(labelList, num);

	forest.CreateForest(fileList);

	std::ifstream testData(argv[4], std::ios::in);
	
	std::vector<std::vector<float> > test;

	std::getline(testData, line);

	std::vector<int> result;

	while(std::getline(testData, line)) {
		std::vector<std::string> ret;
		Split(line, " ", &ret);
		std::vector<float> rowData;
		for(size_t id = 0; id < ret.size(); id++) {
			float tmp;
			sscanf((ret[id]).c_str(), "%f", &tmp);
			rowData.push_back(tmp);
		}
		test.push_back(rowData);
	}
	
	forest.Predictor(test, result);

	std::ofstream output("result.txt", std::ios::out);

	for(size_t i = 0; i < result.size(); i++) {
		output << result[i] << "\n";
	}
	output << std::flush;
	output.close();

	
	label.close();
	input.close();
	testData.close();

	return 0;
}
