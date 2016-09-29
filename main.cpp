#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <math.h>

using namespace std;

#include "CSVParser.h"
#include "TreeUtil.h"
#include "BinaryTree.h"
#include "DecisionTreeCART.h"

int main(int argc, char* argv[])
{
	vector<string> features;
	vector<enum feature_type> featureType;
//	vector<int> labelIndex;
	vector<float> predictValue;
	vector<vector<float> > train;

//	int numClass = CSVParser(argv[1], train, labelIndex, features, featureType);

	CSVParser(argv[1], train, predictValue, features, featureType);

	DecisionTreeCART tree(features, featureType, 10, 1, 0.01, 1.0, 1.0, 0);

	tree.TreeGrowth(train, predictValue);

	tree.Print();
	
	vector<vector<float> > testData;

	ifstream test(argv[2], ios::in);
	
	string line;

	getline(test, line);

	vector<float> result;

	while(getline(test, line)) {
		vector<string> ret;
		Split(line, ",", &ret);
		vector<float> row;
		for(int i = 0; i < ret.size(); i++) {
			float tmp;
			sscanf((ret[i]).c_str(), "%f", &tmp);
			row.push_back(tmp);
		}
		testData.push_back(row);
	}

	cerr << "testing data:\n";
	
	for(int i = 0; i < testData.size(); i++) {
		for(int j = 0; j < testData[i].size(); j++) {
			cerr << testData[i][j] << " ";
		} 
		cerr << "\n";
	}

	tree.Predictor(testData, result);

	int correctNum = 0;

	int classIdx = testData[0].size() - 1;
	for(size_t i = 0; i < testData.size(); i++) {
		float label = 0;
		if(result[i] > 0.5) label = 1;
		if(static_cast<int>(testData[i][classIdx] + 0.5) == label) correctNum++;
	}

	cout << "accurate rate:" << (float)(correctNum) / (float)(testData.size()) << endl;

//	tree.Prune(labelIndex);
//	
//	tree.Print();
//
//	result.clear();
//
//	tree.Predictor(testData, result);
//
//	correctNum = 0;
//
//	for(size_t i = 0; i < testData.size(); i++) {
//		if(static_cast<int>(testData[i][classIdx] + 0.5) == result[i]) correctNum++;
//	}
//
//	cout << "accurate rate:" << (float)(correctNum) / (float)(testData.size()) << endl;
//
////	tree.Delete();
//
//	return 0;
}
