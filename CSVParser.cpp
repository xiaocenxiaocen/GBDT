#include <iostream>
#include <string>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>

using namespace std;
#include "CSVParser.h"
#include "TreeUtil.h"
#include "BinaryTree.h"
#include "DecisionTreeCART.h"

int CSVParser
(
	char* fileName, 
	vector<vector<float> >& trainData, 
	vector<int>& labelIndex, 
	vector<string>& features, 
	vector<enum feature_type>& featureType
)
{
	ifstream train(fileName, ios::in);

	string line;

	getline(train, line);

	Split(line, ",", &features);

	features.pop_back();

	featureType = vector<enum feature_type>(features.size());

	for(int i = 0; i < features.size(); i++) {
		if(features[i][0] == 'd') {
			featureType[i] = DISCRETE;
		} 
		else if(features[i][0] == 'c') {
			featureType[i] = CONTINUOUS;
		}
		features[i] = features[i].substr(1, string::npos);
	}

	while(getline(train, line)) {
		vector<string> ret;
		Split(line, ",", &ret);
		
		vector<float> row;
		for(int i = 0; i < ret.size(); i++) {
			float temp;
			sscanf((ret[i]).c_str(), "%f", &temp);
			row.push_back(temp);
		}
		int ii = static_cast<int>(*(row.end() - 1));
		labelIndex.push_back(ii);
		row.pop_back();
		trainData.push_back(row);
	}

	int numClass = (*max_element(labelIndex.begin(), labelIndex.end())) + 1;
#ifdef DEBUG
//	cerr << "number of class:" << numClass << endl;
//
//	cerr << "data frame:\n";
//	for(int i = 0; i < features.size(); i++) {
//		cerr << features[i] << " ";
//	}
//	cerr << "\n";
//
//	for(int i = 0; i < trainData.size(); i++) {
//		for(int j = 0; j < trainData[i].size() - 1; j++) {
//			cerr << trainData[i][j] << " ";
//		} // end for
//		cerr << labelIndex[i] << "\n";
//	} // end for
#endif
	
	return numClass;	
		
}

void CSVParser
(
	char* fileName, 
	vector<vector<float> >& trainData, 
	vector<float>& residual, 
	vector<string>& features, 
	vector<enum feature_type>& featureType
)
{
	ifstream train(fileName, ios::in);

	string line;

	getline(train, line);

	Split(line, ",", &features);

	features.pop_back();

	featureType = vector<enum feature_type>(features.size());

	for(int i = 0; i < features.size(); i++) {
		if(features[i][0] == 'd') {
			featureType[i] = DISCRETE;
		} 
		else if(features[i][0] == 'c') {
			featureType[i] = CONTINUOUS;
		}
		features[i] = features[i].substr(1, string::npos);
	}

	while(getline(train, line)) {
		vector<string> ret;
		Split(line, ",", &ret);
		
		vector<float> row;
		for(int i = 0; i < ret.size(); i++) {
			float temp;
			sscanf((ret[i]).c_str(), "%f", &temp);
			row.push_back(temp);
		}
		residual.push_back(row[row.size() - 1]);
		row.pop_back();
		trainData.push_back(row);
	}

	cerr << trainData.size() << "\n";

#ifdef DEBUG

//	cerr << "data frame:\n";
//	for(int i = 0; i < features.size(); i++) {
//		cerr << features[i] << " ";
//	}
//	cerr << "\n";
//
//	for(int i = 0; i < trainData.size(); i++) {
//		for(int j = 0; j < trainData[i].size() - 1; j++) {
//			cerr << trainData[i][j] << " ";
//		} // end for
//		cerr << residual[i] << "\n";
//	} // end for
#endif
	
	return;	
		
}
