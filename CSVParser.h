#ifndef CSVPARSER_H
#define CSVPARSER_H

enum feature_type{DISCRETE, CONTINUOUS};

int CSVParser
(
	char* fileName, 
	vector<vector<float> >& trainData, 
	vector<int>& labelIndex, 
	vector<string>& features, 
	vector<enum feature_type>& featureType
);

void CSVParser
(
	char* fileName, 
	vector<vector<float> >& trainData, 
	vector<float>& residual, 
	vector<string>& features, 
	vector<enum feature_type>& featureType
);

#endif
