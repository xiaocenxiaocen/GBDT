#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <cstring>
#include <cmath>

using namespace std;

#include "CSVParser.h"
#include "TreeUtil.h"
#include "BinaryTree.h"
#include "DecisionTreeCART.h"

class GBDT {
private:
	/* number of trees */
	int k; 
	/* number of class */
	int numClass;
	/* random forest */
	vector<DecisionTreeCART> forest;
	vector<string> features;
	vector<enum feature_type> featureType;
	int maxDepth;
	int minItems;
	float descentThreshold;

	/* regular parameter */
	float lambda;

	/* shrinkage parameter a.e. learning rate */
	float nu;

	enum loss_function_type lossFunction;
				
public:
	GBDT(
		int _k, 									/* number of regression trees */ 
		vector<string> _features,					/* name of features 		  */
		vector<enum feature_type> _feature_type,	/* features' type  			  */
		int _max_depth,								/* max depth of trees 		  */
		int _min_items,								/* if the number of items in 
													 * a tree node is less than 
													 * minItems, the splitting of
													 * tree node will stop 
													 */
		float _descent_threshold,					/* the information gain threshold
													 * of tree node splitting
													 */
		float _lambda,								/* the regular parameter of 
													 * regression tree
													 */
		float _nu,									/* the shrinkage parameter	   */
		enum loss_function_type _loss_function = SQUARE_LOSS
													/* the loss function's type, the
 													 * the defaut value is SQUARE_LOSS
 													 * i.e. square loss function
 													 */
		);
	~GBDT() { };
	void ModelFit(const vector<vector<float> >& trainData, const vector<float>& predictValue);
	void Predictor(const vector<vector<float> >& testData, vector<float>& predictValue);
	
};

GBDT::GBDT(
	int _k,
	vector<string> _features,
	vector<enum feature_type> _feature_type,
	int _max_depth,
	int _min_items,
	float _descent_threshold,
	float _lambda,
	float _nu,
	enum loss_function_type _loss_function)
{
	k = _k;

	features = _features;

	featureType = _feature_type;

	maxDepth = _max_depth;
	
	minItems = _min_items;

	descentThreshold = _descent_threshold;

	lambda = _lambda;

	nu = _nu;

	lossFunction = _loss_function;

	for(int i = 0; i < k; i++) {
		forest.push_back(DecisionTreeCART(
			features,
			featureType,
			maxDepth,
			minItems,
			descentThreshold,
			lambda,
			nu,
			0,
			lossFunction));
	}
}

void GBDT::ModelFit(const vector<vector<float> >& trainData, const vector<float>& predictValue)
{
	vector<float> residual;

	if(lossFunction == SQUARE_LOSS) residual = vector<float>(predictValue);
	if(lossFunction == LOGIT_LOSS) residual = vector<float>(predictValue.size(), 0.0f);

	for(int i = 0; i < k; i++) {
		vector<float> predict;
		vector<float> gradient;	

		if(lossFunction == SQUARE_LOSS) {
			gradient = vector<float>(residual);
		}

		if(lossFunction == LOGIT_LOSS) {
			for(int j = 0; j < residual.size(); j++) {
				float y = -1.0;
				if(static_cast<int>(predictValue[j] + 0.5) == 1) y = 1.0;
				gradient.push_back(2.0 * y / ( 1.0 + exp(2.0 * y * residual[j]) ));
			}			
		}

		forest[i].TreeGrowth(trainData, gradient);

		cout << "*****************************************\n";
		cout << "*****************************************\n";
		cout << "*****************************************\n";
		cout << "Regression Tree " << i << ":\n";

		forest[i].Print();

		cout << "*****************************************\n";
		cout << "*****************************************\n";
		cout << "*****************************************\n";

		forest[i].Predictor(trainData, predict);

		if(lossFunction == SQUARE_LOSS) {
			for(int j = 0; j < trainData.size(); j++) {
				residual[j] -= predict[j];
			}
		}

		if(lossFunction == LOGIT_LOSS) {
			for(int j = 0; j < trainData.size(); j++) {
				residual[j] += predict[j];
			}
		}
	}

	return;
}

void GBDT::Predictor(const vector<vector<float> >& testData, vector<float>& predictValue)
{
	predictValue.resize(testData.size());

	memset(&predictValue[0], 0, sizeof(float) * predictValue.size());

	for(int i = 0; i < k; i++) {
		vector<float> predict;
		forest[i].Predictor(testData, predict);

		for(int j = 0; j < testData.size(); j++) {
			predictValue[j] += predict[j];
		} // end for
	} // end for
}

int main(int argc, char* argv[])
{
	vector<string> features;
	vector<enum feature_type> featureType;
	vector<float> predictValue;
	vector<vector<float> > train;

	CSVParser(argv[1], train, predictValue, features, featureType);

	GBDT gbdt(
		500, 			/* number of regression trees */ 
		features,		/* name of features 		  */
		featureType,	/* features' type  			  */
		7,				/* max depth of trees 		  */
		1,				/* if the number of items in 
						 * a tree node is less than 
						 * minItems, the splitting of
						 * tree node will stop 
						 */
		0.1,			/* the minimum 
						 * information gain threshold
						 * of tree node splitting
						 */
		0.01,			/* the regular parameter of 
						 * regression tree
						 */
		0.025,			/* the shrinkage parameter	   */
		LOGIT_LOSS		/* the loss function's type, the
 						 * the defaut value is SQUARE_LOSS
 						 * i.e. square loss function
 						 */
		);

	gbdt.ModelFit(train, predictValue);

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

	gbdt.Predictor(testData, result);

//	ofstream output("result.csv",ios::out);
//
//	for(int i = 0; i < result.size(); i++) {
//		int label = 0;
//		float p = 1.0 / ( 1.0 + exp( - 2.0 * 1 * result[i] ) );
//		if( p > 0.5 ) label = 1;
////		if(result[i] > 0.5) label = 1;
//		output << label << "\n";
//	}	
//	output.flush();
//	output.close();
	
	int correctNum = 0;

	int labelIdx = testData[0].size() - 1;
	for(size_t i = 0; i < testData.size(); i++) {
		int label = 0;
		float p = 1.0 / ( 1.0 + exp( - 2.0 * 1 * result[i] ) );
		if(p >= 0.5) label = 1;
	//	if(result[i] >= 0.5 && result[i] <= 1.5) label = 1;
	//	if(result[i] >= 1.5) label = 2;
		if(static_cast<int>(testData[i][labelIdx] + 0.5) == label) correctNum++;
		cerr << p << " " << testData[i][labelIdx] << "\n";
	}

	cout << "accurate rate:" << (float)(correctNum) / (float)(testData.size()) << endl;

}

