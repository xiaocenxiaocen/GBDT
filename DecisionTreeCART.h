#ifndef DECISIONTREECART_H
#define DECISIONTREECART_H

static const double CHI[18] = {0.004,0.103,0.352,0.711,1.145,1.635,2.167,2.733,3.325,3.94,4.575,5.226,5.892,6.571,7.261,7.962};

enum loss_function_type{SQUARE_LOSS, LOGIT_LOSS};

class DecisionTreeCART : public BinaryTree {
private:
//	friend class RandomForest;
//	friend class GBDT;

	int numFeatures;
	vector<string> features;
	vector<enum feature_type> featureType;
	node* root;
	int numClass;
	int maxDepth;
	int minItems;
	float descentThreshold;	
	
	/* regular parameter */
	float lambda;
	
	/* shrinkage parameter a.e. learning rate */
	float nu;

	enum loss_function_type lossFunction;

	bool FindBestSplit(const vector<vector<float> >& trainData, const vector<float>& residual, node* curNode);
	void FindBestSplit(const vector<vector<float> >& trainData, const vector<int>& labelIndex, node* curNode);
	void Classify(node* curNode, const vector<int>& labelIndex);
	void Evaluate(node* curNode, const vector<float>& residual);
	bool StoppingCondition(node* curNode);
	bool StoppingCondition(node* curNode, const vector<vector<float> >& trainData, const vector<int>& labelIndex);
	void DeleteTree(node* curNode);
	void PrintTree(node* curNode, string prefix);
	void ComputeAlpha(node* curNode);
	void ComputeRt(node* curNode, const vector<int>& labelIndex);
	void SetTreeKey();
	
public:
	DecisionTreeCART();
	DecisionTreeCART(vector<string> _features, vector<enum feature_type> _feature_type, int _max_depth, int _min_items, float _descent_threshold, float _lambda, float _nu, int _num_class = 0, enum loss_function_type = SQUARE_LOSS);
	~DecisionTreeCART();
	void TreeGrowth(const vector<vector<float> >& trainData, const vector<float>& residual);
	void TreeGrowth(const vector<vector<float> >& trainData, const vector<int>& labelIndex);
	void Print();
	void Prune(vector<int>& labelIndex);
	void Predictor(const vector<vector<float> >& testData, vector<int>& labelIndex);
	void Predictor(const vector<vector<float> >& testData, vector<float>& predictValue);
};

#endif
