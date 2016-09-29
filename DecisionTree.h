#ifndef DECISIONTREE_H
#define DECISIONTREE_H

class RandomForest;
class DecisionTree;
class node;

const double CHI[18]={0.004,0.103,0.352,0.711,1.145,1.635,2.167,2.733,3.325,3.94,4.575,5.226,5.892,6.571,7.261,7.962};

enum lType{DISCRETE, CONTINUOUS};

#define UNDEFINED -1

class node {
private:
	friend class DecisionTree;
	int key;
	int depth;
	int labelIdx;
	int classIdx;
	float alpha;
	int leaf_number;
	float r_t;
	float rtree;
	float test_cond; /* test condition of left child */
	std::vector<int> dataId;
	node* parent;
	node* left;
	node* right;
public:
	node(); /* constructor */
	~node() { };
};

class DecisionTree {
private:
	friend class RandomForest;
	std::vector<std::string> label;
	std::vector<enum lType> labelType;
	std::vector<std::vector<float> > data;
	node* root;
	int numClass;
	int max_depth;
	
	void find_best_split(const std::vector<int>& dataId, int& labelIdx, float& test_cond);
	void Classify(node* aNode);
	bool stopping_cond(node* aNode);
	void DeleteTree(node* aNode);
	void PrintTree(node* aNode, std::string prefix);
	void ComputeAlpha(node* aNode);
	void ComputeR_t(node* aNode);
	node* TreeMinimum(node* aNode);
	node* TreeMaximum(node* aNode);
	node* Successor(node* aNode);
	void SetTreeKey();

public:
	DecisionTree();
	~DecisionTree();
	void TreeGrowth();
	void DataParser(std::string fileName);
	//void Delete();
	void Print();
	void Prune();
	void Predictor(const std::vector<std::vector<float> >& predict_data, std::vector<int>& classIdx); 
};

#endif
