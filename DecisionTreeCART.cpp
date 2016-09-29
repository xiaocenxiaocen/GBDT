#include <iostream>
#include <string>
#include <algorithm>
#include <utility>
#include <vector>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <numeric>
#include <queue>

using namespace std;
#include "CSVParser.h"
#include "TreeUtil.h"
#include "BinaryTree.h"
#include "DecisionTreeCART.h"

template<class _Ty1, class _Ty2>
inline bool operator<(const std::pair<float,int>& _Left,
					  const std::pair<float,int>& _Right)
{
	return _Left.first < _Right.first;
}

node::node():
	key(0), depth(0), splitFeatureIndex(0), labelIndex(UNDEFINED), predictValue(0.0f), alpha(0.0f), leafNumber(0), rt(0.0f), rTree(0.0f), splitValue(0.0f), parent(NULL), left(NULL), right(NULL) { }

DecisionTreeCART::DecisionTreeCART():
	root(NULL) { }

DecisionTreeCART::DecisionTreeCART(vector<string> _features, vector<enum feature_type> _feature_type, int _max_depth, int _min_items, float _descent_threshold, float _lambda, float _nu, int _num_class, enum loss_function_type _loss_function)
{
	root = NULL;

	features = vector<string>(_features);
	
	featureType = vector<enum feature_type>(_feature_type);

	maxDepth = _max_depth;
	
	minItems = _min_items;
	
	descentThreshold = _descent_threshold;

	lambda = _lambda;

	nu = _nu;	

	numClass = _num_class;
	
	numFeatures = features.size();

	lossFunction = _loss_function;
}

DecisionTreeCART::~DecisionTreeCART()
{
	DeleteTree(root);
}

bool DecisionTreeCART::FindBestSplit(const vector<vector<float> >& trainData, const vector<float>& residual, node* curNode)
{
	float minInfoGain = 0.0;
	const int numData = curNode->dataIndex.size();

	vector<float> gradients;
	vector<float> hessians;

	/* first, we need to compute the 
 	 * gradient and hessian of loss function
 	 * of each item in the data set of 
 	 * the current node
 	 */
	if(lossFunction == SQUARE_LOSS) {
		for(size_t i = 0; i < numData; i++) {
			gradients.push_back(- residual[curNode->dataIndex[i]]);
			hessians.push_back(1.0);
		}
	}

	if(lossFunction == LOGIT_LOSS) {
		int labelIdx = trainData[0].size() - 1;
		for(size_t i = 0; i < numData; i++) {
			gradients.push_back( residual[curNode->dataIndex[i]] );
			float fabs_r = fabs(residual[curNode->dataIndex[i]]);
			hessians.push_back( fabs_r * ( 2.0 - fabs_r ) );
		}
	}
	float g = accumulate(gradients.begin(), gradients.end(), 0.0);
	float h = accumulate(hessians.begin(), hessians.end(), 0.0);
	float score = - g * g / (h + lambda);
	
	/* second, we should find splitting which feature
 	 * can achieve the minimum information gain 
 	 */
	for(size_t i = 0; i < numFeatures; i++) {
		if(featureType[i] == DISCRETE) {
			float gLeft = 0.0;
			float hLeft = 0.0;
			float gRight = 0.0;
			float hRight = 0.0;
			float infoGain = - score;
			for(size_t j = 0; j < numData; j++) {
				int ii = static_cast<int>(trainData[curNode->dataIndex[j]][i] + 0.5);
				if(ii == 0) {
					gLeft += gradients[j];
					hLeft += hessians[j];
				} 
				else if(ii == 1) {
					gRight += gradients[j];
					hRight += hessians[j];
				}
			}
			float scoreLeft = - gLeft * gLeft / (hLeft + lambda);
			float scoreRight = - gRight * gRight / (gRight + lambda);
			infoGain += scoreLeft + scoreRight;
			if(infoGain < minInfoGain) {
				minInfoGain = infoGain;
				curNode->splitFeatureIndex = i;
				curNode->splitValue = 0.0;
			}	
		} // end if: feature type = discrete  
		else if(featureType[i] == CONTINUOUS) {	
			vector<pair<float, int> > tmp;
			
			/* first, sort items by value of ith feature *
 			 * complexity: O(N * log(N))
 			 */
			/* O(N) */
			for(size_t j = 0; j < numData; j++) {
				tmp.push_back(pair<float, int>(
					trainData[curNode->dataIndex[j]][i],
					j));
			} // end for
			
			/* O(N * log(N)) */
			sort(tmp.begin(), tmp.end());

			/* second, check every split value, find
 			 * the minimum information gain.
 			 * complexity: O(N) 
 			 */
			float gLeft = 0.0;
			float gRight = g;
			float hLeft = 0.0;
			float hRight = h;
			float split;
			float infoGain = 0.0;
			/* O(N) */
			for(size_t j = 0; j < tmp.size() - 1; j++) {
				int jj = tmp[j].second;
				gLeft += gradients[jj];
				gRight -= gradients[jj];
				hLeft += hessians[jj];
				hRight -= hessians[jj];
				split = 0.5 * (tmp[j].first + tmp[j + 1].first);
				
				if(tmp[j].first < tmp[j + 1].first) {
					float scoreLeft = - gLeft * gLeft / (hLeft + lambda);
					float scoreRight = - gRight * gRight / (hRight + lambda);
					infoGain = scoreLeft + scoreRight - score;
					if(infoGain < minInfoGain) {
						minInfoGain = infoGain;
						curNode->splitFeatureIndex = i;
						curNode->splitValue = split;
					} // end if	
				} // end if				
			} // end for	
		} // end else if: feature type = continuous
	} // end for

	if(fabs(minInfoGain) < descentThreshold) {	
		return false;
	} else {
		return true;
	}
}

void DecisionTreeCART::FindBestSplit(const vector<vector<float> >& trainData, const vector<int>& labelIndex, node* curNode)
{
	float minInfoGain = 1.0;
	const int numData = curNode->dataIndex.size();

	for(int i = 0; i < numFeatures; i++) {
		if(featureType[i] == DISCRETE) {
			float i0 = 0.0;
			float i1 = 0.0;
			float infoGain = 0.0;
			int n0 = 0;
			int n1 = 0;
			vector<int> c0(numClass, 0);
			vector<int> c1(numClass, 0);

			for(int j = 0; j < numData; j++) {
				int ii = static_cast<int>(trainData[curNode->dataIndex[j]][i] + 0.5);
				int jj = labelIndex[curNode->dataIndex[j]];
				
				if(ii == 0) {
					n0++;
					c0[jj]++;
				} // end if
				else if(ii == 1) {
					n1++;
					c1[jj]++;
				} // end else if
			} // end for: j
			
			for(int j = 0; j < numClass; j++) {
				//	i0 += ((float)(c0[j])) / ((float)(n0)) * log(((float)(c0[j])) / ((float)(n0)));
				//	i1 += ((float)(c1[j])) / ((float)(n1)) * log(((float)(c1[j])) / ((float)(n1)));
					i0 += ((float)(c0[j])) / ((float)(n0)) * ((float)(c0[j])) / ((float)(n0));
					i1 += ((float)(c1[j])) / ((float)(n1)) * ((float)(c1[j])) / ((float)(n1));
			}
			i0 = 1.0 - i0;
			i1 = 1.0 - i1;
			infoGain = ((float)(n0) / (float)(numData)) * i0 + ((float)(n1) / (float)(numData)) * i1;
			if(infoGain < minInfoGain) {
				curNode->splitFeatureIndex = i;
				minInfoGain = infoGain;
			}
		} // end if
		else if(featureType[i] == CONTINUOUS) {
			vector<pair<float, int> > tmp;
			vector<int> c0(numClass, 0);
			vector<int> c1(numClass, 0);
			
			/* O(N) */
			for(int j = 0; j < numData; j++) {
				int ii = labelIndex[curNode->dataIndex[j]];
				tmp.push_back(pair<float, int>(trainData[curNode->dataIndex[j]][i],ii));
				c1[ii]++;
			}

			/* O(N * log(N)) */
			sort(tmp.begin(), tmp.end());
			
			/* O(N) */
			for(int j = 0; j < numData - 1; j++) {
				float i0 = 0.0;
				float i1 = 0.0;
				c0[tmp[j].second]++;
				c1[tmp[j].second]--;

				if((tmp[j].first < tmp[j + 1].first) && (tmp[j].second != tmp[j + 1].second)) {
					float split = 0.5 * (tmp[j].first + tmp[j + 1].first);
					int n0 = j + 1;
					int n1 = numData - n0;

					for(int k = 0; k < numClass; k++) {
						//	i0 += ((float)(c0[k])) / ((float)(n0)) * log(((float)(c0[k])) / ((float)(n0)));
						//	i1 += ((float)(c1[k])) / ((float)(n1)) * log(((float)(c1[k])) / ((float)(n1)));
							i0 += ((float)(c0[k])) / ((float)(n0)) * ((float)(c0[k])) / ((float)(n0));
							i1 += ((float)(c1[k])) / ((float)(n1)) * ((float)(c1[k])) / ((float)(n1));
					} // end for: k
					i0 = 1.0 - i0;
					i1 = 1.0 - i1;
					float infoGain = ((float)(n0) / (float)(numData)) * i0 + ((float)(n1) / (float)(numData)) * i1;
					if(infoGain < minInfoGain) {
						curNode->splitFeatureIndex = i;
						curNode->splitValue = split;
						minInfoGain = infoGain;
					}
				} // end if 			
			} // end for: j
		} // end else if
	} // end for:i
	
	return;
}

void DecisionTreeCART::TreeGrowth(const vector<vector<float> >& trainData, const vector<float>& residual)
{
	vector<node*> nodeStack;

	root = new node;

	for(size_t i = 0; i < trainData.size(); i++) {
		root->dataIndex.push_back(i);
	}

	root->depth = 0;

	if(FindBestSplit(trainData, residual, root) && !StoppingCondition(root)) {
		nodeStack.push_back(root);
	} else {
		Evaluate(root, residual);

		node* tmpNode = root;
		while(tmpNode->parent != NULL) {
			tmpNode->parent->leafNumber += 1;
			tmpNode = tmpNode->parent;
		}
		return;	
	}
	
	while(!nodeStack.empty()) {
		node* curNode = *(nodeStack.end() - 1);

		curNode->left = new node;
		curNode->right = new node;

		curNode->left->dataIndex.clear();
		curNode->right->dataIndex.clear();

		for(size_t i = 0; i < curNode->dataIndex.size(); i++) {
			if(trainData[curNode->dataIndex[i]][curNode->splitFeatureIndex] <= curNode->splitValue) {
				curNode->left->dataIndex.push_back(curNode->dataIndex[i]);
			} else {
				curNode->right->dataIndex.push_back(curNode->dataIndex[i]);
			} // end if
		} // end for

		curNode->left->parent = curNode;
		curNode->right->parent = curNode;

		nodeStack.pop_back();

		curNode->left->depth = curNode->depth + 1;
		curNode->right->depth = curNode->depth + 1; 

		if(FindBestSplit(trainData, residual, curNode->left) && !StoppingCondition(curNode->left))  {
			nodeStack.push_back(curNode->left);
		} else {
			Evaluate(curNode->left, residual);
			
			node* tmpNode = curNode->left;
			while(tmpNode->parent != NULL) {
				tmpNode->parent->leafNumber += 1;
				tmpNode = tmpNode->parent;
			}
		}

		if(FindBestSplit(trainData, residual, curNode->right) && !StoppingCondition(curNode->right)) {
			nodeStack.push_back(curNode->right);
		} else {
			Evaluate(curNode->right, residual);
			
			node* tmpNode = curNode->left;
			while(tmpNode->parent != NULL) {
				tmpNode->parent->leafNumber += 1;
				tmpNode = tmpNode->parent;
			}
		}
	} // end while

	SetTreeKey();

	return;	
}

void DecisionTreeCART::TreeGrowth(const vector<vector<float> >& trainData, const vector<int>& labelIndex) 
{
	vector<node*> nodeStack;
	
	root = new node;

	root->depth = 0;
	
	for(int i = 0; i < trainData.size(); i++) {
		root->dataIndex.push_back(i);
	}
	
	FindBestSplit(trainData, labelIndex, root);

	ComputeRt(root, labelIndex);

	if(!StoppingCondition(root, trainData, labelIndex)) {
		nodeStack.push_back(root);
	} else {
		Classify(root, labelIndex);
		
		node* tmpNode = root;
		while(tmpNode->parent != NULL) {
			tmpNode->parent->leafNumber += 1;
			tmpNode->parent->rTree += root->rt;
			tmpNode = tmpNode->parent;
		}
		return;
	}
	
	while(!nodeStack.empty()) {
		node* curNode = *(nodeStack.end() - 1);

		curNode->left = new node;
		curNode->right = new node;

		if(featureType[curNode->splitFeatureIndex] == DISCRETE) {
			for(int j = 0; j < curNode->dataIndex.size(); j++) {
				int ii = static_cast<int>(trainData[curNode->dataIndex[j]][curNode->splitFeatureIndex] + 0.5);
				int jj = curNode->dataIndex[j];
				if(ii == 0) {
					curNode->left->dataIndex.push_back(jj);
				} // end if
				else if(ii == 1) {
					curNode->right->dataIndex.push_back(jj);
				} // end else if
			} // end for: j
		} // end if
		else if(featureType[curNode->splitFeatureIndex] == CONTINUOUS) {
			for(int j = 0; j < curNode->dataIndex.size(); j++) {
				int jj = curNode->dataIndex[j];
								
				if(trainData[curNode->dataIndex[j]][curNode->splitFeatureIndex] <= curNode->splitValue) {
					curNode->left->dataIndex.push_back(jj);
				} else {
					curNode->right->dataIndex.push_back(jj);
				} // end if
			} // end for: j
		} // end else if
			
		curNode->left->parent = curNode;
		curNode->right->parent = curNode;

		curNode->left->depth = curNode->depth + 1;
		curNode->right->depth = curNode->depth + 1;

		nodeStack.pop_back();
		
		FindBestSplit(trainData, labelIndex, curNode->left);
		ComputeRt(curNode->left, labelIndex);
		
		if(!StoppingCondition(curNode->left, trainData, labelIndex)) {
			nodeStack.push_back(curNode->left);
		} else {
			Classify(curNode->left, labelIndex);
			
			node* tmpNode = curNode->left;
			while(tmpNode->parent != NULL) {
				tmpNode->parent->leafNumber += 1;
				tmpNode->parent->rTree += curNode->left->rt;
				tmpNode = tmpNode->parent;
			}
		} // end if
		
		FindBestSplit(trainData, labelIndex, curNode->right);
		ComputeRt(curNode->right, labelIndex);
		
		if(!StoppingCondition(curNode->right, trainData, labelIndex)) {
			nodeStack.push_back(curNode->right);
		} else {
			Classify(curNode->right, labelIndex);
			
			node* tmpNode = curNode->right;
			while(tmpNode->parent != NULL) {
				tmpNode->parent->leafNumber += 1;
				tmpNode->parent->rTree += curNode->right->rt;
				tmpNode = tmpNode->parent;
			}
		} // end if
	} // end while
	
	SetTreeKey();
	ComputeAlpha(root);

	return;
}

void DecisionTreeCART::Evaluate(node* curNode, const vector<float>& residual)
{
	curNode->predictValue = 0.0;

	if(lossFunction == SQUARE_LOSS) {
		for(size_t i = 0; i < curNode->dataIndex.size(); i++) {
			curNode->predictValue += nu * residual[curNode->dataIndex[i]];
		}
	
		curNode->predictValue /= (curNode->dataIndex.size() + lambda);
	}

	if(lossFunction == LOGIT_LOSS) {
		float g = 0.0;
		float h = 0.0;
		for(size_t i = 0; i < curNode->dataIndex.size(); i++) {
			g += residual[curNode->dataIndex[i]];
			float fabs_r = fabs(residual[curNode->dataIndex[i]]);
			h += fabs_r * ( 2.0 - fabs_r );
		}
		curNode->predictValue = nu * g / ( h + lambda );
	}

	return;
}

void DecisionTreeCART::Classify(node* curNode, const vector<int>& labelIndex)
{
	vector<int> c(numClass, 0);
	
	for(size_t i = 0; i < curNode->dataIndex.size(); i++) {
		int ii = labelIndex[curNode->dataIndex[i]];
		c[ii]++;
	}

	curNode->labelIndex = static_cast<int>(max_element(c.begin(), c.end()) - c.begin());

	return;
}

void DecisionTreeCART::ComputeRt(node* curNode, const vector<int>& labelIndex)
{
	vector<int> c(numClass, 0);

	for(size_t i = 0; i < curNode->dataIndex.size(); i++) {
		int ii = labelIndex[curNode->dataIndex[i]];
		c[ii]++;
	}

	int labelIdx = static_cast<int>(max_element(c.begin(), c.end()) - c.begin());

	float error_rate = (float)(curNode->dataIndex.size() - c[labelIdx]) / (float)(curNode->dataIndex.size());

	curNode->rt = error_rate * (float)(curNode->dataIndex.size()) / (float)(labelIndex.size());

	return;
}

void DecisionTreeCART::ComputeAlpha(node* curNode)
{
	if(curNode->left == NULL && curNode->right == NULL) {
		curNode->alpha = 0.0f;
	} else {
		curNode->alpha = (curNode->rt - curNode->rTree) / (curNode->leafNumber - 1);
		ComputeAlpha(curNode->left);
		ComputeAlpha(curNode->right);
	} // end else

	return;
}

bool DecisionTreeCART::StoppingCondition(node* curNode)
{
	/* the depth of tree is larger than maxDepth */
	if(curNode->depth > maxDepth) return true;

	/* the number of items in a tree node is smaller
 	 * than the minItems
 	 */
	if(curNode->dataIndex.size() < minItems) return true;
	
	return false;
}

bool DecisionTreeCART::StoppingCondition(node* curNode, const vector<vector<float> >& trainData, const vector<int>& labelIndex)
{
	/* the depth of tree is larger than maxDepth */
	if(curNode->depth > maxDepth) return true;

	/* the number of items in a tree node is smaller
 	 * than the minItems
 	 */
	if(curNode->dataIndex.size() < minItems) return true;

	int c[2][numClass];
	memset(c[0], 0, sizeof(int) * 2 * numClass);

	if(featureType[curNode->splitFeatureIndex] == DISCRETE) {
		for(size_t i = 0; i < curNode->dataIndex.size(); i++) {
			int ii = static_cast<int>(trainData[curNode->dataIndex[i]][curNode->splitFeatureIndex] + 0.5);
			int jj = static_cast<int>(labelIndex[curNode->dataIndex[i]]);

			c[ii][jj]++;
		}
	} 
	else if(featureType[curNode->splitFeatureIndex] == CONTINUOUS) {
		for(size_t i = 0; i < curNode->dataIndex.size(); i++) {
			int jj = labelIndex[curNode->dataIndex[i]];
			int ii = (trainData[curNode->dataIndex[i]][curNode->splitFeatureIndex] <= curNode->splitValue);
			c[ii][jj]++;
		}
	}

	vector<int> rowSum(2);
	vector<int> colSum(numClass);
	
	int totalSum = 0;

	for(int i = 0; i < 2; i++) {
		for(int j = 0; j < numClass; j++) {
			totalSum += c[i][j];
			rowSum[i] += c[i][j];
			colSum[j] += c[i][j];
		}
	} // end for

	double chi = 0.0;

	for(int i = 0; i < 2; i++) {
		for(int j = 0; j < numClass; j++) {
			double excep = 1.0 * rowSum[i] * colSum[j] / totalSum;

			if(fabs(excep) > 1e-8) {
				chi += (c[i][j] - excep) * (c[i][j] - excep) / excep;
			} 
		}
	}	

	if(chi < CHI[numClass - 2]) return true;
	
	return false;

}

void DecisionTreeCART::SetTreeKey()
{
	node* treeMax = TreeMaximum(root);
	node* treeMin = TreeMinimum(root);

	node* curNode = treeMin;
	
	int index = 0;
	
	while(curNode != treeMax) {
		curNode->key = index;
		index++;
		curNode = Successor(curNode);
	} 

	treeMax->key = index;
}

void DecisionTreeCART::Prune(vector<int>& labelIndex) {
	priority_queue<myTriple> pq;

	queue<node*> que;

	que.push(root);

	while(!que.empty()) {
		node* curNode = que.front();

		if(curNode->left != NULL) {
			que.push(curNode->left);
			que.push(curNode->right);
			pq.push(myTriple(curNode->alpha, curNode->leafNumber, curNode->key));
		}

		que.pop();	
	} 
	
	int n = (pq.top()).third;

	cerr << "prune node with key = " << n << "\n";

	node* curNode = root;

	while(curNode != NULL && curNode->key != n) {
		if(curNode->key < n) {
			curNode = curNode->right;
		} else {
			curNode = curNode->left;
		} 
	}

	if(curNode != NULL) {
		DeleteTree(curNode->left);
		DeleteTree(curNode->right);
		curNode->left = NULL;
		curNode->right = NULL;
		
		node* tmpNode = curNode;
		while(tmpNode->parent != NULL) {
			tmpNode->parent->leafNumber -= curNode->leafNumber - 1;
			tmpNode->parent->rTree -= curNode->rTree - curNode->rt;
			tmpNode = tmpNode->parent;
		}

		curNode->leafNumber = 0;
		curNode->rTree = 0.0f;	
		Classify(curNode, labelIndex);
	}

	ComputeAlpha(root);
}

void DecisionTreeCART::Print() 
{
	PrintTree(root, string(""));	
}

void DecisionTreeCART::PrintTree(node* curNode, string prefix)
{
	if(curNode==NULL) return;
	if(curNode->left != NULL && curNode->right != NULL) {
		if(featureType[curNode->splitFeatureIndex] == DISCRETE) {
			cout << "\n";
			cout << prefix << features[curNode->splitFeatureIndex] << "==0";
			cout << " <key: " << curNode->left->key << ">";
			cout << " <rt: " << curNode->left->rt << ">";
			cout << " <rTree: " << curNode->left->rTree << ">";
			cout << " <leaf number: " << curNode->left->leafNumber << ">";
			cout << " <alpha: " << curNode->left->alpha << ">";
			PrintTree(curNode->left, prefix+"|");
			cout << prefix << features[curNode->splitFeatureIndex] << "==1";
			cout << " <key: " << curNode->right->key << ">";
			cout << " <rt: " << curNode->right->rt << ">";
			cout << " <rTree: " << curNode->right->rTree << ">";
			cout << " <leaf number: " << curNode->right->leafNumber << ">";
			cout << " <alpha: " << curNode->right->alpha << ">";
			PrintTree(curNode->right, prefix+"|");
		} 
		else if(featureType[curNode->splitFeatureIndex] == CONTINUOUS) {
			cout << "\n";
			cout << prefix << features[curNode->splitFeatureIndex] << "<=" << curNode->splitValue;
			cout << " <key: " << curNode->left->key << ">";
			cout << " <rt: " << curNode->left->rt << ">";
			cout << " <rTree: " << curNode->left->rTree << ">";
			cout << " <leaf number: " << curNode->left->leafNumber << ">";
			cout << " <alpha: " << curNode->left->alpha << ">";
			PrintTree(curNode->left, prefix+"|");
			cout << prefix << features[curNode->splitFeatureIndex] << ">" << curNode->splitValue;
			cout << " <key: " << curNode->right->key << ">";
			cout << " <rt: " << curNode->right->rt << ">";
			cout << " <rTree: " << curNode->right->rTree << ">";
			cout << " <leaf number: " << curNode->right->leafNumber << ">";
			cout << " <alpha: " << curNode->right->alpha << ">";
			PrintTree(curNode->right, prefix+"|");
		}	
	} else {
		if(curNode->labelIndex != UNDEFINED) {
			cout << ": class " << curNode->labelIndex << "\n";
		} else {
			cout << ": predict value " << curNode->predictValue << "\n";
		} // end if
	} // end if

	return;
}

void DecisionTreeCART::DeleteTree(node* curNode)
{
	if(curNode == NULL) return;
	if(curNode->left == NULL || curNode->right == NULL) {
		delete curNode;
		return;
	} else {
		DeleteTree(curNode->left);
		DeleteTree(curNode->right);
	} // end if

	delete curNode;

	return;
}

void DecisionTreeCART::Predictor(const vector<vector<float> >& testData, vector<int>& labelIndex)
{
	for(int i = 0; i < testData.size(); i++) {
		node* curNode = root;
		int labelIdx = UNDEFINED;
		while(curNode->left != NULL && curNode->right != NULL) {
			if(featureType[curNode->splitFeatureIndex] == DISCRETE) {
				int ii = static_cast<int>(testData[i][curNode->splitFeatureIndex] + 0.5);
				if(ii == 0) curNode = curNode->left;
				else curNode = curNode->right;
			}
			else if(featureType[curNode->splitFeatureIndex] == CONTINUOUS) {
				if(testData[i][curNode->splitFeatureIndex] <= curNode->splitValue) curNode = curNode->left;
				else curNode = curNode->right;
			} // end if
		} // end while

		labelIndex.push_back(curNode->labelIndex);
	} // end for
	return;
}

void DecisionTreeCART::Predictor(const vector<vector<float> >& testData, vector<float>& predictValue)
{
	for(int i = 0; i < testData.size(); i++) {
		node* curNode = root;
		int labelIdx = UNDEFINED;
		while(curNode->left != NULL && curNode->right != NULL) {
			if(featureType[curNode->splitFeatureIndex] == DISCRETE) {
				int ii = static_cast<int>(testData[i][curNode->splitFeatureIndex] + 0.5);
				if(ii == 0) curNode = curNode->left;
				else curNode = curNode->right;
			}
			else if(featureType[curNode->splitFeatureIndex] == CONTINUOUS) {
				if(testData[i][curNode->splitFeatureIndex] <= curNode->splitValue) curNode = curNode->left;
				else curNode = curNode->right;
			} // end if
		} // end while

		predictValue.push_back(curNode->predictValue);
	} // end for
	return;
}
