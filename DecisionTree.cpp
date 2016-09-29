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

#include "util.h"
#include "DecisionTree.h"

template<class _Ty1, class _Ty2>
inline bool operator<(const std::pair<float,int>& _Left,
					  const std::pair<float,int>& _Right)
{
	return _Left.first < _Right.first;
}

node::node():
	depth(0), labelIdx(0), classIdx(UNDEFINED), alpha(0.0f), leaf_number(0), r_t(0.0f), rtree(0.0f), test_cond(0.0f), parent(NULL), left(NULL), right(NULL) { }

DecisionTree::DecisionTree():
	max_depth(20), root(NULL) { }

void DecisionTree::find_best_split(const std::vector<int>& dataId, int& label_index, float& test_cond)
{
	int classIdx = data[0].size() - 1;
	float minDeltaInfo = 1.0;
	int nData = dataId.size();

	for(std::vector<std::string>::iterator il = label.begin(); il != label.end(); il++) {
		int labelIdx = static_cast<int>(il - label.begin());
		if(labelType[labelIdx] == DISCRETE) {
			float i0 = 0.0;
			float i1 = 0.0;
			float deltaInfo = 0.0;
			int n0 = 0;
			int n1 = 0;
			std::vector<int> c0(numClass, 0);
			std::vector<int> c1(numClass, 0);
			for(std::vector<int>::const_iterator id = dataId.begin(); id != dataId.end(); id++) {
				int ii = static_cast<int>(data[*id][labelIdx] + 0.5);
				int jj = static_cast<int>(data[*id][classIdx] + 0.5);
				if(ii == 0) {
					n0++;
					c0[jj]++;
				} // end if
				else if(ii == 1) {
					n1++;
					c1[jj]++;
				} // end if
			} // end for
			for(int ic = 0; ic < numClass; ic++) {
			//	i0 += ((float)(c0[ic])) / ((float)(n0)) * log(((float)(c0[ic])) / ((float)(n0)));
			//	i1 += ((float)(c1[ic])) / ((float)(n1)) * log(((float)(c1[ic])) / ((float)(n1)));
				i0 += ((float)(c0[ic])) / ((float)(n0)) * ((float)(c0[ic])) / ((float)(n0));
				i1 += ((float)(c1[ic])) / ((float)(n1)) * ((float)(c1[ic])) / ((float)(n1));
			}
			i0 = 1.0 - i0;
			i1 = 1.0 - i1;
			deltaInfo = ((float)(n0) / (float)(nData)) * i0 + ((float)(n1) / (float)(nData)) * i1;
			if(deltaInfo < minDeltaInfo) {
				label_index = labelIdx;
				minDeltaInfo = deltaInfo;
				test_cond = 0.0;
			} // end if
		} // end if
		else if(labelType[labelIdx] == CONTINUOUS) {
			std::vector<std::pair<float,int> > tmpValue;
			std::vector<int> c0(numClass, 0);
			std::vector<int> c1(numClass, 0);
			
			/* O(N) */
			for(std::vector<int>::const_iterator id = dataId.begin(); id != dataId.end(); id++) {
				int ii = static_cast<int>(data[*id][classIdx] + 0.5);
				tmpValue.push_back(std::pair<float,int>(data[*id][labelIdx], ii));
				c1[ii]++;
			} // end for
			
			/* O(N * log(N)) */
		//	std::sort(tmpValue.begin(), tmpValue.end(), compare);
			std::sort(tmpValue.begin(), tmpValue.end());

			/* O(N) */
			for(std::vector<std::pair<float,int> >::iterator ite  = tmpValue.begin(); 
															 ite != tmpValue.end() - 1; 
															 ite++) {
				float i0 = 0.0;
				float i1 = 0.0;

				c0[ite->second]++;
				c1[ite->second]--;

			//	for(int i = 0; i < c0.size(); i++) {
			//		std::cerr << c0[i] << " ";
			//	}
			//	
			//	std::cerr << "\n";
			//	
			//	for(int i = 0; i < c0.size(); i++) {
			//		std::cerr << c1[i] << " ";
			//	}
			//	
			//	std::cerr << "\n";

			//	std::cerr << dataId.size() << "\n";
			//	
			//	std::cerr << " end a loop\n";
				
				if(((ite + 1)->first > ite->first) && ((ite + 1)->second != ite->second)) {
					float splitValue = 0.5 * (ite->first + (ite + 1)->first);
					int n0 = static_cast<int>(ite - tmpValue.begin() + 1);
					int n1 = static_cast<int>(tmpValue.size() - n0);
				//	std::cerr << n0 << " " << n1 << "\n";
				//	n0 = std::accumulate(c0.begin(), c0.end(), static_cast<int>(0));
				//	n1 = std::accumulate(c1.begin(), c1.end(), static_cast<int>(0));
				//	std::cerr << n0 << " " << n1 << "\n";
					for(int ic = 0; ic < numClass; ic++) {
					//	i0 += ((float)(c0[ic])) / ((float)(n0)) * log(((float)(c0[ic])) / ((float)(n0)));
					//	i1 += ((float)(c1[ic])) / ((float)(n1)) * log(((float)(c1[ic])) / ((float)(n1)));
						i0 += ((float)(c0[ic])) / ((float)(n0)) * ((float)(c0[ic])) / ((float)(n0));
						i1 += ((float)(c1[ic])) / ((float)(n1)) * ((float)(c1[ic])) / ((float)(n1));
					}
					i0 = 1.0 - i0;
					i1 = 1.0 - i1;
					float deltaInfo = ((float)(n0) / (float)(nData)) * i0 + ((float)(n1) / (float)(nData)) * i1;
					if(deltaInfo < minDeltaInfo) {
						label_index = labelIdx;
						minDeltaInfo = deltaInfo;
						test_cond = splitValue;
					} // end if
				} // end for				
			} // end for
				
		} // end else if
	} // end for
}

bool DecisionTree::stopping_cond(node* aNode)
{
	int c[2][numClass];
	memset(c[0], 0, sizeof(int) * 2 * numClass);
	
	int classIdx = data[0].size() - 1;

	if(labelType[aNode->labelIdx] == DISCRETE) {
		for(std::vector<int>::iterator id = aNode->dataId.begin(); id != aNode->dataId.end(); id++) {
			int ii = static_cast<int>(data[*id][aNode->labelIdx] + 0.5);
			int jj = static_cast<int>(data[*id][classIdx] + 0.5);

			if(ii == 0) {
				c[0][jj]++;
			} // end if
			else if(ii == 1) {
				c[1][jj]++;
			} // end if
		} // end for
	} 
	else if(labelType[aNode->labelIdx] == CONTINUOUS) {
		for(std::vector<int>::iterator id = aNode->dataId.begin(); id != aNode->dataId.end(); id++) {
			int jj = static_cast<int>(data[*id][classIdx] + 0.5);
			if(data[*id][aNode->labelIdx] <= aNode->test_cond) {
				c[0][jj]++;
			} else {
				c[1][jj]++;
			} // end if
		} // end for
	}

//	for(std::vector<int>::iterator id = aNode->dataId.begin(); id != aNode->dataId.end(); id++) {
//		int jj = static_cast<int>(data[*id][classIdx] + 0.5);
//		std::cout << jj << " ";
//	}
//	std::cout << "\n";

	std::vector<int> rowSum(2);
	std::vector<int> colSum(numClass);	
	int totalSum = 0;

//	for(int i = 0; i < 2; i++) {
//		for(int j = 0; j < numClass; j++) {
//			std::cerr << c[i][j] << " ";
//		}
//		std::cerr << "\n";
//	}

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

//	std::cerr << chi << "\n";
	
	if(chi < CHI[numClass - 2]) {
		return true;
	} else {
		return false;
	}
}

void DecisionTree::Classify(node* aNode)
{
	std::vector<int> c(numClass, 0);

	int class_index = data[0].size() - 1;

	for(std::vector<int>::iterator ite = aNode->dataId.begin(); ite != aNode->dataId.end(); ite++) {
		int idx = static_cast<int>(data[*ite][class_index]+0.5);
		c[idx]++;
	}

	aNode->classIdx = static_cast<int>(std::max_element(c.begin(), c.end()) - c.begin());	
}

void DecisionTree::TreeGrowth()
{
	std::vector<node*> nodeStack;

	root = new node;

	for(size_t id = 0; id < data.size(); id++) {
		root->dataId.push_back(id);
	}
	
	find_best_split(root->dataId, root->labelIdx, root->test_cond);

	ComputeR_t(root);

	if(!stopping_cond(root)) {	
		nodeStack.push_back(root);	
	} else {
		Classify(root);

		node* tmp_node = root;
		while(tmp_node->parent != NULL) {
			tmp_node->parent->leaf_number += 1;
			tmp_node->parent->rtree += root->r_t;
			tmp_node = tmp_node->parent;
		}
		return;
	}

	while(!nodeStack.empty()) {
		node* curNode = *(nodeStack.end() - 1);

		curNode->left = new node;
		curNode->right = new node;

		curNode->left->dataId.clear();
		curNode->right->dataId.clear();

		if(labelType[curNode->labelIdx] == DISCRETE) {
			for(std::vector<int>::iterator ite = curNode->dataId.begin(); ite != curNode->dataId.end(); ite++) {
				int ii = static_cast<int>(data[*ite][curNode->labelIdx] + 0.5);
				int jj = *ite;
				if(ii == 0) {
					curNode->left->dataId.push_back(jj);
				}
				else if(ii == 1) {
					curNode->right->dataId.push_back(jj);
				}
			}
		} 
		else if(labelType[curNode->labelIdx] == CONTINUOUS) {
			for(std::vector<int>::iterator ite = curNode->dataId.begin(); ite != curNode->dataId.end(); ite++) {
				int jj = *ite;
				if(data[*ite][curNode->labelIdx] <= curNode->test_cond) {
					curNode->left->dataId.push_back(jj);
				} else {
					curNode->right->dataId.push_back(jj);
				}  // end if
			} // end for
		} // end for
		
		curNode->left->parent = curNode;
		curNode->right->parent = curNode;

		curNode->left->depth = curNode->depth + 1;
		curNode->right->depth = curNode->depth + 1;

		nodeStack.pop_back();

		find_best_split(curNode->left->dataId, curNode->left->labelIdx, curNode->left->test_cond);
		ComputeR_t(curNode->left);
		if(!stopping_cond(curNode->left)) {	
			nodeStack.push_back(curNode->left);	
		} else {
			Classify(curNode->left);

			node* tmp_node = curNode->left;
			while(tmp_node->parent != NULL) {
				tmp_node->parent->leaf_number += 1;
				tmp_node->parent->rtree += curNode->left->r_t;
				tmp_node = tmp_node->parent;
			}
		}

		find_best_split(curNode->right->dataId, curNode->right->labelIdx, curNode->right->test_cond);
		ComputeR_t(curNode->right);
		if(!stopping_cond(curNode->right)) {
			nodeStack.push_back(curNode->right);	
		} else {
			Classify(curNode->right);
			
			node* tmp_node = curNode->right;
			while(tmp_node->parent != NULL) {
				tmp_node->parent->leaf_number += 1;
				tmp_node->parent->rtree += curNode->right->r_t;
				tmp_node = tmp_node->parent;
			}
		}
	}
	
	SetTreeKey();
	ComputeAlpha(root);

	return;
}

void DecisionTree::DataParser(std::string fileName)
{
	std::ifstream tableData(fileName.c_str(), std::ios::in);
	
	std::string line;

	std::getline(tableData, line);

	Split(line, " ", &label);

	label.pop_back();

	labelType = std::vector<enum lType>(label.size());

	for(size_t il = 0; il < label.size(); il++) {
		if(label[il][0] == 'd') {
			labelType[il] = DISCRETE;
		}
		else if(label[il][0] == 'c') {
			labelType[il] = CONTINUOUS;
		}
		label[il] = label[il].substr(1, std::string::npos);
	} 	

	std::vector<int> c_tmp;

	while(std::getline(tableData, line)) {
		std::vector<std::string> ret;
		Split(line, " ", &ret);
		std::vector<float> rowData;
		for(size_t id = 0; id < ret.size(); id++) {
			float tmp;
			sscanf((ret[id]).c_str(), "%f", &tmp);
			rowData.push_back(tmp);
		}
		c_tmp.push_back(rowData[ret.size() - 1]);
		data.push_back(rowData);
	}

	numClass = (*std::max_element(c_tmp.begin(), c_tmp.end())) + 1;
	std::cerr << numClass << std::endl;

#ifdef DEBUG
//	std::cerr << "traing data:\n";
//	for(size_t row = 0; row < label.size(); row++) {
//		std::cerr << label[row] << " ";
//	}
//	std::cerr << "\n";
//
//	for(size_t row = 0; row < data.size(); row++) {
//		for(size_t col = 0; col < data[row].size(); col++) {
//			std::cerr << data[row][col] << " ";
//		} 
//		std::cerr << "\n";
//	}
#endif

	return;	
}

void DecisionTree::Print()
{
	PrintTree(root, std::string(""));
}

void DecisionTree::PrintTree(node* aNode, std::string prefix)
{
	if(aNode==NULL) return;
	if(aNode->classIdx == UNDEFINED) {
		if(labelType[aNode->labelIdx] == DISCRETE) {
			std::cout << "\n";
			std::cout << prefix << label[aNode->labelIdx] << "==0";
			std::cout << " <key: " << aNode->left->key << ">";
			std::cout << " <r_t: " << aNode->left->r_t << ">";
			std::cout << " <rtree: " << aNode->left->rtree << ">";
			std::cout << " <leaf number: " << aNode->left->leaf_number << ">";
			std::cout << " <alpha: " << aNode->left->alpha << ">";
			PrintTree(aNode->left, prefix+"|");
			std::cout << prefix << label[aNode->labelIdx] << "==1";
			std::cout << " <key: " << aNode->right->key << ">";
			std::cout << " <r_t: " << aNode->right->r_t << ">";
			std::cout << " <rtree: " << aNode->right->rtree << ">";
			std::cout << " <leaf number: " << aNode->right->leaf_number << ">";
			std::cout << " <alpha: " << aNode->right->alpha << ">";
			PrintTree(aNode->right, prefix+"|");
		} 
		else if(labelType[aNode->labelIdx] == CONTINUOUS) {
			std::cout << "\n";
			std::cout << prefix << label[aNode->labelIdx] << "<=" << aNode->test_cond;
			std::cout << " <key: " << aNode->left->key << ">";
			std::cout << " <r_t: " << aNode->left->r_t << ">";
			std::cout << " <rtree: " << aNode->left->rtree << ">";
			std::cout << " <leaf number: " << aNode->left->leaf_number << ">";
			std::cout << " <alpha: " << aNode->left->alpha << ">";
			PrintTree(aNode->left, prefix+"|");
			std::cout << prefix << label[aNode->labelIdx] << ">" << aNode->test_cond;
			std::cout << " <key: " << aNode->right->key << ">";
			std::cout << " <r_t: " << aNode->right->r_t << ">";
			std::cout << " <rtree: " << aNode->right->rtree << ">";
			std::cout << " <leaf number: " << aNode->right->leaf_number << ">";
			std::cout << " <alpha: " << aNode->right->alpha << ">";
			PrintTree(aNode->right, prefix+"|");
		}	
	} else {
		std::cout << ": class " << aNode->classIdx << "\n";		
	}

	return;

}

DecisionTree::~DecisionTree() {
	DeleteTree(root);
}

//void DecisionTree::Delete()
//{
//	DeleteTree(root);
//}

void DecisionTree::DeleteTree(node* aNode)
{
	if(aNode==NULL) return;
	if(aNode->left == NULL || aNode->right == NULL) {
//		if(aNode->parent->left == aNode) {
//			aNode->parent->left = NULL;
//		} else {
//			aNode->parent->right = NULL;
//		}
		delete aNode;
		return;
		
	} else {
		DeleteTree(aNode->left);
		DeleteTree(aNode->right);
	}
	
//	if(aNode->parent->left == aNode) {
//		aNode->parent->left = NULL;
//	} else {
//		aNode->parent->right = NULL;
//	}
	delete aNode;

	return;
}

void DecisionTree::Predictor(const std::vector<std::vector<float> >& predict_data, std::vector<int>& classIdx)
{
	for(std::vector<std::vector<float> >::const_iterator ite = predict_data.begin(); ite != predict_data.end(); ite++) {
		node* curNode = root;
		int classIndex = UNDEFINED;
		while(curNode->classIdx == UNDEFINED) {
			if(labelType[curNode->labelIdx] == DISCRETE) {
				int ii = static_cast<int>((*ite)[curNode->labelIdx] + 0.5);
				if(ii == 0) curNode = curNode->left;
				else if(ii == 1) curNode = curNode->right;
			} // end if
			else if(labelType[curNode->labelIdx] == CONTINUOUS) {
				if((*ite)[curNode->labelIdx] <= curNode->test_cond) curNode = curNode->left;
				else curNode = curNode->right;
			} // end else
		} // end while
		classIndex = curNode->classIdx;
		classIdx.push_back(classIndex);
	}
	return;
}

void DecisionTree::ComputeR_t(node* aNode)
{
	std::vector<int> c(numClass, 0);
	int classIndex = data[0].size() - 1;

	for(std::vector<int>::iterator ite = aNode->dataId.begin(); 
								   ite != aNode->dataId.end();
								   ite++) {
		int ii = static_cast<int>(data[*ite][classIndex] + 0.5);
		c[ii]++;
	} // end for

	int classIdx = static_cast<int>(std::max_element(c.begin(), c.end()) - c.begin());

	float error_rate = (float)(aNode->dataId.size() - c[classIdx]) / (float)(aNode->dataId.size());

	aNode->r_t = error_rate * (float)(aNode->dataId.size()) / (float)(data.size());

	return;
}

void DecisionTree::ComputeAlpha(node* aNode)
{
	if(aNode->left == NULL && aNode->right == NULL) {
		aNode->alpha = 0.0f;	
	} else {
		aNode->alpha = (aNode->r_t - aNode->rtree) / (aNode->leaf_number - 1);
		ComputeAlpha(aNode->left);
		ComputeAlpha(aNode->right);
	} 
}

node* DecisionTree::TreeMaximum(node* aNode)
{
	if(aNode == NULL) return aNode;

	while(aNode->right != NULL) {
		aNode = aNode->right;
	}
	return aNode;
}

node* DecisionTree::TreeMinimum(node* aNode)
{
	if(aNode == NULL) return aNode;
	
	while(aNode->left != NULL) {
		aNode = aNode->left;
	}

	return aNode;
}

node* DecisionTree::Successor(node* aNode)
{
	if(aNode == NULL) return aNode;

	if(aNode->right != NULL) {
		return TreeMinimum(aNode->right);
	}

	while(aNode->parent != NULL && aNode == aNode->parent->right) {
		aNode = aNode->parent;
	}

	return aNode->parent;
}

void DecisionTree::SetTreeKey()
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

void DecisionTree::Prune() {
	std::priority_queue<myTriple> pq;
//	std::vector<myTriple> vt;

	std::queue<node*> que;
	
	que.push(root);

	while(!que.empty()) {
		node* curNode = que.front();
		
//		vt.push_back(myTriple(curNode->alpha, curNode->leaf_number, curNode->key));

		if(curNode->left != NULL) {
			que.push(curNode->left);
			que.push(curNode->right);
			pq.push(myTriple(curNode->alpha, curNode->leaf_number, curNode->key));
		}

		que.pop();	
	} 

//	std::sort(vt.begin(), vt.end());
	int n = (pq.top()).third;

	std::cerr << "prune node with key = " << n << "\n";

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
		
		node* tmp_node = curNode;
		while(tmp_node->parent != NULL) {
			tmp_node->parent->leaf_number -= curNode->leaf_number - 1;
			tmp_node->parent->rtree -= curNode->rtree - curNode->r_t;
			tmp_node = tmp_node->parent;
		}

		curNode->leaf_number = 0;
		curNode->rtree = 0.0f;	
		Classify(curNode);
	}

	ComputeAlpha(root);
}

//int main(int argc, char* argv[])
//{
//	DecisionTree tree = DecisionTree();
//	std::string input(argv[1]);
//
//	tree.DataParser(input);
//
//	tree.TreeGrowth();
//
//	tree.Print();
//	
//	std::vector<std::vector<float> > test;
//
//	std::string test_data(argv[2]);
//
//	std::ifstream testData(test_data.c_str(), std::ios::in);
//	
//	std::string line;
//
//	std::getline(testData, line);
//
//	std::vector<int> result;
//
//	while(std::getline(testData, line)) {
//		std::vector<std::string> ret;
//		Split(line, " ", &ret);
//		std::vector<float> rowData;
//		for(size_t id = 0; id < ret.size(); id++) {
//			float tmp;
//			sscanf((ret[id]).c_str(), "%f", &tmp);
//			rowData.push_back(tmp);
//		}
//		test.push_back(rowData);
//	}
//
//	std::cerr << "testing data:\n";
//
//	for(size_t row = 0; row < test.size(); row++) {
//		for(size_t col = 0; col < test[row].size(); col++) {
//			std::cerr << test[row][col] << " ";
//		} 
//		std::cerr << "\n";
//	}
//
//	tree.Predictor(test, result);
//
//	int correctNum = 0;
//
//	int classIdx = test[0].size() - 1;
//	for(size_t i = 0; i < test.size(); i++) {
//		if(test[i][classIdx] == result[i]) correctNum++;
//	}
//
//	std::cerr << "accurate rate:" << (float)(correctNum) / (float)(test.size()) << std::endl;
//
//	tree.Prune();
//	
//	tree.Print();
//
//	result.clear();
//
//	tree.Predictor(test, result);
//
//	correctNum = 0;
//
//	for(size_t i = 0; i < test.size(); i++) {
//		if(test[i][classIdx] == result[i]) correctNum++;
//	}
//
//	std::cerr << "accurate rate:" << (float)(correctNum) / (float)(test.size()) << std::endl;
//
////	tree.Delete();
//
//	return 0;
//}
