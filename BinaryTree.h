#ifndef BINARYTREE_H
#define BINARYTREE_H

class node;
class BinaryTree;
class DecisionTreeCART;

#define UNDEFINED -1

class node {
private:
	friend class BinaryTree;
	friend class DecisionTreeCART;
	/* attribute of binary tree */
	int key;
	int depth;
	node* parent;
	node* left;
	node* right;

	/* attribute of split condition */
	int splitFeatureIndex;
	float splitValue;
	
	/* if the node is a leaf node, 
 	 * the labelIndex determines items
 	 * in this leaf node belong to which
 	 * class
 	 */
	int labelIndex;

	/* if the decision tree 
 	 * is a regression tree,
 	 * the predictValue will
 	 * give the regression value
 	 */
	float predictValue;
	
	/* these attributes store variables
 	 * used in pruning procedure 
 	 */
	float alpha;
	int leafNumber;
	float rt;
	float rTree;

	/* the vector dataIndex stores index
     * of items of this node
     */
	vector<int> dataIndex;

public:
	node(); /* constructor */
	~node() { }
}; // end class node

class BinaryTree {
private:
	node* root;

public:
	BinaryTree();
	virtual ~BinaryTree();	
	virtual void Print();
	node* TreeMinimum(node* curNode);
	node* TreeMaximum(node* curNode);
	node* Successor(node* curNode);
};

#endif
