#include <iostream>
#include <vector>

using namespace std;
#include "BinaryTree.h"

BinaryTree::BinaryTree(): 
	root(NULL) { };

BinaryTree::~BinaryTree()
{
}

void BinaryTree::Print()
{
	cerr << "This is a Binary Tree\n";	

	return;
}

node* BinaryTree::TreeMinimum(node* curNode)
{
	if(curNode == NULL)  return curNode;
	while(curNode->left != NULL) {
		curNode = curNode->left;
	}	

	return curNode;
}

node* BinaryTree::TreeMaximum(node* curNode)
{
	if(curNode == NULL) return curNode;
	while(curNode->right != NULL) {
		curNode = curNode->right;
	}
	
	return curNode;
}

node* BinaryTree::Successor(node* curNode)
{
	if(curNode == NULL) return curNode;

	if(curNode->right != NULL) {
		return TreeMinimum(curNode->right);
	}

	while(curNode->parent != NULL && curNode == curNode->parent->right) {
		curNode = curNode->parent;
	}

	return curNode->parent;
}
