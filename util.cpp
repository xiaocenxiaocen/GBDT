#include <vector>
#include <string>
#include <cstring>
#include <cstdlib>
#include <ctime>

#include "util.h"

void Split(std::string& s, const std::string delim, std::vector< std::string >* ret)  
{  
    size_t last = 0;  
    size_t index=s.find_first_of(delim,last);  
    while (index!=std::string::npos)  
    {  
        ret->push_back(s.substr(last,index-last));  
        last=index+1;  
        index=s.find_first_of(delim,last);  
    }  
    if (index-last>0)  
    {  
        ret->push_back(s.substr(last,index-last));  
    }  
}  

inline void swap(int& a, int& b)
{
	int tmp = a;
	a = b;
	b = tmp;
}

void RandomPerm(int* A, const int k, const int n)
{
	for(int i = 0; i < k ; i++) {
		int j = i + (rand() % (n - i));
		swap(A[i], A[j]);
	}
}
