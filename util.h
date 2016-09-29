#ifndef UTIL_H
#define UTIL_H

void Split(std::string& s, const std::string delim, std::vector< std::string >* ret);  

inline void swap(int& a, int& b);

void RandomPerm(int* A, const int k, const int n);

class myTriple {
public:
	float first;
	int second;
	int third;

	myTriple() {
		first = 0.0f;
		second = 0;
		third = 0;
	} 
	myTriple(float f, int s, int t): first(f), second(s), third(t) { }
	bool operator<(const myTriple& _triple) const {
		if(this->first < _triple.first) {
			return false;
		} else if (this->first > _triple.first) {
			return true;
		} else {
			if(this->second > _triple.second) {
				return false;
			} else {
				return true;
			} 
		}
	}
};	

#endif
