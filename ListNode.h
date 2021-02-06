#include<iostream>
using namespace std;
struct ListNode {
	int val;
	ListNode* next;
	ListNode() :val(0), next(NULL) {};
	ListNode(int x) : val(x), next(NULL) {};
	ListNode(int x, ListNode* p) :val(x), next(p) {};
};