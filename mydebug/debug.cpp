#include<bits/stdc++.h>
using namespace std;
struct ListNode {
	int val;
	ListNode* next;
	ListNode() :val(0), next(NULL) {};
	ListNode(int x) : val(x), next(NULL) {};
	ListNode(int x, ListNode* p) :val(x), next(p) {};
};
struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode() : val(0), left(nullptr), right(nullptr) {}
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
};
int strStr(string haystack, string needle) {
    int h_size=haystack.size(),n_size=needle.size(),i=0,j=0;
    while (j<n_size){
        if(haystack[i]==needle[j]){
            ++i;++j;
        }
        else{
            i=i-j+1;j=0;
            if(i>=h_size)
                return -1;
        }
    }
    return i-n_size;
}
int main(){
    string a="mississippi",b="issip";
    cout << strStr(a,b);
    return 0;
}