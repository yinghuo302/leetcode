#include<bits/stdc++.h>
using namespace std;
struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode() : val(0), left(nullptr), right(nullptr) {}
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
};
bool has_node(TreeNode *root,const int &deep,const int &n){
    int bin=(1<<(deep-2));
    while(bin&&root){
        if(bin&n)
            root=root->right;
        else
            root=root->left;
        bin>>=1;
    }
    return (root!=NULL);
}
int countNodes(TreeNode* root) {
    if(!root)
        return 0;
    int deep=1;
    while(root->left){
        root=root->left;++deep;
    }
    int n1=(1<<(deep-1)),n2=(n1<<1)-1;
    while(n1<n2){
        int mid=(n1-n2+1)/2+n1; 
        if(has_node(root,deep,mid))
            n1=mid;
        else
            n2=mid-1;
    }
    return n1;
}