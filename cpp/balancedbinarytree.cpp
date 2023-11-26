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
int deep_of_tree(TreeNode *root){
    if(!root)
        return 0;
    int a=deep_of_tree(root->left),b=deep_of_tree(root->right);
    return (a>b)? (a+1) : (b+1);
}
bool isBalanced(TreeNode* root) {
    if(!root)
        return true;
    int a=deep_of_tree(root->left),b=deep_of_tree(root->right);
    if((a-b)<-1||(a-b)>1)
        return false;
    return isBalanced(root->left)&&isBalanced(root->right);
}
void tree_to_vector(TreeNode *root,vector<int> &res){
    if(!root){
        tree_to_vector(root->left,res);
        res.push_back(root->val);
        tree_to_vector(root->right,res);
    }
}
TreeNode *vector_to_balancedtree(vector<int>::iterator begin,int len){
    if(len<=0)
        return NULL;
    int x=(len-1)/2;
    TreeNode *root=new TreeNode(*(begin+x));
    root->left=vector_to_balancedtree(begin,x);
    root->right=vector_to_balancedtree(begin+x+1,len-x-1);
    return root;
}
TreeNode* balanceBST(TreeNode* root) {
    vector<int> a;
    tree_to_vector(root,a);
    return vector_to_balancedtree(a.begin(),a.size());
}
ListNode *split_list(ListNode *head){
    ListNode *slow=head,*fast=slow->next;
    while(fast&&fast->next){
        fast=fast->next->next;slow=slow->next;
    }
    ListNode *tem=slow->next;slow->next=NULL;
    return tem;
}
TreeNode* sortedListToBST(ListNode* head) {
    if(!head)
        return NULL;
    if(!head->next)
        return new TreeNode (head->val);
    ListNode *p=split_list(head);
    TreeNode *root=new TreeNode(p->val);
    root->left=sortedListToBST(head);root->right=sortedListToBST(p->next);
    return root;
}