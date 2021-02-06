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
int kth_smallest(TreeNode*root,int &k){
    if(!root)
        return -1;
    if(root->left){
        int a=kth_smallest(root->left,k);
        if(a!=-1)
            return a;
    }
    --k;
    if(k==0)
        return root->val;
    if(root->right){
        int b=kth_smallest(root->right,k);
        if(b!=-1)
            return b;
    }
    return -1;
}
int kthSmallest(TreeNode* root, int k) {
    return kth_smallest(root,k);
}
void convert_bst(TreeNode*root,int &sum){
    if(root->right)
        convert_bst(root->right,sum);
    root->val=root->val+sum;sum=root->val;
    if(root->left)
        convert_bst(root->left,sum);
}
TreeNode* convertBST(TreeNode* root) {
    if(!root)
        return NULL;
    int sum=0;convert_bst(root,sum);
    return root;      
}
TreeNode* deleteNode(TreeNode* root, int key) {
    if(!root)
        return NULL;
    if(root->val==key){
        if(!root->left){//情况1删除节点为无子树或只无左子树
            TreeNode *tem=root->right;delete(root);
            return tem;
        }
        if(!root->right){//情况2删除节点只无右子树
            TreeNode *tem=root->left;delete(root);
            return tem;
        }
        TreeNode *ptr=root->right;//情况3删除节点左右子树均存在，将左子树移至右子树的最左树节点上
        while(ptr->left){
            ptr=ptr->left;
        }
        ptr->left=root->left;
        ptr=root->right;delete(root);
        return ptr;
    }
    if(key<root->val)
        root->left=deleteNode(root->left,key);
    else
        root->right=deleteNode(root->right,key);
    return root;
}
TreeNode* searchBST(TreeNode* root, int val) {
    if(!root)
        return NULL;
    if(root->val==val)
        return root;
    if(val<root->val)
        return searchBST(root->left,val);
    return searchBST(root->right,val);
}
bool tree_smaller_than(TreeNode *root,int k){
    if(root)
        return true;
    if(root->val>=k)
        return false;
    return tree_smaller_than(root->left,k)&&tree_smaller_than(root->right,k);
}
bool tree_larger_than(TreeNode *root,int k){
    if(!root)
        return true;
    if(root->val<=k)
        return false;
    return tree_larger_than(root->left,k)&&tree_larger_than(root->right,k);
}
bool isValidBST(TreeNode* root) {
    if(!root)
        return true;
    if(!tree_smaller_than(root->left,root->val)||!tree_larger_than(root->right,root->val))
        return false;
    return isValidBST(root->left)&&isValidBST(root->right);
}
TreeNode* insertIntoBST(TreeNode* root, int val) {
    if(!root)
        return new TreeNode(val);
    if(val<root->val)
        root->left=insertIntoBST(root->left,val);
    else
        root->right=insertIntoBST(root->right,val);
    return root;
}