#include<bits/stdc++.h>
#include "TreeNode.h"
using namespace std;
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
int main(){
    
    return 0;
}