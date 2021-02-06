#include <bits/stdc++.h>
using namespace std;
struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode() : val(0), left(nullptr), right(nullptr) {}
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
};
TreeNode* invertTree(TreeNode* root) {
    if(!root)
        return NULL;
    TreeNode *t=root->right;
    root->right=invertTree(root->left);root->left=invertTree(t);
    return root;
}
TreeNode *find_right_end(TreeNode *root){//assert root!=NULL
    while(root->right)
        root=root->right;
    return root;
}
void flatten(TreeNode* root) {
    if(root&&root->left){
        TreeNode *tem=root->right;root->right=root->left;
        flatten(root->left);root->left=NULL;
        find_right_end(root->left)->right=tem;
    }
}
bool has_node(TreeNode *root,TreeNode *p){
    if(!root)
        return false;
    if(root==p)
        return true;
    if(has_node(root->right,p)||has_node(root->left,p))
        return true;
    return false;
}
TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {//时间长
    while(root){
        if(root==p||root==q)
            return root;
        if(has_node(root->left,p)&&has_node(root->left,q))
            root=root->left;
        else if(has_node(root->right,p)&&has_node(root->right,q))
            root=root->right;
        else if((has_node(root->right,p)&&has_node(root->left,q))||
        (has_node(root->right,q)&&has_node(root->left,p)))
            return root;
    }
}
/*
*/
int find_max(vector<int>::iterator begin,int size){
    int max=*begin;int pos=0;++begin;
    for(int i=1;i<size;++i,++begin){
        if(*begin>max){
            pos=i;max=*begin;
        }
    }
    return pos;
}
TreeNode* constructMaximumBinaryTree(vector<int>::iterator begin,int size) {
    if(size<=0)
        return NULL;
    if(size==1){
        TreeNode *root=new TreeNode(*begin);
        return root;
    }
    int m=find_max(begin,size);
    TreeNode *root=new TreeNode(*(begin+m),constructMaximumBinaryTree(begin,m),
    constructMaximumBinaryTree(begin+m+1,size-m-1));
    return root;
}
TreeNode* constructMaximumBinaryTree(vector<int>& nums) {
    int size=nums.size();
    return constructMaximumBinaryTree(nums.begin(),size);
}
class Node {
public:
    int val;
    Node* left;
    Node* right;
    Node* next;
    Node() : val(0), left(NULL), right(NULL), next(NULL) {}
    Node(int _val) : val(_val), left(NULL), right(NULL), next(NULL) {}
    Node(int _val, Node* _left, Node* _right, Node* _next)
    : val(_val), left(_left), right(_right), next(_next) {}
};
void connect_node(Node *root){
    if(root&&root->right&&root->left){
        Node *n1=root->left->right,*n2=root->right->left;
        while(n1&&n2){
            n1->next=n2;
            n1=n1->right;n2=n2->left;
        }
    }
}
Node* connect(Node* root) {
    if(!root)
        return NULL;
    if(root->left)
        root->left->next=root->right;
    connect_node(root);
    connect(root->left);connect(root->right);
    return root;
}
int deep_of_tree(TreeNode *root){
    if(!root)
        return 0;
    int a=deep_of_tree(root->left),b=deep_of_tree(root->right);
    return (a>b)? (a+1) : (b+1);
}
string serialize(TreeNode* root) {
    if(!root)
        return "";
    int count=1<<(deep_of_tree(root)-1)-1;
    string s;queue<TreeNode*> node;node.push(root);
    while(count!=0){
        if(node.front()){
            root=node.front();
            s.append(",").append(to_string(root->val));
            node.pop();--count;
            node.push(root->left);node.push(root->right);
        }
        else{
            s.append(",#");
            node.pop();--count;
            node.push(NULL);node.push(NULL);
        }
    }
    s.erase(0,1);
    return s;
}
int read_from_string(string::iterator &str,const string::iterator &end){
    int res=0;
    for(;*str!=','&&str!=end;++str){
        res=res*10+(*str-'0');
    }
    if(str!=end)
        ++str;
    return res;
}
void build_tree(TreeNode *root,int val,int pos){
    if(pos==1){
        root->val=val;return ;
    }
    int len=0,x=pos;
    while(x){
        x>>=1;++len;
    }
    x=(1<<(len-1));
    while(pos){
        if(pos&x){
            if(!root->right)
                root->right=new TreeNode;
            root=root->right;
        }
        else{
            if(!root->left)
                root->left=new TreeNode;
            root=root->left;
        }
        x>>=1;
    }
    root->val=val;
}
TreeNode* deserialize(string data) {//1,2,3,#,#,4,5
    if(data.empty())
        return NULL;
    string::iterator str=data.begin(),end=data.end();
    TreeNode *root=new TreeNode;int i=1;
    while(str!=end){
        if(*str!='#')
            build_tree(root,read_from_string(str,end),i++);
        else{gf
            ++str;
            if(str==end)
                return root;
            else{
                ++str;++i;
            }
        }
    }
    return root;
}