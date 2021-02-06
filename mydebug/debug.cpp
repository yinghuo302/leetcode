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
string serialize(TreeNode* root) {
    if(!root)
        return "";
    int count=(1<<deep_of_tree(root))-1;
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
    while(x){
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
    TreeNode *root=new TreeNode(data[0]-'0');int i=1;
    while(str!=end){
        if(*str!='#'){
            build_tree(root,read_from_string(str,end),i++);
        }
        else{
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
void myprint(TreeNode *root){
    if(root){
        cout << root->val;
        if(root->left)
            myprint(root->left);
        if(root->right)
            myprint(root->left);
    }
}
inline int cmp1(vector<int>& a,vector<int>& b){
    return a[0]<b[0];
}
ListNode *detectCycle(ListNode *head) {
    if (!head)
		return NULL;
	ListNode* fast=head,*slow=head;bool has_cycle=false;
	while (fast) {
		if (fast->next)
			fast = fast->next->next;
		else
			return NULL;
		slow = slow->next;
		if (slow == fast){
			slow=head;has_cycle=true;break;
        }
	}
	if(!has_cycle)
        return NULL;
    while (fast != slow) {
		slow = slow->next; fast = fast->next;
	}
	return slow;
}
int main(){
    string s="1,2,3,#,#,4,5";
    TreeNode *root=deserialize(s);
    myprint(root);cout << endl;
    cout << serialize(root);//1,#,0,#,#,2,3,#,#,#,#,#,#,4,5
    return 0;
}