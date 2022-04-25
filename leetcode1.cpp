#include <bits/stdc++.h>
using namespace std;
int strStr(string haystack, string needle){
    int h_size = haystack.size(), n_size = needle.size(), i = 0, j = 0;
    while (j < n_size){
        if (haystack[i] == needle[j]){
            ++i;
            ++j;
        }
        else{
            i = i - j + 1;
            j = 0;
            if (i >= h_size)
                return -1;
        }
    }
    return i - n_size;
}
//股票价格跨度，股票价格小于或等于今天价格的最大连续日数，包括今天
class StockSpanner{
private:
    stack<int> s;
    vector<int> prices;
public:
    StockSpanner(){}
    int next(int price){
        while (!s.empty() && price >= prices[s.top()])
            s.pop();
        int num = 0;
        if (s.empty())
            num = prices.size() + 1;
        else
            num = prices.size() - s.top();
        prices.push_back(price);
        s.push(prices.size() - 1);
        return num;
    }
};
//设计推特
typedef int User_Id;
typedef int Tweet_time;
typedef int Tweet_Id;
struct TweetNode{
    Tweet_Id id;
    Tweet_time time;
    TweetNode *next;
    TweetNode(int id, int time) : id(id), time(time), next(NULL) {}
    TweetNode() : id(-1), time(-1), next(NULL) {}
};
class User{
private:
    TweetNode *tweets;
public:
    unordered_set<User_Id> followees;
    User() : tweets(NULL) {}
    User(int id) : tweets(NULL) { followees.emplace(id); }
    void like(int like_id){
        followees.emplace(like_id);
    }
    void dislike(int like_id){
        int size = followees.size(), i = 0;
        unordered_set<int>::iterator followee = followees.begin();
        while (followee != followees.end())
            if (like_id == *followee)
                followees.erase(followee);
    }
    TweetNode *get_tweet(){
        return tweets;
    }
    void post_tweet(int tweet_id, int time){
        if (!tweets)
            tweets = new TweetNode(tweet_id, time);
        else
            tweets->next = new TweetNode(tweet_id, time);
    }
};
class Twitter{
private:
    unordered_map<User_Id, User *> users;
    Tweet_time time;
    TweetNode *mergeTwoLists(TweetNode *l1, TweetNode *l2){
        if (l1 == NULL)
            return l2;
        if (l2 == NULL)
            return l1;
        if (l1->time > l2->time){
            l1->next = mergeTwoLists(l1->next, l2);
            return l1;
        }
        else{
            l2->next = mergeTwoLists(l1, l2->next);
            return l2;
        }
    }
    TweetNode *deep_copy(TweetNode *head){
        if (!NULL)
            return NULL;
        TweetNode *root = new TweetNode(head->id, head->time), *p = root;
        while (head){
            p->next = new TweetNode(head->id, head->time);
            p = p->next;
        }
        p = root->next;
        delete (root);
        return p;
    }
    void deep_del(TweetNode *head){
        if (!head)
            return;
        TweetNode *p = head->next;
        while (p){
            delete (head);
            head = p;
            p = p->next;
        }
        delete (head);
    }
public:
    Twitter(){
        time = 0;
    }
    void postTweet(int userId, int tweetId){
        if (users.empty()){
            users.emplace(userId, new User(userId));
            users[userId]->post_tweet(tweetId, time);
            return;
        }
        unordered_map<User_Id, User *>::iterator usr_iter = users.find(userId);
        if (users.end() == usr_iter){
            users.emplace(userId, new User(userId));
            usr_iter->second->post_tweet(tweetId, time);
        }
        else
            usr_iter->second->post_tweet(tweetId, time);
        ++time;
    }
    vector<int> getNewsFeed(int userId){
        TweetNode *r = NULL;
        for (auto &followee : users[userId]->followees){
            r = mergeTwoLists(r, deep_copy(users[followee]->get_tweet()));
        }
        vector<int> res;
        TweetNode *h = r;
        while (res.size() <= 10 && r){
            res.push_back(r->id);
            r = r->next;
        }
        deep_del(h);
        return res;
    }
    void follow(int followerId, int followeeId){
        if (users.empty())
            return;
        unordered_map<User_Id, User *>::iterator usr_iter = users.find(followerId);
        if (users.end() != usr_iter)
            usr_iter->second->like(followeeId);
    }
    void unfollow(int followerId, int followeeId){
        if (users.empty())
            return;
        unordered_map<User_Id, User *>::iterator usr_iter = users.find(followerId);
        if (users.end() != usr_iter)
            usr_iter->second->dislike(followeeId);
    }
};
//在数组中找两个数和为target
vector<int> twoSum(vector<int> &nums, int target){
    unordered_map<int, int> m;
    int size = nums.size();
    for (int i = 0; i < size; ++i){
        unordered_map<int, int>::iterator a = m.find(target - nums[i]);
        if (a != m.end())
            return {i, a->second};
        m.emplace(nums[i], i);
    }
    return {};
}
//连续1的个数
int findMaxConsecutiveOnes(vector<int> &nums){
    int size = nums.size();
    int res = 0, count = 0;
    for (int i = 0; i < size; ++i){
        if (nums[i])
            ++count;
        else if (count > res)
            res = count;
    }
    return (count > res) ? count : res;
}
//如果矩阵上每一条由左上到右下的对角线上的元素都相同，那么这个矩阵是托普利茨矩阵。
bool isToeplitzMatrix(vector<vector<int>> &matrix){
    int n1 = matrix.size(), n2 = matrix[0].size();
    for (int i = 1; i < n1;i++) 
        for (int j = 1; j < n2; j++) 
            if (matrix[i][j] != matrix[i - 1][j - 1]) 
                    return false; 
    return true;
}
//
int longestSubarray(vector<int>& nums, int limit) {
    int size=nums.size(),left=0,right=0,res=0;
    multiset<int> s;
    while(right<size){
        s.insert(nums[right]);
        while(*s.begin()-*s.rbegin()>limit){
            s.erase(s.find(nums[left]));
            ++left;
        }
        if(right-left+1>res)
            res=right-left+1;
        ++right;
    }
    return res;
}
//爱生气的书店老板
int maxSatisfied(vector<int>& customers, vector<int>& grumpy, int X) {
    int size=customers.size(),no_use=0,use=0;
    for(int i=0;i<size;++i)
        if(!grumpy[i])
            no_use+=customers[i];
    int left=0,right=X,tem=0;
    for(int i=0;i<right&&i<size;++i)
        if(grumpy[i])
            tem+=customers[i];
    while(right<size){
        tem=tem+grumpy[right]-grumpy[left];
        if(tem>use)
            use=tem;
        ++right;++right;
    }
    return no_use+use;
}
/*给定一个非空且只包含非负数的整数数组 nums，数组的度的定义是指数组里任一元素出现频数的最大值。
你的任务是在nums中找到与nums拥有相同大小的度的最短连续子数组，返回其长度。*/
int findShortestSubArray(vector<int>& nums) {
    unordered_map<int,vector<int>> m;
    int size=nums.size();
    for(int i=0;i<size;++i){
        unordered_map<int,vector<int>>::iterator a=m.find(nums[i]);
        if(a!=m.end()){
            ++(a->second[0]);
            a->second[2]=i;
        }
        else
            m[nums[i]]={1,i,i};
    }
    unordered_map<int,vector<int>>::iterator i1=m.begin(),i2=m.end();
    int max_d=0,min_l=size;
    while(i1!=i2){
        if(i1->second[0]>max_d){
            max_d=i1->second[0];
            min_l=i1->second[2]-i1->second[1]+1;
        }
        if(i1->second[0]==max_d)
            if(i1->second[2]-i1->second[1]+1<min_l)
                min_l=i1->second[2]-i1->second[1]+1;
        ++i1;
    }
    return min_l;
}
//给定一个由若干0和1组成的数组A，我们最多可以将K个值从0变成1。
int longestOnes(vector<int>& A, int K) {
    int size=A.size(),left=0,right=1,num_of_zero=1,res=0;
    if(A[0])
        num_of_zero=0;
    while(right<size){
        if(!A[right])
            ++num_of_zero;
        while(num_of_zero>K){
            if(!A[left])
                --num_of_zero;
            --left;
        }
        if(right-left+1>res)
            res=right-left+1;
    }
    return res;
}
/*给你一个字符串s，一个字符串t。返回s中涵盖t所有字符的最小子串，时间超限解法：
bool has_letter(const string &s,const string &t,int begin,int end){
    int size=t.size();
    if(end-begin+1<size)
        return false;
    string s1=s.substr(begin,end-begin+1);
    for(int i=0;i<size;++i){
        string::size_type pos=s1.find(t[i]);
        if(pos==s1.npos)
            return false;
        else
            s1.erase(pos,1);
    }
    return true;
}
string minWindow(string s, string t) {
    int size=s.size(),n=t.size(),left=0,right=n-1,min_l=INT_MAX,r_left=0;
    while(right<size){
        left=right-n+1;
        while(left>=0&&!has_letter(s,t,left,right))
            --left;
        if(left>=0&&right-left+1<min_l){
            min_l=right-left+1;
            r_left=left;
        }
        ++right;
    }
    return (min_l==INT_MAX)? string() : s.substr(r_left,min_l);
}
*/
bool has_letter(const string &s,const string &t,int begin,int end){
    int size=t.size();
    if(begin-end+1<size)
        return false;
    for(int i=0;i<size;++i){
        if(s.find(t[i])==s.npos)
            return false;
    }
    return true;
}
/*
string minWindow(string s, string t) {
    int size=s.size(),n=t.size(),left=0,right=n-1,min_l=INT_MAX,r_left=0;
    if(n>size)
        return string();
    unordered_map<char,int> m;
    for(int i=0;i<n;++i)
        m.emplace(t[i],0);
    for(int i=0;i<right;++i){
        unordered_map<char,int>::iterator m_i=m.find(s[i]);
        if(m_i!=m.end())
            ++m_i->second;
    }
    while(right<size){
        left=right-n+1;
        unordered_map<char,int>::iterator m_i=m.find(s[left]);
        if(m_i!=m.end())
            ++m_i->second;
        while(!has_letter(m)&&left>=0)
            --left;
        if(right-left+1<min_l&&left>=0){
            min_l=right-left+1;
            r_left=left;
        }
        ++right;
    }
    return (min_l==INT_MAX)? string() : s.substr(r_left,min_l);
}
*/
//有序数组中找两个数和为target
vector<int> twoSum(vector<int>& numbers, int target) {
    int size=numbers.size(),left=0,right=size-1;
    while(left<right){
        if(numbers[left]+numbers[right]==target)
            return vector<int>{left+1,right+1};
        else if(numbers[left]+numbers[right]>target)
            --right;
        else
            ++left;
    }
    return vector<int>{-1,-1};
}
//水平翻转图像，并反转图像,水平翻转图片是将图片的每一行进行翻转，即逆序,反转图片的意思是图片中的0全部被1替换，1全部被0替换
vector<vector<int>> flipAndInvertImage(vector<vector<int>>& A) {
    int size=A.size();
    for(int i=0;i<size;++i){
        int left=0,right=size-1;
        while (left<right){
            if(A[i][left]==A[i][right]){
                A[i][left]=!A[i][left];
                A[i][right]=!A[i][right];
            }
            ++left;--right;
        }
        if(right==left)
            A[i][left]=!A[i][right];
    }
    return A;
}
//反转字符串
void reverseString(vector<char>& s) {
    if(!s.empty()){
        vector<char>::iterator left=s.begin(),right=s.end()-1;
        while(left<right){
            char tem=*right;*right=*left;*left=tem;
            --right;++left;
        }
    }
}
//常数时间插入，删除，获取随机元素
class RandomizedSet {
private:
    unordered_map<int,int> m;
    vector<int> v;
public:
    RandomizedSet() {
    }
    bool insert(int val) {
        if(m.find(val)==m.end())
            return false;
        v.push_back(val);
        m.emplace(val,v.size());
        return true;
    }
    bool remove(int val) {
        unordered_map<int,int>::iterator i=m.find(val);
        if(i==m.end())
            return false;
        v[i->second]=*v.rbegin();m[*v.rbegin()]=i->second;
        v.pop_back();m.erase(i);
        return true;
    }
    int getRandom() {
        return v[rand()%v.size()];
    }
};
bool check(const int* inWindow,const int* target){
    for(int i=0;i<26;++i)
        if(inWindow[i]!=target[i])
            return false;
    return true;
}
vector<int> findAnagrams(string s, string p) {
    int size=s.size(),n=p.size(),left=0,right=n;
    int inWindow[26]={0},target[26]={0};vector<int> res;
    for(int i=0;i<n;++i)
        ++target[p[i]-'a'];
    for(int i=0;i<right&&i<size;++i)
        ++inWindow[s[i]-'a'];
    if(check(inWindow,target))
        res.push_back(0);
    while (right<size){
        ++inWindow[s[right]-'a'];
        --inWindow[s[left]-'a'];++left;
        if(check(inWindow,target))
            res.push_back(left);
        ++right;
    }
    return res;
}
//转置矩阵
vector<vector<int>> transpose(vector<vector<int>>& matrix) {
    int m=matrix.size(),n=matrix[0].size();
    vector<vector<int>> res(n,vector<int>(m));
    for(int i=0;i<m;++i)
        for(int j=0;j<n;++j)
            res[j][i]=matrix[i][j];
    return res;
}
//最长回文子串
pair<int,int> expand(const string &s,int centor){//assert s.size()>=2
    pair<int,int> res;
    int left=centor,right=centor,size=s.size();
    if(!centor){
        if(s[0]==s[1])
            return {0,2};
        else
            return {0,1};
    }
    while (left>=0&&right<size&&s[left]==s[right]){
        ++right;--left;
    }
    res={left+1,right-left-1};
    if(s[centor]!=s[centor-1])
        return res;
    left=centor-1,right=centor;
    while (left>=0&&right<size&&s[left]==s[right]){
        ++right;--left;
    }
    if(right-left-1>res.second-res.first)
        res={left+1,right-left-1};
    return res;
}
string longestPalindrome(string s) {
    int size=s.size();
    if(size==1)
        return s;
    int len=1,left=0;
    for(int i=0;i<size;++i){
        pair<int,int> tem=expand(s,i);
        cout << tem.second <<" ";
        if(tem.second>len){
            len=tem.second;left=tem.first;
        }
    }
    return s.substr(left,len);
}
//给你一个字符串s和一个整数k，请你找出s中的最长子串，要求该子串中的每一字符出现次数都不少于k
int dfs(const string s,int left,int right,int k){//辅助函数
    int c[26]={0};
    for(int i=left;i<=right;++i)
        ++c[s[i]-'a'];
    char split=0;
    for(int i=0;i<26;++i)
        if(c[i]&&c[i]<k){
            split=i+'a';break;
        }
    if(!split)
        return right-left+1;
    int res=0;
    for(int i=left;i<right;++i){
        if(s[i]==split){
            if(i!=left)
                res=max(res,dfs(s,left,i-1,k));
            left=i+1;
        }
    }
    if(s[right]==split)
        res=max(res,dfs(s,left,right-1,k));
    else
        res=max(res,dfs(s,left,right,k));
    return res;
}
int longestSubstring(string s, int k) {
    return dfs(s,0,s.size()-1,k);
}
//十进制反转数位,若溢出，则返回0
int reverse(int x) {
    int res=0;
    while(!x){
        int tem=x%10;
        x/=10;
        if(res>INT_MAX/10||(res==INT_MAX/10&&tem>7))
            return 0;
        if(res<INT_MIN/10||(res==INT_MIN/10&&tem<-8))
            return 0;
        res=res*10+tem;
    }
    return res;
}
//LRU(最近最少使用)缓存机制,目前存在错误address sanitizer: heap-use-after-free on address
struct LinkList{
    int key;
    int val;
    LinkList* next;
    LinkList* prev;
    LinkList():key(-1),val(-1),next(NULL),prev(NULL){}
    LinkList(int key,int value):key(key),val(value),next(NULL),prev(NULL){}
    LinkList(int key,int value,LinkList* next):key(key),val(value),next(next){}
};
class LRUCache {
private:
    unordered_map<int,LinkList*> data;
    LinkList* head;
    LinkList* tail;
    int size;
    int capacity;
    void move_to_head(LinkList* node){
        if(node!=head){
            if(node==tail){
                tail=tail->prev;
                tail->next=NULL;
                node->prev=NULL;
            }
        }
        else{
            node->prev->next=node->next;
            node->next->prev=node->prev;
        }
        node->next=head;
        head=node;
        head->next->prev=head;
    }
    void add_to_head(LinkList* node){
        node->next=head;
        head=node;
        if(size<capacity){
            if(!size)
                tail=node;
            else
                head->next->prev=head;
            ++size;
        }
        else{
            head->next->prev=head;
            data.erase(tail->key);
            LinkList* tem=tail;
            tail=tail->prev;
            delete(tem);
        }
    }
public:
    LRUCache(int capacity):capacity(capacity),head(NULL),tail(NULL),size(0) {
    }
    int get(int key) {
        unordered_map<int,LinkList*>::iterator i1=data.find(key);
        if(i1!=data.end()){
            move_to_head(i1->second);
            return i1->second->val;
        }
        return -1;
    }
    void put(int key, int value){
        unordered_map<int,LinkList*>::iterator i1=data.find(key);
        if(i1!=data.end()){
            i1->second->val=value;
            move_to_head(i1->second);
        }
        else{
            data.emplace(key,new LinkList(key,value));
            add_to_head(data[key]);
        }
    }
};
//LFU(最不经常使用)缓存机制
class LFUCache {
public:
    LFUCache(int capacity) {
    }
    int get(int key) {
    }
    void put(int key, int value) {
    }
};
//判断是否为单调数组
bool isIncreasing(const vector<int>& A){
    int size=A.size();
    for(int i=0;i<size-1;++i)
        if(A[i]<A[i+1])
            return false;
    return true;
}
bool isDecreasing(const vector<int>& A){
    int size=A.size();
    for(int i=0;i<size-1;++i)
        if(A[i]>A[i+1])
            return false;
    return true;
}
bool isMonotonic(vector<int>& A) {
    return isIncreasing(A)||isDecreasing(A);
}
//字符串转换为整数
int myAtoi(string s) {
    int size=s.size(),i=0,res=0;
    for(;i<size&&s[i]==' ';++i);
    bool is_negative=false;
    if(s[i]=='-'){
        is_negative=true;++i;
    }
    else if(s[i]=='+')
        ++i;
    for(;i<size;++i){
        int tem=s[i]-'0';
        if(tem>=0&&tem<=9){
            if(res>INT_MAX/10||(res==INT_MAX/10&&tem>7))
                return INT_MAX;
            if(res<INT_MIN/10||(res==INT_MIN/10&&tem>8))
                return INT_MIN;
            if(is_negative)
                res=res*10-tem;
            else
                res=res*10+tem;
        }
        else
            break;
    }
    return res;
}
//是否为回文数
bool isPalindrome(int x) {
    if(x<0||(!x&&x%10==0))
        return false;
    int reversed=0;
    while(x>reversed){
        reversed=reversed*10+x%10;
        x/=10;
    }
    return x==reversed||x==(reversed%10);
}
//求数组pos为i到j的元素之和
class NumArray {
private:
    vector<int> sums;
public:
    NumArray(vector<int>& nums) {
        if(!nums.empty()){
            int size=nums.size();
            sums.resize(size+1);
            sums[0]=0;
            for(int i=0;i<size;++i)
                sums[i+1]=nums[i]+sums[i];
        }
    }
    int sumRange(int i, int j) {
        if(!sums.empty())
            return 0;
        else
            return sums[j+1]-sums[i];
    }
};
//罗马数字转整数
int romanToInt(string s) {
    unordered_map<char,int> mp={{'I',1},{'V',5},{'X',10},{'L',50},{'C',100},{'D',500},{'M',1000}};
    int size=s.size(),res=0;
    for(int i=0;i<size-1;++i){
        if(mp[s[i]]<mp[s[i+1]])
            res-=mp[s[i]];
        else
            res+=mp[s[i]];
    }
    res+=mp[s[size-1]];
    return res;
}
//整数转罗马数字
string intToRoman(int num) {
    vector<string> thousands = {"", "M", "MM", "MMM"};
    vector<string> hundreds = {"", "C", "CC", "CCC", "CD", "D", "DC", "DCC", "DCCC", "CM"};
    vector<string> tens = {"", "X", "XX", "XXX", "XL", "L", "LX", "LXX", "LXXX", "XC"};
    vector<string> ones = {"", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX"};
    return thousands[num / 1000] + hundreds[num % 1000 / 100] + tens[num % 100 / 10] + ones[num % 10];
}
//除去排序数组中的重复元素
int removeDuplicates(vector<int>& nums) {

}
//区域检索
class NumMatrix {
private:
    vector<vector<int>> sums;
public:
    NumMatrix(vector<vector<int>>& matrix) {
        if(!matrix.size()){
            int m=matrix.size(),n=matrix[0].size();
            sums.resize(m+1,vector<int>(n+1));
            for(int i=0;i<=m;++i)
                for(int j=0;j<=n;++j)
                    if(!i||!j)
                        sums[i][j]==0;
                    else
                        sums[i][j]=sums[i][j-1]+sums[i-1][j]-sums[i-1][j-1]+matrix[i-1][j-1];
        }
    }
    int sumRegion(int row1, int col1, int row2, int col2) {
        if(!sums.empty())
            return 0;
        else
            return sums[row2+1][col2+1]-sums[row2][col1+1]-sums[row1][col2+1]+sums[row1][col1];
    }
};
//最长公共前缀
string longestCommonPrefix(vector<string>& strs) {
    if(strs.empty())
        return "";
    int m=strs.size(),n=strs[0].size();
    string res;
    for(int i=0;i<n;++i){
        for(int j=1;j<m;++j){
            if(i==strs[j].size()||strs[j][i]!=strs[0][i])
                return res;
        }
        res.push_back(strs[0][i]);
    }
    return res;
}
//对于 0 ≤ i ≤ num 范围中的每个数字 i ，计算其二进制数中的 1 的数目并将它们作为数组返回。
vector<int> countBits(int num) {
    vector<int> res(num+1);
    for(int i=0;i<=num;++i)
        if(!i||i==1)
            res[i]=i;
        else
            res[i]=res[i>>1]+i&1;
    return res;
}
// 多线程，按序打印
class Foo {
    std::mutex m1,m2;
public:
    Foo() {
        m1.lock();
        m2.lock();
    }

    void first(std::function<void()> printFirst) {
        
        // printFirst() outputs "first". Do not change or remove this line.
        printFirst();
        m1.unlock();
    }

    void second(std::function<void()> printSecond) {
        
        // printSecond() outputs "second". Do not change or remove this line.
        m1.lock();
        printSecond();
        m1.unlock();
        m2.unlock();
    }

    void third(std::function<void()> printThird) {
        
        // printThird() outputs "third". Do not change or remove this line.
        m2.lock();
        printThird();
        m2.unlock();
    }
};
// 133克隆图 https://leetcode-cn.com/problems/clone-graph/
class Node {
public:
    int val;
    vector<Node*> neighbors;
    Node() {
        val = 0;
        neighbors = vector<Node*>();
    }
    Node(int _val) {
        val = _val;
        neighbors = vector<Node*>();
    }
    Node(int _val, vector<Node*> _neighbors) {
        val = _val;
        neighbors = _neighbors;
    }
};

Node* cloneGraph(Node* node) {
    if(!node)
        return NULL;
    unordered_map<Node*,Node*> mp;
    mp.emplace(node,new Node(node->val));
    dfs(mp,node);
    unordered_set<Node*> st;
    assist(mp,st,node);
    return mp.find(node)->second;
}
void dfs(unordered_map<Node*,Node*>& mp,Node* cur){
    for(auto &nei : cur->neighbors){
        auto i = mp.find(nei);
        if(i==mp.end()){
            mp.emplace(nei,new Node(nei->val));
            dfs(mp,nei);
        }
    }
}

void assist(unordered_map<Node*,Node*>& mp,unordered_set<Node*>& st,Node *cur){
    if(st.find(cur)!=st.end())
        return ;
    auto copy_node = mp.find(cur)->second;
    st.emplace(cur);
    for(auto &nei:cur->neighbors){
        auto copy_nei = mp.find(nei)->second;
        copy_node->neighbors.emplace_back(copy_nei);
        assist(mp,st,nei);
    }
}
// 207课程表 https://leetcode-cn.com/problems/course-schedule/
// 210课程表 II https://leetcode-cn.com/problems/course-schedule-ii/
// 你这个学期必须选修 numCourses 门课程，记为 0 到 numCourses - 1 。
// 在选修某些课程之前需要一些先修课程。 先修课程按数组 prerequisites 给出，其中 prerequisites[i] = [ai, bi] ，
// 表示如果要学习课程 ai 则 必须 先学习课程  bi 。例如，先修课程对 [0, 1] 表示：想要学习课程 0 ，你需要先完成课程 1 。
// 请你判断是否可能完成所有课程的学习？如果可以，返回 true ；否则，返回 false 。
class Solution {
    int pos;
    bool valid;
public:
    vector<int> findOrder(int numCourses, vector<vector<int>>& prerequisites) {
        vector<vector<int>> edge(numCourses);
        char visited[numCourses];
        for(int i=0;i<numCourses;++i)
            visited[i] = 0;
        vector<int> res(numCourses);
        pos = numCourses;
        valid = true;
        for(auto &prerequire:prerequisites)
            edge[prerequire[1]].push_back(prerequire[0]);
        for(int i=0;(i<numCourses)&&valid;++i)
            if(!visited[i])
                dfs(edge,res,i,visited);
        if(valid)
            return res;
        else
            return {};
    }
    void dfs(const vector<vector<int>>& edge,vector<int>& res,int x,char* visited){
        visited[x] = 1;
        for(auto &y: edge[x]){
            if(!visited[y]){
                dfs(edge,res,y,visited);
                if(!valid)
                    return ;
            }
            else if(visited[y]==1){
                valid =  false;
                return ;
            }
        }
        res[--pos] = x;
        visited[x] = 2;
    }
};
// 310. 最小高度树 https://leetcode-cn.com/problems/minimum-height-trees/
// 给你一棵包含 n 个节点的树，标记为 0 到 n - 1 。给定数字 n 和一个有 n - 1 条无向边的 edges 列表（每一个边都是一对标签），其中 
// edges[i] = [ai, bi] 表示树中节点 ai 和 bi 之间存在一条无向边。可选择树中任何一个节点作为根。当选择节点 x 作为根节点时，设结果
// 树的高度为 h 。在所有可能的树中，具有最小高度的树（即，min(h)）被称为 最小高度树 。
// 请你找到所有的 最小高度树 并按 任意顺序 返回它们的根节点标签列表。 拓扑排序
vector<int> findMinHeightTrees(int n, vector<vector<int>>& edges) {
    if (n == 1)
        return {0};
    if (n == 2)
        return {0, 1};
    int degree[n];
    memset(degree, 0, n<<2);
    vector<vector<int>> graph(n, vector<int>());
    for (vector<int>& edge : edges){
        ++degree[edge[0]];
        ++degree[edge[1]];
        graph[edge[0]].push_back(edge[1]);
        graph[edge[1]].push_back(edge[0]);
    }
    queue<int> q;
    for(int i=0;i<n;++i)
        if(degree[i]==1)
            q.emplace(i);
    while(n>2){
        int size = q.size();
        n -= size;
        for(int i=0;i<size;++i){
            int curr = q.front();
            q.pop();
            degree[curr] = 0; // 无向图，记录是否访问过该节点
            for (int next : graph[curr]){
                if (degree[next] != 0){
                    --degree[next];
                    if (degree[next] == 1)
                        q.push(next);
                }
            }
        }
    }
    vector<int> ans;
    while (!q.empty()){
        ans.push_back(q.front());
        q.pop();
    }
    return ans;
}
// 329矩阵中的最长递增路径 https://leetcode-cn.com/problems/longest-increasing-path-in-a-matrix/
// 给定一个 m x n 整数矩阵 matrix ，找出其中 最长递增路径 的长度。对于每个单元格，你可以往上，下，左，右四个方向移动。
// 你不能 在 对角线 方向上移动或移动到 边界外（即不允许环绕）。 拓扑排序，记忆深度优先搜索
int longestIncreasingPath(vector<vector<int>>& matrix) {
    int m= matrix.size(), n = matrix[0].size();
    int* in_degree[m];
    queue<int> q;
    for(int i=0;i<m;++i){
        in_degree[i] = new int [n];
        for(int j=0;j<n;++j){
            in_degree[i][j] = getDegree(matrix,i,j);
            if(in_degree[i][j]==0)
                q.emplace((i<<8)+j);
        }
    }
    int count = 0;
    while(!q.empty()){
        int size = q.size();
        ++count;
        for(int i=0;i<size;++i){
            int x = q.front();
            q.pop();
            int y = x&0xFF;
            x >>= 8;
            if(x>0&&matrix[x][y]<matrix[x-1][y]){
                --in_degree[x-1][y];
                if(in_degree[x-1][y]==0)
                    q.emplace(((x-1)<<8)+y);
            }
            if(y>0&&matrix[x][y]<matrix[x][y-1]){
                --in_degree[x][y-1];
                if(in_degree[x][y-1]==0)
                    q.emplace((x<<8)+y-1);
            }
            if(x+1<m&&matrix[x][y]<matrix[x+1][y]){
                --in_degree[x+1][y];
                if(in_degree[x+1][y]==0)
                    q.emplace(((x+1)<<8)+y);
            }
            if(y+1<n&&matrix[x][y]<matrix[x][y+1]){
                --in_degree[x][y+1];
                if(in_degree[x][y+1]==0)
                    q.emplace((x<<8)+y+1);
            }
        }
    }
    return count;
}

int getDegree(vector<vector<int>>& matrix,int i,int j){
    int res = 0,m = matrix.size(), n = matrix[0].size();
    if(i>0&&matrix[i][j]>matrix[i-1][j])
        ++res;
    if(j>0&&matrix[i][j]>matrix[i][j-1])
        ++res;
    if(i+1<m&&matrix[i][j]>matrix[i+1][j])
        ++res;
    if(j+1<n&&matrix[i][j]>matrix[i][j+1])
        ++res;
    return res;
}