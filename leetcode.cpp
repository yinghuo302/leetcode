#include <bits/stdc++.h>
using namespace std;
int minPatches(vector<int> &nums, int n)
{
}
int findMaximizedCapital(int k, int W, vector<int> &Profits, vector<int> &Capital)
{
}
int strStr(string haystack, string needle)
{
    int h_size = haystack.size(), n_size = needle.size(), i = 0, j = 0;
    while (j < n_size)
    {
        if (haystack[i] == needle[j])
        {
            ++i;
            ++j;
        }
        else
        {
            i = i - j + 1;
            j = 0;
            if (i >= h_size)
                return -1;
        }
    }
    return i - n_size;
}
//股票价格跨度，股票价格小于或等于今天价格的最大连续日数，包括今天
class StockSpanner
{
private:
    stack<int> s;
    vector<int> prices;

public:
    StockSpanner()
    {
    }
    int next(int price)
    {
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
struct TweetNode
{
    Tweet_Id id;
    Tweet_time time;
    TweetNode *next;
    TweetNode(int id, int time) : id(id), time(time), next(NULL) {}
    TweetNode() : id(-1), time(-1), next(NULL) {}
};
class User
{
private:
    TweetNode *tweets;

public:
    unordered_set<User_Id> followees;
    User() : tweets(NULL) {}
    User(int id) : tweets(NULL) { followees.emplace(id); }
    void like(int like_id)
    {
        followees.emplace(like_id);
    }
    void dislike(int like_id)
    {
        int size = followees.size(), i = 0;
        unordered_set<int>::iterator followee = followees.begin();
        while (followee != followees.end())
            if (like_id == *followee)
                followees.erase(followee);
    }
    TweetNode *get_tweet()
    {
        return tweets;
    }
    void post_tweet(int tweet_id, int time)
    {
        if (!tweets)
            tweets = new TweetNode(tweet_id, time);
        else
            tweets->next = new TweetNode(tweet_id, time);
    }
};
class Twitter
{
private:
    unordered_map<User_Id, User *> users;
    Tweet_time time;
    TweetNode *mergeTwoLists(TweetNode *l1, TweetNode *l2)
    {
        if (l1 == NULL)
            return l2;
        if (l2 == NULL)
            return l1;
        if (l1->time > l2->time)
        {
            l1->next = mergeTwoLists(l1->next, l2);
            return l1;
        }
        else
        {
            l2->next = mergeTwoLists(l1, l2->next);
            return l2;
        }
    }
    TweetNode *deep_copy(TweetNode *head)
    {
        if (!NULL)
            return NULL;
        TweetNode *root = new TweetNode(head->id, head->time), *p = root;
        while (head)
        {
            p->next = new TweetNode(head->id, head->time);
            p = p->next;
        }
        p = root->next;
        delete (root);
        return p;
    }
    void deep_del(TweetNode *head)
    {
        if (!head)
            return;
        TweetNode *p = head->next;
        while (p)
        {
            delete (head);
            head = p;
            p = p->next;
        }
        delete (head);
    }

public:
    Twitter()
    {
        time = 0;
    }
    void postTweet(int userId, int tweetId)
    {
        if (users.empty())
        {
            users.emplace(userId, new User(userId));
            users[userId]->post_tweet(tweetId, time);
            return;
        }
        unordered_map<User_Id, User *>::iterator usr_iter = users.find(userId);
        if (users.end() == usr_iter)
        {
            users.emplace(userId, new User(userId));
            usr_iter->second->post_tweet(tweetId, time);
        }
        else
        {
            usr_iter->second->post_tweet(tweetId, time);
        }
        ++time;
    }
    vector<int> getNewsFeed(int userId)
    {
        TweetNode *r = NULL;
        for (auto &followee : users[userId]->followees)
        {
            r = mergeTwoLists(r, deep_copy(users[followee]->get_tweet()));
        }
        vector<int> res;
        TweetNode *h = r;
        while (res.size() <= 10 && r)
        {
            res.push_back(r->id);
            r = r->next;
        }
        deep_del(h);
        return res;
    }
    void follow(int followerId, int followeeId)
    {
        if (users.empty())
            return;
        unordered_map<User_Id, User *>::iterator usr_iter = users.find(followerId);
        if (users.end() != usr_iter)
            usr_iter->second->like(followeeId);
    }
    void unfollow(int followerId, int followeeId)
    {
        if (users.empty())
            return;
        unordered_map<User_Id, User *>::iterator usr_iter = users.find(followerId);
        if (users.end() != usr_iter)
            usr_iter->second->dislike(followeeId);
    }
};
//在数组中找两个数和为target
vector<int> twoSum(vector<int> &nums, int target)
{
    unordered_map<int, int> m;
    int size = nums.size();
    for (int i = 0; i < size; ++i)
    {
        unordered_map<int, int>::iterator a = m.find(target - nums[i]);
        if (a != m.end())
            return {i, a->second};
        m.emplace(nums[i], i);
    }
    return {};
}
//连续1的个数
int findMaxConsecutiveOnes(vector<int> &nums)
{
    int size = nums.size();
    int res = 0, count = 0;
    for (int i = 0; i < size; ++i)
    {
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
    for(int i=0;i<right;++i)
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