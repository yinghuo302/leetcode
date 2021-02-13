#include<bits/stdc++.h>
using namespace std;
//用队列模拟栈
class MyStack {
private:
    queue<int> s;
public:
    MyStack() {
    }
    void push(int x) {
    s.push(x);
    }
    int pop(){
        int size=s.size();
        for(int i=0;i<size-1;++i){
            s.push(s.front());s.pop();
        }
        size=s.front();s.pop();
        return size;
    }
    int top() {
        int size=s.size();
        for(int i=0;i<size-1;++i){
            s.push(s.front());s.pop();
        }
        s.push(s.front());size=s.front();s.pop();
        return size;
    }
    bool empty() {
        return s.empty();
    }
};
//用栈模拟队列
class MyQueue {
private:
    stack<int> q;
    stack<int> a;
public:
    MyQueue() {
    }
    void push(int x) {
        while(!a.empty())
            a.pop();
        while(!q.empty()){
            a.push(q.top());q.pop();
        }
        a.push(x);
        while(!a.empty()){
            q.push(a.top());a.pop();
        }
    }
    int pop() {
        int x=q.top();q.pop();
        return x;
    }
    int peek() {
        return q.top();
    }
    bool empty() {
        return q.empty();
    }
};
//滑动窗口的中位数
int max(vector<int> &nums){
    int size=nums.size(),max_num=nums[0];
    for(int i=1;i<size;++i)
        if(nums[i]>max_num)
            max_num=nums[i];
    return max_num;
}
vector<int> maxSlidingWindow(vector<int>& nums, int k) {
    int n=nums.size()-k;
    if(k<=0){
        vector<int> res(1);
        res[0]=max(nums);
        return res;
    }
    vector<int> res(n+1);
    priority_queue<pair<int,int> > s; 
    for(int i=0;i<k;++i)
        s.emplace(nums[i],i+k);
    res[0]=s.top().first;
    for(int i=1;i<=n;++i){
        s.emplace(nums[i],i);
        while(s.top().second<i)
            s.pop();
        res[i]=s.top().first;
    }
    return res;
}
//数据流的中位数
class MedianFinder {
private:
    priority_queue<int> q1;//大顶堆
    priority_queue<int,vector<int>,greater<int> > q2;//小顶堆
    int n;
public:
    MedianFinder() {
        n=0;
    }
    void addNum(int num) {
        if(n==1)
            q1.push(num);
        else if(n%2){//第奇数个放入大顶堆
            if(num>q2.top()){
                q1.push(q2.top());
                q2.pop();q2.push(num);
            }
            else
                q1.push(num);
        }
        else{//第偶数个放入小顶堆
            if(num<q1.top()){
                q2.push(q1.top());q1.pop();q1.push(num);
            }
            else
                q2.push(num);
        }
    }
    double findMedian() {
        if(n%2)
            return double(q1.top());
        else
            return (double)(q1.top()+q2.top())/2;
    }
};
//下一个更大的元素，没有重复元素的数组nums1和nums2其中nums1是nums2的子集，
//nums1 中数字 x 的下一个更大元素是指 x 在 nums2 中对应位置的右边的第一个比 x 大的元素。如果不存在，对应位置输出 -1 
vector<int> nextGreaterElement(vector<int>& nums1, vector<int>& nums2) {
    stack<int> s;int size=nums2.size(),i=1;
    s.push(nums2[0]);unordered_map<int,int> a;
    for(int i=1;i<size;++i){
        while((!s.empty())&&nums2[i]>s.top()){
            a.emplace(s.top(),nums2[i]);
            s.pop();
        }
        s.push(nums2[i]);
    }
    while(!s.empty()){
        a.emplace(s.top(),-1);s.pop();
    }
    size=nums1.size();vector<int> res(size);
    for(int j=0;j<size;++j)
        res[j]=a[nums1[j]];
    return res;
}
//下一个更大的元素，数组可以循环，且有重复元素
vector<int> nextGreaterElements(vector<int>& nums) {
    short int size=nums.size();vector<int> res(size);
    stack<pair<short int,int>> s;
    for(short int i=0;i<size;++i){
        while((!s.empty())&&nums[i]>s.top().second){
            res[s.top().first]=nums[i];s.pop();
        }
        s.emplace(i,nums[i]);
    }
    for(short int i=0;i<size;++i){
        while((!s.empty())&&nums[i]>s.top().second){
            res[s.top().first]=nums[i];s.pop();
        }
    }
    while(!s.empty()){
        res[s.top().first]=-1;s.pop();
    }
    return res;
}
//请根据每日气温列表，重新生成一个列表。对应位置的输出为：要想观测到更高的气温，至少需要等待的天数。
vector<int> dailyTemperatures(vector<int>& T) {
    short int size=T.size();vector<int> res(size);
    if(size==1){
        res[0]=0;return res;
    }
    stack<pair<short int,int>> s;
    for(short int i=0;i<size-1;++i){
        while((!s.empty())&&T[i]>s.top().second){
            res[s.top().first]=i-s.top().first;s.pop();
        }
        s.emplace(i,T[i]);
    }
    res[size-1]=0;
    return res;
}
