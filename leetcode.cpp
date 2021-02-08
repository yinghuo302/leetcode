#include<bits/stdc++.h>
using namespace std;
int minPatches(vector<int>& nums, int n) {

}
int findMaximizedCapital(int k, int W, vector<int>& Profits, vector<int>& Capital) {

}
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
//股票价格跨度，股票价格小于或等于今天价格的最大连续日数，包括今天
class StockSpanner {
private:
    stack<int> s;
    vector<int> prices;
public:
    StockSpanner() {
    }
    int next(int price) {
        while(!s.empty()&&price>=prices[s.top()])
            s.pop();
        int num=0;
        if(s.empty())
            num=prices.size()+1;
        else
            num=prices.size()-s.top();
        prices.push_back(price);
        s.push(prices.size()-1);
        return num;
    }
};