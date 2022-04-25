/*
 * @Author: zanilia
 * @Date: 2022-02-17 13:17:36
 * @LastEditTime: 2022-02-17 13:18:19
 * @Descripttion: 
 */
#include <bits/stdc++.h>
using namespace std;
// 330. 按要求补齐数组 https://leetcode-cn.com/problems/patching-array/
// 给定一个已排序的正整数数组 nums，和一个正整数 n 。从 [1, n] 区间内选取任意个数字补充到 nums 中，使得 [1, n] 区间内的任何数字都可以用 nums 中某几个数字的和来表示。请输出满足上述要求的最少需要补充的数字个数。
int minPatches(vector<int> &nums, int n){
    long covered = 0;
    int size = nums.size();
    int ans = 0;
    for(int i=0;i<size;++i){
        if(covered>=n)
            return ans;
        int tem = min(nums[i]-1,n);
        while(covered<tem){
            covered = 2*covered + 1;
            ++ans;
        }
        covered += nums[i];
    }
    while(covered<n){
        covered = 2*covered + 1;
        ++ans;
    }
    return ans;
}
// 502. IPO https://leetcode-cn.com/problems/ipo/
// 假设力扣即将开始IPO.为了以更高的价格将股票卖给风险投资公司，力扣希望在IPO之前开展一些项目以增加其资本。 由于资源有限，它只能在IPO之前完成最多k个不同的项目。帮助力扣设计完成最多k个不同项目后得到最大总资本的方式。给你 n 个项目。对于每个项目 i ，它都有一个纯利润 profits[i] ，和启动该项目需要的最小资本 capital[i].最初，你的资本为 w 。当你完成一个项目时，你将获得纯利润，且利润将被添加到你的总资本中。总而言之，从给定项目中选择 最多 k 个不同项目的列表，以 最大化最终资本 ，并输出最终可获得的最多资本。
int findMaximizedCapital(int k, int w, vector<int> &profits, vector<int> &capital){
    int size = profits.size();
    pair<int,int> projects[size];
    for(int i=0;i<size;++i)
        projects[i] = {capital[i],profits[i]};
    sort(projects,projects+size);
    int p = 0;
    priority_queue<int> hp;
    for(int i=0;i<k;++i){
        while(p<size&&projects[p].first<=w)
            hp.emplace(projects[p++].second);
        if(hp.empty())
            break;
        else{
            w += hp.top();
            hp.pop();
        }
    }
    return w;
}