#include<bits/stdc++.h>
using namespace std;
//用硬币凑成数量所用的最少硬币数
int coinChange(vector<int>& coins, int amount) {
    int dp[amount+1],size=coins.size();
    sort(coins.begin(),coins.end());
    dp[0]=0;
    for(int i=1;i<=amount;++i){
        dp[i]=-1;int tem=(1<<30);
        for(int j=size-1;j>=0;--j)
            if(i-coins[j]>=0&&dp[i-coins[j]]!=-1&&dp[i-coins[j]]<tem)
                tem=dp[i-coins[j]]+1; 
        if(tem!=(1<<30))
            dp[i]=tem;
    }
    return dp[amount];
}
int change(int amount, vector<int>& coins) {
    int dp[amount+1],size=coins.size();
    sort(coins.begin(),coins.end());
    dp[0]=1;
    for(int i=1;i<=amount;++i)
        dp[i]=0;
    for(int j=size-1;j>=0;--j){
        for(int i=1;i<=amount;++i){
            if(i-coins[j]>=0)
                dp[i]+=dp[i-coins[j]];
        }
    }
    return dp[amount];
}
//从一个字符串到另一个字符串的最短编辑距离可以插入删除替换
int min(int a, int b,int c){
    if(a<b){
        if(c<a)
            return c;
        return a;
    }
    if(c<b)
        return c;
    return b;
}
int minDistance(string word1, string word2) {
    int len1=word1.size(),len2=word2.size();
    if(len1==0||len1==0)
        return len1+len2;
    int dp[len1+1][len2+1];
    for(int i=0;i<len1+1;++i)
        dp[i][0]=i;
    for(int j=0;j<len2+1;++j)
        dp[0][j]=j;
    for(int i=1;i<len1+1;++i){
        for(int j=1;j<len2+1;++j){
            if(word1[i-1]==word2[j-1])
                dp[i][j]=min(dp[i-1][j-1],dp[i-1][j]+1,dp[i][j-1]+1);
            else
                dp[i][j]=min(dp[i-1][j-1],dp[i-1][j],dp[i][j-1])+1;
        }
    }
    return dp[len1][len2];
}
//最小编辑距离，只能删除
int minDistance(string word1, string word2) {
    int size1=word1.size(),size2=word2.size();
    int dp[size1+1][size2+1];
    for(int i=0;i<=size1;++i)
        dp[i][0]=i;
    for(int j=0;j<=size2;++j)
        dp[0][j]=j;
    for(int i=1;i<=size1;++i){
        for(int j=1;j<=size2;++j){
            if(word1[i-1]==word2[j-1])
                dp[i][j]=dp[i-1][j-1];
            else
                dp[i][j]=min(dp[i-1][j],dp[i][j-1])+1;
        }
    }
    return dp[size1][size2];
}
//最大子序列和，子序列必须连续
int max(int* src,int size){
    int max_num=*src;
    for(int i=0;i<size;++i,++src)
        if(*src>max_num)
            max_num=*src;
    return max_num;
}
int maxSubArray(vector<int>& nums) {
    int size=nums.size();int pre[size];
    pre[0]=nums[0];
    for(int i=1;i<size;++i)
        pre[i]=(pre[i-1]+nums[i]>nums[i])? (pre[i-1]+nums[i]) : nums[i];
    return max(pre,size);
}
//跳跃问题，能否跳到数组最后一个
bool canJump(vector<int>& nums) {
    int size=nums.size();
    for(int i=0;i<size;++i){
        if(nums[i]==0&&i!=(size-1)){
            bool can_jump=false;
            for(int j=i;j>=0;--j){
                if((nums[j]+j)>i){
                    can_jump=true;break;
                }
            }
            if(!can_jump)
                return  false;
        }
    }
    return true;
}
//跳跃问题，跳到最后一个的最少步数
int jump(vector<int>& nums) {
    int size=nums.size();
    if(size==1)
        return 0;
    int cur_pos=0,n_jump=0,max_next_pos=nums[cur_pos]+cur_pos;
    while(max_next_pos<size-1){
        int next_2th_pos=nums[cur_pos+1]+cur_pos,i=cur_pos+2;++cur_pos;
        for(;i<=max_next_pos;++i){
            if((i+nums[i])>next_2th_pos){
                cur_pos=i;next_2th_pos=i+nums[i];
            }
        }
        max_next_pos=nums[cur_pos]+cur_pos;++n_jump;
    }
    return n_jump+1;
}
//最长递增子序列，子序列不连续但相对位置不变
int lengthOfLIS(vector<int>& nums) {
    int size=nums.size();
    if(size==0||size==1)
        return size;
    int max_len[size];max_len[0]=1;
    for(int i=1;i<size;++i){
        int j=i-1;max_len[i]=1;
        for(;j>=0;--j)
            if(nums[i]>nums[j]&&(max_len[j]+1)>max_len[i])
                max_len[i]=max_len[j]+1;
    }
    return max(max_len,size);
}
//信封套娃问题，一个信封能装下的最多信封数
bool cmp(const vector<int> &a,const vector<int> &b){//a<b
    if((a[0]<b[0])||(a[0]==b[0]&&a[1]>b[1]))
        return true;
    return false;
}
int lengthOfLIS(vector<vector<int>> & nums) {
    int size=nums.size();
    if(size==0||size==1)
        return size;
    int max_len[size];max_len[0]=1;
    for(int i=1;i<size;++i){
        int j=i-1;max_len[i]=1;
        for(;j>=0;--j)
            if(nums[i][1]>nums[j][1]&&(max_len[j]+1)>max_len[i])
                max_len[i]=max_len[j]+1;
    }
    return max(max_len,size);
}
int maxEnvelopes(vector<vector<int>>& envelopes) {
    sort(envelopes.begin(),envelopes.end(),cmp);
    return lengthOfLIS(envelopes);
}
//最长公共子序列，其中子序列不连续相对位置不变
int longestCommonSubsequence(string text1, string text2) {
    int size1=text1.size(),size2=text2.size();
    int dp[size1+1][size2+1];
    for(int i=0;i<=size1;++i)
        dp[i][0]=0;
    for(int i=0;i<=size2;++i)
        dp[0][i]=0;
    for(int i=1;i<=size1;++i)
        for(int j=1;j<=size2;++j){
            if(text1[i-1]==text2[j-1])
                dp[i][j]=dp[i-1][j-1]+1;
            else
                dp[i][j]=max(dp[i][j-1],dp[i-1][j]);
        }
    return dp[size1-1][size2-1];
}
//最长回文子序列，子序列不连续
int longestPalindromeSubseq(string s) {
    int n = s.size();
    int dp[n][n];
    for(int i=0;i<n;++i)
        for(int j=0;j<n;++j)
            dp[i][j]=(i==j)? 1 : 0;
    for (int i = n - 1; i >= 0; i--) {
        for (int j = i + 1; j < n; j++) {
            if (s[i] == s[j])
                dp[i][j] = dp[i + 1][j - 1] + 2;
            else
                dp[i][j] = max(dp[i + 1][j], dp[i][j - 1]);
        }
    }
    return dp[0][n - 1];
}
//零钱兑换的方案数
int sum_vector(const vector<int> &nums){
    int size=nums.size();int sum=0;
    for(int i=0;i<size;++i)
        sum+=nums[i];
    return sum;
}
bool canPartition(vector<int>& nums) {
    int size=nums.size();
    if(size<2)
        return false;
    int sum=sum_vector(nums);
    if(sum%2)
        return false;
    int target=sum/2;
    bool dp[size][target+1];
    for(int i=0;i<size;++i)
        dp[i][0]=true;
    for(int i=1;i<=target;++i)
        dp[0][i]=false;
    if(nums[0]<=target)
        dp[0][nums[0]]=true;
    for(int i=1;i<size;++i){
        for(int j=1;j<=target;++j){
            if(j-nums[i]>=0)
                dp[i][j]=(dp[i-1][j-nums[i]]||dp[i-1][j]);
            else
                dp[i][j]=dp[i-1][j];
        }
    }
    for(int i=0;i<size;++i){
        for(int j=0;j<=target;++j)
            cout << dp[i][j];
    }
    return dp[size-1][target];
}
//无重叠区间需要删去的最小区间数
inline int cmp1(vector<int>& a,vector<int>& b){
    return a[0]<b[0];
}
int eraseOverlapIntervals(vector<vector<int>>& intervals) {
    if(intervals.size()==0)
        return 0;
    sort(intervals.begin(),intervals.end(),cmp1);
    int size=intervals.size();
    int right=intervals[0][1],num=0;
    for(int i=1;i<size;++i){
        if(intervals[i][0]<right){
            if(right>intervals[i][1])
                right=intervals[i][1];
            ++num;
        }
        else
            right=intervals[i][1];
    }
    return num;
}
//把一个字符串变为另一个字符串删除字符的最小ASCII之和
int minimumDeleteSum(string s1, string s2) {
    int n1=s1.size(),n2=s2.size();
    int dp[n1+1][n2+1];
    dp[0][0]=0;
    for(int i=1;i<=n1;++i)
        dp[i][0]=dp[i-1][0]+s1[i-1];
    for(int i=1;i<=n2;++i)
        dp[0][i]=dp[0][i-1]+s2[i-1];
    for(int i=1;i<=n1;++i){
        for(int j=1;j<=n2;++j){
            if(s1[i-1]==s2[j-1])
                dp[i][j]=dp[i-1][j-1];
            else
                dp[i][j]=min(dp[i-1][j]+s1[i-1],dp[i][j-1]+s2[j-1]);
        }
    }
    return dp[n1][n2];
}
//给出每个气球的起始位置和终止位置，求引爆所有气球所需的最少箭数,xstart ≤ x ≤ xend，则该气球会被引爆
//相当于前一个区间右端点与后一个区间左端点相同也算重叠的无区间重叠问题
int findMinArrowShots(vector<vector<int>>& points) {
    if(points.size()==0)
        return 0;
    sort(points.begin(),points.end(),cmp1);
    int size=points.size();
    int right=points[0][1],num=size;
    for(int i=1;i<size;++i){
        if(points[i][0]<=right){
            if(right>points[i][1])
                right=points[i][1];
            --num;
        }
        else
            right=points[i][1];
    }
    return num;
}
bool isMatch(string s, string p) {
    int size1=s.size(),size2=p.size();
    bool dp[size1+1][size2+1];
    dp[0][0]=1;
    for(int i=0;i<=size1;++i)
        dp[i][0]=0;
    for(int i=0;i<size2;++i){

    }
    for(int i=0;i<=size2;++i){
        for(int j=0;j<=size2;++j){

        }
    }
}
//股票问题，你只能选择某一天买入这只股票，并选择在未来的某一个不同的日子卖出该股票。
int maxProfit(vector<int>& prices) {
    int minprice=INT_MAX,n=prices.size(),maxprofit=0;
    for(int i=0;i<n;++i){
        if(prices[i]<minprice)
            minprice=prices[i];
        else if((prices[i]-minprice)>maxprofit)
            maxprofit=prices[i]-minprice;
    }
    return maxprofit;
}
//股票问题，可以多次买卖,你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。
int max(int a,int b){
    return (a>b)? a: b;
}
int max(int a,int b,int c){
    if(a>b){
        if(a>c)
            return a;
        else
            return c;
    }
    else{
        if(c>b)
            return c;
        else
            return b;
    }
}
int maxProfit(vector<int>& prices) {
    int n=prices.size();
    int dp[n][2];dp[0][0]=0;dp[0][1]=-prices[0];
    for(int i=1;i<n;++i){
        dp[i][0]=max(dp[i-1][0],dp[i-1][1]+prices[i]);
        dp[i][1]=max(dp[i-1][0]-prices[i],dp[i-1][1]);
    }
    return dp[n-1][0];
/*  int b=-prices[0],s=0;
    for(int i=1;i<n;++i){
        b=max(b,-prices[i],s-prices[i]);
        s=max(b+prices[i],s);
    }
    return s;*/
}
//你最多可以完成两笔交易。你不能同时参与多笔交易(你必须在再次购买前出售掉之前的股票)
int maxProfit(vector<int>& prices) {
    int n=prices.size();
    int b1=-prices[0],s1=0,b2=-prices[0],s2=0;
    for(int i=1;i<n;i++){
        b1=max(b1,-prices[i]);
        s1=max(s1,b1+prices[i]);//拓展条件可以当天买当天卖
        b2=max(b2,s1-prices[i]);
        s2=max(s2,b2+prices[i]);
    }
    return s2;
}
//最多完成k笔交易
int maxProfit(int k, vector<int>& prices) {
    int size=prices.size(),n=k*2;
    int dp[n];
    for(int i=0;i<n;++i){//dp[i]i为偶数表示买，i为奇数表示卖
        if(i%2)//奇数
            dp[i]=0;
        else
            dp[i]=-prices[0];
    }
    for(int i=1;i<size;++i){
        dp[0]=max(dp[0],-prices[i]);
        for(int j=0;i<n;++j){
            if(j%2)//奇数卖
                dp[j]=max(dp[j],dp[j-1]+prices[i]);
            else//偶数买
                dp[j]=max(dp[j],dp[j-1]-prices[i],-prices[i]);
        }
    }
    return dp[n-1];
}
//股票问题,有一天冷冻期
int maxProfit(vector<int>& prices) {
    int n=prices.size();
    if(n<=1)
        return 0;
    int dp[n][3];//do[i][0]表示当天持股，dp[i][1]表示不持股且当天没卖，dp[i][2]表示不持股且当天卖了
    dp[0][0]=-prices[0];dp[0][1]=0;dp[0][2]=0;
    dp[1][0]=max(dp[0][0],-prices[1]);dp[1][1]=0;dp[1][2]=dp[1][0]+prices[1];
    for(int i=2;i<n;++i){
        dp[i][0]=max(dp[i-1][0],dp[i-1][1]-prices[i],dp[i-2][2]-prices[i]);
        dp[i][1]=max(dp[i-1][1],dp[i-1][2]);
        dp[i][2]=dp[i][0]+prices[i];
    }
    return max(dp[n-1][1],dp[n-1][2]);
}
int maxProfit(vector<int>& prices, int fee) {
    int n=prices.size();
    if(n<=1)
        return 0;
    int buy=-prices[0]-fee,sell=0;
    for(int i=1;i<n;++i){
        buy=max(buy,sell-prices[i]-fee);
        sell=max(sell,buy+prices[i]);
    }
    return sell;
}
//打家劫舍，但不能抢相邻两家
int rob(vector<int>& nums) {
    int n=nums.size();
    if(!n)
        return 0;
    if(n==1)
        return nums[0];
    if(n==2)
        return max(nums[0],nums[1]);
    int dp[n];dp[0]=nums[0];dp[1]=nums[1];dp[2]=nums[2]+nums[0];
    for(int i=3;i<n;++i){
        dp[i]=max(dp[i-2],dp[i-3])+nums[i];
    }
    return max(dp[n-1],dp[n-2]);
}
//比较符号在子数组中的每个相邻元素对之间翻转，则该子数组是湍流子数组，最长湍流子数组(连续)
bool judge(vector<int>& a,int pos){
    if(a[pos]<a[pos-1]&&a[pos]<a[pos+1])
        return true;
    if(a[pos]>a[pos-1]&&a[pos]>a[pos+1])
        return true;
    return false;
}
int maxTurbulenceSize(vector<int>& arr) {
    int n=arr.size();
    if(n<=1)
        return n;
    int dp[n];dp[0]=1;
    dp[1]=(arr[0]==arr[1])? 1 : 2;//dp[i]表示必须以arr[i]为结尾的最长湍流子数组
    for(int i=2;i<n;++i){
        if(judge(arr,i-1))
            dp[i]=dp[i-1]+1;
        else
            dp[i]=(arr[i]==arr[i-1])? 1 : 2;
    }
    return max(dp,n);
}
// 673. 最长递增子序列的个数 https://leetcode-cn.com/problems/number-of-longest-increasing-subsequence/
// 给定一个未排序的整数数组 nums，返回最长递增子序列的个数 。注意 这个数列必须是 严格 递增的。
int findNumberOfLIS(vector<int>& nums) {
    int size = nums.size();
    int cnt[size],dp[size];
    int ans = 0,max_len =0;
    for(int i=0;i<size;++i){
        dp[i] = 1;
        cnt[i] = 1;
        for (int j=0;j<i;++j) {
            if (nums[i]>nums[j]) {
                if (dp[j]+1>dp[i]) {
                    dp[i] = dp[j]+1;
                    cnt[i] = cnt[j];
                } 
                else if(dp[j]+1==dp[i])
                    cnt[i] += cnt[j];
            }
        }
        if(dp[i]>max_len){
            max_len = dp[i];
            ans = cnt[i];
        }
        else if(dp[i]==max_len)
            ans += cnt[i];
    }
    return ans;
}
// 688. 骑士在棋盘上的概率 https://leetcode-cn.com/problems/knight-probability-in-chessboard/
// 在一个 n x n 的国际象棋棋盘上，一个骑士从单元格 (row, column) 开始，并尝试进行 k 次移动。行和列是 从 0 开始 的，所以左上单元格是 (0,0) ，右下单元格是 (n - 1, n - 1) 。象棋骑士有8种可能的走法，如下图所示。每次移动在基本方向上是两个单元格，然后在正交方向上是一个单元格。每次骑士要移动时，它都会随机从8种可能的移动中选择一种(即使棋子会离开棋盘)，然后移动到那里。骑士继续移动，直到它走了 k 步或离开了棋盘。返回 骑士在棋盘停止移动后仍留在棋盘上的概率 。
class Solution {
    int dirs[8][2] = {{-2, -1}, {-2, 1}, {2, -1}, {2, 1}, {-1, -2}, {-1, 2}, {1, -2}, {1, 2}};
public:
    double knightProbability(int n, int k, int row, int column) {
        double dp[k][n][n];
        for (int step=0;step<=k;step++) {
            for (int i=0;i<n;i++) {
                for (int j=0;j<n;j++) {
                    if (step == 0)
                        dp[0][i][j] = 1;
                    else {
                        dp[step][i][j] = 0;
                        for (int a=0;a<8;++a) {
                            int x = i+dirs[a][0], y = j+dirs[a][1];
                            if (x>=0&&x<n&&y>=0&&y<n)
                                dp[step][i][j] += dp[step-1][x][y]/8;
                        }
                    }
                }
            }
        }
        return dp[k][row][column];
    }
};
// 174. 地下城游戏https://leetcode-cn.com/problems/dungeon-game/
// 一些恶魔抓住了公主（P）并将她关在了地下城的右下角。地下城是由 M x N 个房间组成的二维网格。我们英勇的骑士（K）最初被安置在左上角的房间里，他必须穿过地下城并通过对抗恶魔来拯救公主。骑士的初始健康点数为一个正整数。如果他的健康点数在某一时刻降至 0 或以下，他会立即死亡。有些房间由恶魔守卫，因此骑士在进入这些房间时会失去健康点数（若房间里的值为负整数，则表示骑士将损失健康点数）；其他房间要么是空的（房间里的值为 0），要么包含增加骑士健康点数的魔法球（若房间里的值为正整数，则表示骑士将增加健康点数）。为了尽快到达公主，骑士决定每次只向右或向下移动一步。编写一个函数来计算确保骑士能够拯救到公主所需的最低初始健康点数。
int calculateMinimumHP(vector<vector<int>>& dungeon) {
    int m = dungeon.size(), n = dungeon[0].size();
    int dp[n];
    dp[n-1] = max(1-dungeon[m-1][n-1],1);
    for(int i=n-2;i>=0;--i)
        dp[i] = max(dp[i+1]-dungeon[m-1][i],1);
    for(int i=m-2;i>=0;--i){
        dp[n-1] = max(dp[n-1]-dungeon[i][n-1],1);
        for(int j=n-2;j>=0;--j)
            dp[j] = max(min(dp[j],dp[j+1])-dungeon[i][j],1);
    }
    return dp[0];
}