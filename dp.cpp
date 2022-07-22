#include<bits/stdc++.h>
using namespace std;
// 322. 零钱兑换 https://leetcode.cn/problems/coin-change/
// 给你一个整数数组 coins ，表示不同面额的硬币；以及一个整数 amount ，表示总金额。计算并返回可以凑成总金额所需的 最少的硬币个数 。如果没有任何一种硬币组合能组成总金额，返回 -1 。你可以认为每种硬币的数量是无限的。
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
// 518. 零钱兑换 II https://leetcode.cn/problems/coin-change-2/
// 给你一个整数数组 coins 表示不同面额的硬币，另给一个整数 amount 表示总金额。请你计算并返回可以凑成总金额的硬币组合数。如果任何硬币组合都无法凑出总金额，返回 0 。假设每一种面额的硬币有无限个。 题目数据保证结果符合 32 位带符号整数。
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
// 72. 编辑距离 https://leetcode.cn/problems/edit-distance/
//给你两个单词 word1 和 word2， 请返回将 word1 转换成 word2 所使用的最少操作数  。你可以对一个单词进行如下三种操作：插入一个字符，删除一个字符，替换一个字符
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
// 剑指 Offer 42. 连续子数组的最大和 https://leetcode.cn/problems/lian-xu-zi-shu-zu-de-zui-da-he-lcof/
//输入一个整型数组，数组中的一个或连续多个整数组成一个子数组。求所有子数组的和的最大值。要求时间复杂度为O(n)。
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
// 55. 跳跃游戏 https://leetcode.cn/problems/jump-game/
// 给定一个非负整数数组 nums ，你最初位于数组的 第一个下标 。数组中的每个元素代表你在该位置可以跳跃的最大长度。判断你是否能够到达最后一个下标。
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
// 45. 跳跃游戏 II https://leetcode.cn/problems/jump-game-ii/
// 给你一个非负整数数组 nums ，你最初位于数组的第一个位置。数组中的每个元素代表你在该位置可以跳跃的最大长度。你的目标是使用最少的跳跃次数到达数组的最后一个位置。假设你总是可以到达数组的最后一个位置。
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
// 300. 最长递增子序列 https://leetcode.cn/problems/longest-increasing-subsequence/
// 给你一个整数数组 nums ，找到其中最长严格递增子序列的长度。子序列 是由数组派生而来的序列，删除（或不删除）数组中的元素而不改变其余元素的顺序。例如，[3,6,2,7] 是数组 [0,3,1,6,2,2,7] 的子序列。
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
// 354. 俄罗斯套娃信封问题 https://leetcode.cn/problems/russian-doll-envelopes/
// 给你一个二维整数数组 envelopes ，其中 envelopes[i] = [wi, hi] ，表示第 i 个信封的宽度和高度。当另一个信封的宽度和高度都比这个信封大的时候，这个信封就可以放进另一个信封里，如同俄罗斯套娃一样。请计算 最多能有多少个 信封能组成一组“俄罗斯套娃”信封（即可以把一个信封放到另一个信封里面）。注意：不允许旋转信封。
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
// 1143. 最长公共子序列 https://leetcode.cn/problems/longest-common-subsequence/
//给定两个字符串 text1 和 text2，返回这两个字符串的最长 公共子序列 的长度。如果不存在 公共子序列 ，返回 0 。一个字符串的 子序列 是指这样一个新的字符串：它是由原字符串在不改变字符的相对顺序的情况下删除某些字符（也可以不删除任何字符）后组成的新字符串。例如，"ace" 是 "abcde" 的子序列，但 "aec" 不是 "abcde" 的子序列。两个字符串的 公共子序列 是这两个字符串所共同拥有的子序列。
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
// 10. 正则表达式匹配 https://leetcode.cn/problems/regular-expression-matching/
// 给你一个字符串 s 和一个字符规律 p，请你来实现一个支持 '.' 和 '*' 的正则表达式匹配。'.' 匹配任意单个字符,'*' 匹配零个或多个前面的那一个元素所谓匹配，是要涵盖 整个 字符串 s的，而不是部分字符串。
bool isMatch(string s, string p) {
    int size1=s.size(),size2=p.size();
    vector<vector<bool>> dp(size1 + 1, vector<bool>(size2 + 1));
    dp[0][0]=1;
    auto match = [&](int i,int j){
        if(i<0)
            return false;
        if(p[j]=='.')
            return true;
        return s[i]==p[j];
    };
    for(int i=0;i<=size1;++i){
        for(int j=1;j<=size2;++j){
            if(p[j-1]=='*'){
                dp[i][j] = dp[i][j-2];
                if(match(i-1,j-2))
                    dp[i][j] = (dp[i][j]||dp[i-1][j]);
            }
            else if(match(i-1,j-1))
                dp[i][j] = dp[i-1][j-1];
        }
    }
    return dp[size1][size2];
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
// 730. 统计不同回文子序列 https://leetcode.cn/problems/count-different-palindromic-subsequences/
// 给定一个字符串 s，返回 s 中不同的非空「回文子序列」个数 。通过从 s 中删除 0 个或多个字符来获得子序列。如果一个字符序列与它反转后的字符序列一致，那么它是「回文字符序列」。如果有某个 i , 满足 ai != bi ，则两个序列 a1, a2, ... 和 b1, b2, ... 不同。注意：结果可能很大，你需要对 109 + 7 取模 。
int countPalindromicSubsequences(string s) {
    int size = s.size();
    int mod = 1000000007;
    int dp[size][size][4];
    for(int i=size-1;i>=0;++i){
        for(int k=0;k<4;++k)
            dp[i][i][k] = (s[i]==k+'a');
        for(int j=i+1;j<size;++j){
            for(int k=0;k<4;++k){
                char c = k+'a';
                if(s[i]!=c)
                    dp[i][j][k] = dp[i+1][j][k];
                else if(s[j]!=c)
                    dp[i][j][k] = dp[i][j-1][k];
                else{
                    dp[i][j][k] = 2;
                    if(j==i+1)
                        continue;
                    for(int m=0;m<4;++m){
                        dp[i][j][k] += dp[i][j][m];
                        dp[i][j][k] %= mod;
                    }
                }
            }
        }
    }
    int ans = 0;
    for(int i=0;i<4;++i){
        ans += dp[0][size-1][i];
        ans %= mod;
    }
    return ans;
}
// 629. K个逆序对数组 https://leetcode.cn/problems/k-inverse-pairs-array/
// 给出两个整数 n 和 k，找出所有包含从 1 到 n 的数字，且恰好拥有 k 个逆序对的不同的数组的个数。逆序对的定义如下：对于数组的第i个和第 j个元素，如果满i < j且 a[i] > a[j]，则其为一个逆序对；否则不是。由于答案可能很大，只需要返回 答案 mod 109 + 7 的值。
int kInversePairs(int n, int k) {
    vector<vector<int>> dp(2, vector<int>(k + 1));
    int mod = 1000000007;
    dp[0][0] = 1;
    for(int i=1;i<=n;++i){
        int cur = i&1,pre = cur^1;
        dp[cur][0] = dp[pre][0];
        for(int j=1;j<=k;++j){ 
            dp[cur][j] = dp[cur][j-1] + dp[pre][j];
            if(j>=i)
                dp[cur][j] -= dp[pre][j-i];
            if (dp[cur][j] >= mod) 
                dp[cur][j] -= mod;
            else if (dp[cur][j] < 0) 
                dp[cur][j] += mod;
                
        }
    }
    return dp[n&1][k];
}
// 698. 划分为k个相等的子集 https://leetcode.cn/problems/partition-to-k-equal-sum-subsets/
// 给定一个整数数组  nums 和一个正整数 k，找出是否有可能把这个数组分成 k 个非空子集，其总和都相等。
bool canPartitionKSubsets(vector<int>& nums, int k) {
    if(k==1)
        return true;
    int size = nums.size();
    int sum = 0;
    for(auto &num:nums)
        sum += num;
    if(sum%k!=0)
        return false;
    int target = sum / k;
    sort(nums.begin(),nums.end());
    if(nums.back()>target)
        return false;
    int n = 1 << size;
    int dp[n];
    memset(dp,0xFF,sizeof(int)*n);
    dp[0] = 0;
    for(int i=0;i<n;++i){
        if(dp[i]==-1)
            continue;
        for(int j=0;j<size;++j){
            if(i&(1<<j))
                continue;
            int next = i|(1<<j);
            dp[next] = dp[i] + nums[j];
            if(dp[next]>target)
                dp[next] = -1;
            if(dp[next]==target)
                dp[next] = 0;
        }
    }
    return dp[n-1] == 0;
}
// 1595. 连通两组点的最小成本 https://leetcode.cn/problems/minimum-cost-to-connect-two-groups-of-points/
// 给你两组点，其中第一组中有 size1 个点，第二组中有 size2 个点，且 size1 >= size2 。任意两点间的连接成本 cost 由大小为 size1 x size2 矩阵给出，其中 cost[i][j] 是第一组中的点 i 和第二组中的点 j 的连接成本。如果两个组中的每个点都与另一组中的一个或多个点连接，则称这两组点是连通的。换言之，第一组中的每个点必须至少与第二组中的一个点连接，且第二组中的每个点必须至少与第一组中的一个点连接。返回连通两组点所需的最小成本。
int connectTwoGroups(vector<vector<int>>& cost) {
    int size1 = cost.size(),size2 = cost[0].size();
    int n = 1 << size2;
    int dp[size1+1][n];
    for(int i=0;i<=size1;++i)
        for(int j=0;j<n;++j)
            dp[i][j] = 1e9;
    dp[0][0] = 0;
    for(int i=1;i<=size1;++i){
        for(int j=0;j<n;++j){
            for(int k=0;k<size2;++k){
                int alt = min(dp[i][j],dp[i-1][j]) + cost[i-1][k];
                if(alt<dp[i][j|(1<<k)])
                    dp[i][j|(1<<k)] = alt;
            }
        }
    }
    return dp[size1][n-1];
}
// 1745. 回文串分割 IV https://leetcode.cn/problems/palindrome-partitioning-iv/
// 给你一个字符串 s ，如果可以将它分割成三个 非空 回文子字符串，那么返回 true ，否则返回 false 。当一个字符串正着读和反着读是一模一样的，就称其为 回文字符串 。
bool checkPartitioning(string s) {
    int size = s.size();
    bool dp[size+1][size];
    for(int i=size-1;i>0;--i){
        dp[i][i] = 1;
        dp[i+1][i] = 0;
        for(int j=i+1;j<size;++j)
            dp[i][j] = (s[i]==s[j])? dp[i+1][j-1] : false;
    }
    for(int i=1;i<=size;++i)
        for(int j=1;j<=size;++j)
            if(dp[0][i-1]&&dp[i][j-1]&&dp[j][size-1])
                return true;
    return false;
}
// 926. 将字符串翻转到单调递增 https://leetcode.cn/problems/flip-string-to-monotone-increasing/
// 如果一个二进制字符串，是以一些 0（可能没有 0）后面跟着一些 1（也可能没有 1）的形式组成的，那么该字符串是 单调递增 的。给你一个二进制字符串 s，你可以将任何 0 翻转为 1 或者将 1 翻转为 0 。返回使 s 单调递增的最小翻转次数。
int minFlipsMonoIncr(string s) {
    int size = s.size();
    int dp[size][2];
    if(s[0]=='0'){
        dp[0][0] = 0;
        dp[0][1] = 1;
    }
    else{
        dp[0][0] = 1;
        dp[0][1] = 0;
    }
    for(int i=1;i<size;++i){
        if(s[i]=='1'){
            dp[i][0] = dp[i-1][0] + 1;
            dp[i][1] = min(dp[i-1][0],dp[i-1][1]);
        }
        else{
            dp[i][0] = dp[i-1][0];
            dp[i][1] = min(dp[i-1][0],dp[i-1][1]) + 1;
        }
    }
    return min(dp[size-1][0],dp[size-1][1]);
}
// 546. 移除盒子 https://leetcode.cn/problems/remove-boxes/
// 给出一些不同颜色的盒子 boxes ，盒子的颜色由不同的正数表示。你将经过若干轮操作去去掉盒子，直到所有的盒子都去掉为止。每一轮你可以移除具有相同颜色的连续 k 个盒子（k >= 1），这样一轮之后你将得到 k * k 个积分。返回 你能获得的最大积分和 。
int removeBoxes(vector<int>& boxes) {
    int dp[100][100][100];
    memset(dp,0,sizeof(dp));
    function<int(int,int,int)> getDP = [&](int l,int r,int k){
        if(l>r)
            return 0;
        int k1 = k,r1 = r;
        while(r1>l&&boxes[r1-1]==boxes[r1]){
            --r1;
            ++k1;
        }
        if(dp[l][r][k]!=0)
            return dp[l][r][k];
        ++k1;
        dp[l][r][k] = getDP(l,r1-1,0)+(k1*k1);
        for(int i=l;i<r1;++i)
            if(boxes[i]==boxes[r])
                dp[l][r][k] = max(dp[l][r][k],getDP(l,i,k1)+getDP(i+1,r1-1,0));
        return dp[l][r][k];
    };
    return getDP(0,boxes.size()-1,0);
}
// 64. 最小路径和 https://leetcode.cn/problems/minimum-path-sum/
// 给定一个包含非负整数的 m x n 网格 grid ，请找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。说明：每次只能向下或者向右移动一步。
int minPathSum(vector<vector<int>>& grid) {
    int m = grid.size(),n = grid[0].size();
    int dp[m][n];https://leetcode.cn/problems/minimum-path-sum/
    dp[0][0] = grid[0][0];
    for(int i=1;i<n;++i)
        dp[0][i] = dp[0][i-1] + grid[0][i];
    for(int i=1;i<m;++i)
        dp[i][0] = dp[i-1][0] + grid[i][0];
    for(int i=1;i<m;++i)
        for(int j=1;j<n;++j)
            dp[i][j] = min(dp[i-1][j],dp[i][j-1])+grid[i][j];
    return dp[m-1][n-1];
}
// 32. 最长有效括号 https://leetcode.cn/problems/longest-valid-parentheses/
// 给你一个只包含 '(' 和 ')' 的字符串，找出最长有效（格式正确且连续）括号子串的长度
int longestValidParentheses(string s) {
	int size = s.size();
	if(size<2)
		return 0;
	int dp[size];
	memset(dp,0,sizeof(dp));
	dp[1] = s[0]=='('&&s[1]==')'? 2:0;
	for(int i=2;i<size;++i){
		if(s[i]=='(')
			continue;
		if(s[i-1]=='(')
			dp[i] = dp[i-2] + 2;
		if(i-1-dp[i-1]>=0&&s[i-1-dp[i-1]]=='('){
			int alt = dp[i-1]+2;
			if(i-2-dp[i-1]>=0)
				alt += dp[i-2-dp[i-1]];
			dp[i] = max(dp[i],alt);

		}
	}
	int ans = 0;
	for(int i=0;i<size;++i)
		ans = max(dp[i],ans);
	return ans;
}
// 5254. 卖木头块 https://leetcode.cn/problems/selling-pieces-of-wood/
// 给你两个整数 m 和 n ，分别表示一块矩形木块的高和宽。同时给你一个二维整数数组 prices ，其中 prices[i] = [hi, wi, pricei] 表示你可以以 pricei 元的价格卖一块高为 hi 宽为 wi 的矩形木块。每一次操作中，你必须按下述方式之一执行切割操作，以得到两块更小的矩形木块：沿垂直方向按高度 完全 切割木块，或沿水平方向按宽度 完全 切割木块。在将一块木块切成若干小木块后，你可以根据 prices 卖木块。你可以卖多块同样尺寸的木块。你不需要将所有小木块都卖出去。你 不能 旋转切好后木块的高和宽。请你返回切割一块大小为 m x n 的木块后，能得到的 最多 钱数。注意你可以切割木块任意次。
long long sellingWood(int m, int n, vector<vector<int>>& prices) {
	int size = prices.size();
	int price[m+1][n+1];
	memset(price,0,sizeof(price));
	for(auto & p:prices)
		price[p[0]][p[1]] = p[2];
	long long dp[m+1][n+1];
	memset(dp,0,sizeof(dp));
	for(int i=1;i<=m;++i){
		for(int j=1;j<=n;++j){
			dp[i][j] = price[i][j];
			for(int k=1;k<i;++k)
				dp[i][j] = max(dp[i][j], dp[k][j] + dp[i - k][j]);
			for(int k=1;k<j;++k)
				dp[i][j] = max(dp[i][j], dp[i][k] + dp[i][j - k]);
		}
	}
	return dp[m][n];
}
// 115. 不同的子序列 https://leetcode.cn/problems/distinct-subsequences/
// 给定一个字符串 s 和一个字符串 t ，计算在 s 的子序列中 t 出现的个数。字符串的一个 子序列 是指，通过删除一些（也可以不删除）字符且不干扰剩余字符相对位置所组成的新字符串。（例如，"ACE" 是 "ABCDE" 的一个子序列，而 "AEC" 不是）题目数据保证答案符合 32 位带符号整数范围。
int numDistinct(string s, string t) {
    int m = s.size(), n = t.size();
    if(m<n)
        return 0;
    unsigned long long dp[m+1][n+1];
    for(int i=0;i<=n;++i)
        dp[0][i] = 0;
    dp[0][0] = 1;
    for(int i=1;i<=m;++i){
        dp[i][0] = 1;
        for(int j=1;j<=n;++j){
            dp[i][j] = dp[i-1][j];
            if(s[i-1]==t[j-1])
                dp[i][j] += dp[i-1][j-1];
        }
    } 
    return dp[m][n];
}
// 87. 扰乱字符串 https://leetcode.cn/problems/scramble-string/
// 使用下面描述的算法可以扰乱字符串 s 得到字符串 t ：如果字符串的长度为 1 ，算法停止。如果字符串的长度 > 1 ，执行下述步骤：在一个随机下标处将字符串分割成两个非空的子字符串。即，如果已知字符串 s ，则可以将其分成两个子字符串 x 和 y ，且满足 s = x + y 。随机 决定是要「交换两个子字符串」还是要「保持这两个子字符串的顺序不变」。即，在执行这一步骤之后，s 可能是 s = x + y 或者 s = y + x 。在 x 和 y 这两个子字符串上继续从步骤 1 开始递归执行此算法。给你两个 长度相等 的字符串 s1 和 s2，判断 s2 是否是 s1 的扰乱字符串。如果是，返回 true ；否则，返回 false 
bool isScramble(string s1, string s2) {
    int size = s1.size();
    bool dp[size][size][size+1];
    for(int i=size-1;i>=0;--i){
        for(int j=size-1;j>=0;--j){
            dp[i][j][0] = true;
            for(int k=0;i<size;++k){
                if(i+k>size||j+k>size){
                    dp[i][j][k] = false;
                    continue;
                }
                for(int l=1;l<k;++l)
                    dp[i][j][k] = (dp[i][j][l] && dp[i+l][j+l][k-l])||(dp[i][j+k-l][l] && dp[i+l][j][k-l]);
            }
        }
    }
    return dp[0][0][size];
}
// 2318. 不同骰子序列的数目 https://leetcode.cn/problems/number-of-distinct-roll-sequences/
// 给你一个整数 n 。你需要掷一个 6 面的骰子 n 次。请你在满足以下要求的前提下，求出 不同 骰子序列的数目：序列中任意 相邻 数字的 最大公约数 为 1 。序列中 相等 的值之间，至少有 2 个其他值的数字。正式地，如果第 i 次掷骰子的值 等于 第 j 次的值，那么 abs(i - j) > 2 。请你返回不同序列的 总数目 。由于答案可能很大，请你将答案对 109 + 7 取余 后返回。如果两个序列中至少有一个元素不同，那么它们被视为不同的序列。
int distinctSequences(int n) {
	unordered_set<int> st[6] = {{1,2,3,4,5},{0,2,4},{0,1,3,4},{0,2,4},{0,1,2,3,5},{0,4}};
	if(n==1)
		return 6;
	int dp[n-1][6][6];
	int mod = 1e9+7;
	for(int i=0;i<6;++i){
		for(int j=0;j<6;++j){
			if(st[i].count(j))
				dp[0][i][j] = 1;
			else
				dp[0][i][j] = 0;
		}
	}
	for(int i=1;i<n-1;++i){
		for(int j=0;j<6;++j){
			for(int k=0;k<6;++k){
				int cnt = 0;
				if(st[j].count(k)){
					for(int l=0;l<6;++l){
						if(j!=l)
							cnt = (cnt + dp[i-1][k][l])%mod;
					}
				}
				dp[i][j][k] = cnt;
			}
		}
	}
	int ans = 0;
	for(int i=0;i<6;++i){
		for(int j=0;j<6;++j){
			ans =  (ans + dp[n-2][i][j])%mod;
		}
	}
	return ans;
}

class Solution {
    int mod = 1e9+7;
    int dir[4][2] = {{0,1},{1,0},{-1,0},{0,-1}};
    vector<vector<int>> dp;
    vector<vector<int>>* ptr;
    int m,n;
public:
    int dfs(int x,int y){
        auto& grid = *ptr;
        if (dp[x][y])
            return dp[x][y];
        int mx=1;
        for (int i=0;i<4;++i) {
            int tx=x+dir[i][0], ty=y+dir[i][1];
            if (tx>=0&&tx<m&&ty>=0&&ty<n&&grid[tx][ty]>grid[x][y])
                mx = ((long long)mx + dfs(tx,ty))%mod;
        }
        return dp[x][y]=mx;
    }
    int countPaths(vector<vector<int>>& grid) {
        m = grid.size(), n = grid[0].size();
        ptr = & grid;
        dp.resize(m,vector<int>(n));
        int ans = 0;
        for(int i=0;i<m;++i){
            for(int j=0;j<n;++j){
                ans = ((long long)ans + dfs(i,j))%mod;
            }
        }
        return ans;
    }
};