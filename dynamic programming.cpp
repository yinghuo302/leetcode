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
/*
int longestCommonSubsequence(string text1, string text2) {
    int n=text1.size(),m=text2.size();
	int dp[m+1],last=0,temp;
	fill(dp,dp+m+1,0);
    for(int i=1;i<=n;++i,last=0){
		for(int j=1;j<=m;++j){
			temp=dp[j];
			if(text1[i-1]==text2[j-1])	dp[j]=last+1; 
			else	 dp[j]=max(dp[j],dp[j-1]);
			last=temp;
		}
    }
	return dp[m];
}

int longestCommonSubsequence1(string text1, string text2) {
    int size1=text1.size(),size2=text2.size();
    int dp[size1+1];//dp数组存储的是text1的前i位与text2的最长公共子序列
    dp[0]=0;
    for(int i=1;i<=size1;++i){
        int tem=0;
        for(int j=1;j<=size2;++j){
            if(text1[i-1]==text2[j-1])
        }
    }
}
*/
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