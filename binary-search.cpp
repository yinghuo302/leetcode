/*
 * @Author: zanilia
 * @Date: 2022-07-26 10:16:00
 * @LastEditTime: 2022-07-28 17:39:25
 * @Descripttion: 
 */
#include <bits/stdc++.h>
using namespace std;
template<class RAIterator,class Function>
RAIterator upperBound(RAIterator begin,RAIterator end,Function f){
    RAIterator ret = end;
    if(f(--end)<=0)
        return ret;
    while(begin<end){
        RAIterator mid = (end-begin)/2+begin;
        int t = f(mid);
        if(t>0)
            end = mid;
        else
            begin = mid+1;
    }
    return begin;
}
// 1760. 袋子里最少数目的球 https://leetcode.cn/problems/minimum-limit-of-balls-in-a-bag/
// 给你一个整数数组 nums ，其中 nums[i] 表示第 i 个袋子里球的数目。同时给你一个整数 maxOperations 。你可以进行如下操作至多 maxOperations 次：选择任意一个袋子，并将袋子里的球分到 2 个新的袋子中，每个袋子里都有 正整数 个球。比方说，一个袋子里有 5 个球，你可以把它们分到两个新袋子里，分别有 1 个和 4 个球，或者分别有 2 个和 3 个球。你的开销是单个袋子里球数目的 最大值 ，你想要 最小化 开销。请你返回进行上述操作后的最小开销。
int minimumSize(vector<int>& nums, int maxOperations) {
    auto f = [&](int mid){
        int ops = 0;
        for(auto& num:nums)
            ops += (num-1)/mid;
        if(ops>maxOperations)
            return -1;
        return 1;
    };
    int left = 1, right = *max_element(nums.begin(), nums.end());
    return upperBound(left,right+1,f);
}
// 300. 最长递增子序列 https://leetcode.cn/problems/longest-increasing-subsequence/
// 给你一个整数数组 nums ，找到其中最长严格递增子序列的长度。子序列 是由数组派生而来的序列，删除（或不删除）数组中的元素而不改变其余元素的顺序。例如，[3,6,2,7] 是数组 [0,3,1,6,2,2,7] 的子序列。
int lengthOfLIS(vector<int>& nums) {
    int size = nums.size();
    if(!size)
        return 0;
    vector<int> arr(size+1,INT32_MAX);
    arr[0] = -100000;
    int len = 0;
    for(auto& num:nums){
        if(num>arr[len])
            arr[++len] = num;
        else
            *lower_bound(arr.begin(),arr.end(),num) = num;
    }
    return len;
}
// 1552. 两球之间的磁力 https://leetcode.cn/problems/magnetic-force-between-two-balls/
// 在代号为 C-137 的地球上，Rick 发现如果他将两个球放在他新发明的篮子里，它们之间会形成特殊形式的磁力。Rick 有 n 个空的篮子，第 i 个篮子的位置在 position[i] ，Morty 想把 m 个球放到这些篮子里，使得任意两球间 最小磁力 最大。已知两个球如果分别位于 x 和 y ，那么它们之间的磁力为 |x - y| 。给你一个整数数组 position 和一个整数 m ，请你返回最大化的最小磁力。
int maxDistance(vector<int>& position, int m) {
    sort(position.begin(),position.end());
    auto check = [&](int mid){
        int pre = position[0],cnt = 1;
        for(auto&pos:position){
            if(pos-pre>=mid){
                ++cnt;
                pre = pos;
            }
        }
        return cnt<m;
    };
    int left = 1,right = position.back() - position[0];
    return upperBound(left,right+1,check)-1;
}
// 287. 寻找重复数 https://leetcode.cn/problems/find-the-duplicate-number/
// 给定一个包含 n + 1 个整数的数组 nums ，其数字都在 [1, n] 范围内（包括 1 和 n），可知至少存在一个重复的整数。假设 nums 只有 一个重复的整数 ，返回 这个重复的数 。你设计的解决方案必须 不修改 数组 nums 且只用常量级 O(1) 的额外空间。
int findDuplicate(vector<int>& nums) {
    int n = nums.size();
    auto check = [&](int mid){
        int cnt = 0;
        for(auto & num: nums)
            if(num<=mid)
                ++cnt;
        return cnt>mid;
    };
    return upperBound(1,n,check);
}
// 1283. 使结果不超过阈值的最小除数 https://leetcode.cn/problems/find-the-smallest-divisor-given-a-threshold/
// 给你一个整数数组 nums 和一个正整数 threshold  ，你需要选择一个正整数作为除数，然后将数组里每个数都除以它，并对除法结果求和。请你找出能够使上述结果小于等于阈值 threshold 的除数中 最小 的那个。每个数除以除数后都向上取整，比方说 7/3 = 3 ， 10/2 = 5 。题目保证一定有解。
int smallestDivisor(vector<int>& nums, int threshold) {
	auto check = [&](int mid){
		long long sum = 0;
		for(auto &num:nums)
			sum += (num+mid-1)/mid;
		return sum <= threshold;
	};
	return upperBound(1,(int)1e6+1,check);
}
// 6098. 统计得分小于 K 的子数组数目 https://leetcode.cn/problems/count-subarrays-with-score-less-than-k/
// 一个数字的 分数 定义为数组之和 乘以 数组的长度。比方说，[1, 2, 3, 4, 5] 的分数为 (1 + 2 + 3 + 4 + 5) * 5 = 75 。给你一个正整数数组 nums 和一个整数 k ，请你返回 nums 中分数 严格小于 k 的 非空整数子数组数目。子数组 是数组中的一个连续元素序列
long long countSubarrays(vector<int>& nums, long long k) {
    int size = nums.size();
    long long presum[size+1];
    long long ans = 0;
    for(int i=0;i<size;++i)
        presum[i+1] = presum[i] + nums[i];
    auto bisearch = [&](int i){
        int l = i,r = size;
        long long tem = presum[r] - presum[i];
        if(tem<k)
            return size+1;
        while(l<r){
            int mid = (l+r)/2;
            long long tem = presum[mid] - presum[i];
            if(tem>=k)
                r = mid;
            else
                l = mid+1;
        }
        return l;
    };
    for(int i=0;i<size;++i)
        ans += bisearch(i)-i;
    return ans;
}
// 875. 爱吃香蕉的珂珂 https://leetcode.cn/problems/koko-eating-bananas/
// 珂珂喜欢吃香蕉。这里有 n 堆香蕉，第 i 堆中有 piles[i] 根香蕉。警卫已经离开了，将在 h 小时后回来。珂珂可以决定她吃香蕉的速度 k （单位：根/小时）。每个小时，她将会选择一堆香蕉，从中吃掉 k 根。如果这堆香蕉少于 k 根，她将吃掉这堆的所有香蕉，然后这一小时内不会再吃更多的香蕉。  珂珂喜欢慢慢吃，但仍然想在警卫回来前吃掉所有的香蕉。返回她可以在 h 小时内吃掉所有香蕉的最小速度 k（k 为整数）。
int minEatingSpeed(vector<int>& piles, int h) {
    int sum = 0, m = INT32_MIN;
    for(auto &pile:piles){
        if(pile>m)
            m = pile;
        sum += pile;
    }
    auto check = [&](int k){
        int tem = h;
        for(auto& pile:piles){
            tem -= (pile+k-1)/k;
            if(tem<0)
                return false;
        }
        return true;
    };
    
    int l = sum / h, r = m;
    if(l==0) 
        l = 1;
    while(l<r){
        int mid = l + (r-l)/2;
        if(check(mid))
            r = mid;
        else
            l = mid+1;
    }
    return l;
}
// 668. 乘法表中第k小的数 https://leetcode.cn/problems/kth-smallest-number-in-multiplication-table/
// 几乎每一个人都用 乘法表。但是你能在乘法表中快速找到第k小的数字吗？给定高度m 、宽度n 的一张 m * n的乘法表，以及正整数k，你需要返回表中第k 小的数字。
int findKthNumber(int m, int n, int k) {
	int l = 1,r = m*n;
	while(l<r){
		int mid = (r-l)/2+l;
		int row = mid / n;
		int cnt = row*n;
		for(int i=row+1;i<=m;++i)
			cnt += mid / i;
		if(cnt>=k)
			r = mid;
		else
			l = mid +1;
	}
	return l;
}
// 719. 找出第 K 小的数对距离 https://leetcode.cn/problems/find-k-th-smallest-pair-distance/
// 数对 (a,b) 由整数 a 和 b 组成，其数对距离定义为 a 和 b 的绝对差值。给你一个整数数组 nums 和一个整数 k ，数对由 nums[i] 和 nums[j] 组成且满足 0 <= i < j < nums.length 。返回 所有数对距离中 第 k 小的数对距离。
int smallestDistancePair(vector<int>& nums, int k) {
	sort(nums.begin(),nums.end());
	int l = 0, r = nums.back() - nums.front();
	auto begin = nums.begin(),end = nums.end();
	while(l<r){
		int mid = (l+r)/2;
		int cnt = 0;
		for(auto i=begin;i!=end;++i)
			cnt += upper_bound(i+1,end,*i+mid) - i-1;
		if(cnt>=k)
			r = mid;
		else
			l = mid+1;
	}
	return l;
}
// 1898. 可移除字符的最大数目 https://leetcode.cn/problems/maximum-number-of-removable-characters/
// 给你两个字符串 s 和 p ，其中 p 是 s 的一个 子序列 。同时，给你一个元素 互不相同 且下标 从 0 开始 计数的整数数组 removable ，该数组是 s 中下标的一个子集（s 的下标也 从 0 开始 计数）。请你找出一个整数 k（0 <= k <= removable.length），选出 removable 中的 前 k 个下标，然后从 s 中移除这些下标对应的 k 个字符。整数 k 需满足：在执行完上述步骤后， p 仍然是 s 的一个 子序列 。更正式的解释是，对于每个 0 <= i < k ，先标记出位于 s[removable[i]] 的字符，接着移除所有标记过的字符，然后检查 p 是否仍然是 s 的一个子序列。返回你可以找出的 最大 k ，满足在移除字符后 p 仍然是 s 的一个子序列。字符串的一个 子序列 是一个由原字符串生成的新字符串，生成过程中可能会移除原字符串中的一些字符（也可能不移除）但不改变剩余字符之间的相对顺序。
int maximumRemovals(string s, string p, vector<int>& removable) {
    int s_size = s.size();
    int p_size = p.size();
    char state[s_size];
    auto check = [&](int mid){
        memset(state,0,sizeof(state));
        for(int i=0;i<mid;++i)
            state[removable[i]] = 1;
        int j = 0;
        for(int i=0;i<s_size;++i){
            if(!state[i]&&s[i]==p[j])
                ++j;
            if(j==p_size)
                return -1;
        }
        return 1;
    };
    return upperBound(0,(int)removable.size()+1,check)-1;
}

// 给你一个浮点数 hour ，表示你到达办公室可用的总通勤时间。要到达办公室，你必须按给定次序乘坐 n 趟列车。另给你一个长度为 n 的整数数组 dist ，其中 dist[i] 表示第 i 趟列车的行驶距离（单位是千米）。每趟列车均只能在整点发车，所以你可能需要在两趟列车之间等待一段时间。例如，第 1 趟列车需要 1.5 小时，那你必须再等待 0.5 小时，搭乘在第 2 小时发车的第 2 趟列车。返回能满足你准时到达办公室所要求全部列车的 最小正整数 时速（单位：千米每小时），如果无法准时到达，则返回 -1 。生成的测试用例保证答案不超过 107 ，且 hour 的 小数点后最多存在两位数字 。
int minSpeedOnTime(vector<int>& dist, double hour) {
    int size = dist.size();
    int end = 1e7+1;
    auto check = [&](int mid){
        double sum = 0;
        for(int i=size-2;i>=0;--i)
            sum += (dist[i]+mid-1)/mid;
        sum += (double)dist.back() / mid;
        return (sum<=hour)? 1 :-1;
    }; 
    int ans = upperBound(1,end,check);
    return (ans == end)? -1 : ans;
}
// 1482. 制作 m 束花所需的最少天数 https://leetcode.cn/problems/minimum-number-of-days-to-make-m-bouquets/
// 给你一个整数数组 bloomDay，以及两个整数 m 和 k 。现需要制作 m 束花。制作花束时，需要使用花园中 相邻的 k 朵花 。花园中有 n 朵花，第 i 朵花会在 bloomDay[i] 时盛开，恰好 可以用于 一束 花中。请你返回从花园中摘 m 束花需要等待的最少的天数。如果不能摘到 m 束花则返回 -1 。
int minDays(vector<int>& bloomDay, int m, int k) {
    auto check = [&](int mid){
        int cnt = 0,ans = 0;
        for(auto& day:bloomDay){
            if(day<=mid)
                ++cnt;
            else{
                ans += cnt / k;
                cnt = 0;
            }
        }
        ans += cnt/k;
        return (ans >= m)? 1 : -1;
    };
    int left = INT32_MAX,right = 0;
    for(auto& day:bloomDay){
        left = min(day,left);
        right = max(right,day);
    }
    int ans = upperBound(left,right+1,check);
    return ans>right? -1 : ans;
}

// 1818. 绝对差值和 https://leetcode.cn/problems/minimum-absolute-sum-difference/
// 给你两个正整数数组 nums1 和 nums2 ，数组的长度都是 n 。数组 nums1 和 nums2 的 绝对差值和 定义为所有 |nums1[i] - nums2[i]|（0 <= i < n）的 总和（下标从 0 开始）。你可以选用 nums1 中的 任意一个 元素来替换 nums1 中的 至多 一个元素，以 最小化 绝对差值和。在替换数组 nums1 中最多一个元素 之后 ，返回最小绝对差值和。因为答案可能很大，所以需要对 109 + 7 取余 后返回。|x| 定义为：如果 x >= 0 ，值为 x ，或者如果 x <= 0 ，值为 -x
int minAbsoluteSumDiff(vector<int>& nums1, vector<int>& nums2) {
    int diff = 0,sum= 0,tmp;
    const int size =nums1.size();
    vector<int> nums= nums1;
    sort(nums.begin(),nums.end());
    for(int i =0;i<size;++i){
        tmp = abs(nums1[i]-nums2[i]);
        sum = (sum +tmp)%1000000007;
        int j = lower_bound(nums.begin(),nums.end(),nums2[i])-nums.begin();
        if(j<size)
            diff = max(diff,tmp-(nums[j]-nums2[i]));
        if(j>0)
            diff = max(diff,tmp-(nums2[i]-nums[j-1]));
    }
    return (sum -diff+1000000007)%1000000007;
}