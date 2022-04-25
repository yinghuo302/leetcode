/*
 * @Author: zanilia
 * @Date: 2022-02-17 16:17:01
 * @LastEditTime: 2022-02-17 21:56:15
 * @Descripttion: 
 */
#include <bits/stdc++.h>
using namespace std;
// 307. 区域和检索 - 数组可修改 https://leetcode-cn.com/problems/range-sum-query-mutable/
// 给你一个数组 nums ，请你完成两类查询。其中一类查询要求 更新 数组 nums 下标对应的值另一类查询要求返回数组 nums 中索引 left 和索引 right 之间（ 包含 ）的nums元素的 和 ，其中 left <= right实现 NumArray 类：NumArray(int[] nums) 用整数数组 nums 初始化对象void update(int index, int val) 将 nums[index] 的值 更新 为 valint sumRange(int left, int right) 返回数组 nums 中索引 left 和索引 right 之间（ 包含 ）的nums元素的 和 （即，nums[left] + nums[left + 1], ..., nums[right]）
class NumArray {
private:
    int* arr;
    int size;
public:
    NumArray(vector<int>& nums) {
        size = nums.size();
        arr = new int[size*2];
        for(int i=0;i<size;++i)
            arr[i+size] = nums[i];
        for(int i=size-1;i>0;--i)
            arr[i] = arr[i*2] + arr[i*2+1];
    }
    void update(int index, int val) {
        int pos = index + size;
        while(pos>0){
            arr[pos/2] = arr[pos] + arr[pos^1];
            pos /= 2;
        }
    }
    int sumRange(int left, int right) {
        int begin = left + size,end = right + size;
        int sum = 0;
        while(begin<=end){
            if(begin&1){
                sum += arr[begin];
                ++begin;
            }
            if(!(end&1)){
                sum += arr[end];
                --end;
            }
            begin /= 2;
            end /= 2;
        }
        return sum;
    }
};
// 315. 计算右侧小于当前元素的个数https://leetcode-cn.com/problems/count-of-smaller-numbers-after-self/
// 给你一个整数数组 nums ，按要求返回一个新数组 counts 。数组 counts 有该性质： counts[i] 的值是  nums[i] 右侧小于 nums[i] 的元素的数量。
class BIT{
    int* arr;
	unsigned size;
	static int lowbit(int x){
        return x&(-x);
    }
public:
    BIT(unsigned _size):arr(new int[_size+1]),size(_size){
		memset(arr,0,(_size+1)*4);
	}
    void update(int pos,int inc){
        while(pos<=size){
            arr[pos] += inc;
            pos += lowbit(pos);
        }
    }
    int getSum(int pos){
        int sum = 0;
        while(pos>0){
            sum += arr[pos];
            pos -= lowbit(pos);
        }
        return sum;
    }
};
class Solution {
private:
    vector<int> a;
    int getId(int val){
        return lower_bound(a.begin(),a.end(),val) - a.begin() + 1;
    }
public:
    vector<int> countSmaller(vector<int>& nums) {
        a = nums;
        sort(a.begin(),a.end());
        int size = nums.size();
		BIT bit(size);
        vector<int> res(size);
        for(int i=size-1;i>=0;--i){
            int id = getId(nums[i]);
            res[i] = bit.getSum(id-1);
            bit.update(id,1);
        }
        return res;
    }
};
// 1409. 查询带键的排列 https://leetcode-cn.com/problems/queries-on-a-permutation-with-key/
// 给你一个待查数组 queries ，数组中的元素为 1 到 m 之间的正整数。 请你根据以下规则处理所有待查项 queries[i]（从 i=0 到 i=queries.length-1）：一开始，排列 P=[1,2,3,...,m]。对于当前的 i ，请你找出待查项 queries[i] 在排列 P 中的位置（下标从 0 开始），然后将其从原位置移动到排列 P 的起始位置（即下标为 0 处）。注意， queries[i] 在 P 中的位置就是 queries[i] 的查询结果。请你以数组形式返回待查数组  queries 的查询结果。
vector<int> processQueries(vector<int>& queries, int m) {
	int size = queries.size();
	vector<int> ans(size);
	int pos[m+1];
	BIT bit(m+size);
	for(int i=1;i<=m;++i){
		pos[i] = i+size;
		bit.update(i+size,1);
	}
	for(int i=0;i<size;++i){
		int query = queries[i];
		int cur_pos = pos[query];
		bit.update(cur_pos,-1);
		ans[i] = bit.getSum(cur_pos);
		cur_pos = size-i;
		pos[query] = cur_pos;
		bit.update(cur_pos,1);
	}
	return ans;
}
// 1395. 统计作战单位数 https://leetcode-cn.com/problems/count-number-of-teams/
//  n 名士兵站成一排。每个士兵都有一个 独一无二 的评分 rating 。每 3 个士兵可以组成一个作战单位，分组规则如下：从队伍中选出下标分别为 i、j、k 的 3 名士兵，他们的评分分别为 rating[i]、rating[j]、rating[k]作战单位需满足： rating[i] < rating[j] < rating[k] 或者 rating[i] > rating[j] > rating[k] ，其中  0 <= i < j < k< n请你返回按上述条件可以组建的作战单位数量。每个士兵都可以是多个作战单位的一部分。
class Solution {
	unordered_map<int,int> dist;
	void discretize(const vector<int>& rating){
		vector<int> tmp = rating;
		sort(tmp.begin(),tmp.end());
		int size = tmp.size();
		dist.emplace(tmp[0],1);
		for(int i=1;i<size;++i)
			if(tmp[i]!=tmp[i-1])
				dist.emplace(tmp[i],i+1);
	}
public:
    int numTeams(vector<int>& rating) {
		discretize(rating);
		int size = rating.size();
		BIT bit(size);
		BIT inc(size);
		BIT dec(size);
		int ans = 0;
		for(int i=0;i<size;++i){
			int id = dist[rating[i]];
			bit.update(id,1);
			inc.update(id,bit.getSum(id-1));
			dec.update(id,bit.getSum(size)-bit.getSum(id));
			ans += inc.getSum(id-1);
			ans += (dec.getSum(size)-dec.getSum(id));
		}
		return ans;
    }
};
// 1157. 子数组中占绝大多数的元素 https://leetcode-cn.com/problems/online-majority-element-in-subarray/
// 设计一个数据结构，有效地找到给定子数组的 多数元素 。子数组的 多数元素 是在子数组中出现 threshold 次数或次数以上的元素。实现 MajorityChecker 类:MajorityChecker(int[] arr) 会用给定的数组 arr 对 MajorityChecker 初始化。int query(int left, int right, int threshold) 返回子数组中的元素  arr[left...right] 至少出现 threshold 次数，如果不存在这样的元素则返回 -1。
struct Count{
	int val;
	int cnt;
	Count() = default;
	Count(int _val,int _cnt):val(_val),cnt(_cnt){}
	Count operator+(Count a){
		if(a.val==val)
			return {val,a.cnt+cnt};
		else if(a.cnt<=cnt)
			return {val,cnt-a.cnt};
		else
			return {a.val,a.cnt-cnt};
	}
	Count& operator+=(Count a){
		if(a.val==val)
			cnt += a.cnt;
		else if(a.cnt<=cnt)
			cnt-=a.cnt;
		else{
			val = a.val;
			cnt = a.cnt -cnt;
		}
		return *this;
	}
};
class SegTree {
private:
   Count* arr;
   int size;
public:
   SegTree(vector<int>& nums) {
       size = nums.size();
       arr = new Count[size*2];
       for(int i=0;i<size;++i)
           arr[i+size] = {nums[i],1};
       for(int i=size-1;i>0;--i)
           arr[i] = arr[i*2] + arr[i*2+1];
	}
   void update(int index, int val) {
       int pos = index + size;
       while(pos>0){
           arr[pos/2] = arr[pos] + arr[pos^1];
           pos /= 2;
       }
   }
   int sumRange(int left, int right) {
       int begin = left + size,end = right + size;
       Count sum = {0,0};
       while(begin<=end){
           if(begin&1){
               sum += arr[begin];
               ++begin;
           }
           if(!(end&1)){
               sum += arr[end];
               --end;
           }
           begin /= 2;
           end /= 2;
       }
       return sum.val;
   }
};
class MajorityChecker {
	unordered_map<int,vector<int>> dist;
	SegTree seg;
public:
    MajorityChecker(vector<int>& arr): seg(arr){
		int size = arr.size();
		for(int i=0;i<size;++i)
			dist[arr[i]].push_back(i);		
    }
    int query(int left, int right, int threshold) {
		auto val = seg.sumRange(left,right);
		auto& tem = dist[val];
		int cnt = upper_bound(tem.begin(),tem.end(),right)-lower_bound(tem.begin(),tem.end(),left);
		return (cnt>=threshold)? val:-1;
    }
};
// 1505. 最多 K 次交换相邻数位后得到的最小整数 https://leetcode-cn.com/problems/minimum-possible-integer-after-at-most-k-adjacent-swaps-on-digits/
// 给你一个字符串 num 和一个整数 k 。其中，num 表示一个很大的整数，字符串中的每个字符依次对应整数上的各个 数位 。你可以交换这个整数相邻数位的数字 最多 k 次。请你返回你能得到的最小整数，并以字符串形式返回。
string minInteger(string num, int k) {

}