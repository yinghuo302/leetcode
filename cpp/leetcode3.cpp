#include <bits/stdc++.h>
using namespace std;
// 352. 将数据流变为多个不相交区间 https://leetcode.cn/problems/data-stream-as-disjoint-intervals/
// 给你一个由非负整数 a1, a2, ..., an 组成的数据流输入，请你将到目前为止看到的数字总结为不相交的区间列表。实现 SummaryRanges 类：SummaryRanges() 使用一个空数据流初始化对象。void addNum(int val) 向数据流中加入整数 val 。int[][] getIntervals() 以不相交区间 [starti, endi] 的列表形式返回对数据流中整数的总结。
class SummaryRanges {
	map<int,int> intervals;
public:
    SummaryRanges() {
		intervals.emplace(-2,-2);
	}
    
    void addNum(int val) {
		auto i = --intervals.upper_bound(val);
		if(i->second>=val)
            return ;
		else if(i->second==val-1)
			i->second = val;
        else
			i = intervals.emplace(val,val).first;
        auto next = i;next++;
        if(next!=intervals.end()&&next->first==val+1){
            i->second = next->second;
            intervals.erase(next);
        }
    }
    
    vector<vector<int>> getIntervals() {
		auto begin = ++intervals.begin(),end = intervals.end();
		vector<vector<int>> ans;
		while(begin!=end){
			ans.push_back({begin->first,begin->second});
			++begin;
		}
		return ans;
    }
};
// 36. 有效的数独 https://leetcode.cn/problems/valid-sudoku/
// 请你判断一个 9 x 9 的数独是否有效。只需要 根据以下规则 ，验证已经填入的数字是否有效即可。一个有效的数独（部分已被填充）不一定是可解的。只需要根据以上规则，验证已经填入的数字是否有效即可。空白格用 '.' 表示。
class IsValidSudoku{
	int row_state[9];
	int col_state[9];
	int block_state[3][3];
	bool setState(int i, int j, int digit) {
		int tem = (1 << digit);
		if((row_state[i]&tem)||(col_state[j]&tem)||(block_state[i/3][j/3]&tem))
			return false;
        row_state[i] |= tem;
        col_state[j] |= tem;
        block_state[i/3][j/3] |= tem;
		return true;
    }
public:
	bool isValidSudoku(vector<vector<char>>& board) {
		memset(row_state,0,sizeof(row_state));
		memset(col_state,0,sizeof(col_state));
		memset(block_state,0,sizeof(block_state));
		for(int i=0;i<9;++i)
			for(int j=0;j<9;++j)
				if(board[i][j]!='.'&&!setState(i,j,board[i][j]-'0'))
					return false;
		return true;
	}
};
// 37. 解数独 https://leetcode.cn/problems/sudoku-solver/
// 编写一个程序，通过填充空格来解决数独问题。数独的解法需 遵循如下规则：数字 1-9 在每一行只能出现一次。数字 1-9 在每一列只能出现一次。数字 1-9 在每一个以粗实线分隔的 3x3 宫内只能出现一次。（请参考示例图）数独部分空格内已填入了数字，空白格用 '.' 表示。
class Solution {
	int row_state[9] = {0,0,0,0,0,0,0,0,0};
    int col_state[9] ={0,0,0,0,0,0,0,0,0};
    int block_state[3][3] = {{0,0,0},{0,0,0},{0,0,0}};
	bool solved = false;
	vector<vector<char>>* board_ptr;
	bool setState(int i, int j, int digit) {
		int tem = (1 << digit);
		if((row_state[i]&tem)||(col_state[j]&tem)||(block_state[i/3][j/3]&tem))
			return false;
        row_state[i] |= tem;
        col_state[j] |= tem;
        block_state[i/3][j/3] |= tem;
		return true;
    }
	void resetState(int i, int j, int digit) {
		int tem = ~(1 << digit);
        row_state[i] &= tem;
        col_state[j] &= tem;
        block_state[i/3][j/3] &= tem;
    }
	void dfs(int x,int y){
		if(x==9){
			solved = true;
			return ;
		}
		int next_x = x ,next_y = y+1;
		if(next_y==9){
			next_y = 0;
			next_x++;
		}
		if((*board_ptr)[x][y]!='.'){
			dfs(next_x,next_y);
			return ;
		}
		for(int i=1;i<=9;++i){
			if(!setState(x,y,i))
				continue;
			(*board_ptr)[x][y] = i+'0';
			dfs(next_x,next_y);
			if(solved)
				return ;
			(*board_ptr)[x][y] = '.';
			resetState(x,y,i);
		}
	}
public:
    void solveSudoku(vector<vector<char>>& board) {
		board_ptr = &board;
		for(int i=0;i<9;++i)
			for(int j=0;j<9;++j)
				if(board[i][j]!='.')
					setState(i,j,board[i][j]-'0');
		dfs(0,0);
    }
};
// 42. 接雨水 https://leetcode.cn/problems/trapping-rain-water/
// 给定 n 个非负整数表示每个宽度为 1 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水。
int trap(vector<int>& height) {
	stack<pair<int,int>> stk;
	stk.emplace(make_pair(-1,INT32_MAX));
	int size = height.size();
	int ans = 0;
	for(int i=0;i<size;++i){
		pair<int,int> top = stk.top();
		while(top.first!=-1&&height[top.first]<=height[i]){
			ans += (i-top.first-1)*(height[top.first]-top.second);
			stk.pop();
			stk.top().second = height[top.first];
			top = stk.top();
		}
		if(top.first!=-1){
			ans += (i-top.first-1)*(height[i]-top.second);
			stk.top().second = height[i];
		}
		stk.emplace(make_pair(i,height[i]));
	}
	return ans;
}
// 41. 缺失的第一个正数 https://leetcode.cn/problems/first-missing-positive/
// 给你一个未排序的整数数组 nums ，请你找出其中没有出现的最小的正整数。请你实现时间复杂度为 O(n) 并且只使用常数级别额外空间的解决方案。
int firstMissingPositive(vector<int>& nums) {
	int size = nums.size();
	int m = size;
	for(int i=0;i<size;++i){
		if(nums[i]==i||nums[i]<=0)
			continue;
		if(nums[i]>size)
			nums[i] = -1;
		int cur = nums[i];
		while(1){
			int tem = nums[cur-1];
			if(cur==tem)
				break;
			nums[cur-1] = cur;
			if(tem<=0||tem>size)
				break;
		}
	}
	for(int i=0;i<size;++i)
		if(nums[i]!=i+1)
			return i+1;
	return size+1;
}
// 60. 排列序列 https://leetcode.cn/problems/permutation-sequence/
// 给出集合 [1,2,3,...,n]，其所有元素共有 n! 种排列。按大小顺序列出所有排列情况，并一一标记，当 n = 3 时, 所有排列如下："123","132","213","231","312","321",给定 n 和 k，返回第 k 个排列。
string getPermutation(int n, int k) {
	int factorial[n];
	factorial[0] = 1;
	for(int i=1;i<n;++i)
		factorial[i] = factorial[i-1]*i;
	vector<int> nums(n+1);
	for(int i=0;i<=n;++i)
		nums[i] = i;
	string ans(n,' ');
	--k;
	for(int i=1;i<=n;++i){
		int order = k/factorial[n-i]+1;
		ans[i-1] = '0' + nums[order];
		nums.erase(nums.begin()+order);
		k %= factorial[n-i];
	}
	return ans;
}
// 1089. 复写零 https://leetcode.cn/problems/duplicate-zeros/
// 给你一个长度固定的整数数组 arr，请你将该数组中出现的每个零都复写一遍，并将其余的元素向右平移。注意：请不要在超过该数组长度的位置写入元素。要求：请对输入的数组 就地 进行上述修改，不要从函数返回任何东西。
void duplicateZeros(vector<int>& arr) {
	int size = arr.size(),l = 0, r =0;
	while(r<size){
		if(arr[l]==0)
			++r;
		++l;++r;
	}
	if(r>size){
		arr[--size] = 0;
		--l;
	}
	while(size>0){
		arr[--size] = arr[--l];
		if(arr[l]==0&&size>0)
			arr[--size] = 0;
	}       
}
// 50. Pow(x, n) https://leetcode.cn/problems/powx-n/
// 实现 pow(x, n) ，即计算 x 的整数 n 次幂函数（即，xn ）
double myPow(double x, int n) {
	double x_pow = x;
	double ans = 1.0;
	bool flag = false;
	if(n<0){
		flag = true;
		n++;
		n = -n;
	}
	while(n>0){
		if(n&1)
			ans *= x_pow;
		x_pow *= x_pow;
		n /= 2;
	}
	if(flag)
		ans = 1.0/(ans*x);
	return ans;
}
// 1044. 最长重复子串 https://leetcode.cn/problems/longest-duplicate-substring/
// 给你一个字符串 s ，考虑其所有 重复子串 ：即 s 的（连续）子串，在 s 中出现 2 次或更多次。这些出现之间可能存在重叠。返回 任意一个 可能具有最长长度的重复子串。如果 s 不含重复子串，那么答案为 "" 。
class LongestDupSubstring {
public:
    static long long mypow(int a, int m, int mod) {
        long long ans = 1;
        long long contribute = a;
        while (m > 0){
            if (m % 2==1){
                ans = ans * contribute % mod;
                if (ans < 0)
                    ans += mod;
            }
            contribute = contribute * contribute % mod;
            if (contribute < 0)
                contribute += mod;
            m /= 2;
        }
        return ans;
    }
    string longestDupSubstring(string s) {
        int size = s.size();
        int base1= 29, base2 = 31;
        int mod1 = int(1e9 + 9), mod2 = 998244353;
        auto check = [&](int len){
            unordered_set<long long> str_hash;
            long long t1 = mypow(base1,len,mod1),t2 = mypow(base2,len,mod2);
            long long last_hash1 = 0,last_hash2 = 0;
            for(int i=0;i<size;++i){
                if(i<len){
                    last_hash1 = (last_hash1*base1 + s[i])%mod1;
                    last_hash2 = (last_hash2*base2 + s[i])%mod2;
                }
                if(i==len-1)
                    str_hash.emplace( (last_hash1<< 32) | last_hash2);
                if(i>=len){
                    last_hash1 = (last_hash1*base1 - s[i-len]*t1+ s[i])%mod1;
                    last_hash2 = (last_hash2*base2 - s[i-len]*t2+ s[i])%mod2;
                    if(last_hash1<0)
                        last_hash1 += mod1;
                    if(last_hash2<0)
                        last_hash2 += mod2;
                    long long tem = (last_hash1<< 32) | last_hash2;
                    if(str_hash.count(tem))
                        return i-len+1;
                    str_hash.emplace(tem);
                }
            }
            return -1;
        };
        int l = 0,r = s.size()-1;
        int start = -1,length;
        while(l<=r){
            int mid = (l+r)/2;
            int tem = check(mid);
            if(tem!=-1){
                length = mid;
                l = mid+1;
                start = tem;
            }
            else
                r = mid-1;
        }
        return start==-1? "" :s.substr(start,length);
    }
};
// 5242. 兼具大小写的最好英文字母 https://leetcode.cn/problems/greatest-english-letter-in-upper-and-lower-case/
// 给你一个由英文字母组成的字符串 s ，请你找出并返回 s 中的 最好 英文字母。返回的字母必须为大写形式。如果不存在满足条件的字母，则返回一个空字符串。最好 英文字母的大写和小写形式必须 都 在 s 中出现。英文字母 b 比另一个英文字母 a 更好 的前提是：英文字母表中，b 在 a 之 后 出现。
string greatestLetter(string s) {
    int size = s.size();
	if(size==1)
		return "";
	char flag[26];
	memset(flag,0,sizeof(flag));
	for(auto& ch:s){
		if(ch>='a'&&ch<='z')
			flag[ch-'a'] |= 1;
		if(ch>='A'&&ch<='Z')
			flag[ch-'A'] |= 2;
	}
	for(int i=25;i>=0;--i)
		if(flag[i]==3)
			return string(1,char('A'+i));
	return "";
}
// 5218. 个位数字为 K 的整数之和 https://leetcode.cn/problems/sum-of-numbers-with-units-digit-k/
// 给你两个整数 num 和 k ，考虑具有以下属性的正整数多重集：每个整数个位数字都是 k 。所有整数之和是 num 。返回该多重集的最小大小，如果不存在这样的多重集，返回 -1 。注意：多重集与集合类似，但多重集可以包含多个同一整数，空多重集的和为 0 。个位数字 是数字最右边的数位。
int minimumNumbers(int num, int k) {
    int m = num / k;
	int need = num%10;
	int t = k;
	for(int i=1;i<=m;++i,t+=k)
		if((t%10)==need)
			return i;
	return -1;
}
// 6099. 小于等于 K 的最长二进制子序列 https://leetcode.cn/problems/longest-binary-subsequence-less-than-or-equal-to-k/
// 给你一个二进制字符串 s 和一个正整数 k 。请你返回 s 的 最长 子序列，且该子序列对应的 二进制 数字小于等于 k 。注意：子序列可以有 前导 0 。空字符串视为 0 。子序列 是指从一个字符串中删除零个或者多个字符后，不改变顺序得到的剩余字符序列。
int longestSubsequence(string s, int k) {
	int size = s.size();
	int maxlen = 32- __builtin_clz(k);
	if(maxlen>size)
		return size;
	int tem = 0;
	int cnt = 0;
	for(int i=size-maxlen;i<size;++i)
		tem = (tem<<1) + s[i] - '0';
	for(int i=0;i<size-maxlen;++i)
		if(s[i]=='0')
			++cnt;
	if(tem<=k)
		return cnt + maxlen;
	else
		return cnt + maxlen-1;
}
// 715. Range 模块 https://leetcode.cn/problems/range-module/
/*
 * Range模块是跟踪数字范围的模块。设计一个数据结构来跟踪表示为 半开区间 的范围并查询它们。半开区间 [left, right) 表示所有 left <= x< right 的实数 x 。
 * 实现 RangeModule 类:
 * RangeModule() 初始化数据结构的对象。
 * void addRange(int left, int right) 添加 半开区间 [left, right)，跟踪该区间中的每个实数。添加与当前跟踪的数字部分重叠的区间时，应当添加在区间 [left, right) 中尚未跟踪的任何数字到该区间中。
 * boolean queryRange(int left, int right) 只有在当前正在跟踪区间 [left, right) 中的每一个实数时，才返回 true ，否则返回 false 。
 * void removeRange(int left, int right) 停止跟踪 半开区间 [left, right) 中当前正在跟踪的每个实数。
 */
class RangeModule {
	map<int,int> ranges;
public:
    RangeModule() {
		ranges.emplace(-1,-1);
		ranges.emplace(INT32_MAX,INT32_MAX);
	}
    void addRange(int left, int right) {
		auto i = --ranges.upper_bound(left);
		if(i->second<left)
			++i;
		left = min(i->first,left);
		while(right>=i->first){
			right = max(i->second,right);
			i = ranges.erase(i);
		}
		ranges.emplace(left,right);
    }
    bool queryRange(int left, int right) {
		auto i = --ranges.upper_bound(right);
		if(i->second<=left)
			return false;
		return (i->first<=left)&&(i->second>=right);
    }
    void removeRange(int left, int right) {
		auto i = --ranges.upper_bound(left);
		if(i->second<=left)
			++i;
		int l = min(i->first,left),r = right;
		while(right>i->first){
			r = max(i->second,r);
			i = ranges.erase(i);
		}
		if(l!=left)
			ranges.emplace(l,left);
		if(r!=right)
			ranges.emplace(right,r);
    }
};
/* 
 * 有效数字（按顺序）可以分成以下几个部分：一个 小数 或者 整数（可选）一个 'e' 或 'E' ，后面跟着一个 整数
 * 小数（按顺序）可以分成以下几个部分：（可选）一个符号字符（'+' 或 '-'）
 * 下述格式之一：
 * 至少一位数字，后面跟着一个点 '.',
 * 至少一位数字，后面跟着一个点 '.' ，后面再跟着至少一位数字
 * 一个点 '.' ，后面跟着至少一位数字
 * 整数（按顺序）可以分成以下几个部分：
 * （可选）一个符号字符（'+' 或 '-'）,至少一位数字
 * 部分有效数字列举如下：["2", "0089", "-0.1", "+3.14", "4.", "-.9", "2e10", "-90E3", "3e+7", "+6e-1", "53.5e93", "-123.456e789"]
 * 部分无效数字列举如下：["abc", "1a", "1e", "e3", "99e2.5", "--6", "-+3", "95a54e53"]给你一个字符串 s ，如果 s 是一个 有效数字 ，请返回 true 。
 */
bool isNumber(string s) {
	bool flag[2] = {false,false}, flage = false,flagp = false,flag_pn = false;
	for(auto &ch:s){
		if(ch=='+'||ch=='-'){
			if(flag_pn)
				return false;   
		}
		else if('0'<=ch&&ch<='9')
			flag[flagp] = true;
		else if(ch=='.'){
			if(flagp||flage)
				return false;
			flagp = true;
		}
		else if(ch=='E'||ch=='e'){
			if(flage)
				return false;
			if(flag[0]||flag[1]){
				flage = true;
				flag[0] = false; flag[1] = false;
				flag_pn = false;
			}
			else
				return false;
		}
		else
			return false;
		flag_pn = (ch!='E'&&ch!='e');
	}
	return flag[0]||flag[1];
}
// 68. 文本左右对齐 https://leetcode.cn/problems/text-justification/
// 给定一个单词数组 words 和一个长度 maxWidth ，重新排版单词，使其成为每行恰好有 maxWidth 个字符，且左右两端对齐的文本。你应该使用 “贪心算法” 来放置给定的单词；也就是说，尽可能多地往每行中放置单词。必要时可用空格 ' ' 填充，使得每行恰好有 maxWidth 个字符。要求尽可能均匀分配单词间的空格数量。如果某一行单词间的空格不能均匀分配，则左侧放置的空格数要多于右侧的空格数。文本的最后一行应为左对齐，且单词之间不插入额外的空格。注意:单词是指由非空格字符组成的字符序列。每个单词的长度大于 0，小于等于 maxWidth。输入单词数组 words 至少包含一个单词。
vector<string> fullJustify(vector<string>& words, int maxWidth) {
	int pos = 0,size = words.size(),cur;
	vector<string> ans;
	while(pos<size){
		int prev = pos;
		cur = -1;
		while(pos<size&&cur+words[pos].size()+1<=maxWidth){
			cur += words[pos].size()+1;
			++pos;
		}
		string tem = words[prev];
		if(prev+1==pos||pos==size){
			for(int i=prev+1;i<pos;++i)
				tem += (' '+words[i]);
			tem += string(maxWidth-tem.size(),' ');
			ans.push_back(tem);
			continue;
		}
		int space = maxWidth - cur,k = space/(pos-prev-1);
		space = space%(pos-prev-1);
		for(int i=prev+1;i<pos;++i){
			if(space!=0){
				tem += string(k+2,' ')+words[i];
				--space;
			}
			else
				tem += string(k+1,' ')+words[i];
		}
		ans.push_back(tem);
	}
	return ans;
}
// 2315. 统计星号 https://leetcode.cn/problems/count-asterisks/
// 给你一个字符串 s ，每 两个 连续竖线 '|' 为 一对 。换言之，第一个和第二个 '|' 为一对，第三个和第四个 '|' 为一对，以此类推。请你返回 不在 竖线对之间，s 中 '*' 的数目。注意，每个竖线 '|' 都会 恰好 属于一个对。
int countAsterisks(string s) {
    bool flag = true;
    int cnt = 0;
    for(auto& ch:s){
        if(ch=='|')
            flag = !flag;
        else if(ch=='*'&&flag)
            ++cnt;
    }      
    return cnt; 
}
// 2316. 统计无向图中无法互相到达点对数 https://leetcode.cn/problems/count-unreachable-pairs-of-nodes-in-an-undirected-graph/
// 给你一个整数 n ，表示一张 无向图 中有 n 个节点，编号为 0 到 n - 1 。同时给你一个二维整数数组 edges ，其中 edges[i] = [ai, bi] 表示节点 ai 和 bi 之间有一条 无向 边。请你返回 无法互相到达 的不同 点对数目 。
class UnionSet{
    int* rank;
    int* fa;
public:
    UnionSet():rank(new int[20]),fa(new int [20]){
        for(int i=0;i<20;++i){
            rank[i] = 1;
            fa[i] = i;
        }
    }
    UnionSet(int n):rank(new int[n]),fa(new int [n]){
        for(int i=0;i<n;++i){
            rank[i] = 1;
            fa[i] = i;
        }
    }
    int find(int x){
        if(fa[x]!=x)
            fa[x] = find(fa[x]);
        return fa[x];
    }
    bool join(int x,int y){
        int fx = find(x),fy= find(y);
        if(fx==fy)
            return false;
        if(rank[fx]<rank[fy]){
            int tem = fy;
            fy = fx;
            fx = tem;
        }
        rank[fx] += rank[fy];
        fa[fy] = fx;
        return true;
    }
    int getCount(int i){
        return rank[i];
    }
};
class Solution {
public:
    long long countPairs(int n, vector<vector<int>>& edges) {
        long long ans = (long long)n*(n-1)/2;
        UnionSet us(n);
        for(auto& edge:edges)
            us.join(edge[0],edge[1]);
        for(int i=0;i<n;++i){
            if(us.find(i)==i){
                int cnt = us.getCount(i);
                ans -= (long long)cnt* (cnt-1)/2; 
            }
        }
        return ans;
    }
};
// 2317. 操作后的最大异或和 https://leetcode.cn/problems/maximum-xor-after-operations/
// 给你一个下标从 0 开始的整数数组 nums 。一次操作中，选择 任意 非负整数 x 和一个下标 i ，更新 nums[i] 为 nums[i] AND (nums[i] XOR x) 。注意，AND 是逐位与运算，XOR 是逐位异或运算。请你执行 任意次 更新操作，并返回 nums 中所有元素 最大 逐位异或和。
int maximumXOR(vector<int>& nums) {
    char flag[32];
    memset(flag,0,sizeof(flag));
    for(auto &num:nums){
        for(int i=0;i<32;++i,num /= 2){
            if(num&1)
                flag[i] = 1;
        }
    } 
    int ans = 0;    
    for(int i=31;i>=0;--i)
            ans = (ans<<1) + flag[i];
    return ans;
}
// 522. 最长特殊序列 II https://leetcode.cn/problems/longest-uncommon-subsequence-ii/
// 给定字符串列表 strs ，返回其中 最长的特殊序列 的长度。如果最长特殊序列不存在，返回 -1 。特殊序列 定义如下：该序列为某字符串 独有的子序列（即不能是其他字符串的子序列）。 s 的 子序列可以通过删去字符串 s 中的某些字符实现。例如，"abc" 是 "aebdc" 的子序列，因为您可以删除"aebdc"中的下划线字符来得到 "abc" 。"aebdc"的子序列还包括"aebdc"、 "aeb" 和 "" (空字符串)。
int findLUSlength(vector<string>& strs) {
	auto check = [](const string& a,const string& b){
		int m = a.size(),n =  b.size();
		int j = 0;
		for(int i=0;i<n;++i){
			if(b[i]==a[j])
				++j;
			if(j==m)
				break;
		}
		return j==m;
	};
	int size = strs.size();
	int ans = 0;
	for(int i=0;i<size;++i){
		bool flag = true;
		for(int j=0;j<size;++j){
			if(j!=i&&check(strs[i],strs[j]))
				flag = false;
		}
		if(flag)
			ans = max(ans,(int)strs[i].size());
	}
	return ans;
}
// 6108. 解密消息 https://leetcode.cn/problems/decode-the-message/
// 给你字符串 key 和 message ，分别表示一个加密密钥和一段加密消息。解密 message 的步骤如下：使用 key 中 26 个英文小写字母第一次出现的顺序作为替换表中的字母 顺序 。将替换表与普通英文字母表对齐，形成对照表。按照对照表 替换 message 中的每个字母。空格 ' ' 保持不变。例如，key = "happy boy"（实际的加密密钥会包含字母表中每个字母 至少一次），据此，可以得到部分对照表（'h' -> 'a'、'a' -> 'b'、'p' -> 'c'、'y' -> 'd'、'b' -> 'e'、'o' -> 'f'）。返回解密后的消息。
string decodeMessage(string key, string message) {
	char mp[26];
	memset(mp,0,sizeof(mp));
	char val = 'a';
	for(auto& ch:key){
		if(ch==' ')
			continue;
		char tem = ch-'a';
		if(mp[tem]==0)
			mp[tem] = val++;
	}
	for(auto& ch:message){
		if(ch==' ')
			continue;
		ch = mp[ch-'a'];
	}
	return message;
}
struct ListNode {
	int val;
	ListNode* next;
	ListNode() :val(0), next(NULL) {};
	ListNode(int x) : val(x), next(NULL) {};
	ListNode(int x, ListNode* p) :val(x), next(p) {};
};
// 6111. 螺旋矩阵 IV https://leetcode.cn/problems/spiral-matrix-iv/
// 给你两个整数：m 和 n ，表示矩阵的维数。另给你一个整数链表的头节点 head 。请你生成一个大小为 m x n 的螺旋矩阵，矩阵包含链表中的所有整数。链表中的整数从矩阵 左上角 开始、顺时针 按 螺旋 顺序填充。如果还存在剩余的空格，则用 -1 填充。返回生成的矩阵。
vector<vector<int>> spiralMatrix(int m, int n, ListNode* head) {
	vector<vector<int>> ans(m,vector<int>(n,-1));
	int top = 0,bottom = m-1,left = 0,right = n-1;
	while(head&&top<bottom&&left<right){
		for(int i=left;i<right;++i,head = head->next){
			if(!head)
				break;
			ans[top][i] = head->val;
		}
		for(int i=top;i<bottom;++i,head = head->next){
			if(!head)
				break;
			ans[i][right] = head->val;
		}
		for(int i=right;i>left;--i,head=head->next){
			if(!head)
				break;
			ans[bottom][i] = head->val;
		}
		for(int i=bottom;i>top;--i,head = head->next){
			if(!head)
				break;
			ans[i][left] = head->val;
		}
		++top;--bottom;++left;--right;
	}
	if(head&&left==right){
		for(int i=top;i<=bottom;++i,head = head->next){
			if(!head)
				break;
			ans[i][left] = head->val;
		}
	}
	if(head&&top==bottom){
		for(int i=left;i<=right;++i,head = head->next){
			if(!head) 
				break;
			ans[top][i] = head->val;
		}
	}
	return ans;
}
// 6109. 知道秘密的人数 https://leetcode.cn/problems/number-of-people-aware-of-a-secret/
// 在第 1 天，有一个人发现了一个秘密。给你一个整数 delay ，表示每个人会在发现秘密后的 delay 天之后，每天 给一个新的人 分享 秘密。同时给你一个整数 forget ，表示每个人在发现秘密 forget 天之后会 忘记 这个秘密。一个人 不能 在忘记秘密那一天及之后的日子里分享秘密。给你一个整数 n ，请你返回在第 n 天结束时，知道秘密的人数。由于答案可能会很大，请你将结果对 109 + 7 取余 后返回。
int peopleAwareOfSecret(int n, int delay, int forget) {
	long long cnt[forget];
	cnt[0] = 1;
	int begin = 0;
	for(int i=2;i<n;++i){
		long long new_discover = 0;
		for(int j=delay;j<forget;++j)
			new_discover += cnt[(begin+delay-1)%forget];
		--begin;
		if(begin<0)
			begin = forget-1;
		cnt[begin] = new_discover;
	}
	long long ans = 0;
	for(int i=0;i<forget;++i)
		ans += cnt[i];
	return ans;
}
// 648. 单词替换 https://leetcode.cn/problems/replace-words/
// 在英语中，我们有一个叫做 词根(root) 的概念，可以词根后面添加其他一些词组成另一个较长的单词——我们称这个词为 继承词(successor)。例如，词根an，跟随着单词 other(其他)，可以形成新的单词 another(另一个)。现在，给定一个由许多词根组成的词典 dictionary 和一个用空格分隔单词形成的句子 sentence。你需要将句子中的所有继承词用词根替换掉。如果继承词有许多可以形成它的词根，则用最短的词根替换它。你需要输出替换之后的句子。
string replaceWords(vector<string>& dictionary, string sentence) {
    unordered_set<long long> st;
     int base1= 29, base2 = 31;
    int mod1 = int(1e9 + 9), mod2 = 998244353;
    for(auto dict:dictionary){
        long long hash1 = 0,hash2 = 0;
        for(auto& ch:dict){
            hash1 = (hash1*base1 + ch)%mod1;
            hash2 = (hash2*base2 + ch)%mod2;
        }
        st.emplace((hash1<<32) + hash2);
    }
    long long hash1 = 0,hash2 = 0;
    int last = 0;
    int size = sentence.size();
    bool flag = false;
    string ans;
    ans.reserve(size);
    for(int i=0;i<size;++i){
        if(sentence[i]==' '){
            hash1 = 0;
            hash2 = 0;
            if(!flag){
                if(last==0)
                    ans.append(sentence.substr(last,i-last));
                else
                    ans.append(' '+sentence.substr(last,i-last));
            }
            else
                flag = false;
            last = i + 1;
            continue;
        }
        if(flag)
            continue;
        hash1 = (hash1*base1 + sentence[i])%mod1;
        hash2 = (hash2*base2 + sentence[i])%mod2;
        long long hash = (hash1 << 32)+ hash2;
        if(st.count(hash)){
            flag = true;
            if(last==0)
                ans.append(sentence.substr(last,i-last+1));
            else
                ans.append(' '+sentence.substr(last,i-last+1));
        }
    }
    if(!flag){
        if(last==0)
            ans.append(sentence.substr(last,size-last));
        else
            ans.append(' '+sentence.substr(last,size-last));
    }
    return ans;
}
// zj-future01. 信号接收 https://leetcode.cn/contest/zj-future2022/problems/WYKGLO/
// 假设有若干信号发射源定时发送信号， signals[i] = [start, end) 表示第 i 个信号发射源运作的开始时间 start 和停止时间 end 。若调度员的接收设备同一时刻仅能接收一个发射源发出的信号，请判断调度员能否收到所有发射源的完整信号。注意：只有接收了一个信号源从开始到结束的所有信号才算完整信号。
bool canReceiveAllSignals(vector<vector<int>>& intervals) {
    sort(intervals.begin(),intervals.end(),[](const vector<int>& a,const vector<int> &b){
        return a[0]<b[0];
    });
    int right = -1;
    for(auto& interval:intervals){
        if(interval[0]<right)
            return false;
        right = max(right,interval[1]);
    }
    return true;
}
// zj-future02. 黑白棋游戏 https://leetcode.cn/contest/zj-future2022/problems/GVbKaI/
// 现有一个黑白棋游戏，初始时给出一排棋子，记作数组 chess，其中白色棋子记作 0，黑色棋子记作 1。用户可以每次交换 任意位置 的两颗棋子的位置。为了使得所有黑色棋子相连，请返回最少需要交换多少次。
int minSwaps(vector<int>& chess) {
    int b_cnt = 0;
    for(auto& chs:chess)
        if(chs==1)
            ++b_cnt;
    int window = 0;
    for(int i=0;i<b_cnt;++i)
        if(!chess[i])
            ++window;
    int ans = window,left = 0,right = b_cnt;
    int size = chess.size();
    while(right<size){
        if(!chess[left++])
            --window;
        if(!chess[right++])
            ++window;
        ans = min(ans,window);
    }
    return ans;
}
// zj-future03. 快递中转站选址 https://leetcode.cn/contest/zj-future2022/problems/kflZMc/
// 某区域地图记录在 k 二维数组 area，其中 0 表示空地，1 表示快递分发点。若希望选取一个地点设立中转站，使得中转站至各快递分发点的「曼哈顿距离」总和最小。请返回这个 最小 的距离总和。注意：曼哈顿距离：distance(p1, p2) = |p2.x - p1.x| + |p2.y - p1.y| 所有位置均可作为快递中转站的设立点。
int buildTransferStation(vector<vector<int>>& area) {
    unordered_set<long long> st;
    int m = area.size(), n = area.size();
    for(int i=0;i<m;++i){
        for(int j=0;j<n;++j){
            if(area[i][j]==1)
                st.emplace(((long long)i<<32)+j);
        }
    }
    int min_x = 0,min_y = 0,min = INT16_MAX;
    for(int i=0;i<m;++i){
        int dis = 0;
        for(auto& point:st)
            dis += (point>>32) + abs((point&0xFFFFFFFF)-i);
        if(dis<min){
            min = dis;
            min_y = i;
        }
    }
    for(int i=0;i<n;++i){
        int dis = 0;
        for(auto& point:st)
            dis += abs((point>>32)-i) + (point&0xFFFFFFFF);
        if(dis<min){
            min = dis;
            min_x = i;
        }
    }
    int ans = 0;
    for(auto& point:st)
        ans += abs((point>>32)-min_x) + abs((point&0xFFFFFFFF)-min_y);
    return ans;
}
// zj-future04. 门店商品调配 https://leetcode.cn/contest/zj-future2022/problems/NBCXIp/
// 某连锁店开设了若干门店，门店间允许进行商品借调以应对暂时性的短缺。本月商品借调的情况记于数组 distributions，其中 distributions[i] = [from,to,num]，表示从 from 门店调配了 num 件商品给 to 门店。若要使得每一个门店最终借出和借入的商品数量相同，请问至少还需要进行多少次商品调配。注意：一次商品调配以三元组 [from, to, num] 表示，并有 from ≠ to 且 num > 0。
int minTransfers(vector<vector<int>>& distributions) {
    int size = distributions.size();
    int cnt[12] = {0};
    for(auto& distribution:distributions){
        cnt[distribution[0]] -= distribution[2];
        cnt[distribution[1]] += distribution[2];
    }
    int tem[12];
    multiset<int,greater<int>> st;
    for(int i=0;i<12;++i)
        if(cnt[i])
            st.emplace(cnt[i]);
    int ans = INT32_MAX;
    function<void(int)> bfs = [&](int depth){
        if(depth>=ans)
            return ;
        if(st.empty())
            ans = min(ans,depth);
        int max = *st.begin();
        auto i = st.upper_bound(0);
        int j = 0;
        while(i!=st.end()){
            int val = -(*i);
            st.erase(max);
            st.erase(val);
            st.emplace(max-val);
            bfs(depth+1);    
            st.erase(max-val);
            st.emplace(val);
            st.emplace(max);
            ++j;
            auto i = st.upper_bound(0);
            for(int k=0;k<j&&i!=st.end();++k)
                ++i;
        }
    };
    bfs(0);
    return ans;
}
struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode() : val(0), left(nullptr), right(nullptr) {}
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
};
// 2331. 计算布尔二叉树的值 https://leetcode.cn/problems/evaluate-boolean-binary-tree/
// 给你一棵 完整二叉树 的根，这棵树有以下特征：叶子节点 要么值为 0 要么值为 1 ，其中 0 表示 False ，1 表示 True 。非叶子节点 要么值为 2 要么值为 3 ，其中 2 表示逻辑或 OR ，3 表示逻辑与 AND 。计算 一个节点的值方式如下：如果节点是个叶子节点，那么节点的 值 为它本身，即 True 或者 False 。否则，计算 两个孩子的节点值，然后将该节点的运算符对两个孩子值进行 运算 。返回根节点 root 的布尔运算值。完整二叉树 是每个节点有 0 个或者 2 个孩子的二叉树。叶子节点 是没有孩子的节点。
bool evaluateTree(TreeNode* root) {
    if(!root->left)
        return root->val;
    if(root->val==2)
        return evaluateTree(root->left)||evaluateTree(root->right);
    else
        return evaluateTree(root->left)&&evaluateTree(root->right);
}
// 2332. 坐上公交的最晚时间 https://leetcode.cn/problems/the-latest-time-to-catch-a-bus/
// 给你一个下标从 0 开始长度为 n 的整数数组 buses ，其中 buses[i] 表示第 i 辆公交车的出发时间。同时给你一个下标从 0 开始长度为 m 的整数数组 passengers ，其中 passengers[j] 表示第 j 位乘客的到达时间。所有公交车出发的时间互不相同，所有乘客到达的时间也互不相同。给你一个整数 capacity ，表示每辆公交车 最多 能容纳的乘客数目。每位乘客都会搭乘下一辆有座位的公交车。如果你在 y 时刻到达，公交在 x 时刻出发，满足 y <= x  且公交没有满，那么你可以搭乘这一辆公交。最早 到达的乘客优先上车。返回你可以搭乘公交车的最晚到达公交站时间。你 不能 跟别的乘客同时刻到达。注意：数组 buses 和 passengers 不一定是有序的。
int latestTimeCatchTheBus(vector<int>& buses, vector<int>& passengers, int capacity) {
    sort(buses.begin(),buses.end());
    sort(passengers.begin(),passengers.end());
    int i = 0,cnt;
    int p_size = passengers.size();
    for(auto &bus:buses){
        cnt = capacity;
        while(i<p_size&&cnt&&passengers[i]<=bus){
            ++i;
            --cnt;
        }
    }
    --i;
    int ans = cnt? buses.back():passengers[i];
    while(i>=0&&passengers[i--]==ans)
        --ans;
    return ans;
}
// 2333. 最小差值平方和 https://leetcode.cn/problems/minimum-sum-of-squared-difference/
// 给你两个下标从 0 开始的整数数组 nums1 和 nums2 ，长度为 n 。数组 nums1 和 nums2 的 差值平方和 定义为所有满足 0 <= i < n 的 (nums1[i] - nums2[i])2 之和。同时给你两个正整数 k1 和 k2 。你可以将 nums1 中的任意元素 +1 或者 -1 至多 k1 次。类似的，你可以将 nums2 中的任意元素 +1 或者 -1 至多 k2 次。请你返回修改数组 nums1 至多 k1 次且修改数组 nums2 至多 k2 次后的最小 差值平方和 。注意：你可以将数组中的元素变成 负 整数。
long long minSumSquareDiff(vector<int>& nums1, vector<int>& nums2, int k1, int k2) {
    int n = nums1.size();
    vector<int> diff(n+1);
    for(int i=0;i<n;++i)
        diff[i] = abs(nums1[i]-nums2[i]);
    diff[n] = 0;
    sort(diff.begin(),diff.end());
    int k = k1 + k2;
    for(int i=n;i>0;--i){
        long long d = ((long long)(n-i+1)) *(diff[i]-diff[i-1]);
        if(d<=k)
            k -= d;
        else{
            int t = diff[i] - k/(n-i+1);
            for(int j=i;j<=n;++j)
                diff[j] = t;
            t = k %(n-i+1);
            for(int j=0;j<t;++j)
                --diff[i+j];
            long long ans = 0;
            for (int j = 1;j<=n;j++) 
                ans += ((long long)diff[j])*diff[j];
            return ans;
        }
    }
    return 0;
}
// 2334. 元素值大于变化阈值的子数组 https://leetcode.cn/problems/subarray-with-elements-greater-than-varying-threshold/
// 给你一个整数数组 nums 和一个整数 threshold 。找到长度为 k 的 nums 子数组，满足数组中 每个 元素都 大于 threshold / k 。请你返回满足要求的 任意 子数组的 大小 。如果没有这样的子数组，返回 -1 。子数组 是数组中一段连续非空的元素序列。
// 并查集解法
class Solution {
    class UnionSet{
        int* rank;
        int* fa;
    public:
        UnionSet(int n):rank(new int[n]),fa(new int [n]){
            for(int i=0;i<n;++i){
                rank[i] = 1;
                fa[i] = i;
            }
        }
        ~UnionSet(){
            delete[] rank;
            delete[] fa;
        }
        int find(int x){
            if(fa[x]!=x)
                fa[x] = find(fa[x]);
            return fa[x];
        }
        int join(int x,int y){
            int fx = find(x),fy= find(y);
            if(fx==fy)
                return rank[fx];
            if(rank[fx]<rank[fy]){
                int tem = fy;
                fy = fx;
                fx = tem;
            }
            rank[fx] += rank[fy];
            fa[fy] = fx;
            return rank[fx];
        }
    };
    struct Node{
        int val;
        int pos;
        Node(int val,int pos):val(val),pos(pos){}
        bool operator< (const Node& other) const {
            return val < other.val;
        }
    };
public:
    int validSubarraySize(vector<int>& nums, int threshold) {
        int n = nums.size();
        UnionSet us(n+1);
        int fa[n + 1], sz[n + 1];
        iota(fa, fa + n + 1, 0);
        for (int i = 0; i <= n; ++i) sz[i] = 1;
        int ids[n];
        iota(ids, ids + n, 0);
        sort(ids, ids + n, [&](int i, int j) { 
            return nums[i] > nums[j]; 
        });
        for (int i : ids) {
            int size = us.join(i,i+1);
            if (nums[i] > threshold/(size-1)) 
                return size-1;
        }
        return -1;
    }
};
// 单调栈解法
int validSubarraySize(vector<int> &nums, int threshold) {
    int n = nums.size();
    int left[n]; 
    stack<int> s;
    for (int i = 0; i < n; ++i) {
        while(!s.empty()&&nums[s.top()]>=nums[i]) 
            s.pop();
        left[i] = s.empty() ? -1 : s.top();
        s.push(i);
    }
    int right[n];
    s = stack<int>();
    for (int i = n - 1; i >= 0; --i) {
        while (!s.empty() && nums[s.top()] >= nums[i]) 
            s.pop();
        right[i] = s.empty() ? n : s.top();
        s.push(i);
    }
    for (int i = 0; i < n; ++i) {
        int k = right[i] - left[i] - 1;
        if (nums[i] > threshold / k) 
            return k;
    }
    return -1;
}
// 2344. 使数组可以被整除的最少删除次数 https://leetcode.cn/problems/minimum-deletions-to-make-array-divisible/
// 给你两个正整数数组 nums 和 numsDivide 。你可以从 nums 中删除任意数目的元素。请你返回使 nums 中 最小 元素可以整除 numsDivide 中所有元素的 最少 删除次数。如果无法得到这样的元素，返回 -1 。如果 y % x == 0 ，那么我们说整数 x 整除 y 。
int gcd(int a,int b){
    if(a<b) return gcd(b,a);
    if(b==0) return a;
    return gcd(b,a%b);
}
int minOperations(vector<int>& nums, vector<int>& numsDivide) {
    int g  = 0;
    for(auto& d:numsDivide)
        g = gcd(g,d);
    sort(nums.begin(),nums.end());
    int size = nums.size();
    for(int i=0;i<size;++i)
        if(g%nums[i]==0)
            return i;
    return -1;
}