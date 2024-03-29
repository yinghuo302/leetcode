/*
 * @Author: zanilia
 * @Date: 2022-07-03 22:27:55
 * @LastEditTime: 2023-02-28 10:47:03
 * @Descripttion: 
 */
#include <bits/stdc++.h>
using namespace std;
// 2335. 装满杯子需要的最短总时长 https://leetcode.cn/problems/minimum-amount-of-time-to-fill-cups/
// 现有一台饮水机，可以制备冷水、温水和热水。每秒钟，可以装满 2 杯 不同 类型的水或者 1 杯任意类型的水。给你一个下标从 0 开始、长度为 3 的整数数组 amount ，其中 amount[0]、amount[1] 和 amount[2] 分别表示需要装满冷水、温水和热水的杯子数量。返回装满所有杯子所需的 最少 秒数。
int fillCups(vector<int>& amount) {       
    sort(amount.begin(),amount.end());
    if(amount[0] + amount[1] < amount[2])
        return amount[2];
    int num = min(amount[0],(amount[2]-amount[1]-amount[0])/2);
    return min(amount[0]+amount[1]-num,amount[2]) + num;
}
// 2336. 无限集中的最小数字 https://leetcode.cn/problems/smallest-number-in-infinite-set/
/*
 * 现有一个包含所有正整数的集合 [1, 2, 3, 4, 5, ...] 。实现 SmallestInfiniteSet 类：
 * SmallestInfiniteSet() 初始化 SmallestInfiniteSet 对象以包含 所有 正整数。
 * int popSmallest() 移除 并返回该无限集中的最小整数。
 * void addBack(int num) 如果正整数 num 不 存在于无限集中，则将一个 num 添加 到该无限集中。
*/
class SmallestInfiniteSet {
	map<int,int> intervals;
public:
    SmallestInfiniteSet() {
		intervals.emplace(-2,-2);
        intervals.emplace(1,INT32_MAX);
	}
    void addBack(int val) {
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
    int popSmallest() {
        auto i = (++intervals.begin()) ;
        int begin = i->first+1, end = i->second;
        intervals.erase(i);
        if(begin<=end)
            intervals.emplace(begin,end);
    }
};
// 2337. 移动片段得到字符串 https://leetcode.cn/problems/move-pieces-to-obtain-a-string/
// 给你两个字符串 start 和 target ，长度均为 n 。每个字符串 仅 由字符 'L'、'R' 和 '_' 组成，其中：字符 'L' 和 'R' 表示片段，其中片段 'L' 只有在其左侧直接存在一个 空位 时才能向 左 移动，而片段 'R' 只有在其右侧直接存在一个 空位 时才能向 右 移动。字符 '_' 表示可以被 任意 'L' 或 'R' 片段占据的空位。如果在移动字符串 start 中的片段任意次之后可以得到字符串 target ，返回 true ；否则，返回 false 。
class CanChange{ 
    struct Node{
        char ch;
        int pos;
        Node(char ch,int pos):ch(ch),pos(pos){}
    };
    static vector<Node> getNode(const string& s){
        vector<Node> ans;
        int size = s.size();
        for(int i=0;i<size;++i)
            if(s[i]=='L'||s[i]=='R')
                ans.push_back(Node(s[i],i));
        return ans;
    }
    bool canChange(string start, string target) {
        vector<Node> from = getNode(start),to =  getNode(target);
        int size = from.size();
        if(size!=to.size())
            return false;
        for(int i=0;i<size;++i){
            if(from[i].ch!=to[i].ch)
                return false;
            if(from[i].ch=='L'&&from[i].pos<to[i].pos)
                return false;
            if(from[i].ch=='R'&&from[i].pos>to[i].pos)
                return false;
        }
        return true;
    }
};
// 2338. 统计理想数组的数目 https://leetcode.cn/problems/count-the-number-of-ideal-arrays/
// 给你两个整数 n 和 maxValue ，用于描述一个 理想数组 。对于下标从 0 开始、长度为 n 的整数数组 arr ，如果满足以下条件，则认为该数组是一个 理想数组 ：每个 arr[i] 都是从 1 到 maxValue 范围内的一个值，其中 0 <= i < n 。每个 arr[i] 都可以被 arr[i - 1] 整除，其中 0 < i < n 。返回长度为 n 的 不同 理想数组的数目。由于答案可能很大，返回对 109 + 7 取余的结果。
class IdealArrays{
    const int MOD = 1000000007;
    vector<long long> inv;
    long long C(int a, int b) {
        if (b > a) return 0;
        long long ret = 1;
        for (int i = 1; i <= b; i++) 
            ret = (ret * (a - i + 1) % MOD * inv[i]) % MOD;
        return ret;
    }
public:
    int idealArrays(int n, int maxValue) {
        vector<vector<int>> f( maxValue+ 1);
        int mx = 0;
        for (int i = 2; i <= maxValue; i++) {
            if (f[i].empty()) {  // 说明i是质数
                for (int j = i; j <= maxValue; j += i) {
                    int x = j, y = 0;
                    for (; x % i == 0; x /= i) 
                        y++;
                    f[j].push_back(y);
                    mx = max(mx, y);
                }
            }
        }
        inv.resize(mx + 5);
        inv[1] = 1;
        for (int i = 2; i <= mx; i++) 
            inv[i] = (MOD - MOD / i) * inv[MOD % i] % MOD;
        long long ans = 0;
        for (int i=1;i<=maxValue; i++) {
            long long t = 1;
            for (int x : f[i]) 
                t = (t * C(n + x - 1, x)) % MOD;
            ans = (ans + t) % MOD;
        }
        return ans;
    }
};
// 2341. 数组能形成多少数对 https://leetcode.cn/problems/maximum-number-of-pairs-in-array/
// 给你一个下标从 0 开始的整数数组 nums 。在一步操作中，你可以执行以下步骤：从 nums 选出 两个 相等的 整数从 nums 中移除这两个整数，形成一个 数对请你在 nums 上多次执行此操作直到无法继续执行。返回一个下标从 0 开始、长度为 2 的整数数组 answer 作为答案，其中 answer[0] 是形成的数对数目，answer[1] 是对 nums 尽可能执行上述操作后剩下的整数数目。
vector<int> numberOfPairs(vector<int>& nums) {
    vector<int> ans = {0,0};
    int cnt[101];
    memset(cnt,0,sizeof(cnt));
    for(auto& num:nums){
        ++cnt[num];
        if(!(cnt[num]&1))
            ++ans[0];
    }
    ans[1] = nums.size() - ans[0]*2;
    return ans;
}
// 2342. 数位和相等数对的最大和 https://leetcode.cn/problems/max-sum-of-a-pair-with-equal-sum-of-digits/
// 给你一个下标从 0 开始的数组 nums ，数组中的元素都是 正 整数。请你选出两个下标 i 和 j（i != j），且 nums[i] 的数位和 与  nums[j] 的数位和相等。请你找出所有满足条件的下标 i 和 j ，找出并返回 nums[i] + nums[j] 可以得到的 最大值 。
int maximumSum(vector<int>& nums) {
    unordered_map<int,vector<int>> mp;
    for(auto& num:nums){
        int sum = 0,tem = num;
        while(tem!=0){
            sum += sum%10;
            sum /= 10;
        }
        mp[sum].push_back(num);
    }
    int ans = -1;
    for(auto& p:mp){
        auto& arr = p.second;
        if(arr.size()<2)
            continue;
        int first = INT32_MIN,second = INT32_MIN;
        for(auto &num:arr){
            if(num>=first){
                second = first;
                first = num;
            }
        }
        ans = max(ans,second+first);
    }
    return ans;
}
// 2343. 裁剪数字后查询第 K 小的数字 https://leetcode.cn/problems/query-kth-smallest-trimmed-number/
// 给你一个下标从 0 开始的字符串数组 nums ，其中每个字符串 长度相等 且只包含数字。再给你一个下标从 0 开始的二维整数数组 queries ，其中 queries[i] = [ki, trimi] 。对于每个 queries[i] ，你需要：将 nums 中每个数字 裁剪 到剩下 最右边 trimi 个数位。在裁剪过后的数字中，找到 nums 中第 ki 小数字对应的 下标 。如果两个裁剪后数字一样大，那么下标 更小 的数字视为更小的数字。将 nums 中每个数字恢复到原本字符串。请你返回一个长度与 queries 相等的数组 answer，其中 answer[i]是第 i 次查询的结果。提示：裁剪到剩下 x 个数位的意思是不断删除最左边的数位，直到剩下 x 个数位。nums 中的字符串可能会有前导 0 。
class Solution {
    struct Node{
        int pos;
        int k;
        int trim;
        Node(int pos,int k,int trim):pos(pos),k(k),trim(trim){}
    };
public:
    vector<int> smallestTrimmedNumbers(vector<string>& nums, vector<vector<int>>& queries) {
        int size = queries.size();
        vector<Node> q;
        vector<int> ans(size);
        q.reserve(size);
        for(int i=0;i<size;++i)
            q.emplace_back(Node(i,queries[i][0],queries[i][1]));
        sort(q.begin(),q.end(),[](const Node&a,const Node& b){
            if(a.trim<b.trim)
                return true;
            else if(a.trim==b.trim)
                return a.k < b.k;
            return false;
        });
        list<int> l[2][10];
        int size1 = nums.size(),size2 = nums[0].size();
        int pos = size2 - 1;
        for(int i=0;i<size1;++i)
            l[pos&1][nums[i][pos]-'0'].push_back(i);
        int m = min(q.back().trim,size2-1);
        int q_pos = 0;
        vector<int> tem(size1);
        for(int i=1;i<=m;++i){
            --pos;
            int tem_pos = 0;
            int cur = pos&1, pre = cur^1;
            for(int j=0;j<10;++j){
                while(!l[pre][j].empty()){
                    int t = l[pre][j].back();
                    l[pre][j].pop_back();   
                    tem[tem_pos++] = t;
                    l[cur][nums[t][pos]-'0'].push_front(t);
                }
            }
            while(q_pos<size&&q[q_pos].trim==i){
                ans[q[q_pos].pos] = tem[q[q_pos].k-1];
                ++q_pos;
            }
        }
        if(q.back().trim==size2){
            int tem_pos = 0;
            for(int i=0;i<10;++i){
                while(!l[0][i].empty()){
                    int t = l[0][i].back();
                    l[0][i].pop_back(); 
                    tem[tem_pos++] = t;
                }
            }
            while(q_pos<size){
                ans[q[q_pos].pos] = tem[q[q_pos].k-1];
                ++q_pos;
            }
        }
        return ans;
    }
};
// 658. 找到 K 个最接近的元素 https://leetcode.cn/problems/find-k-closest-elements/
// 给定一个 排序好 的数组 arr ，两个整数 k 和 x ，从数组中找到最靠近 x（两数之差最小）的 k 个数。返回的结果必须要是按升序排好的。整数 a 比整数 b 更接近 x 需要满足：|a - x| < |b - x| 或者|a - x| == |b - x| 且 a < b
vector<int> findClosestElements(vector<int>& arr, int k, int x) {
    int size = arr.size();
    if(arr.back()<=x)
        return vector<int>(arr.begin()+size-k,arr.begin()+size);
    if(arr[0]>=x)
        return vector<int>(arr.begin(),arr.begin()+k);
    int idx = upper_bound(arr.begin(),arr.end(),x) - arr.begin();
    int left = max(0,idx-k-1),right = min(size-1,idx+k-1);
    while(right-left+1>k){
        if(x-arr[left]>arr[right]-x)
            ++left;
        else
            --right;
    }
    return vector<int>(arr.begin()+left,arr.begin()+right+1);
}
// 1894. 找到需要补充粉笔的学生编号 https://leetcode.cn/problems/find-the-student-that-will-replace-the-chalk/
// 一个班级里有 n 个学生，编号为 0 到 n - 1 。每个学生会依次回答问题，编号为 0 的学生先回答，然后是编号为 1 的学生，以此类推，直到编号为 n - 1 的学生，然后老师会重复这个过程，重新从编号为 0 的学生开始回答问题。给你一个长度为 n 且下标从 0 开始的整数数组 chalk 和一个整数 k 。一开始粉笔盒里总共有 k 支粉笔。当编号为 i 的学生回答问题时，他会消耗 chalk[i] 支粉笔。如果剩余粉笔数量 严格小于 chalk[i] ，那么学生 i 需要 补充 粉笔。请你返回需要 补充 粉笔的学生 编号 。
int chalkReplacer(vector<int>& chalk, int k) {
    int size = chalk.size();
    if(chalk[0]>k)
        return 0;
    for(int i=1;i<size;++i){
        chalk[i] += chalk[i-1];
        if(chalk[i]>k)
            return i;
    }
    k %= chalk.back();
    return upper_bound(chalk.begin(),chalk.end(),k) - chalk.begin();
}
// 287. 寻找重复数 https://leetcode.cn/problems/find-the-duplicate-number/
// 给定一个包含 n + 1 个整数的数组 nums ，其数字都在 [1, n] 范围内（包括 1 和 n），可知至少存在一个重复的整数。假设 nums 只有 一个重复的整数 ，返回 这个重复的数 。你设计的解决方案必须 不修改 数组 nums 且只用常量级 O(1) 的额外空间。
int findDuplicate(vector<int>& nums) {
    int slow = 0, fast = 0;
    do {
        slow = nums[slow];
        fast = nums[nums[fast]];
    } while (slow != fast);
    slow = 0;
    while (slow != fast) {
        slow = nums[slow];
        fast = nums[fast];
    }
    return slow;
}
// 1488. 避免洪水泛滥 https://leetcode.cn/problems/avoid-flood-in-the-city/
// 你的国家有无数个湖泊，所有湖泊一开始都是空的。当第 n 个湖泊下雨前是空的，那么它就会装满水。如果第 n 个湖泊下雨前是 满的 ，这个湖泊会发生 洪水 。你的目标是避免任意一个湖泊发生洪水。给你一个整数数组 rains ，其中：rains[i] > 0 表示第 i 天时，第 rains[i] 个湖泊会下雨。rains[i] == 0 表示第 i 天没有湖泊会下雨，你可以选择 一个 湖泊并 抽干 这个湖泊的水。请返回一个数组 ans ，满足：ans.length == rains.length。如果 rains[i] > 0 ，那么ans[i] == -1 。如果 rains[i] == 0 ，ans[i] 是你第 i 天选择抽干的湖泊。如果有多种可行解，请返回它们中的 任意一个 。如果没办法阻止洪水，请返回一个 空的数组 。请注意，如果你选择抽干一个装满水的湖泊，它会变成一个空的湖泊。但如果你选择抽干一个空的湖泊，那么将无事发生。
vector<int> avoidFlood(vector<int>& rains) {
    int size = rains.size();
    unordered_map<int,int> last;
    vector<int> ans(size,1);
    set<int> water;
    for(int i = 0; i < size; i++) {
        if (rains[i] == 0){
            water.emplace(i);
            continue;
        }
        else if (last.count(rains[i])){
            auto iter= water.upper_bound(last[rains[i]]);
            if(iter==water.end())
                return vector<int>();
            ans[*iter] = rains[i];
            water.erase(iter);
            last[rains[i]] = i;
        }
        else
            last.emplace(rains[i],i);
        ans[i] = -1;
    }
    return ans;
}
// 6160. 和有限的最长子序列 https://leetcode.cn/problems/longest-subsequence-with-limited-sum/
// 给你一个长度为 n 的整数数组 nums ，和一个长度为 m 的整数数组 queries 。返回一个长度为 m 的数组 answer ，其中 answer[i] 是 nums 中 元素之和小于等于 queries[i] 的 子序列 的 最大 长度  。子序列 是由一个数组删除某些元素（也可以不删除）但不改变剩余元素顺序得到的一个数组。
vector<int> answerQueries(vector<int>& nums, vector<int>& queries) {
    sort(nums.begin(),nums.end());
    int size = nums.size();
    long long pre_sum[size+1];
    pre_sum[0] = 0;
    for(int i=0;i<size;++i)
        pre_sum[i+1] = pre_sum[i]+nums[i];
    vector<int> ans;
    ans.reserve(queries.size());
    for(auto &query:queries)
        ans.push_back(upper_bound(pre_sum,pre_sum+size+1,query)-pre_sum);
    return ans;
}
// 6161. 从字符串中移除星号 https://leetcode.cn/problems/removing-stars-from-a-string/
// 给你一个包含若干星号 * 的字符串 s 。在一步操作中，你可以：选中 s 中的一个星号。移除星号 左侧 最近的那个 非星号 字符，并移除该星号自身。返回移除 所有 星号之后的字符串。注意：生成的输入保证总是可以执行题面中描述的操作。可以证明结果字符串是唯一的。
string removeStars(string s) {
    int size = s.size();
    char ans[size+1];
    memset(ans,0,sizeof(ans));
    int i=0,j=0;
    for(int i=0;i<size;++i){
        if(s[i]=='*'){
            --j;
        }
        else
            ans[j++] = s[i];
    }
    ans[j] = '\0';
    return string(ans);
}
// 6162. 收集垃圾的最少总时间 https://leetcode.cn/problems/minimum-amount-of-time-to-collect-garbage/
// 给你一个下标从 0 开始的字符串数组 garbage ，其中 garbage[i] 表示第 i 个房子的垃圾集合。garbage[i] 只包含字符 'M' ，'P' 和 'G' ，但可能包含多个相同字符，每个字符分别表示一单位的金属、纸和玻璃。垃圾车收拾 一 单位的任何一种垃圾都需要花费 1 分钟。同时给你一个下标从 0 开始的整数数组 travel ，其中 travel[i] 是垃圾车从房子 i 行驶到房子 i + 1 需要的分钟数。城市里总共有三辆垃圾车，分别收拾三种垃圾。每辆垃圾车都从房子 0 出发，按顺序 到达每一栋房子。但它们 不是必须 到达所有的房子。任何时刻只有 一辆 垃圾车处在使用状态。当一辆垃圾车在行驶或者收拾垃圾的时候，另外两辆车 不能 做任何事情。请你返回收拾完所有垃圾需要花费的 最少 总分钟数。
int garbageCollection(vector<string>& garbage, vector<int>& travel) {
    int m_cnt = 0,p_cnt = 0,g_cnt = 0;
    int m_travel = 0,p_travel = 0,g_travel = 0;
    int m = 0,p = 0,g = 0; 
    int size = garbage.size();   
    for(auto ch:garbage[0]){
        switch (ch){
            case 'M':m_cnt++;
            case 'P':p_cnt++;
            case 'G':g_cnt++;
        }
    }
    for(int i=1;i<size;++i){
        m += travel[i-1];
        p += travel[i-1];
        g += travel[i-1];
        auto& str = garbage[i];
        for(auto ch:str){
            switch (ch){
                case 'M':m_cnt++; m_travel += m;m=0;
                case 'P':p_cnt++; p_travel += p;p=0;
                case 'G':g_cnt++; g_travel += g;g=0;
            }
        }
    }
    return m_cnt + p_cnt + g_cnt + m_travel+ p_travel+g_travel;
}
// 6163. 给定条件下构造矩阵 https://leetcode.cn/problems/build-a-matrix-with-conditions/
// 给你一个 正 整数 k ，同时给你：一个大小为 n 的二维整数数组 rowConditions ，其中 rowConditions[i] = [abovei, belowi] 和一个大小为 m 的二维整数数组 colConditions ，其中 colConditions[i] = [lefti, righti] 。两个数组里的整数都是 1 到 k 之间的数字。你需要构造一个 k x k 的矩阵，1 到 k 每个数字需要 恰好出现一次 。剩余的数字都是 0 。矩阵还需要满足以下条件：对于所有 0 到 n - 1 之间的下标 i ，数字 abovei 所在的 行 必须在数字 belowi 所在行的上面。对于所有 0 到 m - 1 之间的下标 i ，数字 lefti 所在的 列 必须在数字 righti 所在列的左边。返回满足上述要求的 任意 矩阵。如果不存在答案，返回一个空的矩阵。
vector<vector<int>> buildMatrix(int k, vector<vector<int>>& rowConditions, vector<vector<int>>& colConditions) {
    vector<int> edges[k+1];
    int top_row[k+1],top_col[k+1],*top,top_num = 0;
    char color[k+1]; bool valid = true;
    function<void(int)> dfs = [&](int pos){
        color[pos] = 1;
        for(auto& next:edges[pos]){
            if(valid&&color[next]==0)
                dfs(next);
            else if(color[next]==1)
                valid = false;
        }
        top[pos] = (top_num++);
        color[pos] = 2;
    };
    memset(color,0,sizeof(color));
    top = top_row;
    for(auto& rowCondition:rowConditions)
        edges[rowCondition[1]].emplace_back(rowCondition[0]);
    for(int i=1;i<=k;++i)
        if(valid&&color[i]==0)
            dfs(i);
    if(!valid)
        return vector<vector<int>>();
    memset(color,0,sizeof(color));
    top = top_col;
    top_num = 0;
    valid = true;
    for(int i=1;i<=k;++i)
        edges[i].clear();
    for(auto& colCondition:colConditions)
        edges[colCondition[1]].emplace_back(colCondition[0]);
    for(int i=1;i<=k;++i)
        if(valid&&color[i]==0)
            dfs(i);
    if(!valid)
        return vector<vector<int>>();
    vector<vector<int>> matrix(k,vector<int>(k,0));
    for(int i=1;i<=k;++i)
        matrix[top_row[i]][top_col[i]] = i;
    return matrix;
}

struct VecHash{
    size_t operator()(const vector<int>& nums){
        size_t ret = 0;
        for(auto& num:nums)
            ret ^= hash<int>()(num);
        return ret;
    }
};
struct VecEqual{
    bool operator()(const vector<int>& a,const vector<int>& b){
        int size = a.size();
        for(int i=0;i<size;++i){
            if(a[i]!=b[i])
                return false;
        }
        return true;
    }
};
// 九坤-04. 筹码游戏 https://leetcode.cn/contest/ubiquant2022/problems/I3Gm2h/
// 九坤很喜欢玩德州扑克，但是有一个神奇的周五，大家都去加班了，于是九坤只能研究起了桌上的筹码。他把所有的筹码都放入了一个纸箱中，并按以下规则向外抽取筹码：每次抽取仅取出 1 个筹码。如果对本次取出的筹码不满意，会将该筹码放回并重新抽取，直到确定想要这个筹码。对于取出的筹码，他会将相同面值的筹码放在一堆。例如：抽取了 6 个筹码，3 个 10，2 个 5，1个 1，那么他就会把这些筹码放成三堆，数量分别是3、2、1。纸箱中共有 kind 种面值的筹码。现给定九坤取出筹码的最终目标为 nums， nums[i] 表示第 i 堆筹码的数量。设每种面值的筹码都有无限多个，且九坤总是遵循最优策略，使得他达成目标的操作次数最小化。请返回九坤达成目标的情况下，需要取出筹码次数的期望值。注意：最终取出的筹码中，对于任意两堆筹码的面值都是不同的。不需要考虑筹码堆的顺序（例如，[3,1,1]、[1,1,3] 这两个筹码堆是相同的）
double chipGame(vector<int>& nums, int kind) {
    unordered_map<vector<int>,double,VecHash,VecEqual> mp;
    nums.resize(kind);
    auto state = nums;
    function<double()> dfs = [&](){
        auto i = mp.find(state);
        if (i!=mp.end())
            return i->second;
        int cnt = 0;
        int cnt = 0;
        double ret = 0;
        for(int i = 0, j; i < kind; i = j) {
            for(j = i; j < kind && state[i] == state[j]; ++j);
            if(state[j - 1] == nums[j - 1])
                continue;
            ++state[j - 1];
            double adt = dfs();
            --state[j - 1];
            cnt += j - i;
            ret += (j - i) * adt;
        }
        ret = (ret + kind) / cnt;
        return mp[state] = ret;
    };
    return dfs();
}

int temperatureTrend(vector<int>& temperatureA, vector<int>& temperatureB) {
    int size = temperatureA.size();
    auto get = [](int a,int b){
        if (a>b){
            return 1;
        }else if(a==b){
            return 0;
        }
        return -1;
    };
    int cnt = 0,ans = 0;
    for(int i=0;i<size;++i){
        if(get(temperatureA[i],temperatureA[i-1])==get(temperatureB[i],temperatureB[i-1])){
            ++cnt; ans = max(cnt,ans);
        }else
            cnt = 0;
    }
}
int transportationHub(vector<vector<int>>& path) {
    int in_degree[1001];
    int out_degree[1001];
    int cnt = 0;
    memset(in_degree,0,sizeof(in_degree));
    memset(out_degree,0,sizeof(out_degree));
    for(auto& p:path){
        if(!out_degree[p[0]]&&!in_degree[p[0]])
            ++cnt;
        if(!out_degree[p[1]]&&!in_degree[p[1]])
            ++cnt;
        ++out_degree[p[0]];
        ++in_degree[p[1]];
    }
    for(int i=0;i<=1000;++i){
        if(in_degree[i]==cnt-1&& out_degree[i]==0){
            return i;
        }
    }
    return -1;
}
// 6189. 按位与最大的最长子数组 https://leetcode.cn/problems/longest-subarray-with-maximum-bitwise-and/
// 给你一个长度为 n 的整数数组 nums 。考虑 nums 中进行 按位与（bitwise AND）运算得到的值 最大 的 非空 子数组。换句话说，令 k 是 nums 任意 子数组执行按位与运算所能得到的最大值。那么，只需要考虑那些执行一次按位与运算后等于 k 的子数组。返回满足要求的 最长 子数组的长度。数组的按位与就是对数组中的所有数字进行按位与运算。子数组 是数组中的一个连续元素序列。
int longestSubarray(vector<int>& nums) {
    vector<int> max_pos;
    int size = nums.size();
    int m = 0;
    for(int i=0;i<size;i++){
        if(nums[i]>m){
            max_pos.clear();
            max_pos.push_back(i);
        }else if(nums[i]==m){
            max_pos.push_back(i);
        }
    }       
    int n = max_pos.size(),ans = 1,cnt = 1;
    for(int i=1;i<n;i++){
        if(max_pos[i]!=max_pos[i-1])
            cnt = 1;
        else
            cnt++;
        ans = max(ans,cnt);
    }
    return ans;
}
// 6212. 删除字符使频率相同 模拟 https://leetcode.cn/problems/remove-letter-to-equalize-frequency/
// 给你一个下标从 0 开始的字符串 word ，字符串只包含小写英文字母。你需要选择 一个 下标并 删除 下标处的字符，使得 word 中剩余每个字母出现 频率 相同。如果删除一个字母后，word 中剩余所有字母的出现频率都相同，那么返回 true ，否则返回 false 。注意：字母 x 的 频率 是这个字母在字符串中出现的次数。你 必须 恰好删除一个字母，不能一个字母都不删除。
bool equalFrequency(string word) {
    int cnt[26] = {0};
    for(auto ch:word)
        cnt[ch-'a']++;
    auto check = [&](){
        int a = -1;
        for(int i=0;i<26;++i){
            if(cnt[i]==0) continue;
            if(a==-1) a = cnt[i];
            if(a!=cnt[i]) return false;
        }
        return true;
    };
    for(auto ch:word){
        cnt[ch-'a']--;
        if(check())
            return true;
        cnt[ch-'a']++;
    }
    return false;
}
// 6197. 最长上传前缀 模拟 https://leetcode.cn/problems/longest-uploaded-prefix/
/* 
 * 给你一个 n 个视频的上传序列，每个视频编号为 1 到 n 之间的 不同 数字，你需要依次将这些视频上传到服务器。请你实现一个数据结构，在上传的过程计算 最长上传前缀 。如果 闭区间 1 到 i 之间的视频全部都已经被上传到服务器，那么我们称 i 是上传前缀。最长上传前缀指的是符合定义的 i 中的 最大值 。请你实现 LUPrefix 类：
 * LUPrefix(int n) 初始化一个 n 个视频的流对象。
 * void upload(int video) 上传 video 到服务器。
 * int longest() 返回上述定义的 最长上传前缀 的长度。
*/
class LUPrefix {
    bitset<int(1e5+5)> bs;
    int p;
public:
    LUPrefix(int n):p(1) {}
    
    void upload(int video) {
        bs.set(video);
        while(bs.test(p))
            p++;
    }
    
    int longest() {
        return p;
    }
};
// 6213. 所有数对的异或和 找规律 https://leetcode.cn/problems/bitwise-xor-of-all-pairings/
// 给你两个下标从 0 开始的数组 nums1 和 nums2 ，两个数组都只包含非负整数。请你求出另外一个数组 nums3 ，包含 nums1 和 nums2 中 所有数对 的异或和（nums1 中每个整数都跟 nums2 中每个整数 恰好 匹配一次）。请你返回 nums3 中所有整数的 异或和 。
int xorAllNums(vector<int>& nums1, vector<int>& nums2) {
    int m = nums1.size(), n = nums2.size();
    m = m & 1; n = n & 1;
    int ans = 0;
    for(auto& num:nums1){
        if(n)
            ans ^= num;
    }
    for(auto& num:nums2){
        if(m)
            ans ^= num;
    }
    return ans;
}
class BIT{
    int* arr;
	unsigned size;
	static int lowbit(int x){
        return x&(-x);
    }
public:
    BIT(unsigned _size):arr(new int[_size+1]),size(_size){
		memset(arr,0,(_size+1)*sizeof(int));
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
// 6198. 满足不等式的数对数目 树状数组 https://leetcode.cn/problems/number-of-pairs-satisfying-inequality/
// 给你两个下标从 0 开始的整数数组 nums1 和 nums2 ，两个数组的大小都为 n ，同时给你一个整数 diff ，统计满足以下条件的 数对 (i, j) ：0 <= i < j <= n - 1 且nums1[i] - nums1[j] <= nums2[i] - nums2[j] + diff.请你返回满足条件的 数对数目 。
long long numberOfPairs(vector<int>& nums1, vector<int>& nums2, int diff) {
    int size = nums1.size();
    long long cnt = 0;
    BIT b(int(1e5));
    const int m = int(3*1e4+5);
    for(int i=0;i<size;i++){
        int t = nums1[i] - nums2[i] + m;
        cnt += b.getSum(t+diff);
        b.update(t,1);
    }      
    return cnt;
}
// 2465. 不同的平均值数目 https://leetcode.cn/problems/number-of-distinct-averages/
// 给你一个下标从 0 开始长度为 偶数 的整数数组 nums 。只要 nums 不是 空数组，你就重复执行以下步骤：找到 nums 中的最小值，并删除它。找到 nums 中的最大值，并删除它。计算删除两数的平均值。两数 a 和 b 的 平均值 为 (a + b) / 2 。比方说，2 和 3 的平均值是 (2 + 3) / 2 = 2.5 。返回上述过程能得到的 不同 平均值的数目。注意 ，如果最小值或者最大值有重复元素，可以删除任意一个。
int distinctAverages(vector<int>& nums) {
    sort(nums.begin(),nums.end());
    int size = nums.size();
    unordered_set<int> st;
    for(int i=0;i<size/2;i++){
        st.emplace(nums[i]+nums[size-1-i]);
    }       
    return st.size();
}
// 2466. 统计构造好字符串的方案数 https://leetcode.cn/problems/count-ways-to-build-good-strings/
// 给你整数 zero ，one ，low 和 high ，我们从空字符串开始构造一个字符串，每一步执行下面操作中的一种：将 '0' 在字符串末尾添加 zero  次。将 '1' 在字符串末尾添加 one 次。以上操作可以执行任意次。如果通过以上过程得到一个 长度 在 low 和 high 之间（包含上下边界）的字符串，那么这个字符串我们称为 好 字符串。请你返回满足以上要求的 不同 好字符串数目。由于答案可能很大，请将结果对 109 + 7 取余 后返回。
int countGoodStrings(int low, int high, int zero, int one) {
    int mod = 1e9+ 7;
    int dp[high+1];
    memset(dp,0,(high+1)*sizeof(int));
    dp[0] = 1;
    for(int i=0;i<=high;i++){
        if(i>=zero) dp[i] = (dp[i] +  dp[i-zero])%mod;
        if(i>=one)  dp[i] = (dp[i] +  dp[i-one])%mod; 
    } 
    int ans = 0;
    for(int i=low;i<=high;i++){
        ans = (ans+dp[i])%mod;
    }
    return ans;
}
struct Node{
    int node;
    int depth;
    Node(){}
    Node(int node,int depth):node(node),depth(depth){}
};  
// 2467. 树上最大得分和路径 https://leetcode.cn/problems/most-profitable-path-in-a-tree/
// 一个 n 个节点的无向树，节点编号为 0 到 n - 1 ，树的根结点是 0 号节点。给你一个长度为 n - 1 的二维整数数组 edges ，其中 edges[i] = [ai, bi] ，表示节点 ai 和 bi 在树中有一条边。在每一个节点 i 处有一扇门。同时给你一个都是偶数的数组 amount ，其中 amount[i] 表示：如果 amount[i] 的值是负数，那么它表示打开节点 i 处门扣除的分数。如果 amount[i] 的值是正数，那么它表示打开节点 i 处门加上的分数。游戏按照如下规则进行：一开始，Alice 在节点 0 处，Bob 在节点 bob 处。每一秒钟，Alice 和 Bob 分别 移动到相邻的节点。Alice 朝着某个 叶子结点 移动，Bob 朝着节点 0 移动。对于他们之间路径上的 每一个 节点，Alice 和 Bob 要么打开门并扣分，要么打开门并加分。注意：如果门 已经打开 （被另一个人打开），不会有额外加分也不会扣分。如果 Alice 和 Bob 同时 到达一个节点，他们会共享这个节点的加分或者扣分。换言之，如果打开这扇门扣 c 分，那么 Alice 和 Bob 分别扣 c / 2 分。如果这扇门的加分为 c ，那么他们分别加 c / 2 分。如果 Alice 到达了一个叶子结点，她会停止移动。类似的，如果 Bob 到达了节点 0 ，他也会停止移动。注意这些事件互相 独立 ，不会影响另一方移动。请你返回 Alice 朝最优叶子结点移动的 最大 净得分。
int mostProfitablePath(vector<vector<int>>& edges, int bob, vector<int>& amount) {
    int size = edges.size();
    vector<int> graph[size];
    for(auto &edge:edges){
        graph[edge[0]].emplace_back(edge[1]);
        graph[edge[1]].emplace_back(edge[0]);
    }  
    Node queue[size+1];
    int prev[size+1];
    prev[bob] = -1;
    int begin = 0,end = 0;
    int depth = 0;
    queue[end++] = Node(bob,0);
    char color[size+1];
    memset(color,0,size+1);
    while(queue[begin].node!=0){
        for(auto next:graph[queue[begin].node]){
            if(color[next]!=0)
                continue;
            color[next] = 1;
            prev[next] = queue[begin].node;
            queue[end++] = Node(queue[begin].node,queue[begin].depth+1);
        }
        begin++;
    }
    unsigned int flag[size+1];
    memset(flag,0xFF,(size+1)*sizeof(int));
    int cur = prev[0],depth = queue[begin].depth;
    while(cur!=-1){
        flag[cur] = depth;
        cur = prev[cur];
        depth--;
    }
    int opened[size+1];
    memset(opened,0,(size+1)*sizeof(int));
    char color[size+1];
    memset(color,0,size+1);
    int val = 0,ans = 0;
    depth = -1;
    function<void(int,int)> dfs;
    dfs = [&](int node,int prev){
        depth++;
        int inc = 0;
        if(depth<flag[node]) inc = amount[node];
        if(depth==flag[node]) inc = amount[node]/2;
        val += inc;
        if(graph[node].size()==1) ans = max(val,ans);
        for(auto next:graph[node]){
            if(next==prev) continue;
            dfs(next,node);
        }
        val -= inc;
        depth--;
    };
    dfs(0,-1);
    return ans;
}
// 38. 外观数列 https://leetcode.cn/problems/count-and-say/
// 给定一个正整数 n ，输出外观数列的第 n 项。「外观数列」是一个整数序列，从数字 1 开始，序列中的每一项都是对前一项的描述。你可以将其视作是由递归公式定义的数字字符串序列：countAndSay(1) = "1"，countAndSay(n) 是对 countAndSay(n-1) 的描述，然后转换成另一个数字字符串。要 描述 一个数字字符串，首先要将字符串分割为 最小 数量的组，每个组都由连续的最多 相同字符 组成。然后对于每个组，先描述字符的数量，然后描述字符，形成一个描述组。要将描述转换为数字字符串，先将每组中的字符数量用数字替换，再将所有描述组连接起来。
string countAndSay(int n) {
	if(n==1) return "1";
	string prev = countAndSay(n-1);
	string ans = "";
	char last=prev[0]; int cnt = 0;
	for(auto ch: prev){
		if(ch!=last){
			ans.append( to_string(cnt)).push_back(last);
			last = ch; cnt = 1;
		}
		else
			cnt++;
	}
	if(cnt!=0)
		ans.append( to_string(cnt)).push_back(last);
	return ans;
}
// 39. 组合总和 https://leetcode.cn/problems/combination-sum/
// 给你一个 无重复元素 的整数数组 candidates 和一个目标整数 target ，找出 candidates 中可以使数字和为目标数 target 的 所有 不同组合 ，并以列表形式返回。你可以按 任意顺序 返回这些组合。candidates 中的 同一个 数字可以 无限制重复被选取 。如果至少一个数字的被选数量不同，则两种组合是不同的。 对于给定的输入，保证和为 target 的不同组合数少于 150 个。
vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
	vector<vector<int>> ans; ans.reserve(150);
	vector<int> tem; tem.reserve(20);  
	function<void(int,int)> dfs;
	dfs = [&](int val,int idx){
		if(val==0) candidates.emplace_back(tem);
		if(idx==candidates.size()||val<=0) return ;
		dfs(val,idx+1);
		if(val-candidates[idx]>=0){
			tem.emplace_back(candidates[idx]);
			dfs(val-candidates[idx],idx);
			tem.pop_back();
		}
	};
	dfs(target,0);
	return ans;
}
// 931. 下降路径最小和 https://leetcode.cn/problems/minimum-falling-path-sum/
// 给你一个 n x n 的 方形 整数数组 matrix ，请你找出并返回通过 matrix 的下降路径 的 最小和 。下降路径 可以从第一行中的任何元素开始，并从每一行中选择一个元素。在下一行选择的元素和当前行所选元素最多相隔一列（即位于正下方或者沿对角线向左或者向右的第一个元素）。具体来说，位置 (row, col) 的下一个元素应当是 (row + 1, col - 1)、(row + 1, col) 或者 (row + 1, col + 1) 。
int minFallingPathSum(vector<vector<int>>& matrix) {
	int n = matrix.size();
	vector<vector<int>> dp = vector<vector<int>>(2,vector<int>(n));
	int curr,prev;
	for(int i=0;i<n;i++){
		curr = (i&1), prev = curr ^ 1;
		for(int j=0;j<n;j++){
			int val = dp[prev][j];
			if(j>0&&dp[prev][j-1]<val)
				val = dp[prev][j-1];
			if(j+1<n&&dp[prev][j+1]<val)
				val = dp[prev][j+1];
			dp[curr][j] = val + matrix[i][j];
		}
	}
	int res = dp[curr][0];
	for(int i=0;i<n;i++)
		if(dp[curr][i] < res)
			res = dp[curr][i];
	return res;
}
// 6929. 数组的最大美丽值 https://leetcode.cn/problems/maximum-beauty-of-an-array-after-applying-operation/
// 给你一个下标从 0 开始的整数数组 nums 和一个 非负 整数 k 。在一步操作中，你可以执行下述指令：在范围 [0, nums.length - 1] 中选择一个 此前没有选过 的下标 i 。将 nums[i] 替换为范围 [nums[i] - k, nums[i] + k] 内的任一整数。数组的 美丽值 定义为数组中由相等元素组成的最长子序列的长度。对数组 nums 执行上述操作任意次后，返回数组可能取得的 最大 美丽值。注意：你 只 能对每个下标执行 一次 此操作。数组的 子序列 定义是：经由原数组删除一些元素（也可能不删除）得到的一个新数组，且在此过程中剩余元素的顺序不发生改变。
int maximumBeauty(vector<int>& nums, int k) {
	const int offset = 100000;
	int diff[300002] = {0};
	for(auto num: nums){
		diff[num-k]++;
		diff[num+k+1]++;
	} 
	int tmp = 0, ans = 0;
	for(int i=0;i<300002;i++){
		tmp += diff[i];
		if(tmp>ans) ans = tmp;
	}
	return ans;
}
// 6927. 合法分割的最小下标 https://leetcode.cn/problems/minimum-index-of-a-valid-split/
// 如果元素 x 在长度为 m 的整数数组 arr 中满足 freq(x) * 2 > m ，那么我们称 x 是 支配元素 。其中 freq(x) 是 x 在数组 arr 中出现的次数。注意，根据这个定义，数组 arr 最多 只会有 一个 支配元素. 给你一个下标从 0 开始长度为 n 的整数数组 nums ，数据保证它含有一个支配元素。你需要在下标 i 处将 nums 分割成两个数组 nums[0, ..., i] 和 nums[i + 1, ..., n - 1] ，如果一个分割满足以下条件，我们称它是 合法 的：0 <= i < n - 1. nums[0, ..., i] 和 nums[i + 1, ..., n - 1] 的支配元素相同。这里， nums[i, ..., j] 表示 nums 的一个子数组，它开始于下标 i ，结束于下标 j ，两个端点都包含在子数组内。特别地，如果 j < i ，那么 nums[i, ..., j] 表示一个空数组。请你返回一个 合法分割 的 最小 下标。如果合法分割不存在，返回 -1 。
int minimumIndex(vector<int>& nums) {
	int size = nums.size();
	int prime[size],prime_rev[size];
	unordered_map<int,int> cnt;
	int max_num = nums[0];
	for(int i=0;i<size;i++){
		cnt[nums[i]]++;
		if(cnt[nums[i]] >cnt[max_num]) max_num = nums[i];
		prime[i] = cnt[max_num]*2 >(i+1) ? max_num :-1;
	}
	cnt.clear();
	for(int i=size-1;i>=0;i--){
		cnt[nums[i]]++;
		if(cnt[nums[i]] >cnt[max_num]) max_num = nums[i];
		prime_rev[i] = cnt[max_num]*2>(size-i)? max_num :-1;
	}
	for(int i=0;i<size-1;i++)
		if(prime[i]!=-1&&prime[i]==prime_rev[i+1])
			return i;
	return -1;
}
// 834. 树中距离之和 https://leetcode.cn/problems/sum-of-distances-in-tree/
// 给定一个无向、连通的树。树中有 n 个标记为 0...n-1 的节点以及 n-1 条边 。给定整数 n 和数组 edges ， edges[i] = [ai, bi]表示树中的节点 ai 和 bi 之间有一条边。返回长度为 n 的数组 answer ，其中 answer[i] 是树中第 i 个节点与所有其他节点之间的距离之和。
vector<int> sumOfDistancesInTree(int n, vector<vector<int>>& edges) {
	vector<int> ans(n,0);
	vector<int> sz(n, 1);
	vector<vector<int>> graph(n, vector<int>());
	for (auto& edge: edges) {
		int u = edge[0], v = edge[1];
		graph[u].emplace_back(v);
		graph[v].emplace_back(u);
	}
	function<void(int,int)>  dfs;
	dfs  = [&](int node,int pa){
		for(auto next :graph[node]){
			if(next==pa) continue;
			dfs(next,node);
			sz[node] += sz[next];
			ans[node] += ans[next] + sz[next];
		}
	};
	function<void(int,int)> reroot;
	reroot = [&](int node,int pa){
		for(auto next :graph[node]){
			if(next==pa) continue;
			ans[next] = ans[node] + n -2 * sz[next];
			reroot(next,node);
		}
	};
	dfs(0, -1);
	reroot(0, -1);
	return ans;
}   
// 1499. 满足不等式的最大值 https://leetcode.cn/problems/max-value-of-equation/
// 给你一个数组 points 和一个整数 k 。数组中每个元素都表示二维平面上的点的坐标，并按照横坐标 x 的值从小到大排序。也就是说 points[i] = [xi, yi] ，并且在 1 <= i < j <= points.length 的前提下， xi < xj 总成立。请你找出 yi + yj + |xi - xj| 的 最大值，其中 |xi - xj| <= k 且 1 <= i < j <= points.length。题目测试数据保证至少存在一对能够满足 |xi - xj| <= k 的点。
int findMaxValueOfEquation(vector<vector<int>>& points, int k) {
	int n = points.size();
	int begin = 0, end = 0;
	pair<int,int> q[n];
	int ans = INT32_MIN;
	for(auto &point: points){
		while(begin<end&&point[0] - q[begin].first>k) begin++;
		if(begin<end) ans = max(ans,point[0] + q[begin].second + point[1]);
		while(begin<end&&point[1]-point[0]>q[end-1].second) end--;
		q[end++] = pair(point[0],point[1]-point[0]);
	}
	return ans;
}
// 860. 柠檬水找零 https://leetcode.cn/problems/lemonade-change/
// 在柠檬水摊上，每一杯柠檬水的售价为 5 美元。顾客排队购买你的产品，（按账单 bills 支付的顺序）一次购买一杯。每位顾客只买一杯柠檬水，然后向你付 5 美元、10 美元或 20 美元。你必须给每个顾客正确找零，也就是说净交易是每位顾客向你支付 5 美元。注意，一开始你手头没有任何零钱。给你一个整数数组 bills ，其中 bills[i] 是第 i 位顾客付的账。如果你能给每位顾客正确找零，返回 true ，否则返回 false 。
bool lemonadeChange(vector<int>& bills) {
	int cnt_five = 0,cnt_ten = 0;
	for(auto bill: bills){
		if(bill==5) cnt_five++;
		else if(bill==10){
			if(cnt_five==0) return false;
			cnt_five--;
			cnt_ten++;
		}else{
			if(cnt_ten!=0){
				cnt_ten--;
				if(cnt_five==0) return false;
				cnt_five--;
			}else if(cnt_five>=3){
				cnt_five -= 3;
			}else{
				return false;
			}
		}
	}
	return true;
}
// 31. 下一个排列 https://leetcode.cn/problems/next-permutation/
// 整数数组的一个 排列  就是将其所有成员以序列或线性顺序排列。例如，arr = [1,2,3] ，以下这些都可以视作 arr 的排列：[1,2,3]、[1,3,2]、[3,1,2]、[2,3,1] 。整数数组的 下一个排列 是指其整数的下一个字典序更大的排列。更正式地，如果数组的所有排列根据其字典顺序从小到大排列在一个容器中，那么数组的 下一个排列 就是在这个有序容器中排在它后面的那个排列。如果不存在下一个更大的排列，那么这个数组必须重排为字典序最小的排列（即，其元素按升序排列）。例如，arr = [1,2,3] 的下一个排列是 [1,3,2] 。类似地，arr = [2,3,1] 的下一个排列是 [3,1,2] 。而 arr = [3,2,1] 的下一个排列是 [1,2,3] ，因为 [3,2,1] 不存在一个字典序更大的排列。给你一个整数数组 nums ，找出 nums 的下一个排列。必须 原地 修改，只允许使用额外常数空间。
void nextPermutation(vector<int>& nums) {
	int size = nums.size(),k=-1,l;
	auto reverse = [](vector<int>& nums,int i,int j){
		while(i<j){
			int tem = nums[i];
			nums[i++] = nums[j];
			nums[j--] = tem;
		}
	};
	for(int i=size-2;i>=0;i--)
		if(nums[i]<nums[i+1]){
			k = i;
			break;
		}
	if(k==-1) return reverse(nums,0,size-1);
	for(int i=size-1;i>=0;i--)
		if(nums[k]<nums[i]){
			l = i;
			break;
		}
	swap(nums[k],nums[l]);
	return reverse(nums,k+1,size-1);
}
// 2787. 将一个数字表示成幂的和的方案数 https://leetcode.cn/problems/ways-to-express-an-integer-as-sum-of-powers/
// 给你两个 正 整数 n 和 x 。请你返回将 n 表示成一些 互不相同 正整数的 x 次幂之和的方案数。换句话说，你需要返回互不相同整数 [n1, n2, ..., nk] 的集合数目，满足 n = n1x + n2x + ... + nkx 。由于答案可能非常大，请你将它对 109 + 7 取余后返回。比方说，n = 160 且 x = 3 ，一个表示 n 的方法是 n = 23 + 33 + 53 。
int numberOfWays(int n, int x) {
	function<int(int,int)> solve; 
	unordered_map<int,int> mp;
	solve = [&](int sum,int num){
		if(num==0) return (sum==0)? 1 : 0;
		int tem = (sum << 16) + num;
		if(mp.count(tem)) return mp[tem];
		int ans = solve(sum,num-1);
		if(sum-pow(num,x)>0) 
			ans += solve(sum-pow(num,x),num-1);
		mp[tem] = ans;
		return ans;
	};
	return solve(n,pow(n,1.0/x) + 2);
}
// 2788. 按分隔符拆分字符串 https://leetcode.cn/problems/split-strings-by-separator/description/
// 给你一个字符串数组 words 和一个字符 separator ，请你按 separator 拆分 words 中的每个字符串。返回一个由拆分后的新字符串组成的字符串数组，不包括空字符串 。注意separator 用于决定拆分发生的位置，但它不包含在结果字符串中。拆分可能形成两个以上的字符串。结果字符串必须保持初始相同的先后顺序。
vector<string> splitWordsBySeparator(vector<string>& words, char separator) {
	vector<string> ans; string tem;
	for(auto &word: words){
		stringstream ss(word);
		while(getline(ss,tem,separator)){
			if(!tem.empty()) 
				ans.emplace_back(tem);
		}
	}
	return ans;
}
// 2789. 合并后数组中的最大元素 https://leetcode.cn/problems/largest-element-in-an-array-after-merge-operations/description/
// 给你一个下标从 0 开始、由正整数组成的数组 nums 。你可以在数组上执行下述操作 任意 次：选中一个同时满足 0 <= i < nums.length - 1 和 nums[i] <= nums[i + 1] 的整数 i 。将元素 nums[i + 1] 替换为 nums[i] + nums[i + 1] ，并从数组中删除元素 nums[i] 。返回你可以从最终数组中获得的 最大 元素的值。
long long maxArrayValue(vector<int>& nums) {
	int size = nums.size();
    long long dp[size];
	dp[size-1] = nums[size-1];
	long long ans = nums[size-1];
	for(int i=size-2;i>=0;i--){
		dp[i] = nums[i];
		if(nums[i] <= nums[i+1] || nums[i] <= dp[i+1]) 
			dp[i] += max((long long)nums[i+1],dp[i+1]);
		ans = max(ans,dp[i]);
	}
	return ans;
}
// 2785. 将字符串中的元音字母排序 https://leetcode.cn/problems/sort-vowels-in-a-string/
// 给你一个下标从 0 开始的字符串 s ，将 s 中的元素重新 排列 得到新的字符串 t ，它满足：所有辅音字母都在原来的位置上。更正式的，如果满足 0 <= i < s.length 的下标 i 处的 s[i] 是个辅音字母，那么 t[i] = s[i] 。元音字母都必须以他们的 ASCII 值按 非递减 顺序排列。更正式的，对于满足 0 <= i < j < s.length 的下标 i 和 j  ，如果 s[i] 和 s[j] 都是元音字母，那么 t[i] 的 ASCII 值不能大于 t[j] 的 ASCII 值。请你返回结果字母串。元音字母为 'a' ，'e' ，'i' ，'o' 和 'u' ，它们可能是小写字母也可能是大写字母，辅音字母是除了这 5 个字母以外的所有字母。
string sortVowels(string s) {
	string vowels = "AEIOUaeiou";
	int cnt[10] = {0};
	int size = s.size();
	vector<int> idxs;
	idxs.reserve(size);
	for(int i =0;i<size;i++){
		auto pos = vowels.find(s[i]);
		if(pos!=vowels.npos){
			idxs.emplace_back(i);
			cnt[pos]++;
		}
	}
	int j = 0;
	for(auto &idx:idxs){
		while(cnt[j]==0) j++;
		s[idx] = vowels[j];
	}  
	return s;
}
// 2786. 访问数组中的位置使分数最大 https://leetcode.cn/problems/visit-array-positions-to-maximize-score/description/
// 给你一个下标从 0 开始的整数数组 nums 和一个正整数 x 。你 一开始 在数组的位置 0 处，你可以按照下述规则访问数组中的其他位置：如果你当前在位置 i ，那么你可以移动到满足 i < j 的 任意 位置 j 。对于你访问的位置 i ，你可以获得分数 nums[i] 。如果你从位置 i 移动到位置 j 且 nums[i] 和 nums[j] 的 奇偶性 不同，那么你将失去分数 x 。请你返回你能得到的 最大 得分之和。注意 ，你一开始的分数为 nums[0] 。
long long maxScore(vector<int>& nums, int x) {
	int size = nums.size();
	long long dp[size]; dp[0] = nums[0];
	long long ans = nums[0];
	long long prev[2] = {0};
	bool flag = nums[0] & 1;
	prev[flag] = nums[0]; 
	prev[!flag] = nums[0] -x;
	for(int i=1;i<size;i++){
		bool flag = nums[i] & 1;
		dp[i] = max(prev[flag],prev[!flag]-x) + nums[i];
		prev[flag] = max(prev[flag],dp[i]);
		ans = max(ans,dp[i]);
	}
	return ans;
}
// 2784. 检查数组是否是好的 https://leetcode.cn/problems/check-if-array-is-good/
// 给你一个整数数组 nums ，如果它是数组 base[n] 的一个排列，我们称它是个 好 数组。base[n] = [1, 2, ..., n - 1, n, n] （换句话说，它是一个长度为 n + 1 且包含 1 到 n - 1 恰好各一次，包含 n  两次的一个数组）。比方说，base[1] = [1, 1] ，base[3] = [1, 2, 3, 3] 。如果数组是一个好数组，请你返回 true ，否则返回 false 。注意：数组的排列是这些数字按任意顺序排布后重新得到的数组。
bool isGood(vector<int>& nums) {
	int size = nums.size();
	int cnt[size];
	for(int i=0;i<size;i++)
		cnt[i] = 0;
	for(int i=0;i<size;i++){
		if(nums[i]<size) 
			cnt[nums[i]] += 1;
		else 
			return false;
	}
	for(int i=1;i<size-1;i++)
		if(cnt[i]!=1)
			return false;
	return cnt[size-1]==2;
}