/*
 * @Author: zanilia
 * @Date: 2021-12-04 14:03:29
 * @LastEditTime: 2022-02-21 22:22:14
 * @Descripttion: 
 */
#include <bits/stdc++.h>
using namespace std;
typedef unordered_map<string, priority_queue<string, vector<string>, std::greater<string>>> Map;
// 332. 重新安排行程 https://leetcode-cn.com/problems/reconstruct-itinerary/
// 给你一份航线列表 tickets ，其中 tickets[i] = [fromi, toi] 表示飞机出发和降落的机场地点。请你对该行程进行重新规划排序。所有这些机票都属于一个从 JFK（肯尼迪国际机场）出发的先生，所以该行程必须从 JFK 开始。如果存在多种有效的行程，请你按字典序返回最小的行程组合。例如，行程 ["JFK", "LGA"] 与 ["JFK", "LGB"] 相比就更小，排序更靠前。假定所有机票至少存在一种合理的行程。且所有的机票必须都用一次 且 只能用一次。 
vector<string> findItinerary(vector<vector<string>>& tickets) {
    Map edge;
    vector<string> stk;
    for(auto &ticket:tickets)
        edge[ticket[0]].emplace(ticket[1]);
    dfs(edge,stk,"JFK");
    reverse(stk.begin(),stk.end());
    return stk;
}
void dfs(Map& edge,vector<string>& stk,const string& cur){
    while(edge.count(cur)&&!edge[cur].empty()){
        string tem = edge[cur].top();
        edge[cur].pop();
        dfs(edge,stk,tem);
    }
    stk.emplace_back(cur);
}
// 753. 破解保险箱 https://leetcode-cn.com/problems/cracking-the-safe/
// 有一个需要密码才能打开的保险箱。密码是 n 位数, 密码的每一位是 k 位序列 0, 1, ..., k-1 中的一个 。你可以随意输入密码，保险箱
// 会自动记住最后 n 位输入，如果匹配，则能够打开保险箱。请返回一个能打开保险箱的最短字符串.
string crackSafe(int n, int k) {
    int flag = pow(10, n - 1);
    unordered_set<int> visited;
    std::string ans;
    dfs(0,k,flag,visited,ans);
    ans += string(n-1,'0');
    return ans;
}
void dfs(int node,int k,int flag,unordered_set<int> &visited,string& ans) {
    for (int x = 0; x < k; ++x) {
        int nei = node * 10 + x;
        if (!visited.count(nei)) {
            visited.insert(nei);
            dfs(nei%flag,k,flag,visited,ans);
            ans += (x + '0');
        }
    }
}
// 1020 飞地数量https://leetcode-cn.com/problems/number-of-enclaves/
// 给你一个大小为mxn的二进制矩阵grid,其中0表示一个海洋单元格、1表示一个陆地单元格。一次移动是指从一个陆地单元格走到另一个相邻(上、下、左、右)的陆地单元格或跨过grid的边界。返回网格中 无法 在任意次数的移动中离开网格边界的陆地单元格的数量。
class Solution {
    vector<vector<int>>* g;
    vector<vector<bool>>* visited;
    int m;
    int n;
    void dfs(int i,int j){
        if(i<0||j<0||i==m||j==n||(*g)[i][j]==0||(*visited)[i][j])
            return ;
        (*visited)[i][j] = true;
        dfs(i-1,j);
        dfs(i,j-1);
        dfs(i+1,j);
        dfs(i,j+1);
    }
public:
    int numEnclaves(vector<vector<int>>& grid) {
        m = grid.size();
        n = grid[0].size();
        g = &grid;
        int ans = 0;
        vector<vector<bool>> tem(m,vector<bool>(n,0));
        visited = &tem;
        for(int i=0;i<m;++i){
            dfs(i,0);
            dfs(i,n-1);
        }
        for(int j=1;j<n-1;++j){
            dfs(0,j);
            dfs(m-1,j);
        }
        for(int i=1;i<m-1;++i)
            for(int j=1;j<n-1;++j)
                if(!(*visited)[i][j]&&grid[i][j])
                    ++ans;
        return ans;
    }
};
// 743 网络延迟时间https://leetcode-cn.com/problems/network-delay-time/
// 有 n 个网络节点，标记为 1 到 n。给你一个列表 times，表示信号经过 有向 边的传递时间。 times[i] = (ui, vi, wi)，其中 ui 是源节点，vi 是目标节点，wi 是一个信号从源节点传递到目标节点的时间。现在，从某个节点 K 发出一个信号。需要多久才能使所有节点都收到信号？如果不能使所有节点收到信号，返回 -1 。
struct Cmp{
    bool operator()(const pair<int,int>& a,const pair<int,int>& b){
        return a.first > b.first;
    }
};
int networkDelayTime(vector<vector<int>>& times, int n, int k) {
    vector<pair<int,int>> edges[n];
    for(auto &time:times)
        edges[time[0]-1].push_back({time[1]-1,time[2]});
    unsigned dist[n];
    for(int i=0;i<n;++i)
        dist[i] = 0xFFFFFFFF;
    dist[k-1] = 0;
    priority_queue<pair<int, int>, vector<pair<int, int>>,Cmp> hp;
    hp.emplace(0, k-1);
    while (!hp.empty()) {
        auto p = hp.top();
        hp.pop();
        int cost = p.first, next = p.second;
        if (dist[next] < cost)
            continue;
        for (auto &e : edges[next]) {
            int alt = dist[next] + e.second;
            if (dist[next]!=0xFFFFFFFF&&alt < dist[e.first]) {
                dist[e.first] = alt;
                hp.emplace(alt, e.first);
            }
        }
    }
    unsigned ans = 0;
    for(int i=0;i<n;++i)
        if(dist[i]>ans)
            ans = dist[i];
    return ans;
}
// 765. 情侣牵手https://leetcode-cn.com/problems/couples-holding-hands/
// n 对情侣坐在连续排列的 2n 个座位上，想要牵到对方的手。人和座位由一个整数数组 row 表示，其中 row[i] 是坐在第 i 个座位上的人的 ID。情侣们按顺序编号，第一对是 (0, 1)，第二对是 (2, 3)，以此类推，最后一对是 (2n-2, 2n-1)。返回 最少交换座位的次数，以便每对情侣可以并肩坐在一起。 每次交换可选择任意两人，让他们站起来交换座位。
int minSwapsCouples(vector<int>& row) {
    int size = row.size();
    int mp[size];
    for(int i=0;i<size;++i)
        mp[row[i]] = i;
    int count = 0;
    for(int i=0;i<size;i+=2){
        int p = row[i] ^ 1;
        if(row[i+1]==p)
            continue;
        row[mp[p]] = row[i+1];
        mp[row[i+1]] = mp[p];
        row[i+1] = p;
        ++count;
    }
    return count;
}
// 785. 判断二分图 https://leetcode-cn.com/problems/is-graph-bipartite/
bool dfs(vector<vector<int>>& graph,char* colors,int pos,char color){
    char ncolor = ~color;
    if(colors[pos]==ncolor)
        return false;
    for(auto next:graph[pos])
        if(!dfs(graph,colors,next,ncolor))
            return false;
    return true;
}
bool isBipartite(vector<vector<int>>& graph) {
    int size = graph.size();
    char colors[size];
    for(int i=0;i<size;++i)
        colors[i] = 0;
    bool flag = true;
    for(int i=0;i<size&&flag;++i){
        if(!colors[i])
            flag &= dfs(graph,colors,i,0xF0); 
    }
    return flag;
}
// 797. 所有可能的路径 https://leetcode-cn.com/problems/all-paths-from-source-to-target/
// 给你一个有 n 个节点的 有向无环图（DAG），请你找出所有从节点 0 到节点 n-1 的路径并输出（不要求按特定顺序）graph[i] 是一个从节点 i 可以访问的所有节点的列表（即从节点 i 到节点 graph[i][j]存在一条有向边）。
class Solution {
private:
    vector<vector<int>> ans;
    vector<int> stk;
    int n;
    void dfs(vector<vector<int>>& g,int pos){
        stk.push_back(pos);
        if(pos==n){
            ans.push_back(stk);
            return ;
        }
        for(auto next:g[pos]){
            dfs(g,next);
            stk.pop_back();
        }
    }
public:
    vector<vector<int>> allPathsSourceTarget(vector<vector<int>>& graph) {
        n = graph.size();
        dfs(graph,0);
        return ans;
    }
};
// 802. 找到最终的安全状态 https://leetcode-cn.com/problems/find-eventual-safe-states/
// 在有向图中，以某个节点为起始节点，从该点出发，每一步沿着图中的一条有向边行走。如果到达的节点是终点（即它没有连出的有向边），则停止。对于一个起始节点，如果从该节点出发，无论每一步选择沿哪条有向边行走，最后必然在有限步内到达终点，则将该起始节点称作是 安全 的。返回一个由图中所有安全的起始节点组成的数组作为答案。答案数组中的元素应当按 升序 排列。该有向图有 n 个节点，按 0 到 n - 1 编号，其中 n 是 graph 的节点数。图以下述形式给出：graph[i] 是编号 j 节点的一个列表，满足 (i, j) 是图的一条有向边。
class Solution {
private:
    char* color;
    bool dfs(vector<vector<int>>& g,int pos){
        if(color[pos])
            return (color[pos] == 2);
        color[pos] = 1;
        for(auto next:g[pos]){
            if(!dfs(g,next))
                return false;
        }
        color[pos] = 2;
        return true;
    }
public:
    vector<int> eventualSafeNodes(vector<vector<int>> &graph) {
        int size = graph.size();
        color = new char[size];
        memset(color,0,size);
        vector<int> ans;
        for(int i=0;i<size;++i)
            if(dfs(graph,i))
                ans.push_back(i);
        return ans;
    }
};

// 剑指 Offer 51. 数组中的逆序对https://leetcode-cn.com/problems/shu-zu-zhong-de-ni-xu-dui-lcof/
// 在数组中的两个数字，如果前面一个数字大于后面的数字，则这两个数字组成一个逆序对。输入一个数组，求出这个数组中的逆序对的总数。
int mergeSort(vector<int>& nums,int* tmp,int left,int right){
    if(left>=right)
        return 0;
    int mid = (left + right)/2;
    int count = mergeSort(nums,tmp,left,mid) + mergeSort(nums,tmp,mid+1,right);
    int i = left,j=mid+1,pos = left;
    while(i<=mid&&j<=right){
        if(nums[i]<=nums[j]){
            tmp[pos++] = nums[i++];
            count += (j-mid-1);
        }
        else
            tmp[pos++] = nums[j++];
    }
    while(i<=mid){
        tmp[pos++] = nums[i++];
        count += (j-mid-1);
    }
    while(j<=right)
        tmp[pos++] = nums[j++];
    for(;left<=right;++left)
        nums[left] = tmp[left];
    return count;
}
int reversePairs(vector<int>& nums) {
    if(nums.empty())
        return 0;
    int* tmp = new int[nums.size()];
    int count =  mergeSort(nums,tmp,0,nums.size()-1);
    delete[] tmp;
    return count;
}
// 493. 翻转对 https://leetcode-cn.com/problems/reverse-pairs/
// 给定一个数组 nums ，如果 i < j 且 nums[i] > 2*nums[j] 我们就将 (i, j) 称作一个重要翻转对。你需要返回给定数组中的重要翻转对的数量。
int mergeCount(vector<int>& nums,int* tmp,int left,int right){
    if(left==right)
        return 0;
    int mid = (left+right)/2;
    int count = mergeCount(nums,tmp,left,mid) + mergeCount(nums,tmp,mid+1,right);
    int i = left;
    int l = left;
    int r = mid + 1;
    while(l<=mid){
        while(r<=right&&(long long)nums[l]>2*(long long)nums[r]) 
            r++;
        count += (r-mid-1);
        l++;
    }
    l = left; r = mid+1;
    while(l<=mid&&r<=right){
        if(nums[l]<=nums[r])
            tmp[i++] = nums[l++];
        else
            tmp[i++] = nums[r++];
    }
    while(l<=mid)
        tmp[i++] = nums[l++];
    while(r<=right)
        tmp[i++] = nums[r++];
    for(;left<=right;++left)
        nums[left] = tmp[left];
    return count;
}
int reversePairs(vector<int>& nums) {
    int size = nums.size();
    if(size==0)
        return 0;
    int tmp[size];
    return mergeCount(nums,tmp,0,size-1);
}
// 327. 区间和的个数 https://leetcode-cn.com/problems/count-of-range-sum/
// 给你一个整数数组 nums 以及两个整数 lower 和 upper 。求数组中，值位于范围 [lower, upper] （包含 lower 和 upper）之内的 区间和的个数 。区间和 S(i, j) 表示在 nums 中，位置从 i 到 j 的元素之和，包含 i 和 j (i ≤ j)。
class Solution{
private:
    long* sum;
    long* tmp;
    int low;
    int up;
    int mergeCount(int left,int right){
        if(left==right)
            return 0;
        int mid = (left+right)/2;
        int count = mergeCount(left,mid) + mergeCount(mid+1,right);
        int i = left;
        int l = mid + 1;
        int r = mid + 1;
        while (i<=mid) {
            while(l<=right&&sum[l]-sum[i]<low)
                l++;
            while(r<=right&&sum[r]-sum[i]<=up) 
                r++;
            count+=(r-l);
            i++;
        }
        i = left;l = left; r = mid+1;
        while(l<=mid&&r<=right){
            if(sum[l]<=sum[r])
                tmp[i++] = sum[l++];
            else
                tmp[i++] = sum[r++];
        }
        while(l<=mid)
            tmp[i++] = sum[l++];
        while(r<=right)
            tmp[i++] = sum[r++];
        for(;left<=right;++left)
            sum[left] = tmp[left];
        return count;
    } 
public:
    int countRangeSum(vector<int>& nums, int lower, int upper) {
        int size = nums.size();
        low = lower;
        up = upper;
        sum = new long[size+1];
        sum[0] = 0;
        for(int i=0;i<size;++i)
            sum[i+1] = sum[i] + nums[i];
        tmp = new long[size+1];
        int ans = mergeCount(0,size-1);
        delete[] sum;
        delete[] tmp;
        return ans;
    }
};
// 787. K 站中转内最便宜的航班 https://leetcode-cn.com/problems/cheapest-flights-within-k-stops/
// 有 n 个城市通过一些航班连接。给你一个数组 flights ，其中 flights[i] = [fromi, toi, pricei] ，表示该航班都从城市 fromi 开始，以价格 pricei 抵达 toi。现在给定所有的城市和航班，以及出发城市 src 和目的地 dst，你的任务是找到出一条最多经过 k 站中转的路线，使得从 src 到 dst 的 价格最便宜 ，并返回该价格。 如果不存在这样的路线，则输出 -1。
// #define INF ;
int findCheapestPrice(int n, vector<vector<int>>& flights, int src, int dst, int k) {
    int nums[2][n];
    for(int i=0;i<n;++i)
        nums[0][i] = 1010001;
    nums[0][src] = 0;
    ++k;
    int ans = 1010001;
    for(int i=1;i<=k;++i){
        int curr = i&1,prev = curr^1;
        for(int i=0;i<n;++i)
            nums[curr][i] = 1010001;
        for(auto &flight:flights){
            int i = flight[0], j=flight[1],cost=flight[2];
            nums[curr][j] = min(nums[curr][j],nums[curr][i]+cost);
        }
        ans = min(ans,nums[curr][dst]);
    }
    return (ans==1010001)? -1:ans;
}
// 341. 扁平化嵌套列表迭代器https://leetcode-cn.com/problems/flatten-nested-list-iterator/
class NestedInteger {
public:
    bool isInteger() const;
    int getInteger() const;
    const vector<NestedInteger> &getList() const;
};
struct Node{
    typedef vector<NestedInteger>::iterator iter;
    iter cur;
    iter end;
    Node(iter _cur,iter _end):end(_end),cur(_cur){}
};
class NestedIterator {
private:
    stack<Node> stk;
public:
    NestedIterator(vector<NestedInteger> &nestedList) {
        stk.emplace(nestedList.begin(),nestedList.end());
    }
    int next() {
        return (stk.top().cur++)->getInteger();
    }
    bool hasNext() {
        while(!stk.empty()){
            auto &i = stk.top();
            if(i.cur==i.end){
                stk.pop();
                continue;
            }
            if(i.cur->isInteger())
                return true;
            auto &_list = (i.cur++)->getList();
            stk.emplace(_list.begin(),_list.end());
        }
        return false;
    }
};
// 1380. 矩阵中的幸运数https://leetcode-cn.com/problems/lucky-numbers-in-a-matrix/
// 给你一个 m * n 的矩阵，矩阵中的数字 各不相同 。请你按 任意 顺序返回矩阵中的所有幸运数。幸运数是指矩阵中满足同时下列两个条件的元素：在同一行的所有元素中最小在同一列的所有元素中最大
struct MinInfo{
    int val;
    vector<int> pos;
    MinInfo():val(INT32_MAX){}
};
vector<int> luckyNumbers (vector<vector<int>>& matrix) {
    int m = matrix.size(), n =matrix[0].size();
    MinInfo row_min[m];
    int col_max[n];
    for(int i=0;i<n;++i)
        col_max[i] = INT32_MIN;
    for(int i=0;i<m;++i){
        for(int j=0;j<n;++j){
            int tmp = matrix[i][j];
            if(tmp<row_min[i].val){
                row_min[i].val = tmp;
                row_min[i].pos.clear();
                row_min[i].pos.push_back(j);
            }
            else if(tmp==row_min[i].val)
                row_min[i].pos.push_back(j);
            if(tmp>col_max[j])
                col_max[j] = tmp;
        }
    }
    vector<int> ans;
    for(int i=0;i<m;++i){
        auto& pos = row_min[i].pos;
        int size = pos.size();
        for(int j=0;j<size;++j)
            if(row_min[i].val==col_max[pos[j]])
                ans.push_back(row_min[i].val);
    }
    return ans;
}
// 面试题 10.10. 数字流的秩 https://leetcode-cn.com/problems/rank-from-stream-lcci/
// 假设你正在读取一串整数。每隔一段时间，你希望能找出数字 x 的秩(小于或等于 x 的值的个数)。请实现数据结构和算法来支持这些操作，也就是说：实现 track(int x) 方法，每读入一个数字都会调用该方法；实现 getRankOfNumber(int x) 方法，返回小于或等于 x 的值的个数。
class StreamRank {
private:
	vector<int> arr;
public:
    StreamRank(){}
    void track(int x) {		
		auto pos = upper_bound(arr.begin(),arr.end(),x);
		if(pos==arr.end())
			arr.push_back(x);
		else
			arr.insert(pos,x);
	}
    int getRankOfNumber(int x) {
		return upper_bound(arr.begin(),arr.end(),x)-arr.begin();
    }
};
// 969. 煎饼排序 https://leetcode-cn.com/problems/pancake-sorting/
// 给你一个整数数组 arr ，请使用 煎饼翻转 完成对数组的排序。一次煎饼翻转的执行过程如下：选择一个整数 k ，1 <= k <= arr.length反转子数组 arr[0...k-1]（下标从 0 开始）例如，arr = [3,2,1,4] ，选择 k = 3 进行一次煎饼翻转，反转子数组 [3,2,1] ，得到 arr = [1,2,3,4] 。以数组形式返回能使 arr 有序的煎饼翻转操作所对应的k值序列。任何将数组排序且翻转次数在 10 * arr.length 范围内的有效答案都将被判断为正确。
vector<int> pancakeSort(vector<int>& arr) {
    int size = arr.size();
    vector<int> ans;
    for(int i=size;i;--i){
        auto iter = arr.begin();
        int pos = max_element(iter,iter+i)-iter+1;
        if(pos!=i){
            reverse(iter,iter+pos);
            reverse(iter,iter+i);
            ans.push_back(pos);
            ans.push_back(i);
        }
    }
    return ans;
}
