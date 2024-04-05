#include <bits/stdc++.h>
using namespace std; 
// 2790. 长度递增组的最大数目 https://leetcode.cn/problems/maximum-number-of-groups-with-increasing-length/
// 给你一个下标从 0 开始、长度为 n 的数组 usageLimits 。你的任务是使用从 0 到 n - 1 的数字创建若干组，并确保每个数字 i 在 所有组 中使用的次数总共不超过 usageLimits[i] 次。此外，还必须满足以下条件：每个组必须由 不同 的数字组成，也就是说，单个组内不能存在重复的数字。每个组（除了第一个）的长度必须 严格大于 前一个组。在满足所有条件的情况下，以整数形式返回可以创建的最大组数。
int maxIncreasingGroups(vector<int>& usageLimits) {
	int size = usageLimits.size();
	sort(usageLimits.begin(),usageLimits.end());
	int prev_idx[size];
	prev_idx[0] = 0;
	for(int i=1;i<size;i++){
		if(usageLimits[i]==usageLimits[i-1])
			prev_idx[i] = prev_idx[i-1];
		else
			prev_idx[i] = i;
	}
	int m = usageLimits.back();
	for(int i=1;i<=m;i++){
		int pos = size-1,prev_pos = size-1,cnt = 0;
		while(pos>=0&&cnt<i){
			cnt += (pos - prev_idx[pos] + 1);
			prev_pos = pos;
			pos = prev_idx[pos]-1;
			usageLimits[prev_pos]--;
		}
		if(cnt<i||usageLimits[prev_pos]<0) return i-1;
		if(cnt>i){
			prev_idx[prev_pos] = prev_pos - (cnt - i) + 1;
			usageLimits[prev_pos - (cnt - i)] = usageLimits[prev_pos];
			usageLimits[prev_pos]++;
			if(usageLimits[prev_pos - (cnt - i)]==usageLimits[pos])
				prev_idx[prev_pos - (cnt - i)] = prev_idx[pos];
			else
				prev_idx[prev_pos - (cnt - i)] = pos+1;
		}
	}
	return m;
}
// LCP 63. 弹珠游戏 https://leetcode.cn/problems/EXvqDp/description/
// 欢迎各位来到「力扣嘉年华」，接下来将为各位介绍在活动中广受好评的弹珠游戏。N*M 大小的弹珠盘的初始状态信息记录于一维字符串型数组 plate 中，数组中的每个元素为仅由 "O"、"W"、"E"、"." 组成的字符串。其中："O" 表示弹珠洞（弹珠到达后会落入洞中，并停止前进）；"W" 表示逆时针转向器（弹珠经过时方向将逆时针旋转 90 度）；"E" 表示顺时针转向器（弹珠经过时方向将顺时针旋转 90 度）；"." 表示空白区域（弹珠可通行）。游戏规则要求仅能在边缘位置的 空白区域 处（弹珠盘的四角除外）沿 与边缘垂直 的方向打入弹珠，并且打入后的每颗弹珠最多能 前进 num 步。请返回符合上述要求且可以使弹珠最终入洞的所有打入位置。你可以 按任意顺序 返回答案。注意：若弹珠已到达弹珠盘边缘并且仍沿着出界方向继续前进，则将直接出界。
class BallGame {
    vector<string> *p;
    int _num;
    int m;
    int n;
    static constexpr int dirs[4][2] = {{1,0},{0,1},{-1,0},{0,-1}};
    static constexpr int clockwise[4] = {3,0,1,2};
    static constexpr int anticlockwise[4] = {1,2,3,0};
    bool canArrive(int i,int j,int dir){
        auto& plate = *p;
        if(plate[i][j]!='.')   // 则要求仅能在边缘位置的空白区域处
            return false;
        for(int k=0;k<_num;++k){
            if(plate[i][j]=='W')
                dir = anticlockwise[dir];
            else if (plate[i][j]=='E')
                dir = clockwise[dir];
            i += dirs[dir][0]; j += dirs[dir][1];
            if(i<0||i>=m||j<0||j>=n)
                return false;
            if(plate[i][j]=='O')
                return true;
        }
        return false;
    }
public:
    vector<vector<int>> ballGame(int num, vector<string>& plate) {
        m = plate.size(); n = plate[0].size();
        p = &plate; _num = num;
        vector<vector<int>> ans;
        for(int i=1;i<m-1;++i){
            if(canArrive(i,0,1))
                ans.push_back({i,0});
            if(n!=1&&canArrive(i,n-1,3))
                ans.push_back({i,n-1});
        }
        for(int i=1;i<n-1;++i){
            if(canArrive(0,i,0))
                ans.push_back({0,i});
            if(m!=1&&canArrive(m-1,i,2))
                ans.push_back({m-1,i});
        }
        return ans;
    }
};

struct TreeNode{
    int val;
    TreeNode* left;
    TreeNode* right;
    TreeNode(int val):val(val){}
};
// LCP 64. 二叉树灯饰 https://leetcode.cn/problems/U7WvvU/description/
// 「力扣嘉年华」的中心广场放置了一个巨型的二叉树形状的装饰树。每个节点上均有一盏灯和三个开关。节点值为 0 表示灯处于「关闭」状态，节点值为 1 表示灯处于「开启」状态。每个节点上的三个开关各自功能如下：开关 1：切换当前节点的灯的状态；开关 2：切换 以当前节点为根 的子树中，所有节点上的灯的状态；开关 3：切换 当前节点及其左右子节点（若存在的话） 上的灯的状态；给定该装饰的初始状态 root，请返回最少需要操作多少次开关，可以关闭所有节点的灯。
int closeLampInTree(TreeNode* root) {
	int INF = 1e9;
	unordered_map<long long,int> mp;
	function<int(long long)> dfs;
	dfs = [&](long long node){
		TreeNode* root =(TreeNode*)(node&(~3));
		if(!root)
			return 0;
		if(mp.count(node))
			return mp[node];
		bool turn = node&1,need_state = node&2;
		int cur_state = root->val;
		int ans = INF;
		for(int i=0;i<8;i++){
			int flag = 0;
			bool op1 = i&1,op2 = i&2,op3 = i&3; //op1改当前，op2改子树所有，op3改当前及子孙
			if(op2) need_state = !need_state;
			if(turn^op3^op1) cur_state ^= 1; 
			if(root->val!=need_state) continue;
			if(op3) flag |= 2;
			ans = min(ans,op1+op2+op3+dfs((long long)(root->left)|flag|need_state)+dfs((long long)(root->right)|flag|need_state));
		}
		return ans;
	};
	return dfs((long long)root);
}
// 985. 查询后的偶数和 https://leetcode.cn/problems/sum-of-even-numbers-after-queries/description/
// 给出一个整数数组 A 和一个查询数组 queries。对于第 i 次查询，有 val = queries[i][0], index = queries[i][1]，我们会把 val 加到 A[index] 上。然后，第 i 次查询的答案是 A 中偶数值的和。（此处给定的 index = queries[i][1] 是从 0 开始的索引，每次查询都会永久修改数组 A。）返回所有查询的答案。你的答案应当以数组 answer 给出，answer[i] 为第 i 次查询的答案。
vector<int> sumEvenAfterQueries(vector<int>& nums, vector<vector<int>>& queries) {
	int sum = 0;
	for(auto num :nums){
		if((num&1)==0) sum += num;
	}
	vector<int> ans; ans.reserve(queries.size());
	for(auto &query:queries) {
		int index = query[1], val = query[0];
		if ((nums[index]&1)==0) sum -= nums[index];
		nums[index] += val;
		if ((nums[index]&1)==0) sum += nums[index];
		ans.push_back(sum);
	}
	return ans;
}

// 2034. 股票价格波动 https://leetcode.cn/problems/stock-price-fluctuation/description/
// 给你一支股票价格的数据流。数据流中每一条记录包含一个 时间戳 和该时间点股票对应的 价格 。不巧的是，由于股票市场内在的波动性，股票价格记录可能不是按时间顺序到来的。某些情况下，有的记录可能是错的。如果两个有相同时间戳的记录出现在数据流中，前一条记录视为错误记录，后出现的记录 更正 前一条错误的记录。请你设计一个算法，实现：更新 股票在某一时间戳的股票价格，如果有之前同一时间戳的价格，这一操作将 更正 之前的错误价格。找到当前记录里 最新股票价格 。最新股票价格 定义为时间戳最晚的股票价格。找到当前记录里股票的 最高价格 。找到当前记录里股票的 最低价格 。请你实现 StockPrice 类：StockPrice() 初始化对象，当前无股票价格记录。void update(int timestamp, int price) 在时间点 timestamp 更新股票价格为 price 。int current() 返回股票 最新价格 。int maximum() 返回股票 最高价格 。int minimum() 返回股票 最低价格 。
class StockPrice {
	int curTime;
	int curPrice;
	unordered_map<int,int> mp;
	multiset<int> st;
public:
    StockPrice():curTime(0),curPrice(0) {}
    
    void update(int timestamp, int price) {
		if(timestamp>=curTime){
			curTime = timestamp;
			curPrice = price;
		}
		if(mp.count(timestamp)){
			auto it = st.find(mp[timestamp]);
			if(it!=st.end()) st.erase(it);
		}
		mp[timestamp] = price;
		st.insert(price);
    }
    
    int current() {
		return curPrice;
    }
    
    int maximum() {
		return *st.cbegin();
    }
    
    int minimum() {
		return *st.begin();
    }
};

// 2731. 移动机器人 https://leetcode.cn/problems/movement-of-robots/
// 有一些机器人分布在一条无限长的数轴上，他们初始坐标用一个下标从 0 开始的整数数组 nums 表示。当你给机器人下达命令时，它们以每秒钟一单位的速度开始移动。给你一个字符串 s ，每个字符按顺序分别表示每个机器人移动的方向。'L' 表示机器人往左或者数轴的负方向移动，'R' 表示机器人往右或者数轴的正方向移动。当两个机器人相撞时，它们开始沿着原本相反的方向移动。请你返回指令重复执行 d 秒后，所有机器人之间两两距离之和。由于答案可能很大，请你将答案对 109 + 7 取余后返回。注意：对于坐标在 i 和 j 的两个机器人，(i,j) 和 (j,i) 视为相同的坐标对。也就是说，机器人视为无差别的。当机器人相撞时，它们 立即改变 它们的前进方向，这个过程不消耗任何时间。当两个机器人在同一时刻占据相同的位置时，就会相撞。例如，如果一个机器人位于位置 0 并往右移动，另一个机器人位于位置 2 并往左移动，下一秒，它们都将占据位置 1，并改变方向。再下一秒钟后，第一个机器人位于位置 0 并往左移动，而另一个机器人位于位置 2 并往右移动。例如，如果一个机器人位于位置 0 并往右移动，另一个机器人位于位置 1 并往左移动，下一秒，第一个机器人位于位置 0 并往左行驶，而另一个机器人位于位置 1 并往右移动。
int sumDistance(vector<int>& nums, string s, int d) {
	int size = nums.size();
	vector<long long> arr(size,0);
	const long long mod = 1e9+7;
	long long sum = 0;
	for(int i=0;i<size;i++){
		if(s[i]=='R') arr[i] = nums[i] + d;
		else arr[i] = nums[i] - d;
	}
	sort(arr.begin(),arr.end());
	for(int i=1;i<size;i++){
		sum = (sum + (arr[i] - arr[i - 1]) * i % mod * (size - i) % mod) %mod;
	}
	return int(sum);
}
// 2512. 奖励最顶尖的 K 名学生 https://leetcode.cn/problems/reward-top-k-students
// 给你两个字符串数组 positive_feedback 和 negative_feedback ，分别包含表示正面的和负面的词汇。不会 有单词同时是正面的和负面的。一开始，每位学生分数为 0 。每个正面的单词会给学生的分数 加 3 分，每个负面的词会给学生的分数 减  1 分。给你 n 个学生的评语，用一个下标从 0 开始的字符串数组 report 和一个下标从 0 开始的整数数组 student_id 表示，其中 student_id[i] 表示这名学生的 ID ，这名学生的评语是 report[i] 。每名学生的 ID 互不相同。给你一个整数 k ，请你返回按照得分 从高到低 最顶尖的 k 名学生。如果有多名学生分数相同，ID 越小排名越前。
vector<int> topStudents(vector<string>& positive_feedback, vector<string>& negative_feedback, vector<string>& report, vector<int>& student_id, int k) {
	unordered_map<string,int> mp;
	for(auto &str: positive_feedback)
		mp[str] = 3;
	for(auto &str:negative_feedback)
		mp[str] = -1;
	struct Stu{
		int score;
		int id;
		bool operator< (const Stu& other) const {
			if (other.score==score) return id < other.id;
			return score > other.score;
		}
		Stu() = default;
		Stu(int id,int score):id(id),score(score){}
	};
	int size = student_id.size();
	vector<Stu> arr; arr.reserve(size);
	for(int i=0;i<size;i++) {
		stringstream ss(report[i]);
		string tmp; int score = 0;
		while(ss >> tmp)
			score += mp[tmp];
		arr.emplace_back(student_id[i],score);
	}
	sort(arr.begin(),arr.end());
	vector<int> ans(k,0);
	for(int i=0;i<k;i++){
		ans[i] = arr[i].id;
	}
	return ans;
}


vector<int> missingRolls(vector<int>& rolls, int mean, int n) {
	int sum = 0, m = rolls.size();
	for(auto roll: rolls)
		sum += roll;
	sum = mean * (m+n) - sum;
	if(sum<n || sum >6*n) return {};
	int a = sum / n, cnt = sum - a * n; 
	vector<int> ans;
	ans.insert(ans.end(),(n-cnt),a);
	ans.insert(ans.end(),cnt,a+1);
}
// 1402. 做菜顺序 https://leetcode.cn/problems/reducing-dishes/
// 一个厨师收集了他 n 道菜的满意程度 satisfaction ，这个厨师做出每道菜的时间都是 1 单位时间。一道菜的 「 like-time 系数 」定义为烹饪这道菜结束的时间（包含之前每道菜所花费的时间）乘以这道菜的满意程度，也就是 time[i]*satisfaction[i] 。返回厨师在准备了一定数量的菜肴后可以获得的最大 like-time 系数 总和。你可以按 任意 顺序安排做菜的顺序，你也可以选择放弃做某些菜来获得更大的总和。
int maxSatisfaction(vector<int>& satisfaction) {
	sort(satisfaction.begin(),satisfaction.end(),greater<int>());
	int size = satisfaction.size();
	int sum = 0,ans = 0;
	for(int i=0;i<size;i++){
		sum += satisfaction[i];
		if(sum<0) break;
		ans += sum;
	}
	return ans;
}
// 117. 填充每个节点的下一个右侧节点指针 II https://leetcode.cn/problems/populating-next-right-pointers-in-each-node-ii
/*
给定一个二叉树：
	struct Node {
	int val;
	Node *left;
	Node *right;
	Node *next;
	}
填充它的每个 next 指针，让这个指针指向其下一个右侧节点。如果找不到下一个右侧节点，则将 next 指针设置为 NULL 。初始状态下，所有 next 指针都被设置为 NULL 。
*/
class Connect {
	class Node {
	public:
		int val;
		Node* left;
		Node* right;
		Node* next;
		Node() : val(0), left(NULL), right(NULL), next(NULL) {}
		Node(int _val) : val(_val), left(NULL), right(NULL), next(NULL) {}
		Node(int _val, Node* _left, Node* _right, Node* _next)
			: val(_val), left(_left), right(_right), next(_next) {}
	};
	struct Info{
		Node* node;
		int depth;
		Info()=default;
		Info(Node* node,int depth):node(node),depth(depth){}
	};
public:
    Node* connect(Node* root) {
		if(!root) return root;
        queue<Info> que;
		que.emplace(root,0);
		while(1){
			Info cur = que.front();
			que.pop();
			if(cur.node->left) que.emplace(cur.node->left,cur.depth+1);
			if(cur.node->right) que.emplace(cur.node->right,cur.depth+1);
			if(que.empty()) break;
			const Info& next = que.front();
			if(next.depth==cur.depth) cur.node->next = next.node;
		}
		return root;
    }
};
// https://leetcode.cn/problems/find-closest-node-to-given-two-nodes/description/
// 给你一个 n 个节点的 有向图 ，节点编号为 0 到 n - 1 ，每个节点 至多 有一条出边。有向图用大小为 n 下标从 0 开始的数组 edges 表示，表示节点 i 有一条有向边指向 edges[i] 。如果节点 i 没有出边，那么 edges[i] == -1 。同时给你两个节点 node1 和 node2 。请你返回一个从 node1 和 node2 都能到达节点的编号，使节点 node1 和节点 node2 到这个节点的距离 较大值最小化。如果有多个答案，请返回 最小 的节点编号。如果答案不存在，返回 -1 。注意 edges 可能包含环。
int closestMeetingNode(vector<int>& edges, int node1, int node2) {
	int n = edges.size();
	auto dfs = [&](int s){
		vector<int> dist(n,-1); 
		int cur = 0; dist[s] = 0; s = edges[s];
		do{
			dist[s] = cur++;
			s = edges[s];
		}while(s!=-1);
		return dist;
	};
	auto d1 = dfs(node1), d2 = dfs(node2);
}
// 1410. HTML 实体解析器 https://leetcode.cn/problems/html-entity-parser/
// 「HTML 实体解析器」 是一种特殊的解析器，它将 HTML 代码作为输入，并用字符本身替换掉所有这些特殊的字符实体。HTML 里这些特殊字符和它们对应的字符实体包括：双引号：字符实体为 &quot; ，对应的字符是 " 。单引号：字符实体为 &apos; ，对应的字符是 ' 。与符号：字符实体为 &amp; ，对应对的字符是 & 。大于号：字符实体为 &gt; ，对应的字符是 > 。小于号：字符实体为 &lt; ，对应的字符是 < 。斜线号：字符实体为 &frasl; ，对应的字符是 / 。给你输入字符串 text ，请你实现一个 HTML 实体解析器，返回解析器解析后的结果。
string entityParser(string text) {
	struct Info{
        char ch;
        int pos;
        Info() = default;
        Info(char ch,int pos):ch(ch),pos(pos){}
    };
	auto check = [&](int pos){
		if(text[pos]!='&') return Info(text[pos],pos+1);
		auto str = text.substr(pos,4);
		if(str=="&gt;") return Info('>',pos+4);
		if(str=="&lt;") return Info('<',pos+4);
		if(text.substr(pos,5)=="&amp;") return Info('&',pos+5);
		str = text.substr(pos,6);
		if(str=="&quot;") return Info('"',pos+6);
		if(str=="&apos;") return Info('\'',pos+6);
		if(text.substr(pos,7)=="&frasl;") return Info('/',pos+7);
		return Info('&',pos+1);
	};
	int i=0,size = text.size();
	string ans; ans.reserve(size);
	while(i<size){
		auto ret = check(i);
		ans.push_back(ret.pos!=i+1? ret.ch: text[i]);
		i = ret.pos;
	}
	return ans;
}
// 100137. 统计最大元素出现至少 K 次的子数组 https://leetcode.cn/problems/count-subarrays-where-max-element-appears-at-least-k-times
// 给你一个整数数组 nums 和一个 正整数 k 。请你统计有多少满足 「 nums 中的 最大 元素」至少出现 k 次的子数组，并返回满足这一条件的子数组的数目。子数组是数组中的一个连续元素序列。
long long countSubarrays(vector<int>& nums, int k) {
	int mx = 0,left = 0,right = 0,cnt = 0,size = nums.size();
	long long ans = 0;
	for(auto num: nums)
		mx = max(mx,num);
	while(left < size) {
		while(right<size&&cnt< k)
			cnt += (nums[right++] == mx) ? 1 : 0;
		if(cnt<k) break;
		ans += (size -right+1);
		cnt -= (nums[left++] == mx) ? 1 : 0;
	} 
	return ans ;
}
// 2276. 统计区间中的整数数目 https://leetcode.cn/problems/count-integers-in-intervals
// 给你区间的 空 集，请你设计并实现满足要求的数据结构：新增：添加一个区间到这个区间集合中。统计：计算出现在 至少一个 区间中的整数个数。实现 CountIntervals 类：CountIntervals() 使用区间的空集初始化对象void add(int left, int right) 添加区间 [left, right] 到区间集合之中。int count() 返回出现在 至少一个 区间中的整数个数。注意：区间 [left, right] 表示满足 left <= x <= right 的所有整数 x 。
class CountIntervals {
	struct Node{
		int right;
		int val;
		Node(int right,int val):right(right),val(val){}
	};
	map<int,Node> ranges;
	int sum;
	typedef map<int,Node>::iterator iter;
	iter split(int x){
		auto it = ranges.lower_bound(x);
		if(it!=ranges.end()&&it->first==x) 
			return it;
		it--;
		int l = it->first,right = it->second.right;
		auto node = it->second;
		ranges.erase(it);
		node.right = x-1;
		ranges.emplace(l,node);
		node.right = right;
		return ranges.emplace(x,node).first;
	}
public:
    CountIntervals():sum(0) {
		ranges.emplace(0,Node(1e9+7,0));
    }
    
    void add(int left, int right) {
		auto itr = split(right+1),itl = split(left);
		for(auto it = itl;it!=itr;it++){
			sum -= (it->second.val!=0) ? (it->second.right-it->first+1):0;
		}
		ranges.emplace(left,Node(right,1));
    }
    
    int count() {
		return sum;
    }
};