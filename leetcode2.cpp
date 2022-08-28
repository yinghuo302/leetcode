/*
 * @Author: zanilia
 * @Date: 2021-12-04 14:03:29
 * @LastEditTime: 2022-07-26 10:24:03
 * @Descripttion: 
 */
#include <bits/stdc++.h>
using namespace std;
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
// 1036. 逃离大迷宫 https://leetcode.cn/problems/escape-a-large-maze/
// 在一个 106 x 106 的网格中，每个网格上方格的坐标为 (x, y) 。现在从源方格 source = [sx, sy] 开始出发，意图赶往目标方格 target = [tx, ty] 。数组 blocked 是封锁的方格列表，其中每个 blocked[i] = [xi, yi] 表示坐标为 (xi, yi) 的方格是禁止通行的。每次移动，都可以走到网格中在四个方向上相邻的方格，只要该方格 不 在给出的封锁列表 blocked 上。同时，不允许走出网格。只有在可以通过一系列的移动从源方格 source 到达目标方格 target 时才返回 true。否则，返回 false。
class Solution {
    static constexpr int dirs[4][2] = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
public:
    bool isEscapePossible(vector<vector<int>>& blocked, vector<int>& source, vector<int>& target) {
        int size = blocked.size();
        if(size<2)
            return true;
        vector<int>& row = *(new vector<int>(size+2));
        vector<int>& col = *(new vector<int>(size+2));
        row[0] = source[0],row[1] = target[0];
        col[0] = source[1],col[1] = target[1];
        int i = 2;
        for(auto &block:blocked){
            row[i] = block[0];
            col[i++] = block[1];
        }
        sort(row.begin(),row.end());
        sort(col.begin(),col.end());
        unordered_map<int,int> row_mp;
        unordered_map<int,int> col_mp;
        int rid = (row[0] == 0 ? 0 : 1);
        row_mp[row[0]] = rid;
        for(int i=0;i<=size;++i){
            if(row[i+1]==row[i])
                continue;
            rid += (row[i+1] == row[i] + 1 ? 1 : 2);
            row_mp[row[i+1]] = rid;
        }
        if(row.back()!=999999)
            ++rid;
        int cid = (col[0] == 0 ? 0 : 1);
        col_mp[col[0]] = cid;
        for(int i=0;i<=size;++i){
            if(col[i+1]==col[i])
                continue;
            cid += (col[i+1] == col[i] + 1 ? 1 : 2);
            col_mp[col[i+1]] = cid;
        }
        if(col.back()!=999999)
            ++cid;
        delete(&row);delete(&col);
        vector<vector<int>> g(rid+1,vector<int>(cid+1));
        for(auto &block:blocked)
            g[row_mp[block[0]]][col_mp[block[1]]] = 1;
        int sx=row_mp[source[0]],sy=col_mp[source[1]];
        int tx=row_mp[target[0]],ty=col_mp[target[1]];
        queue<pair<int, int>> q;
        q.emplace(sx, sy);
        g[sx][sy] = 1;
        while(!q.empty()) {
            int x = q.front().first,y = q.front().second;
            q.pop();
            for (int i=0;i<4;++i) {
                int nx=x+dirs[i][0],ny=y+dirs[i][1];
                if (nx>=0&&nx<=rid&&ny>=0&&ny<=cid&&g[nx][ny] != 1) {
                    if(nx==tx&&ny==ty)
                        return true;
                    q.emplace(nx, ny);
                    g[nx][ny] = 1;
                }
            }
        }
        return false;
    }
};
// 829. 连续整数求和 https://leetcode.cn/problems/consecutive-numbers-sum/
// 给定一个正整数 n，返回 连续正整数满足所有数字之和为 n 的组数 。 
int consecutiveNumbersSum(int n) {
    int tem = n*2;
    int limit = sqrt(tem+0.25)-0.5;
    int res = 1;
    for(int i=2;i<=limit;++i){
        if(i&1)
            res += (n%i==0);
        else
            res += ((n%i!=0)&&(tem%i==0));
    }
    return res;
}
// 729. 我的日程安排表 I https://leetcode.cn/problems/my-calendar-i/
// 实现一个 MyCalendar 类来存放你的日程安排。如果要添加的日程安排不会造成 重复预订 ，则可以存储这个新的日程安排。当两个日程安排有一些时间上的交叉时（例如两个日程安排都在同一时间内），就会产生 重复预订 。日程可以用一对整数 start 和 end 表示，这里的时间是半开区间，即 [start, end), 实数 x 的范围为，  start <= x < end 。实现 MyCalendar 类：MyCalendar() 初始化日历对象。boolean book(int start, int end) 如果可以将日程安排成功添加到日历中而不会导致重复预订，返回 true 。否则，返回 false 并且不要将该日程安排添加到日历中。
class MyCalendar {
    map<int,int> mp;
public:
    MyCalendar() {
        mp.emplace(-1,-1);
    }
    bool book(int start, int end) {
        auto i = --mp.lower_bound(end);
        if(i->second<=start){
            mp.emplace(start,end);
            return true;
        }
        return false;
    }
};
// 731. 我的日程安排表 II https://leetcode.cn/problems/my-calendar-ii/
// 实现一个 MyCalendar 类来存放你的日程安排。如果要添加的时间内不会导致三重预订时，则可以存储这个新的日程安排。MyCalendar 有一个 book(int start, int end)方法。它意味着在 start 到 end 时间内增加一个日程安排，注意，这里的时间是半开区间，即 [start, end), 实数 x 的范围为，  start <= x < end。当三个日程安排有一些时间上的交叉时（例如三个日程安排都在同一时间内），就会产生三重预订。每次调用 MyCalendar.book方法时，如果可以将日程安排成功添加到日历中而不会导致三重预订，返回 true。否则，返回 false 并且不要将该日程安排添加到日历中。请按照以下步骤调用MyCalendar 类: MyCalendar cal = new MyCalendar();MyCalendar.book(start, end)
class MyCalendarTwo {
    map<int,int> cnt;
public:
    MyCalendarTwo() {}  
    bool book(int start, int end) {
        int tem = 0;
        cnt[start]++;
        cnt[end]--;
        for(auto &i:cnt){
            tem += i.second;
            if(tem>=3){
                cnt[start]--;
                cnt[end]++;
                return false;
            }
        }
        return true;
    }
};
// 732. 我的日程安排表 III https://leetcode.cn/problems/my-calendar-iii/
// 当 k 个日程安排有一些时间上的交叉时（例如 k 个日程安排都在同一时间内），就会产生 k 次预订。给你一些日程安排 [start, end) ，请你在每个日程安排添加后，返回一个整数 k ，表示所有先前日程安排会产生的最大 k 次预订。实现一个 MyCalendarThree 类来存放你的日程安排，你可以一直添加新的日程安排。MyCalendarThree() 初始化对象。int book(int start, int end) 返回一个整数 k ，表示日历中存在的 k 次预订的最大值。
class MyCalendarThree {
    map<int,int> cnt;
public:
    MyCalendarThree() {}   
    int book(int start, int end) {
        int ans = 0,tem = 0;
        cnt[start]++;
        cnt[end]--;
        for(auto &i:cnt){
            tem += i.second;
            if(tem>ans)
                ans = tem;
        }
        return ans;
    }
};
// 630. 课程表 III https://leetcode.cn/problems/course-schedule-iii/
// 这里有 n 门不同的在线课程，按从 1 到 n 编号。给你一个数组 courses ，其中 courses[i] = [durationi, lastDayi] 表示第 i 门课将会 持续 上 durationi 天课，并且必须在不晚于 lastDayi 的时候完成。你的学期从第 1 天开始。且不能同时修读两门及两门以上的课程。返回你最多可以修读的课程数目。
int scheduleCourse(vector<vector<int>>& courses) {
    sort(courses.begin(),courses.end(),[](const vector<int>&a,const vector<int>&b){
        return a[1] < b[1];
    });
    int total = 0;
    priority_queue<int> hp;
    for(auto &course:courses){
        int duration = course[0],last = course[1];
        total += duration;
        hp.emplace(duration);
        if(total>last){
            total -= hp.top();
            hp.pop();
        }
            hp.emplace(duration);
    }
    return hp.size();
}
// 587. 安装栅栏 https://leetcode.cn/problems/erect-the-fence/
// 在一个二维的花园中，有一些用 (x, y) 坐标表示的树。由于安装费用十分昂贵，你的任务是先用最短的绳子围起所有的树。只有当所有的树都被绳子包围时，花园才能围好栅栏。你需要找到正好位于栅栏边界上的树的坐标。
vector<vector<int>> outerTrees(vector<vector<int>>& trees) {
    int size = trees.size();
    if(size<4)
        return trees;
    sort(trees.begin(),trees.end(),[](const vector<int>& a,const vector<int>& b){
        return a[0]<b[0]||(a[0]==b[0]&&a[1]<b[1]);
    });
    auto cross = [](const vector<int> & p, const vector<int> & q, const vector<int> & r) {
        return (q[0] - p[0]) * (r[1] - q[1]) - (q[1] - p[1]) * (r[0] - q[0]);
    };
    vector<int> hull;
    bool used[size];
    for(int i=0;i<size;++i)
        used[i] = false;
    hull.push_back(0);
    for(int i=1;i<size;++i){
        while(hull.size()>1&&cross(trees[hull[hull.size()-2]],trees[hull.back()],trees[i])<0){
            used[hull.back()] = false;
            hull.pop_back();
        }
        hull.push_back(i);
        used[i] = true;
    }
    int m = hull.size();
    for(int i=size-2;i>=0;--i){
        if(used[i])
            continue;
        while(hull.size()>m&&cross(trees[hull[hull.size()-2]],trees[hull.back()],trees[i])<0){
            used[hull.back()] = false;
            hull.pop_back();
        }
        hull.push_back(i);
        used[i] = true;
    }
    hull.pop_back();
    vector<vector<int>> ans;
    ans.reserve(hull.size());
    for(auto &h:hull)
        ans.push_back(trees[h]);
    return ans;
}
// 497. 非重叠矩形中的随机点 https://leetcode.cn/problems/random-point-in-non-overlapping-rectangles/
// 给定一个由非重叠的轴对齐矩形的数组 rects ，其中 rects[i] = [ai, bi, xi, yi] 表示 (ai, bi) 是第 i 个矩形的左下角点，(xi, yi) 是第 i 个矩形的右上角点。设计一个算法来随机挑选一个被某一矩形覆盖的整数点。矩形周长上的点也算做是被矩形覆盖。所有满足要求的点必须等概率被返回。在给定的矩形覆盖的空间内的任何整数点都有可能被返回。请注意 ，整数点是具有整数坐标的点。实现 Solution 类:Solution(int[][] rects) 用给定的矩形数组 rects 初始化对象。int[] pick() 返回一个随机的整数点 [u, v] 在给定的矩形所覆盖的空间内。
class Solution {
    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution;
    vector<int> arr;
    vector<vector<int>>& rects;
public:
    Solution(vector<vector<int>>& rects): rects(rects){
        int size = rects.size();
        arr.reserve(size+1);
        arr.push_back(0);
        int sum = 0;
        for(auto &rect:rects){
            sum += (rect[2]-rect[0]+1)*(rect[3]-rect[1]+1);
            arr.push_back(sum);
        }
        distribution = uniform_int_distribution<int>(0,sum-1);
    }
    vector<int> pick() {
        int k = distribution(generator);
        int index = upper_bound(arr.begin(),arr.end(),k)- arr.begin()-1;
        k -= arr[index];
        int a = rects[index][0],b = rects[index][1],y = rects[index][3];
        int col = y - b + 1;
        return {a+k/col,b+k%col};  
    }
};
// 1739. 放置盒子 https://leetcode.cn/problems/building-boxes/
// 有一个立方体房间，其长度、宽度和高度都等于 n 个单位。请你在房间里放置 n 个盒子，每个盒子都是一个单位边长的立方体。放置规则如下：你可以把盒子放在地板上的任何地方。如果盒子 x 需要放置在盒子 y 的顶部，那么盒子 y 竖直的四个侧面都 必须 与另一个盒子或墙相邻。给你一个整数 n ，返回接触地面的盒子的 最少 可能数量。
int minimumBoxes(int n) {
    int h = pow(6l*n,1.0/3);
    long t = h*(h+1)/2;
    long tmp = t*(h+2)/3;
    if(tmp==n)
        return t;
    if(tmp>n){
        tmp -= t;
        t -= h;
        h -= 1;
    }
    long t0 = n - tmp;
    long t1 = sqrt(2*t0+0.25)-0.25;
    long t2 = t1*(t1+1)/2;
    if(t2<t0)
        return t+t1+1;
    return t+t1;
}
// 2183. 统计可以被 K 整除的下标对数目 https://leetcode.cn/problems/count-array-pairs-divisible-by-k/
// 给你一个下标从 0 开始、长度为 n 的整数数组 nums 和一个整数 k ，返回满足下述条件的下标对 (i, j) 的数目：0 <= i < j <= n - 1 且nums[i] * nums[j] 能被 k 整除。
long long countPairs(vector<int>& nums, int k) {
    function<int(int,int)> gcd = [&](int a,int b){
        return a<=b?(b%a==0?a:gcd(b%a,a)):gcd(b,a);
    };
    int size = nums.size();
    unordered_map<int,int> mp;
    long long ans = 0;
    for(auto &num:nums){
        if(((long long)num)*num%k==0)
            --ans;
        int t = gcd(num,k);
        auto i = mp.find(t);
        if(i==mp.end())
            mp.emplace(t,1);
        else
            i->second++;
    }
    for(auto &x:mp)
        for(auto &y:mp)
            if(((long long)x.first)*y.first%k==0)
                ans += ((long long)x.second) *y.second;
    return ans/2;
}
// 6096. 咒语和药水的成功对数 https://leetcode.cn/problems/successful-pairs-of-spells-and-potions/
// 给你两个正整数数组 spells 和 potions ，长度分别为 n 和 m ，其中 spells[i] 表示第 i 个咒语的能量强度，potions[j] 表示第 j 瓶药水的能量强度。同时给你一个整数 success 。一个咒语和药水的能量强度 相乘 如果 大于等于 success ，那么它们视为一对 成功 的组合。请你返回一个长度为 n 的整数数组 pairs，其中 pairs[i] 是能跟第 i 个咒语成功组合的 药水 数目
vector<int> successfulPairs(vector<int>& spells, vector<int>& potions, long long success) {
    sort(potions.begin(),potions.end());
    vector<int> ans;
    ans.reserve(spells.size());
    auto end = potions.end();
    for(auto& spell:spells){
        int need = (success+spell-1) / spell;
        ans.push_back(end-lower_bound(potions.begin(),potions.end(),need));
    }
    return ans;
}
// 6097. 替换字符后匹配 https://leetcode.cn/problems/match-substring-after-replacement/
// 给你两个字符串 s 和 sub 。同时给你一个二维字符数组 mappings ，其中 mappings[i] = [oldi, newi] 表示你可以替换 sub 中任意数目的 oldi 个字符，替换成 newi 。sub 中每个字符 不能 被替换超过一次。如果使用 mappings 替换 0 个或者若干个字符，可以将 sub 变成 s 的一个子字符串，请你返回 true，否则返回 false 。一个 子字符串 是字符串中连续非空的字符序列。
bool matchReplacement(string s, string sub, vector<vector<char>>& mappings) {
    int size1 = s.size(), size2 = sub.size();
    unordered_map<char,unordered_set<char>> mp;
    for(auto &maps:mappings)
        mp[maps[0]].emplace(maps[1]);
    for(int i=0;i<=size1-size2;++i){
        int j=0;
        for(;j<size2;++j)
            if(s[i+j]!=sub[j]&&!mp[sub[j]].count(s[i+j]))
                break;
        if(j==size2)
            return true;
    }
    return false;
}
// 5259. 计算应缴税款总额 https://leetcode.cn/problems/calculate-amount-paid-in-taxes/
// 给你一个下标从 0 开始的二维整数数组 brackets ，其中 brackets[i] = [upperi, percenti] ，表示第 i 个税级的上限是 upperi ，征收的税率为 percenti 。税级按上限 从低到高排序（在满足 0 < i < brackets.length 的前提下，upperi-1 < upperi）。税款计算方式如下：不超过 upper0 的收入按税率 percent0 缴纳接着 upper1 - upper0 的部分按税率 percent1 缴纳然后 upper2 - upper1 的部分按税率 percent2 缴纳以此类推给你一个整数 income 表示你的总收入。返回你需要缴纳的税款总额。与标准答案误差不超 10-5 的结果将被视作正确答案
double calculateTax(vector<vector<int>>& brackets, int income) {
    int prev = 0;
    double ans = 0;
    for(auto &bracket:brackets){
        if(income<=bracket[0]){
            ans += (income - prev)*((double)bracket[1]/100);
            break;
        }
        ans += (bracket[0] - prev)*((double)bracket[1]/100);
        prev = bracket[0];
    }
    return ans;
}
// 5270. 网格中的最小路径代价 https://leetcode.cn/problems/minimum-path-cost-in-a-grid/
// 给你一个下标从 0 开始的整数矩阵 grid ，矩阵大小为 m x n ，由从 0 到 m * n - 1 的不同整数组成。你可以在此矩阵中，从一个单元格移动到 下一行 的任何其他单元格。如果你位于单元格 (x, y) ，且满足 x < m - 1 ，你可以移动到 (x + 1, 0), (x + 1, 1), ..., (x + 1, n - 1) 中的任何一个单元格。注意： 在最后一行中的单元格不能触发移动。每次可能的移动都需要付出对应的代价，代价用一个下标从 0 开始的二维数组 moveCost 表示，该数组大小为 (m * n) x n ，其中 moveCost[i][j] 是从值为 i 的单元格移动到下一行第 j 列单元格的代价。从 grid 最后一行的单元格移动的代价可以忽略。grid 一条路径的代价是：所有路径经过的单元格的 值之和 加上 所有移动的 代价之和 。从 第一行 任意单元格出发，返回到达 最后一行 任意单元格的最小路径代价。
int minPathCost(vector<vector<int>>& grid, vector<vector<int>>& moveCost) {
    int m = grid.size(), n = grid[0].size();
        int dp[m][n];
        for(int i=0;i<n;++i)
            dp[0][i] = grid[0][i];
        for(int i=1;i<m;++i){
            for(int j=0;j<n;++j){
                dp[i][j] = 1e7;
                for(int k=0;k<n;++k){
                    dp[i][j] = min(dp[i][j],dp[i-1][k]+moveCost[grid[i-1][k]][j]+grid[i][j]);
                }
            }
        }
        int ans = INT32_MAX;
        --m;
        for(int i=0;i<n;++i)
            ans = min(ans,dp[m][i]);
        return ans;
}
// 5289. 公平分发饼干 https://leetcode.cn/problems/fair-distribution-of-cookies/
// 给你一个整数数组 cookies ，其中 cookies[i] 表示在第 i 个零食包中的饼干数量。另给你一个整数 k 表示等待分发零食包的孩子数量，所有 零食包都需要分发。在同一个零食包中的所有饼干都必须分发给同一个孩子，不能分开。分发的 不公平程度 定义为单个孩子在分发过程中能够获得饼干的最大总数。返回所有分发的最小不公平程度。
int distributeCookies(vector<int>& cookies, int k) {
    int size = cookies.size();
	int cnt[k];
	int ans = INT32_MAX;
	memset(cnt,0,sizeof(cnt));
	function<void(int)> dfs = [&](int i){
		if(i==size){
			int m = INT32_MIN;
			for(int j=0;j<k;++j)
				m = max(cnt[j],m);
			ans = min(m,ans);
			return ;
		}
		for(int j=0;j<k;++j){
			cnt[j] += cookies[i];
			dfs(i+1);
			cnt[j] -= cookies[i];
		}
	};
	dfs(0);
	return ans;
}
// 6094. 公司命名 https://leetcode.cn/problems/naming-a-company/
// 给你一个字符串数组 ideas 表示在公司命名过程中使用的名字列表。公司命名流程如下：从 ideas 中选择 2 个 不同 名字，称为 ideaA 和 ideaB 。交换 ideaA 和 ideaB 的首字母。如果得到的两个新名字 都 不在 ideas 中，那么 ideaA ideaB（串联 ideaA 和 ideaB ，中间用一个空格分隔）是一个有效的公司名字。否则，不是一个有效的名字。返回 不同 且有效的公司名字的数目。
long long distinctNames(vector<string>& ideas) {
    unordered_map<string,int> idea_str;
    for(auto &idea:ideas)
        idea_str[idea.substr(1)] |= 1 <<(idea[0]-'a');
    int cnt[26][26];
    memset(cnt,0,sizeof(cnt));
    long long ans = 0;
    for(auto &idea:idea_str){
        for(int i=0;i<26;++i){
            if(idea.second&(1<<i)){
                for(int j=0;j<26;++j)
                    if(!(idea.second&(1<<j)))
                        ++cnt[i][j];
            }
            else{
                for(int j=0;j<26;++j)
                    if((idea.second&(1<<j)))
                        ans += cnt[i][j];
            }
        }
    }
    return ans *2;
}
// 890. 查找和替换模式 https://leetcode.cn/problems/find-and-replace-pattern/
// 你有一个单词列表 words 和一个模式  pattern，你想知道 words 中的哪些单词与模式匹配。如果存在字母的排列 p ，使得将模式中的每个字母 x 替换为 p(x) 之后，我们就得到了所需的单词，那么单词与模式是匹配的。（回想一下，字母的排列是从字母到字母的双射：每个字母映射到另一个字母，没有两个字母映射到同一个字母。）返回 words 中与给定模式匹配的单词列表。你可以按任何顺序返回答案。
vector<string> findAndReplacePattern(vector<string>& words, string pattern) {
    int size = pattern.size();
    auto check =[](const string& w,const string& p){
        int size = p.size();
        unordered_map<char,char> mp;
        for(int i=0;i<size;++i){
            if(!mp.count(p[i]))
                mp[p[i]] = w[i];
            else if(mp[p[i]]!=w[i])
                return false;
        }
        return true;
    };
    vector<string> ans;
    for(auto& word:words)
        if(check(word,pattern)&&check(pattern,word))
            ans.push_back(word);
    return ans;
}
// 6. Z 字形变换 https://leetcode.cn/problems/zigzag-conversion/
// 将一个给定字符串 s 根据给定的行数 numRows ，以从上往下、从左到右进行 Z 字形排列。之后，你的输出需要从左往右逐行读取，产生出一个新的字符串，请你实现这个将字符串进行指定行数变换的函数：
string convert(string s, int numRows) {
    vector<string> arr(numRows,string());
    int size = s.size();
    int n = numRows*2-2;
    for(int i=0;i<size;++i){
        int m = i%n;
        if(m>=numRows)
            m = numRows - 1 - (m-numRows);
        arr[m].push_back(s[i]);
    }
    string ans = "";
    for(auto &str:arr)
        ans += str;
    return ans;
}
// 16. 最接近的三数之和 https://leetcode.cn/problems/3sum-closest/
// 给你一个长度为 n 的整数数组 nums 和 一个目标值 target。请你从 nums 中选出三个整数，使它们的和与 target 最接近。返回这三个数的和。假定每组输入只存在恰好一个解。
int threeSumClosest(vector<int>& nums, int target) {
    sort(nums.begin(),nums.end());
    int size = nums.size();
    int ans = 1e7;
    for(int i=0;i<size;++i){
        if(i>=0&&nums[i]==nums[i-1])
            continue;
        int j = i+1,k = size-1;
        while(j<k){
            int sum = nums[i] + nums[j] +nums[k];
            if(sum==target)
                return target;
            if(abs(sum-target)<abs(ans-target))
                ans = sum;
            if(sum>target)
                --k;
            else
                ++j;
        }
    }
    return ans;
}
// 51. N 皇后 https://leetcode.cn/problems/n-queens/
// 按照国际象棋的规则，皇后可以攻击与之处在同一行或同一列或同一斜线上的棋子。n 皇后问题 研究的是如何将 n 个皇后放置在 n×n 的棋盘上，并且使皇后彼此之间不能相互攻击。给你一个整数 n ，返回所有不同的 n 皇后问题 的解决方案。每一种解法包含一个不同的 n 皇后问题 的棋子放置方案，该方案中 'Q' 和 '.' 分别代表了皇后和空位。
vector<vector<string>> solveNQueens(int n) {
    vector<vector<string>> ans;
    int column = 0,diagonal1 = 0,diagonal2 =0; 
    vector<string> tem(n,string(n,'.'));
    function<void(int)> dfs = [&](int i){
        if(i==n){
            ans.push_back(tem);
            return ;
        }
        for(int j=0;j<n;++j){
            int c = 1<<j, d1 = 1<<(i-j+n),d2 = 1<<(i+j);
            if(column&c||d1&diagonal1||d2&diagonal2)
                continue;
            column |= c;
            diagonal1 |= d1;
            diagonal2 |= d2;
            tem[i][j] = 'Q';
            dfs(i+1);
            tem[i][j] = '.';
            column &= ~c;
            diagonal1 &= ~d1;
            diagonal2 &= ~d2;
        }
    };
    dfs(0);
    return ans;
}
// 76. 最小覆盖子串 https://leetcode.cn/problems/minimum-window-substring/
// 给你一个字符串 s 、一个字符串 t 。返回 s 中涵盖 t 所有字符的最小子串。如果 s 中不存在涵盖 t 所有字符的子串，则返回空字符串 "" 。
string minWindow(string s, string t) {
	int m = s.size(),n = t.size();
	unordered_map<char,int> ch_mp;
	for(auto& ch:t)
		ch_mp[ch]++;
	int l = 0, r = n; // [l,r)
	int need = ch_mp.size();
	auto end = ch_mp.end();
	for(int i=0;i<n;++i){
		auto iter = ch_mp.find(s[i]);
		if(iter!=end){
			--iter->second;
			if(iter->second==0)
				--need;
		}
	}
	int len = INT32_MAX,pos = -1;
	while(r<=m){
		while(need>0&&r<=m){
			auto iter = ch_mp.find(s[r++]);
			if(iter!=end){
				--iter->second;
				if(iter->second==0)
					--need;
			}
		}
		while(need==0){
			if(r-l<len){
				len = r - l;
				pos = l;
			}
			auto iter = ch_mp.find(s[l++]);
			if(iter!=end){
				++iter->second;
				if(iter->second==1)
					++need;
			}
		}
	}
	return (len==INT32_MAX)? "" : s.substr(pos,len);
}
// 30. 串联所有单词的子串 https://leetcode.cn/problems/substring-with-concatenation-of-all-words/
// 给定一个字符串 s 和一些 长度相同 的单词 words 。找出 s 中恰好可以由 words 中所有单词串联形成的子串的起始位置。注意子串要与 words 中的单词完全匹配，中间不能有其他字符 ，但不需要考虑 words 中单词串联的顺序。
vector<int> findSubstring(string s, vector<string>& words) {
    unordered_map<string,int> word_mp;
	int m = words.size(), n =words[0].size();
	for(int i=0;i<m;++i)
		word_mp[words[i]]++;
	int size = s.size();
	int all = m*n;
	int state = 0;
	auto& word0 = words[0];
	vector<int> ans; 
	auto end = word_mp.end();
	for(int i=0;i<=size-all;++i){
		auto index = word_mp.find(s.substr(i,n));
		if(index==end)
			continue;
		unordered_map<string,int> tem = word_mp;
		tem[index->first]--;
		int j=1;
		for(;j<m;++j){
			index = tem.find(s.substr(i+j*n,n));
			bool flag = false;
			if(index==end)
				break;
			if(index->second==0)
				break;
			index->second--;
		}
		if(j==m)
			ans.push_back(i);
	}
	return ans;
}
// 22. 括号生成 https://leetcode.cn/problems/generate-parentheses/
// 数字 n 代表生成括号的对数，请你设计一个函数，用于能够生成所有可能的并且 有效的 括号组合。
vector<string> generateParenthesis(int n) {
	string str(n*2,'(');
	vector<string> ans;
	int l_cnt = 0,r_cnt = 0; 
	function<void(int)> dfs = [&](int i){
		if(i==n*2){
			ans.push_back(str);
			return ;
		}
		if(l_cnt!=n){
			++l_cnt;
			dfs(i+1);
			--l_cnt;
		}
		if(l_cnt!=r_cnt){
			++r_cnt;
			str[i] = ')';
			dfs(i+1);
			str[i] = '(';
			--r_cnt;
		}
	};
	dfs(0);
	return ans;
}
// 84. 柱状图中最大的矩形 https://leetcode.cn/problems/largest-rectangle-in-histogram/
// 给定 n 个非负整数，用来表示柱状图中各个柱子的高度。每个柱子彼此相邻，且宽度为 1 。求在该柱状图中，能够勾勒出来的矩形的最大面积。
int largestRectangleArea(vector<int>& heights) {
    int size = heights.size();
    stack<int> stk;
	int left[size];
    int ans = 0;
    stk.emplace(-1);
	int pos,alt;
    for(int i=0;i<size;++i){	
		while((pos=stk.top())!=-1&&heights[pos]>=heights[i]){
			if((alt = (i-left[pos]-1)*heights[pos])>ans)
				ans = alt;
			stk.pop();
		}
		left[i] = stk.top();
		stk.emplace(i);
    }
	while((pos=stk.top())!=-1){
		if((alt = (size-left[pos]-1)*heights[pos])>ans)
			ans = alt;
		stk.pop();
	}
    return ans;
}
// 84. 柱状图中最大的矩形 https://leetcode.cn/problems/largest-rectangle-in-histogram/
// 给定 n 个非负整数，用来表示柱状图中各个柱子的高度。每个柱子彼此相邻，且宽度为 1 。求在该柱状图中，能够勾勒出来的矩形的最大面积。
int largestRectangleArea(vector<int>& heights) {
    int size = heights.size();
    stack<int> stk;
	int left[size];
    int ans = 0;
    stk.emplace(-1);
	int pos,alt;
    for(int i=0;i<size;++i){	
		while((pos=stk.top())!=-1&&heights[pos]>=heights[i]){
			if((alt = (i-left[pos]-1)*heights[pos])>ans)
				ans = alt;
			stk.pop();
		}
		left[i] = stk.top();
		stk.emplace(i);
    }
	while((pos=stk.top())!=-1){
		if((alt = (size-left[pos]-1)*heights[pos])>ans)
			ans = alt;
		stk.pop();
	}
    return ans;
}
// 85. 最大矩形 https://leetcode.cn/problems/maximal-rectangle/
// 给定一个仅包含 0 和 1 、大小为 rows x cols 的二维二进制矩阵，找出只包含 1 的最大矩形，并返回其面积。
int maximalRectangle(vector<vector<char>>& matrix) {
    int m = matrix.size(),n = matrix[0].size();
	vector<int> heights[2];
	for(int i=0;i<n;++i)
		heights[0][i] = matrix[0][i]-'0';
	int ans = largestRectangleArea(heights[0]);
	for(int i=1;i<m;++i){
		int curr = i&1,prev = curr^1;
		for(int j=0;j<n;++j)
			heights[curr][j] = (matrix[i][j]=='1'? heights[prev][j]+1:0);
		ans = max(ans,largestRectangleArea(heights[curr]));
	}
	return ans;
}
// 29. 两数相除 https://leetcode.cn/problems/divide-two-integers/
// 给定两个整数，被除数 dividend 和除数 divisor。将两数相除，要求不使用乘法、除法和 mod 运算符。返回被除数 dividend 除以除数 divisor 得到的商。整数除法的结果应当截去（truncate）其小数部分，例如：truncate(8.345) = 8 以及 truncate(-2.7335) = -2
int divide(int dividend, int divisor) {
	if (dividend == INT_MIN){
		if (divisor == 1)
			return INT_MIN;
		if (divisor == -1)
			return INT_MAX;
	}
	if (divisor == INT_MIN)
		return dividend == INT_MIN ? 1 : 0;
	if (dividend == 0)
		return 0;        
	bool flag = false;
	if (dividend > 0) {
		dividend = -dividend;
		flag = !flag;
	}
	if (divisor > 0) {
		divisor = -divisor;
		flag = !flag;
	}
	vector<int> candidates = {divisor};
	candidates.reserve(32);
	while (candidates.back() >= dividend - candidates.back())
		candidates.push_back(candidates.back() + candidates.back());
	int ans = 0;
	for (int i = candidates.size() - 1; i >= 0; --i) {
		if (candidates[i] >= dividend) {
			ans += (1 << i);
			dividend -= candidates[i];
		}
	}
	return flag ? -ans : ans;
}
// 2003. 每棵子树内缺失的最小基因值 https://leetcode.cn/problems/smallest-missing-genetic-value-in-each-subtree/
// 有一棵根节点为 0 的 家族树 ，总共包含 n 个节点，节点编号为 0 到 n - 1 。给你一个下标从 0 开始的整数数组 parents ，其中 parents[i] 是节点 i 的父节点。由于节点 0 是 根 ，所以 parents[0] == -1 。总共有 105 个基因值，每个基因值都用 闭区间 [1, 105] 中的一个整数表示。给你一个下标从 0 开始的整数数组 nums ，其中 nums[i] 是节点 i 的基因值，且基因值 互不相同 。请你返回一个数组 ans ，长度为 n ，其中 ans[i] 是以节点 i 为根的子树内 缺失 的 最小 基因值。节点 x 为根的 子树 包含节点 x 和它所有的 后代 节点。
vector<int> smallestMissingValueSubtree(vector<int>& parents, vector<int>& nums) {
	bitset<100002> exist;
	int m = INT32_MIN;
	int size = parents.size();
	vector<int> ans(size,1);
	int one_index = 0;
	for(;one_index<size;++one_index)
		if(nums[one_index]==1)
			break;
	if(one_index==size)
		return ans;
	vector<vector<int>> tree(size,vector<int>());
	for(int i=1;i<size;++i)
		tree[parents[i]].emplace_back(i);
	function<void(int,int)> dfs = [&](int curr,int prev){
		exist.set(nums[curr]);
        for(auto &next:tree[curr])
            if(next!=prev)
                dfs(next,curr);
	};
	int lack = 1;
	int prev = -1;
	while(one_index!=-1){
		dfs(one_index,prev);
		while(exist.test(lack))
			++lack;
		ans[one_index] = lack;
		prev = one_index;
		one_index = parents[one_index];
	}
	return ans;
}
// 710. 黑名单中的随机数 https://leetcode.cn/problems/random-pick-with-blacklist/
// 给定一个整数 n 和一个 无重复 黑名单整数数组 blacklist 。设计一种算法，从 [0, n - 1] 范围内的任意整数中选取一个 未加入 黑名单 blacklist 的整数。任何在上述范围内且不在黑名单 blacklist 中的整数都应该有 同等的可能性 被返回。优化你的算法，使它最小化调用语言 内置 随机函数的次数。实现 Solution 类:Solution(int n, int[] blacklist) 初始化整数 n 和被加入黑名单 blacklist 的整数int pick() 返回一个范围为 [0, n - 1] 且不在黑名单 blacklist 中的随机整数
class Solution {
	unordered_map<int,int> mp;
	int m;
public:
    Solution(int n, vector<int>& blacklist) {
		int size = blacklist.size();
		unordered_set<int> st;
		mp.reserve(size);
		m = n - size;
		int i = m;
		for(auto& black:blacklist)
			if(black>=m)
				st.emplace(black);
		for(auto& black:blacklist){
			if(black>=m)
				continue;
			while(st.count(i))
				++i;
			mp[black] = i++;
		}
    }
    int pick() {
		int p = rand()%m;
		return mp.count(p)? mp[p]:p;
    }
};