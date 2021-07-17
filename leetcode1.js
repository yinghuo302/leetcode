/*
 * @Author: zanilia
 * @Date: 2021-06-27 21:38:56
 * @LastEditTime: 2021-07-15 10:55:30
 * @Descripttion:
*/
function ListNode(val) {
    this.val = val;
    this.next = null;
}
// 编写代码，移除未排序链表中的重复节点。保留最开始出现的节点。
/*
 * @param {ListNode} head
 * @return {ListNode}
*/
var removeDuplicateNodes = function(head) {
    if(head===null)
        return head;
    var p = head
    const val_existed = new Set();
    val_existed.add(head.val);
    while(p.next){
        if(!val_existed.has(p.next.val)){
            val_existed.add(p.next.val);
            p = p.next;
        }
        else
            p.next = p.next.next;
    }
    return head;
};
/*
 * @param {string} pattern
 * @param {string} s
 * @return {boolean}
 */
var wordPattern = function(pattern, s) {
    var char_to_string = new Map();
    var string_to_char = new Map();
    var words = s.split(' ');
    if(words.length!=pattern.length)
        return false;
    var size = pattern.length;
    for(let i=0;i<size;++i){
        if((char_to_string.has(pattern[i])&&char_to_string.get(pattern[i])!=words[i])||
                (string_to_char.has(words[i])&&string_to_char.get(words[i])!=pattern[i]))
            return false;
        char_to_string.set(pattern[i],words[i]);
        string_to_char.set(words[i],pattern[i]);
    }
    return true;
};
// 给你一个 无重叠的 ，按照区间起始端点排序的区间列表。
// 在列表中插入一个新的区间，你需要确保列表中的区间仍然有序且不重叠（如果有必要的话，可以合并区间）。
/*
 * @param {number[][]} intervals
 * @param {number[]} newInterval
 * @return {number[][]}
*/
var insert = function(intervals, newInterval) {
    const res = [];
    const size = intervals.length;
    var i=0;
    for(;i<size&&intervals[i][1]<newInterval[0];++i)
        res.push(intervals[i]);
    for(;i<size&&intervals[i][0]<=newInterval[1];++i){
        newInterval[0] = newInterval[0]>intervals[i][0]? intervals[i][0]:newInterval[0];
        newInterval[1] = newInterval[1]>intervals[i][1]? newInterval[1]:intervals[i][1];
    }
    res.push(newInterval);
    for(;i<size;++i)
        res.push(intervals[i]);
    return res;
};
// 给你一个数组 routes ，表示一系列公交线路，其中每个 routes[i] 表示一条公交线路，第 i 辆公交车将会在上面循环行驶。
// 例如，路线 routes[0] = [1, 5, 7] 表示第 0 辆公交车会一直按序列 1 -> 5 -> 7 -> 1 -> 5 -> 7 -> 1 -> ... 这样的车站路线行驶。
// 现在从 source 车站出发（初始时不在公交车上），要前往 target 车站。 期间仅可乘坐公交车。
// 求出 最少乘坐的公交车数量 。如果不可能到达终点车站，返回 -1 。
/*
 * @param {number[][]} routes
 * @param {number} source
 * @param {number} target
 * @return {number}
*/
var numBusesToDestination = function(routes, source, target) {
    if(source==target)
        return 0;
    const size = routes.length;
    const edge = new Array(n).fill(0).map(() => new Array(n).fill(0));
    const rec = new Map();
    for(let i=0;i<size;++i){
        for(const site of routes[i]){
            let list = (rec.get(site)||[]);
            for(const j of list)
                edge[i][j] = edge[j][i] = true;
            list.push(i);
            rec.set(site,list);
        }
    }
    const distance = new Array(n).fill(-1);
    const queue = [];
    for (const position of (rec.get(source) || [])) {
        distance[position] = 1;
        queue.push(position);
    }
    while (queue.length) {
        const x = queue.shift();
        for (let y = 0; y < n; y++) {
            if (edge[x][y] && distance[y] === -1) {
                distance[y] = distance[x] + 1;
                queue.push(y);
            }
        }
    }
    let ret = Number.MAX_VALUE;
    for (const position of (rec.get(target) || [])) {
        if (dis[position] !== -1) {
            ret = Math.min(ret, distance[position]);
        }
    }
    return ret === Number.MAX_VALUE ? -1 : ret;
};
// !error return NaN
// 给你一个points 数组，表示 2D 平面上的一些点，其中 points[i] = [xi, yi] 。
// 连接点 [xi, yi] 和点 [xj, yj] 的费用为它们之间的 曼哈顿距离 ：|xi - xj| + |yi - yj| ，其中 |val| 表示 val 的绝对值。
// 请你返回将所有点连接的最小总费用。只有任意两点之间 有且仅有 一条简单路径时，才认为所有点都已连接。
function UnionSet(n){
    this.size = n;
    this.rank = new Array(n).fill(1);
    this.f = new Array(n);
    for(let i=0;i<n;++i)
        this.f[i] = i;
    this.find = function(x){
        if (this.f[x] === x) 
            return x;
        this.f[x] = this.find(this.f[x]);
        return this.f[x];
    }
    this.join = function(x,y){
        let fx = this.find(x), fy = this.find(y);
        if (fx === fy) 
            return false;
        if (this.rank[fx] < this.rank[fy])
            [fx, fy] = [fy, fx];
        this.rank[fx] += this.rank[fy];
        this.f[fy] = fx;
        return true;
    }
}
/*
 * @param {number[][]} points
 * @return {number}
*/
var minCostConnectPoints = function(points) {
    const size = points.length;
    const unoin_set = new UnionSet(size);
    const edge = [];
    var distance = function(i,j){
        return Math.abs(points[i][0]-points[j][0])+Math.abs(points[i][1]-points[j][1]);
    }
    for(let i=0;i<size;++i)
        for(let j=i+1;j<size;++j)
            edge.push([i,j,distance[i,j]]);
    edge.sort((a,b)=>a[2]-b[2]);
    var ret = 0,num = 1;
    for(const [i,j,length] of edge){
        if(unoin_set.join(i,j)){
            ret += length;
            ++num;
            if(num==size)
                break;
        }
    }
    return ret;
};
// 给你二叉树的根结点 root ，请你将它展开为一个单链表：
// 展开后的单链表应该同样使用 TreeNode ，其中 right 子指针指向链表中下一个结点，而左子指针始终为 null 。
// 展开后的单链表应该与二叉树 先序遍历 顺序相同。
function TreeNode(val, left, right) {
    this.val = (val===undefined ? 0 : val)
    this.left = (left===undefined ? null : left)
    this.right = (right===undefined ? null : right)
}
var flatten = function(root) {
    if(!root)
        return ;
    var p = root.right;
    root.right = root.left;
    flatten(root.left);
    root.left = null;
    flatten(p);
    while(root.right)
        root = root.right;
    root.right = p;
};
/*
 * @param {number} columnNumber
 * @return {string}
*/
var convertToTitle = function(columnNumber) {
    let ret = [];
    while(columnNumber!=0){
        let num1 = (columnNumber-1)%26+1;
        columnNumber /= 26;
        ret.push(String.fromCharCode(num1-1+'A'.charCodeAt()));
        columnNumber = Math.floor((columnNumber-a0)/26);
    }
    ret.reverse();
    return ret.join('');
};
// Remove duplicate elements
/*
 * @param {number[]} nums
 * @param {number} val
 * @return {number}
*/
var removeElement = function(nums, val) {
    var left = 0,right = nums.length;
    while(left<right){
        if(nums[left]===val)
            nums[left] = nums[--right];
        else
            ++left;
    }
    return left;
};
// 二叉树序列化和反序列化
/*
 * Encodes a tree to a single string.
 * @param {TreeNode} root
 * @return {string}
*/
var serialize = function(root) {
    ret = new String();
    serialize_assist(root,ret);
    return ret;
};
var serialize_assist = function(root,res){
    if(!root)
        res += ",null";
    else{
        res += (root.val + '' + ",");
        serialize_assist(root.left,res);
        serialize_assist(root.right,res);
    }
};
/*
 * Decodes your encoded data to tree.
 *
 * @param {string} data
 * @return {TreeNode}
*/
var deserialize = function(data) {
    dataArray = data.split(',');
    return deserialize_assist(dataArray,root);
};

var deserialize_assist = function(dataArray){
    if(dataArray[0]=='null'){
        dataArray.shift();
        return null;
    }
    var root = new TreeNode(parseInt(dataArray[0]));
    root.left = deserialize_assist(dataArray);
    root.right = deserialize_assist(dataArray);
    return root;
}
// coins可以买到的最多的icecrean数
/*
 * Your functions will be called as such:
 * deserialize(serialize(root));
*/
var maxIceCream = function(costs, coins) {
    costs.sort();
    var size = costs.length,num = 0;
    for(let i=0;i<size;++i){
        coins -= costs[i];
        if(coins<0)
            return num;
        ++num;
    }
    return num;
};
// 如果字符串中不含有任何 'aaa'，'bbb' 或 'ccc' 这样的字符串作为子串，那么该字符串就是一个「快乐字符串」。
// 给你三个整数 a，b ，c，请你返回 任意一个 满足下列全部条件的字符串 s：
/*
 * @param {number} a
 * @param {number} b
 * @param {number} c
 * @return {string}
*/
var longestDiverseString = function(a, b, c) {
    var num = [[a,'a'],[b,'b'],[c,'c']],ret = "";
    var sort_order = function(a,b){return b[0]-a[0];}
    num.sort(sort_order);
    while(num[0][0]!=0){
        if(num[0][0]===1){
            ret += num[0][1];
            --num[0][0];
        }
        else{
            ret += num[0][1]
            ret += num[0][1];
            num[0][0] -= 2;
        }
        num.sort(sort_order);
    }
    return ret;
};
// 数组排序
/*
 * @param {number[]} nums
 * @return {number[]}
*/
var sortArray = function(nums) {

};
// 给定一个字符串，请将字符串里的字符按照出现的频率降序排列。
// 桶排序
/**
 * @param {string} s
 * @return {string}
*/
var frequencySort = function(s) {
    const frequency = new Map();
    var size = s.length;
    var max_frequency = 0;
    for(let i=0;i<size;++i){
        let frequency_tem = (frequency.get(ch)||0)+1;
        frequency.set(s[i],frequency_tem);
        if(frequency_tem>max_frequency)
            max_frequency = frequency_tem;
    }
    const bucket = new Array(max_frequency+1).fill(null);
    for(const [ch,ch_frequency] of frequency){
        if(!bucket[ch_frequency])
            bucket[ch_frequency] = new ListNode(ch);
        else{
            let p = bucket[ch_frequency];
            while(p.next)
                p = p.next;
            p.next = new ListNode(ch);
        }
    }
    const ret = "";
    for(let i= max_frequency;i>=0;--i){
        let p = bucket[i];
        while(p){
            for(let j = 0;j<i;++j)
                ret += p.val;
            p = p.next;
        }
    }
    return ret;
};
// 判断两个数组是否能通过交换变为同一个数组
/*
 * @param {number[]} target
 * @param {number[]} arr
 * @return {boolean}
 */
var canBeEqual = function(target, arr) {
    const arr_num_count = new Map(),target_num_count = new Map();
    let size = target.length;
    for(let i =0;i<size;++i){
        arr_num_count.set(arr[i],(arr_num_count.get(arr[i])||0)+1);
        target_num_count.set(target[i],(target_num_count.get(target[i])||0)+1);
    }
    for(const [num,count] of arr_num_count){
        if(target_num_count.get(num)!=count)
            return false;
    }
    return true;
};
/*
 * @param {number[]} nums
 * @return {number}
*/
var partitionDisjoint = function(nums) {

};
/*
 * @param {number[]} nums
 * @return {number[]}
*/
var findErrorNums = function(nums) {
    var size = nums.length,ret = new Array(2),counter = new Array(size+1).fill(0),;
    for(const num of nums)
        ++counter[num];
    for(let i=1;i<=size;++i){
        if(counter[i]===0)
            num[1] = i;
        if(counter[i]===2)
            num[i]= i;
    }
    return ret;
};
// 我们把数组 A 中符合下列属性的任意连续子数组 B 称为 “山脉”：B.length >= 3
// 存在 0 < i < B.length - 1 使得 B[0] < B[1] < ... B[i-1] < B[i] > B[i+1] > ... > B[B.length - 1]
// （注意：B 可以是 A 的任意子数组，包括整个数组 A。） 给出一个整数数组 A，返回最长 “山脉” 的长度。
/*
 * @param {number[]} arr
 * @return {number}
*/
 var longestMountain = function(arr) {
    var size = arr.length;
    if(size<=2)
        return 0;
    var left = new Array(size).fill(0),right = new Array(size).fill(0);
    for(let i =1;i<size-1;++i){
        for(let j=i-1;j>=0;--j){
            if(arr[j]<arr[j+1])
                ++left[i];
            else
                break;
        }
        for(let j=i;j<size;++j){
            if(arr[j]>arr[j+1])
                ++right[i];
            else
                break;
        }
    }
    var ret = 0;
    for(let i=1;i<size-1;++i)
        if(left[i]!=0&&right[i]!=0)
            ret = Math.max(ret,left[i]+right[i]+1);
    return ret;
};
// 寻找主要元素，在数组出现次数大于n/2
/*
 * @param {number[]} nums
 * @return {number}
*/
var majorityElement = function(nums) {
    var cnt = 0,tmp = 0;
    for(const num of nums){
        if(cnt===0){
            cnt = 1;
            tmp = num;
        }
        else
            cnt += (num===tmp? 1: -1);
    }
    cnt = 0;
    for(const num of nums)
        if(num===tmp)
            ++cnt;
    if(cnt>nums.length/2)
        return tmp;
    return -1;
};
// 给定一个长度为偶数的整数数组arr,只有对arr进行重组后可以满足“对于每个0<=i<len(arr)/2,都有arr[2*i+1]=2*arr[2*i]”时,返回true
/*
 * @param {number[]} arr
 * @return {boolean}
*/
var canReorderDoubled = function(arr){
    const cntmap = new Map();
    for(const num of arr)
        cntmap.set(num,cntmap.get(num)?cntmap.get(num)+1:1);
    arr.sort((a,b)=>{return Math.abs(a) - Math.abs(b);});
    for(const num of arr){
        if (cntmap.get(num) == 0) 
            continue;
        if (cntmap.get(2 * num) === 0 || cntmap.get(2 * num) == undefined) 
            return false;
        cntmap.set(num, cntmap.get(num) - 1);
        cntmap.set(2 * num, cntmap.get(2 * num) - 1);
    }
    return true;
};
// 超市里正在举行打折活动，每隔n个顾客会得到discount的折扣。
/*
 * @param {number} n
 * @param {number} discount
 * @param {number[]} products
 * @param {number[]} prices
*/
var Cashier = function(n, discount, products, prices) {
    this.n = n;
    this.products_map = new Map();
    this.customer = 0;
    var size = products.length;
    for(let i=0;i<size;++i)
        this.products_map.set(products[i],prices[i]);
    this.discount = 1 - discount/100;
};

/** 
 * @param {number[]} product 
 * @param {number[]} amount
 * @return {number}
 */
Cashier.prototype.getBill = function(product, amount) {
    var product_num = product.length;
    var sum = 0;
    for(let i =0;i<product_num;++i)
        sum += this.products_map.get(product[i])*amount[i];
    ++this.customer;
    if(this.customer === this.n){
        this.customer = 0;
        return sum * this.discount;
    }
    return sum;
};
/*
 * Your Cashier object will be instantiated and called as such:
 * var obj = new Cashier(n, discount, products, prices)
 * var param_1 = obj.getBill(product,amount)
*/
// 最大子数组的和
/*
 * @param {number[]} nums
 * @return {number}
*/
var maxSubArray = function(nums) {
    const dp = new Array(nums.length);
    var size = nums.length;
    dp[0] = nums[0];
    for(let i =1;i<size;++i){
        if(dp[i-1]>=0)
            dp[i] = dp[i-1] + nums[i];
        else
            dp[i] = nums[i];
    }
    return Math.max(...dp);
};
// 
/*
 * @param {number[]} nums
 * @return {number}
*/
var findShortestSubArray = function(nums) {
    var size = nums.length;
    var cnt_map = new Map();
    for(let i = 0;i<size;++i){
        let tmp = cnt_map.get(nums[i]);
        if(tmp){
            ++tmp[0];
            tmp[2] = i;
            cnt_map.set(nums[i],tmp)
        }
        else
            cnt_map.set(nums[i],[1,i,0]);
    }
    var max_cnt = 0,min_len = 0;
    for(const [num,cnt] of cnt_map){
        if(cnt[0]>max_cnt){
            max_cnt = cnt[0];
            min_len = cnt[2] - cnt[1];
        }
        else if(cnt[0]===max_cnt){
            max_cnt = cnt[0];
            if(min_len<=cnt[2]-cnt[1]){
                min_len = cnt[2] - cnt[1];
                console.log(min_len);
            }
        }
    }    
    return min_len;
};
/*
 * @param {number[]} nums
 * @return {number}
*/
var numberOfArithmeticSlices = function(nums) {
    var dp = 0,sum = 0;
    for (let i = 2; i < nums.length;++i) {
        if (nums[i]-nums[i-1]==nums[i-1]-nums[i-2]) {
            dp = 1 + dp;
            sum += dp;
        }
        else
            dp = 0;
    }
    return sum;
};
/**
 * // This is the interface that allows for creating nested lists.
 * // You should not implement it, or speculate about its implementation
 * function NestedInteger() {
 *
 *     Return true if this NestedInteger holds a single integer, rather than a nested list.
 *     @return {boolean}
 *     this.isInteger = function() {
 *         ...
 *     };
 *
 *     Return the single integer that this NestedInteger holds, if it holds a single integer
 *     Return null if this NestedInteger holds a nested list
 *     @return {integer}
 *     this.getInteger = function() {
 *         ...
 *     };
 *
 *     Set this NestedInteger to hold a single integer equal to value.
 *     @return {void}
 *     this.setInteger = function(value) {
 *         ...
 *     };
 *
 *     Set this NestedInteger to hold a nested list and adds a nested integer elem to it.
 *     @return {void}
 *     this.add = function(elem) {
 *         ...
 *     };
 *
 *     Return the nested list that this NestedInteger holds, if it holds a nested list
 *     Return null if this NestedInteger holds a single integer
 *     @return {NestedInteger[]}
 *     this.getList = function() {
 *         ...
 *     };
 * };
 */
/*
 * @param {string} s
 * @return {NestedInteger}
*/
var deserialize = function(s) {

};
// 
/*
 * Initialize your data structure here.
*/
var TimeMap = function() {
    this.time_map = new Map();
};
/* 
 * @param {string} key 
 * @param {string} value 
 * @param {number} timestamp
 * @return {void}
*/
TimeMap.prototype.set = function(key, value, timestamp) {
    var tmp = this.time_map.get(key);
    if(tmp)
        tmp.push([value,timestamp]);
    else
        this.time_map.set(key,[[value,timestamp]]);
};

/** 
 * @param {string} key 
 * @param {number} timestamp
 * @return {string}
 */
TimeMap.prototype.get = function(key, timestamp) {
    var tmp = this.time_map.get(key);
    if(tmp){
        let left = 0,right = tmp.length;
        while(left<=right){
            let mid = Math.floor((right-left)/2)+left;
            if(tmp[mid][1]>timestamp) 
                right = mid - 1;
            else if(tmp[mid][1] < timestamp) 
                left = mid + 1;
            else
                return tmp[mid][0];
        }
    }
    return "";
};

/*
 * Your TimeMap object will be instantiated and called as such:
 * var obj = new TimeMap()
 * obj.set(key,value,timestamp)
 * var param_2 = obj.get(key,timestamp)
*/
// 
/*
 * @param {number[]} heights
 * @return {number}
*/
var heightChecker = function(heights) {
    const arr = new Array(101).fill(0);
    for(const height of heights)
        ++arr[height];
    var count = 0,size = arr.length;
    for(let i=1,j=0;i<size;++i)
        while (arr[i]-- > 0) 
            if (heights[j++] != i) 
                ++count;
    return count;
};
/*
 * Definition for a binary tree node.
 * function TreeNode(val) {
 *     this.val = val;
 *     this.left = this.right = null;
 * }
*/
// 树的深度
/*
 * @param {TreeNode} root
 * @return {number}
*/
var maxDepth = function(root) {
    if(!root)
        return 0;
    return Math.max(maxDepth(root.left),maxDepth(root.right))+1;
};
// 
/*
 * @param {number} n
 * @param {number[]} primes
 * @return {number}
*/
var nthSuperUglyNumber = function(n, primes) {
    const size = primes.length;
    const arr= new Array(size).fill(1);
    const ugly_num= new Array(n+1).fill(0);
    ugly_num[1]=1;
    for(let i=2;i<=n;++i){
        let min=Infinity
        for(let j=0;j<size;++j)
            min=Math.min(min,ugly_num[arr[j]]*primes[j])
        for(let k=0;k<size;++k)
            if(min===ugly_num[arr[k]]*primes[k])
                ++arr[k];
        ugly_num[i]=min;
    }
    return ugly_num[n];
};
/*
 * @param {number[]} machines
 * @return {number}
*/
var findMinMoves = function(machines) {
    var average = 0,size = machines.length;
    for(const machine of machines)
        average += machine;
    if(average%size!=0)
        return -1;
    average = Math.floor(average/size);
    for(let i=0;i<size;++i)
        machines[i] -= average;
    var cur_sum = 0, max_sum = 0, tmp_res = 0, res = 0;
    for (const m of machines) {
        cur_sum += m;
        max_sum = Math.max(max_sum, Math.abs(cur_sum));
        tmp_res = Math.max(max_sum, m);
        res = Math.max(res, tmp_res);
    }
    return res;
}
//h 指数的定义: “h 代表“高引用次数”（high citations），一名科研人员的 h 指数是指他（她）的 （N 篇论文中）
// 总共有 h 篇论文分别被引用了至少 h 次。（其余的 N - h 篇论文每篇被引用次数不多于 h 次。）"
// citations为引用次数
/*
 * @param {number[]} citations
 * @return {number}
*/
var hIndex = function(citations) {
    var size = citations.length;
    const arr = new Array(size+1).fill(0);
    for(const citation of citations){
        if(citation<=size)
            ++arr[citation];
        else
            ++arr[n];
    }
    var total = 0;
    for(let i =n;i>=0;++i){
        total += arr[i];
        if(total>=i)
            return i;
    }
    return 0;
};
// citations为引用次数，已经为升序
/*
 * @param {number[]} citations
 * @return {number}
*/
var hIndex = function(citations) {
    var n = citations.length;
    var left =0, right = n-1;
    while(left<=right){
        let mid=left+Math.floor((right-left)/2);
        if (citations[mid] >=n-mid) 
            right = mid - 1;
        else
            left = mid + 1;
    }
    return n-left;
};
/*
 * @param {number[]} arr
 * @return {number}
*/
var sumOddLengthSubarrays = function(arr) {
    var res = 0,size = arr.length;
    for(let i=0;i<size;++i){
        var left = i + 1, right=size-i,
            left_even= Math.floor((left+1)/2),right_even=Math.floor((right+1)/2),
            left_odd=Math.floor(left/2),right_odd=Math.floor(right/2);
        res += (left_even * right_even + left_odd * right_odd) * arr[i];
    }
    return res;
};
// 二分搜索
/*
 * @param {number[]} nums
 * @param {number} target
 * @return {number}
*/
var search = function(nums, target) {
    var left =0,right = nums.length-1;
    while(left<=right){
        let mid = Math.floor((right-left)/2)+left;
        if(nums[mid]===target)
            return mid;
        if(nums[mid]<target)
            left = mid +1;
        else
            right = mid-1;
    }
    return -1;
};
/**
 * Definition for isBadVersion()
 * 
 * @param {integer} version number
 * @return {boolean} whether the version is bad
 * isBadVersion = function(version) {
 *     ...
 * };
 */
/*
 * @param {function} isBadVersion()
 * @return {function}
*/
var solution = function(isBadVersion) {
    /**
     * @param {integer} n Total versions
     * @return {integer} The first bad version
     */
    return function(n) {
        if(n===1)
            return 1;
        var left = 1,right = n;
        while(left+1!=right){
            let mid = Math.floor((right-left)/2)+left;
            if(isBadVersion(mid))
                right = mid;
            else
                left = mid;
        }
        return right;
    };
};
/*
 * @param {number[]} nums
 * @param {number} target
 * @return {number}
*/
var searchInsert = function(nums, target) {
    var left = 0,right = nums.length;
    if(!right||nums[0]>=target)
        return 0;
    --right;
    while(right!=left+1){
        let mid = Math.floor((right-left)/2)+left;
        if(nums[mid]===target)
            return mid;
        if(nums[mid]<target)
            left = mid;
        else
            right = mid;
    }
    if(nums[right]===target)
        return right;
    else
        return left;
};
// 三数之和为0的三元组
/*
 * @param {number[]} nums
 * @return {number[][]}
*/
var threeSum = function(nums) {
    // nums.sort();
    // var ans = [],size = nums.length;
    // for(let first =0;first<size;++first){
    //     if(first>0&&nums[first]===nums[first-1])
    //         continue;
    //     let third = size-1,target = -nums[first];
    //     for(let second = first+1;second<size;++second){
    //         if(second>first+1&&nums[second]===nums[second-1])
    //             continue;
    //         while (second<third&&nums[second]+nums[third]>target)
    //             --third;
    //         if (second === third)
    //             break;
    //         if (nums[second] + nums[third] === target) 
    //             ans.push([nums[first],nums[second],nums[third]]);
            
    //     }
    // }
    // return ans;
    let ans = [];
    const size = nums.length;
    if(nums==null||size < 3)
        return ans;
    nums.sort((a, b) => a - b); 
    for (let i=0;i<size;++i) {
        if(nums[i]>0) 
            break;
        if(i > 0 && nums[i] == nums[i-1]) 
            continue;
        let left = i+1,right = size-1;
        while(left<right){
            let sum = nums[i] + nums[left] + nums[right];
            if(sum===0){
                ans.push([nums[i],nums[left],nums[right]]);
                while (left<right && nums[left] == nums[left+1]) 
                    left++; 
                while (left<right && nums[right] == nums[right-1]) 
                    right--; 
                left++;
                right--;
            }
            else if (sum < 0) 
                left++;
            else if (sum > 0) 
                right--;
        }
    }        
    return ans;
};
// 给你一个按非递减顺序排序的整数数组nums,返回每个数字的平方组成的新数组,要求也按非递减顺序排序。
/*
 * @param {number[]} nums
 * @return {number[]}
*/
var sortedSquares = function(nums) {
    var negative = -1,size = nums.length;
    for(let i = 0;i<size;++i)
        if(nums[i]<0)
            ++negative;
        else
            break;
    var ans = new Array(size);
    var left = negative,right = negative +1,j=0;
    while(left>=0&&right<size){
        if(nums[right]+nums[left]>=0){
            ans[j]=nums[left]*nums[left];
            --left;
            ++j;
        }
        else{
            ans[j] = nums[right]*nums[right];
            ++right;
            ++j;
        }
    }
    while(left>=0){
        ans[j]= nums[left]*nums[left];
        --left;
        ++j;
    }
    while(right<size){
        ans[j]= nums[right]*nums[right];
        ++right;
        ++j;
    }
    return ans;
};
/*
 * @param {number[]} nums
 * @param {number} k
 * @return {void} Do not return anything, modify nums in-place instead.
*/
var rotate = function(nums, k) {
    var size = nums.length;
    k %= size;
    const gcd = (x, y) => y ? gcd(y, x % y) : x;
    var count = gcd(size,k);
    for(let i =0;i<count;++i){
        let current = i;
        let prev = nums[i];
        do {
            let next = (current + k) % size;
            let temp = nums[next];
            nums[next] = prev;
            prev = temp;
            current = next;
        } while (i !== current);
    }
};
/*
 * @param {number[]} nums
 * @return {void} Do not return anything, modify nums in-place instead.
*/
var sortColors = function(nums) {
    var left = 0,right = nums.length-1,tem;
    for (let i=0; i<=right;++i) {
        while(i<=right&&nums[i]===2) {
            tem = nums[i];
            nums[i] = nums[right];
            nums[right] = tem;
            --right;
        }
        if (nums[i] == 0) {
            tem = nums[i];
            nums[i] = nums[left];
            nums[left] = tem;
            ++left;
        }
    }
};
/*
 * @param {number[][]} intervals
 * @return {number[][]}
*/
var merge = function(intervals) {
    var size = intervals.length;
    if(size===0)
        return [];
    intervals.sort();
    var merged = new Array();
    for(let i =0;i<size;++i){
        let L = intervals[i][0], R = intervals[i][1];
        if (merged.length!==0 || merged[merged.length-1][1] < L)
            merged.push([L, R]);
        else
            merged[merged.length-1][1] = max(merged[merged.length-1][1], R);
    }
    return merged;
};
// 返回杨辉三角的第rowindex行
/*
 * @param {number} rowIndex
 * @return {number[]}
*/
var getRow = function(rowIndex) {
    const ans = new Array(rowIndex + 1);
    ans[0] = 1;
    for (let i=1; i<=rowIndex;++i) 
        ans[i] = ans[i-1]*(rowIndex-i+1)/i;
    return ans;
};