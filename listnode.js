/*
 * @Author: zanilia
 * @Date: 2021-06-27 21:38:56
 * @LastEditTime: 2021-06-28 23:44:01
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