/*
 * @Author: zanilia
 * @Date: 2021-07-23 11:45:23
 * @LastEditTime: 2021-07-29 12:30:33
 * @Descripttion: 
*/
function ListNode(val, next) {
    this.val = (val===undefined ? 0 : val)
    this.next = (next===undefined ? null : next)
}
/*
 * Initialize your data structure here.
*/
var MyLinkedList = function() {
    this.head = null;
    this.tail = null;
    this.size = 0;  
};
/*
 * Get the value of the index-th node in the linked list. If the index is invalid, return -1. 
 * @param {number} index
 * @return {number}
*/
MyLinkedList.prototype.get = function(index) {
    if(index>=this.size||index<0)
        return -1;
    var p = this.head;
    while(index){
        p = p.next;
        --index;
    }
    return p.val;
};
/*
 * Add a node of value val before the first element of the linked list. After the insertion, the new node will be the first node of the linked list. 
 * @param {number} val
 * @return {void}
*/
MyLinkedList.prototype.addAtHead = function(val) {
    if(!this.head)
        this.head = this.tail = new ListNode(val);
    else
        this.head = new ListNode(val,this.head);
    ++this.size;
};
/*
 * Append a node of value val to the last element of the linked list. 
 * @param {number} val
 * @return {void}
*/
MyLinkedList.prototype.addAtTail = function(val) {
    if(!this.tail)
        this.head = this.tail = new ListNode(val);
    else{
        this.tail.next = new ListNode(val);
        this.tail = this.tail.next;
    }
    ++this.size;
};
/*
 * Add a node of value val before the index-th node in the linked list. If index equals to the length of linked list, the node will be appended to the end of linked list. If index is greater than the length, the node will not be inserted. 
 * @param {number} index 
 * @param {number} val
 * @return {void}
*/
MyLinkedList.prototype.addAtIndex = function(index, val) {
    if(index>this.size)
        return ;
    if(index<=0){
        if(!this.head)
            this.head = this.tail = new ListNode(val);
        else
            this.head = new ListNode(val,this.head);
    }
    else if(index===this.size){
        this.tail.next = new ListNode(val);
        this.tail = this.tail.next;
    }
    else{
        var p = this.head;
        --index;
        while(index){
            p = p.next;
            --index;
        }
        p.next = new ListNode(val,p.next);
    }
    ++this.size;
};
/*
 * Delete the index-th node in the linked list, if the index is valid. 
 * @param {number} index
 * @return {void}
*/
MyLinkedList.prototype.deleteAtIndex = function(index) {
    if(index>=this.size||index<0)
        return ;
    if(index===0){
        this.head = this.head.next;
        if(!this.head)
            this.tail = null;
        --this.size;
        return ;
    }
    var p = this.head,index_copy = index-1;
    while(index_copy){
        p = p.next;
        --index_copy;
    }
    p.next = p.next.next;
    if(index===this.size-1)
        this.tail = p;
    --this.size;
};
/*
 * Your MyLinkedList object will be instantiated and called as such:
 * var obj = new MyLinkedList()
 * var param_1 = obj.get(index)
 * obj.addAtHead(val)
 * obj.addAtTail(val)
 * obj.addAtIndex(index,val)
 * obj.deleteAtIndex(index)
*/
/*
 * @param {number} n
 * @return {boolean}
*/
var isPowerOfTwo = function(n) {
    return n > 0 && (n & (n - 1)) === 0;
};
// 数组排序
/*
 * @param {number[]} nums
 * @return {number[]}
*/
var sortArray = function(nums) {
    
};
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
            edge.push([i,j,distance(i,j)]);
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
/*
 * @param {number[]} nums
 * @return {boolean}
*/
var isPossible = function(nums) {
    
};
// 从相邻元素还原处原数组，只需返回一个答案即可
/*
 * @param {number[][]} adjacentPairs
 * @return {number[]}
*/
var restoreArray = function(adjacentPairs) {
    const adj_map = new Map();
    for(const pair of adjacentPairs){
        adj_map.get(pair[0])? adj_map.get(pair[0]).push(pair[1]) :adj_map.set(pair[0],[pair[1]]);
        adj_map.get(pair[1])? adj_map.get(pair[1]).push(pair[0]) :adj_map.set(pair[1],[pair[0]]);
    }
    const size = adjacentPairs.length+1,ans = new Array(size);
    for(const [num,adj] of adj_map){
        if(adj.length===1){
            ans[0] = num;
            break;
        }
    }
    ans[1] = adj_map.get(ans[0])[0];
    for(let i=2;i<size;++i){
        const a = adj_map.get(ans[i-1]);
        ans[i] = (ans[i-2]==a[0])? a[1]:a[0];
    }
    return ans;
};
// 设计一个支持 push ，pop ，top 操作，并能在常数时间内检索到最小元素的栈。
/*
 * initialize your data structure here.
*/
var MinStack = function() {
    this.stack = [];
    this.min_stack = [Infinity];
};
/* 
 * @param {number} val
 * @return {void}
*/
MinStack.prototype.push = function(val) {
    this.stack.push(val);
    this.min_stack.push(Math.min(this.min_stack[this.min_stack.length-1],val));
};
/*
 * @return {void}
*/
MinStack.prototype.pop = function() {
    this.min_stack.pop();
    this.stack.pop();
};
/*
 * @return {number}
*/
MinStack.prototype.top = function() {
    return this.stack[this.stack.length-1];
};
/*
 * @return {number}
*/
MinStack.prototype.getMin = function() {
    return this.min_stack[this.min_stack.length-1];
};
/*
 * Your MinStack object will be instantiated and called as such:
 * var obj = new MinStack()
 * obj.push(val)
 * obj.pop()
 * var param_3 = obj.top()
 * var param_4 = obj.getMin()
*/
// 共有 n 名小伙伴一起做游戏。小伙伴们围成一圈，按 顺时针顺序 从 1 到 n 编号。确切地说，从第 i 名小伙伴顺时针移动一位会到达第(i+1)名小伙伴
// 的位置,其中1<=i<n,从第n名小伙伴顺时针移动一位会回到第1名小伙伴的位置.游戏遵循如下规则:从第1名小伙伴所在位置开始 。
// 沿着顺时针方向数 k 名小伙伴，计数时需要 包含 起始时的那位小伙伴。逐个绕圈进行计数，一些小伙伴可能会被数过不止一次。你数到的最后一名小伙伴
// 需要离开圈子，并视作输掉游戏。如果圈子中仍然有不止一名小伙伴，从刚刚输掉的小伙伴的 顺时针下一位 小伙伴 开始，回到步骤 2 继续执行。
// 否则，圈子中最后一名小伙伴赢得游戏。给你参与游戏的小伙伴总数 n ，和一个整数 k ，返回游戏的获胜者。
/*
 * @param {number} n
 * @param {number} k
 * @return {number}
*/
var findTheWinner = function(n, k) {
    if(n === 1)
        return 1;
    var ans = 0
    for(let i=2;i<=n;++i)
        ans = (ans + k) % i
    return ans + 1;
};
// 给你一个由(,)和小写字母组成的字符串s,你需要从字符串中删除最少数目的'('或者')'(可以删除任意位置的括号)使得剩下的括号字符串有效。
/*
 * @param {string} s
 * @return {string}
*/
var minRemoveToMakeValid = function(s) {
    const arr = s.split(''),size = arr.length;
    const stack = new Array();
    for(let i =0;i<size;++i){
        if(arr[i]==='(')
            stack.push(i);
        if(arr[i]===')'){
            if(stack.length>0)
                stack.pop();
            else
                arr[i] = '';
        }
    }
    for(const pos of stack)
        arr[pos] = '';
    return arr.join('');
};
// 编写一个高效的算法来判断 m x n 矩阵中，是否存在一个目标值。该矩阵具有如下特性：
// 每行中的整数从左到右按升序排列。每行的第一个整数大于前一行的最后一个整数。
/**
 * @param {number[][]} matrix
 * @param {number} target
 * @return {boolean}
 */
var searchMatrix = function(matrix, target) {
    const m = matrix.length, n = matrix[0].length;
    var low = 0, right = m * n - 1,mid,tmp;
    while (low <= right) {
        mid = Math.floor((right - low) / 2) + low;
        tmp = matrix[Math.floor(mid / n)][mid % n];
        if (tmp < target) 
            low = mid + 1;
        else if (tmp > target)
            right = mid - 1;
        else
            return true;
    }
    return false;
};
// 有序数组转二叉搜索树
/**
 * @param {number[]} nums
 * @return {TreeNode}
 */
var sortedArrayToBST = function(nums) {
    const sortedArrayToBSTHelper = function(nums,left,right){
        if(left>right)
            return null;
        var mid = Math.floor((left+right+1)/2);
        var root = new TreeNode(nums[mid]);
        root.left = sortedArrayToBSTHelper(nums,left,mid-1);
        root.right = sortedArrayToBSTHelper(nums,mid+1,right);
        return root;
    }
    return sortedArrayToBSTHelper(nums,0,nums.length-1);
};
// 给定一个二叉树，返回其节点值的锯齿形层序遍历。（即先从左往右，再从右往左进行下一层遍历，以此类推，层与层之间交替进行）。
/**
 * @param {TreeNode} root
 * @return {number[][]}
 */
var zigzagLevelOrder = function(root) {
    if(!root)
        return [];
    const nodes = [],ans = [];
    var left_order = true;
    while(nodes.length){
        let tmp_list = [];
        const size = nodes.length;
        for(let i=0;i<size;++i){
            const node = nodes.shift();
            if(left_order)
                tmp_list.push(node.val);
            else
                tmp_list.unshift(node.val);
            if(node.left)
                nodes.push(node.left);
            if(node.right)
                nodes.push(node.right);
        }
        ans.push(tmp_list);
        left_order = !left_order;
    }
    return ans;
};
// 峰值元素是指其值大于左右相邻值的元素.给你一个输入数组nums,找到峰值元素并返回其索引.数组可能包含多个峰值返回任何一个峰值所在位置即可。
/**
 * @param {number[]} nums
 * @return {number}
 */
var findPeakElement = function(nums) {
    const size = nums.length;
    for(let i=0;i<size;++i){
        if(nums[i]>nums[i+1])
            return i;
    }
    return size-1;
};
// 整数数组nums按升序排列,数组中的值互不相同.在传递给函数之前,nums在预先未知的某个下标k(0<=k<nums.length)上进行了旋转，
// 使数组变为 [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]（下标 从 0 开始 计数）
/**
 * @param {number[]} nums
 * @param {number} target
 * @return {number}
 */
var search = function(nums, target) {
    var size = nums.length;
    if(size===0)
        return -1;
    if(size===1)
        return (nums[0]===target)? 0 :-1;
    var left = 0,right = size-1,mid;
    while(left<=right){
        mid = Math.floor((left+right)/2);
        if(nums[mid]===target)
            return mid;
        if (nums[0]<= nums[mid]) 
            if (nums[0]<=target&&target<nums[mid])
                right = mid - 1;
            else
                left = mid + 1;
        else
            if (nums[mid]<target&&target <= nums[size-1]) 
                left = mid + 1;
            else 
                right = mid - 1;
    }
    return -1;
};
// 已知一个长度为n的数组，预先按照升序排列，经由1到n次旋转后，得到输入数组。例如，原数组 nums=[0,1,2,4,5,6,7] 在变化后可能得到：
// 若旋转 4 次，则可以得到 [4,5,6,7,0,1,2]若旋转 7 次，则可以得到 [0,1,2,4,5,6,7]
// 注意，数组 [a[0], a[1], a[2], ..., a[n-1]] 旋转一次 的结果为数组 [a[n-1], a[0], a[1], a[2], ..., a[n-2]] 。
// 给你一个元素值 互不相同 的数组 nums ，它原来是一个升序排列的数组，并按上述情形进行了多次旋转。请你找出并返回数组中的 最小元素 。
/**
 * @param {number[]} nums
 * @return {number}
 */
var findMin = function(nums) {
    var left = 0,right = nums.length - 1;
    while (left < right) {
        const pivot = left + Math.floor((right - left) / 2);
        if (nums[pivot] < nums[right])
            right = pivot;
        else
            left = pivot + 1;
    }
    return nums[left];
};

var minOperations = function(target, arr) {
    const size = target.length;
    const pos_map = new Map();
    for (let i = 0; i < size; ++i)
        pos_map.set(target[i], i);
    const nums = [];
    for (const val of arr) {
        if (pos_map.has(val)) {
            const index = pos_map.get(val);
            const it = binarySearch(nums, index);
            if (it !== nums.length)
                nums[it] = index;
            else 
                nums.push(index);
        }
    }
    return size - nums.length;
};
const binarySearch = (nums, target) => {
    const size = nums.length;
    if (size === 0 || nums[size - 1] < target)
        return size;
    let left = 0, right = size - 1;
    while (left < right) {
        const mid = Math.floor((right - left) / 2) + left;
        if(nums[mid]<target)
            left = mid + 1;
        else
            right = mid;
    }
    return left;
};
/**
 * @param {TreeNode} root
 * @return {number}
*/
var findSecondMinimumValue = function(root) {
    var ans = -1;
    const dfs = function(r){
        if(!r||!r.left)
            return ;
        else
            if(r.left.val<r.right.val)
                ans = r.right.val;
            else if(r.left.val>r.right.val)
                ans = r.left.val;
            else
                ans = -1;
    }
    dfs(root);
    return ans;
};
// 给定一个二叉树的 根节点 root，想象自己站在它的右侧，按照从顶部到底部的顺序，返回从右侧所能看到的节点值。
/**
 * @param {TreeNode} root
 * @return {number[]}
 */
var rightSideView = function(root) {
    if(!root)
        return [];
    var ans = [],deepth = -1;
    const queue = [[root,0]];
    while(queue.length){
        const node = queue.shift();
        if(node[1]===deepth+1){
            ++deepth;
            ans.push(node[0].val);
        }
        if(node[0].right)
            queue.push([node[0].right,node[1]+1]);
        if(node[0].left)
            queue.push([node[0].left,node[1]+1]);
    }
    return ans;
};
/**
 * @param {TreeNode} root
 * @param {number} targetSum
 * @return {number[][]}
 */
// 给你二叉树的根节点root和一个整数目标和targetSum,找出所有从根节点到叶子节点路径总和等于给定目标和的路径。
var pathSum = function(root, targetSum) {
    const tmp = new Array(),ans = new Array();
    var path_length = 0;
    const pathSumAssist = (node)=>{
        if(!node)
            return ;
        path_length += node.val;
        tmp.push(node.val);
        if(!node.left&&!node.right&&path_length===targetSum)
            ans.push(Array.from(tmp));
        pathSumAssist(node.left);
        pathSumAssist(node.right);
        path_length -= node.val;
        tmp.pop();
    }
    pathSumAssist(root);
    return ans;
};
// 给定一个二叉搜索树的根节点 root 和一个值 key，删除二叉搜索树中的 key 对应的节点，并保证二叉搜索树的性质不变。
// 返回二叉搜索树（有可能被更新）的根节点的引用。
/**
 * @param {TreeNode} root
 * @param {number} key
 * @return {TreeNode}
 */
 var deleteNode = function(root, key) {
    if(!root)
        return null;
    if(key<root.val)
        root.left = deleteNode(root.left,key);
    else if(key>root.val)
        root.right = deleteNode(root.right,key);
    else{
        if(!root.right)
            return root.left;
        if(!root.left)
            return root.right;
        const min_pos = getMin(root.right);
        root.val = min_pos.val;
        root.right = deleteNode(root.right,min_pos.val);
    }
    return root;
};
const getMin = (root)=>{
    if(!root)
        return null;
    var p = root;
    while(p.left)
        p = p.left;
    return p;
}
// 删除root中的最小节点
const deleteMin = (root) =>{
    if(!root)
        return null;
    if(!root.left)
        return root.right;
    var p = root;
    while(p.left.left)
        p = p.left;   
    p.left = null;
    return root; 
}
var deleteNode = (root,key) =>{
    if(root.val===key)
        return null;
    var p = root;
    while(p){
        
    }
}
// 二叉搜索树的迭代器
/**
 * @param {TreeNode} root
 */
 var BSTIterator = function(root) {
    this.cur = root;
    this.stack = new Array();
};

/**
 * @return {number}
 */
BSTIterator.prototype.next = function() {
    while(this.cur){
        this.stack.push(this.cur);
        this.cur = this.cur.left;
    }
    this.cur = this.stack.pop();
    var ret = this.cur.val;
    this.cur = this.cur.right;
    return ret;
};

/**
 * @return {boolean}
 */
BSTIterator.prototype.hasNext = function() {
    return this.cur||(this.stack.length!==0);
};

/**
 * Your BSTIterator object will be instantiated and called as such:
 * var obj = new BSTIterator(root)
 * var param_1 = obj.next()
 * var param_2 = obj.hasNext()
 */
// 二叉搜索树中第k小的元素
/**
 * @param {TreeNode} root
 * @param {number} k
 * @return {number}
 */
var kthSmallest = function(root, k) {
    var tmp = new Array();
    const  midOrder = (node) =>{
        if(tmp.length===k||!root)
            return ;
        midOrder(node.left);
        tmp.push(node.val);
        midOrder(node.right);
    }
    return tmp[k-1];
};
// 检查括号是否有效
/**
 * @param {string} s
 * @return {boolean}
 */
 var isValid = function(s) {
    const stack = new Array();let c = '(';
    for(const ch of s){
        if(ch==='('||ch==='{'||ch==='[')
            stack.push(ch);
        else{
            if(ch!==check(stack.pop()))
                return false;
        }
    }
    return stack.length===0;
};
const check = (ch) => {
    if(ch==undefined)
        return ' ';
    if(ch==='(')
        return ')';
    if(ch==='[')
        return ']';
    return '}';
}
// 带退格的字符串比较
/**
 * @param {string} s
 * @param {string} t
 * @return {boolean}
 */
var backspaceCompare = function(s, t) {
    const m = s.length,n = t.length;
    var p1 = m-1, p2 = n-1,skips = 0,skipt = 0;
    while(p1>=0||p2>=0){
        while(p1>=0){
            if(s[p1]==='#'){
                ++skips;--p1;                
            }
            else if(skips>0){
                --skips;--p1;
            }
            else
                break;
        }
        while(p2>=0){
            if(t[p2]==='#'){
                ++skipt;--p2;
            }
            else if(skipt>0){
                --skipt;--p2;
            }
            else
                break;
        }
        if(p1>=0&&p2>=0){
            if(s[p1]!==t[p2])
                return false;
        }
        else if(p1>=0||p2>=0)
            return false;
        --p1;--p2;
    }
    return true;
};
// 给你 n 个非负整数 a1，a2，...，an，每个数代表坐标中的一个点 (i, ai) 。在坐标内画 n 条垂直线，垂直线 i 的两个端点分别为 
// (i, ai) 和 (i, 0) 。找出其中的两条线，使得它们与 x 轴共同构成的容器可以容纳最多的水。
/**
 * @param {number[]} height
 * @return {number}
 */
var maxArea = function(height) {
    var left = 0,right = height.length-1;
    var ans = -1;
    while(left<=right){
        ans = Math.max(ans,Math.min(height[left],height[right])*(right-left));
        if(height[left]<=height[right])
            ++left;
        else
            --right;
    }
    return ans;
};
// 给定一个二叉树（具有根结点 root）， 一个目标结点 target ，和一个整数值 K 。
// 返回到目标结点 target 距离为 K 的所有结点的值的列表。 答案可以以任何顺序返回。
/**
 * @param {TreeNode} root
 * @param {TreeNode} target
 * @param {number} k
 * @return {number[]}
 */
var distanceK = function(root, target, k) {
    const ans = new Array();
    const parents = new Map();
    const findParent = (node,parents)=>{
        if(!node)
            return ;
        if(node.left)
            parents.set(node.left.val,node);
        if(node.right)
            parents.set(node.right.val,node);
        findParent(node.left,parents);
        findParent(node.right,parents);
    }
    findParent(root,parents);
    const findAns = (curr,prev,n)=>{
        if(!curr)
            return ;
        if(n===k){
            ans.push(curr.val);
            return ;
        }
        if(curr.left!==prev)
            findAns(curr.left,curr,n+1);
        if(curr.right!==prev)
            findAns(curr.right,curr,n+1);
        if(parents.get(curr.val)!==prev)
            findAns(parents.get(curr.val),curr,n+1);
    }
    findAns(target,null,0);
    return ans;
};
// 给你一个整数数组 nums ，数组中共有 n 个整数。132 模式的子序列 由三个整数 nums[i]、nums[j] 和 nums[k] 组成，并同时满足：
// i < j < k 和 nums[i] < nums[k] < nums[j] 。如果 nums 中存在 132 模式的子序列 ，返回 true ；否则，返回 false 。
/**
 * @param {number[]} nums
 * @return {boolean}
 */
var find132pattern = function(nums) {
    const size = nums.length;
    const k_tem = [nums[size - 1]];
    let max_k = -Number.MAX_SAFE_INTEGER;
    for (let i = size - 2; i >= 0; --i) {
        if (nums[i] < max_k)
            return true;
        while (k_tem.length && nums[i] > k_tem[k_tem.length - 1]) {
            max_k = k_tem[k_tem.length - 1];
            k_tem.pop();
        }
        if (nums[i] > max_k)
            k_tem.push(nums[i]);
    }
    return false;
};
/**
 * @param {number[][]} firstList
 * @param {number[][]} secondList
 * @return {number[][]}
 */
var intervalIntersection = function(firstList, secondList) {
    const m = firstList.length,n = secondList.length;
    var p1 = 0,p2 = 0;
    const ans = new Array();
    while(p1<m&&p2<n){
        let low = Math.max(firstList[p1][0],secondList[p2][0]);
        let high = Math.min(firstList[p1][1],secondList[p2][1]);
        if(low<=high)
            ans.push([low,high]);
        if(firstList[p1][1]<secondList[p2][1])
            ++p1;
        else
            ++p2;
    }
    return ans;
};