/*
 * @Author: zanilia
 * @Date: 2021-06-27 21:38:56
 * @LastEditTime: 2021-07-25 13:15:13
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
    const arr_num_count = new Map();
    const target_num_count = new Map();
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
/* 
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
// 带时间戳的map
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
/*
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
    var negative = -1;
    var size = nums.length;
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
// 给定一个包含红色(0),白色(1)和蓝色(2)，一共n个元素的数组,原地对它们进行排序,使得相同颜色的元素相邻,并按照红色,白色,蓝色顺序排列。
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
/*
 * @param {ListNode} head
 * @return {ListNode}
*/
var swapPairs = function(head) {
    if(!head)
        return null;
    var ret,tmp;
    if(head.next){
        ret = head.next;
        head.next =ret.next;
        ret.next = head;
    }
    while(head.next){
        if(head.next.next){
            tmp = head.next;
            head.next = tmp.next;
            tmp.next = head.next.next
            head.next.next = tmp;
            head = head.next.next;
        }
        else
            head = head.next;
    }
    return ret;
};
// 给定一个三角形 triangle ，找出自顶向下的最小路径和。
// 每一步只能移动到下一行中相邻的结点上。相邻的结点 在这里指的是下标与上一层结点下标相同或者等于上一层结点下标+1的两个结点。
/*
 * @param {number[][]} triangle
 * @return {number}
*/
var minimumTotal = function(triangle) {
    const size = triangle.length;
    if(size===1)
        return triangle[0][0];
    const dp = new Array(2);
    dp[0] = new Array(size);dp[1] = new Array(size);
    dp[0][0] = triangle[0][0];
    var curr, prev;
    for(let i=1;i<size;++i){
        curr = i%2;prev = 1 - curr;
        dp[curr][0] = dp[prev][0] + triangle[i][0];
        for(let j=1;j<i;++j)
            dp[curr][j] = Math.min(dp[prev][j],dp[prev][j-1])+triangle[i][j];
        dp[curr][i] = dp[prev][i-1] + triangle[i][i];
    }
    return Math.min(...dp[curr]);
}
// 空间压缩
var minimumTotal = function(triangle) {
    const size = triangle.length;
    const dp = new Array(size);
    dp[0] = triangle[0][0];
    for(let i=1;i<size;++i){
        dp[i] = dp[i-1] + triangle[i][i];
        for(let j=i-1;j>0;--j)
            dp[j] = Math.min(dp[j],dp[j-1])+triangle[i][j];
        dp[0]  += triangle[i][0];
    }
    return Math.min(...dp);
};
/*
 * @param {string} time
 * @return {string}
*/
var maximumTime = function(time) {
    const arr = Array.from(time);
    if (arr[0] === '?') 
        arr[0] = ('4' <= arr[1] && arr[1] <= '9') ? '1' : '2';
    if (arr[1] === '?') 
        arr[1] = (arr[0] == '2') ? '3' : '9';
    if (arr[3] === '?') 
        arr[3] = '5';
    if (arr[4] === '?') 
        arr[4] = '9';
    return arr.join('');
};
// 给定一个数组 A，将其划分为两个连续子数组 left 和 right， 使得：left 中的每个元素都小于或等于 right 中的每个元素。
// left 和 right 都是非空的。left 的长度要尽可能小。返回left子数组的长度
//  @param {number[]} nums @return {number}
var partitionDisjoint = function(nums) {
    const size = nums.length;
    const left_max = new Array(size),right_min = new Array(size);
    left_max[0] = nums[0];
    for(let i=0;i<size;++i)
        left_max[0] = Math.max(left_max[i-1],nums[i]);
    right_min[size-1] = nums[size-1];
    for(let i=size-2;i>=0;--i)
        right_min[i] = Math.min(right_min[i+1],nums[i]); 
    for(let i=1;i<size;++i)
        if(left_max[i-1]<=right_min[i])
            return i;
};
// 颠倒给定的 32 位无符号整数的二进制位。分治方法
// @param {number} n - a positive integer @return {number} - a positive integer
var reverseBits = function(n) {
    const M1 = 0x55555555,M2 = 0x33333333,M4 = 0x0f0f0f0f,M8 = 0x00ff00ff; 
    n = n >>> 1 & M1 | (n & M1) << 1;
    n = n >>> 2 & M2 | (n & M2) << 2;
    n = n >>> 4 & M4 | (n & M4) << 4;
    n = n >>> 8 & M8 | (n & M8) << 8;
    return (n >>> 16 | n << 16) >>> 0;
};