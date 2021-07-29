/*
 * @Author: zanilia
 * @Date: 2021-07-14 20:31:29
 * @LastEditTime: 2021-07-24 16:18:49
 * @Descripttion: 
 */
// 将数组中0移动至末尾，其他元素相对位置不变
/*
 * @param {number[]} nums
 * @return {void} Do not return anything, modify nums in-place instead.
*/
var moveZeroes = function(nums) {
    var p1 = 0,p2 = 0,size = nums.length,tem;
    while(p2<size){
        if(nums[p2]!==0){
            if(p1!==p2){
                tem = nums[p2];
                nums[p2] = nums[p1];
                nums[p1] = tem;
            }
            ++p1;++p2;
        }
        else
            ++p2;
    }
    while(p1<size){
        nums[p1] = 0;
        ++p1;
    }
};
// numbers升序，寻找两个数相加为target，下标从1开始
/*
 * @param {number[]} numbers
 * @param {number} target
 * @return {number[]}
*/
var twoSum = function(numbers, target) {
    var left = 0,right = numbers.length-1,tem;
    while(left<right){
        tem = numbers[left]+numbers[right];
        if(tem<target)
            ++left;
        else if(tem>target)
            --right;
        else
            return [left,right];
    }
};
// 给定一个n×n的二维矩阵matrix表示一个图像.请你将图像顺时针旋转90度。
/*
 * @param {number[][]} matrix
 * @return {void} Do not return anything, modify matrix in-place instead.
*/
var rotate = function(matrix) {
    var size = matrix.length,tem;
    for (let i=0;i<size/2;++i){
        for (let j=0;j<size;++j) {
            tem = matrix[size-i-1][j];
            matrix[size-i-1][j] = matrix[i][j]; 
            matrix[i][j] = tem;
        }
    }
    for (let i=0;i<size;++i) {
        for (let j=0;j<i;++j) {
            tem = matrix[i][j];
            matrix[i][j] = matrix[j][i];
            matrix[j][i] = tem;
        }
    }
};
// 给你一个正整数n,生成一个包含1到n2所有元素，且元素按顺时针顺序螺旋排列的nxn正方形矩阵matrix。
/*
 * @param {number} n
 * @return {number[][]}
*/
var generateMatrix = function(n) {
    const ans = new Array(n).map(() => new Array(n).fill(0));
    var top = 0,bottom = n-1,left =0 ,right = n-1;
    var num = 0;
    while(top<=bottom){
        for (let column = left; column <= right; column++)
            ans[top][column] = ++num;
        for (let row = top + 1; row <= bottom; row++) 
            ans[row][right] = ++num;
        if (left<right&&top<bottom) {
            for (let column =right-1;column>left;--column)
                ans[bottom][column] = ++num;
            for (let row = bottom;row>top;--row)
                ans[row][left] = ++num;
        }
        left++;
        right--;
        top++;
        bottom--;
    }
    return ans;
};
// 反转字符串
/*
 * @param {character[]} s
 * @return {void} Do not return anything, modify s in-place instead.
*/
var reverseString = function(s) {
    var left = 0,right = s.length-1,tem;
    while(left<right){
        tem = s[left];
        s[left] = s[right];
        s[right] = tem;
        ++left;
        --right;
    }
};
// 将字符串中每一个单词中的字母反转
/*
 * @param {string} s
 * @return {string}
*/
var reverseWords = function(s) {
    var left = 0,right = s.indexOf(' '),left_tem,right_tem;
    const ans = new Array(s.length);
    while(right!==-1){
        left_tem = left;
        right_tem = right-1;
        while(left_tem<right_tem)
            ans[left_tem++] = s[right_tem--];
        ans[left_tem] = ' ';
        left = right+1;
        right = s.indexOf(' ',left);
    }
    left = right+1;
    right = s.length-1;
    while(left<right)
        ans[left++] = s[right--];
    return ans.join('');
};
// 矩阵按行，列分别为升序，查找是否存在target
/*
 * @param {number[][]} matrix
 * @param {number} target
 * @return {boolean}
*/
var searchMatrix = function(matrix, target) {
    var row = matrix.length-1,col = 0,len = matrix[0].length;
    while (row>=0&&col<len) {
        if (matrix[row][col] > target)
            row--;
        else if (matrix[row][col] < target) 
            col++;
        else 
            return true;
    }
    return false;
};
/*
 * @param {number[]} arr
 * @return {number}
*/
var maximumElementAfterDecrementingAndRearranging = function(arr) {
    const size = arr.length;
    const counter = new Array(size+1);
    for(const num of arr){
        if(num>=size)
            ++counter[size];
        else
            ++counter[num];
    }
    var miss =0;
    for(let i =0;i<size;++i){
        if(counter[i]===0)
            ++miss;
        else
            miss -= Math.min(miss,counter[i]-1);
    }
    return size - miss;
};
// n个左右括号组合生成所有合理的括号对
/*
 * @param {number} n
 * @return {string[]}
*/
var generateParenthesis = function(n) {
    const dp = new Array(n+1);
    dp[0] = [""];dp[1] = ["()"];
    for(let i=2;i<=n;++i)
        for(let j=0;j<n;++j)
            for(const p1 of dp[j])
                for(const p2 of dp[i-j-1])
                    dp[i].push("("+p1+")"+p2);
    return dp[n];
};
// 链表的中间节点
function ListNode(val, next) {
    this.val = (val===undefined ? 0 : val)
    this.next = (next===undefined ? null : next)
}
/*
 * @param {ListNode} head
 * @return {ListNode}
*/
var middleNode = function(head) {
    var slow = head,fast = head.next;
    while(fast){
        if(fast.next.next){
            fast = fast.next.next;
            slow = slow.next;
        }
        else
            return slow.next;
    }
    return slow;
};
// 在数组中查找升序三元组
/*
 * @param {number[]} nums
 * @return {boolean}
*/
var increasingTriplet = function(nums) {
    var small = Infinity, mid = Infinity;
    for(const num of nums){
        if(num<=small)
            small = num;
        else if(num<=mid)
            mid = num;
        else
            return true;
    }
    return false;
};
// 统计一个数字在排序数组中出现的次数。
/*
 * @param {number[]} nums
 * @param {number} target
 * @return {number}
*/
var search = function(nums, target) {
    var left =0,right = nums.length-1,pos = -1,ret = 1;
    const size = nums.length;
    while(left<=right){
        let mid = Math.floor((right-left)/2)+left;
        if(nums[mid]===target){
            pos = mid;break;
        }
        if(nums[mid]<target)
            left = mid +1;
        else
            right = mid-1;
    }
    if(pos===-1)
        return 0;
    for(let i =pos-1;i>=0;--i)
        if(nums[i]===target)
            ++ret;
        else
            break;
    for(let i =pos+1;i<=size;++i)
        if(nums[i]===target)
            ++ret;
        else
            break;
    return ret;
};
// 给你一个长度为n的整数数组nums,其中n>1,返回输出数组output,其中output[i]等于nums中除nums[i]之外其余各元素的乘积。
/*
 * @param {number[]} nums
 * @return {number[]}
*/
var productExceptSelf = function(nums) {
    const size = nums.length-1;
    const ans = new Array(size);
    ans[0] = 1;
    for(let i =0;i<size;++i)
        ans[i+1] = nums[i]*ans[i];
    var right = 1;
    for(let i = size;i>=0;--i){
        ans[i] *= right;
        right *= nums[i];
    }
    return ans;
};
// 给定一个整数数组和一个整数k,你需要找到该数组中和为k的连续的子数组的个数。
/*
 * @param {number[]} nums
 * @param {number} k
 * @return {number}
*/
var subarraySum = function(nums, k) {
    const prefix_sum = new Map();
    var count = 0, sum = 0;
    prefix_sum.set(0,1);
    for(const num of nums){
        sum += num;
        if(prefix_sum.has(sum-k))
            count += prefix_sum.get(sum-k);
        prefix_sum.set(sum,(prefix_sum.has(sum)? prefix_sum.get(sum)+1 :1));
    }
    return count;
};
// 无重复字符的最长子串
/*
 * @param {string} s
 * @return {number}
*/
var lengthOfLongestSubstring = function(s) {
    const size = s.length;
    const char_set = new Set();
    var right = -1,ret = 0;
    for(let i = 0;i<size;++i){
        if(i!=0)
            char_set.delete(s.charAt(i-1));
        while(right+1<size&&!char_set.has(s.charAt(right+1)))
            char_set.add(s.charAt(++right));
        ret = Math.max(ret,right-i-1);  
    }
    return ret;
};
// 字符串相加
/*
 * @param {string} num1
 * @param {string} num2
 * @return {string}
*/
var addStrings = function(num1, num2) {
    const size1 = num1.length,size2 = num2.length;
    if(size1<size2)
        return addStrings(num2,num1);
    var i = num1.length-1, j = num2.length-1,tem,need_carry = 0;
    const ans = new Array(size1+1);
    while(i>=0){
        if(j<0){
            if(need_carry){
                tem = num1.charAt(i)-'0'+need_carry;
                if(tem>=10){
                    ans[i+1] = tem-10;
                    need_carry=1;
                }
                else{
                    ans[i+1] = tem;
                    need_carry=0;
                }
            }
            else
                ans[i+1]=num1.charAt(i)-'0';
            --i;
        }
        else{
            tem=(num1.charAt(i)-'0')+(num2.charAt(j)-'0')+need_carry;
            if(tem>=10){
                ans[i+1] = tem-10;
                need_carry = 1;
            }
            else{
                ans[i+1] = tem;
                need_carry=0;
            }
            --i;--j;
        }
    }
    if(need_carry)
        ans[0] = need_carry;
    return ans.join('');
};
/*
 * @param {string} s1
 * @param {string} s2
 * @return {boolean}
*/
var checkInclusion = function(s1, s2) {
    const size1 = s1.length,size2 = s2.length;
    if(size1>size2)
        return false;
    var diff = 0;
    const cnt = new Array(26).fill(0);
    for(let i =0;i<size1;++i){
        --cnt[s1[i].charCodeAt()-'a'.charCodeAt()];
        ++cnt[s2[i].charCodeAt()-'a'.charCodeAt()];
    }
    for(const count of cnt)
        if(count!==0)
            ++diff;
    if(diff===0)
        return true;
    var right,left;
    for(let i=size1;i<size2;++i){
        right = s2[i].charCodeAt()-'a'.charCodeAt();
        left = s2[i-size1].charCodeAt()-'a'.charCodeAt();
        if(right===left)
            continue;
        if(cnt[right]===0)
            ++diff;
        ++cnt[right];
        if(cnt[right]===0)
            --diff;
        if(cnt[left]===0)
            ++diff;
        --cnt[left];
        if(cnt[left]===0)
          --diff;
        if(diff===0)
            return true;
    }   
    return false;
};
// 一个字符串s更改顺序，删除字母后能组成最长的回文字符串长度
/*
 * @param {string} s
 * @return {number}
*/
var longestPalindrome = function(s) {
    const char_count = new Map();
    for(const ch of s)
        char_count.set(ch,(char_count.has(ch))?char_count.get(ch):1);
    var odd = 0,ret = 0;
    for(const [char,count] of char_count){
        if(count%2){
            ret += count-1;
            odd = 1;
        }
        else
            ret += count;
    }
    return ret;
};
// 按是否为异位词分组
/*
 * @param {string[]} strs
 * @return {string[][]}
*/
var groupAnagrams = function(strs) {
    const cnt_map = new Map();
    for(const str of strs){
        const cnt = new Array(26).fill(0);
        for(const ch of str)
            ++cnt[ch.charCodeAt()-'a'.charCodeAt()];
        cnt_map[cnt]?cnt_map[cnt].push(str):cnt_map[cnt] = [str];
    }
    return Object.values(cnt_map);
};
/*
 * @param {number[][]} image
 * @param {number} sr
 * @param {number} sc
 * @param {number} newColor
 * @return {number[][]}
*/
var floodFill = function(image, sr, sc, newColor) {
    if(image[sr][sc]===newColor)
        return image;
    var tmp = image[sr][sc];
    image[sr][sc] = newColor;
    if(sr>=1&&image[sr-1][sc]===tmp)
        floodFill(image,sr-1,sc,newColor);
    if(sc>=1&&image[sr][sc-1]===tmp)
        floodFill(image,sr,sc-1,newColor);
    if(sr+1<image.length&&image[sr+1][sc]===tmp)
        floodFill(image,sr+1,sc,newColor);
    if(sc+1<image[0].length&&image[sr][sc+1]===tmp)
        floodFill(image,sr,sc+1,newColor);
    return image;
};
// 给定一个包含了一些 0 和 1 的非空二维数组 grid 。
// 一个 岛屿 是由一些相邻的 1 (代表土地) 构成的组合，这里的「相邻」要求两个 1 必须在水平或者竖直方
// 向上相邻。你可以假设 grid 的四个边缘都被 0（代表水）包围着。
// 找到给定的二维数组中最大的岛屿面积。(如果没有岛屿，则返回面积为 0 。)
/*
 * @param {number[][]} grid
 * @return {number}
*/
var maxAreaOfIsland = function(grid) {
    var ret = 0,m = grid.length,n = grid[0].length;
    for(let i = 0; i < m; ++i)
        for(let j = 0; j < n; ++j)
            if(grid[i][j] == 1)
                ret = Math.max(ret,dfs(i,j,grid));
    return ret;
};
var dfs = function(i,j,grid){
    if(i<0 || j<0 || i>= grid.length || j>=grid[i].length || grid[i][j] == 0)
        return 0;
    grid[i][j] =0;
    let num = 1;
    num += dfs(i+1,j,grid);
    num += dfs(i-1,j,grid);
    num += dfs(i,j+1,grid);
    num += dfs(i,j-1,grid);
    return num;
}
// 字符串S由小写字母组成.我们要把这个字符串划分为尽可能多的片段,同一字母最多出现在一个片段中.返回一个表示每个字符串片段的长度的列表
/*
 * @param {string} s
 * @return {number[]}
*/
var partitionLabels = function(s) {
    const size = s.length,last_pos = new Array(),code_a = 'a'.codePointAt(0);
    for(let i =0;i<size;++i)
        ++last_pos[s.codePointAt(i)-code_a];
    const ans =[];
    var start = 0,end = 0;
    for(let i=0;i<size;++i){
        end = Math.max(end,last_pos[s.codePointAt(i)-code_a]);
        if(i===end){
            ans.push(end-start+1);
            start = end+1;
        }
    }
    return ans;
};
// 给你两个正整数数组 nums1 和 nums2 ，数组的长度都是 n 。
// 数组 nums1 和 nums2 的 绝对差值和 定义为所有 |nums1[i] - nums2[i]|（0 <= i < n）的 总和（下标从 0 开始）。
// 你可以选用 nums1 中的 任意一个 元素来替换 nums1 中的 至多 一个元素，以 最小化 绝对差值和。
// 在替换数组 nums1 中最多一个元素 之后 ，返回最小绝对差值和。因为答案可能很大，所以需要对1e9+7取余 后返回。
/*
 * @param {number[]} nums1
 * @param {number[]} nums2
 * @return {number}
 */
 var minAbsoluteSumDiff = function(nums1, nums2) {
    var diff = 0,sum= 0,tmp;
    const size =nums1.length,nums=[...nums1];
    const binarySearch = (rec, target) => {
        let left = 0, right = rec.length - 1;
        if (rec[right] < target) {
            return right + 1;
        }
        while (left < right) {
            const mid = Math.floor((right - left) / 2) + left;
            if (rec[mid] < target) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        return left;
    }
    for(let i =0;i<size;++i){
        tmp = Math.abs(nums1[i]-nums2[i]);
        sum = (sum +tmp)%1000000007;
        let j = binarySearch(nums,nums2[i]);
        if(j<size)
            diff = Math.max(diff,tmp-(nums[j]-nums2[i]));
        if(j>0)
            diff = Math.max(diff,tmp-(nums2[i]-rec[j-1]));
    }
    return (sum +diff+1000000007)%1000000007;
};
// 实现一个算法，确定一个字符串 s 的所有字符是否全都不同。
/*
 * @param {string} astr
 * @return {boolean}
*/
var isUnique = function(astr) {
    var mark= 0,code_a = 'a'.codePointAt(0),move;
    for(const ch of ast){
        move = 1<<(ch.codePointAt(0)-code_a);
        if(mark&move!==0)
            return false;
        else
            mark |= 1<<m;
    }
    return true;
};
function TreeNode(val, left, right) {
    this.val = (val===undefined ? 0 : val)
    this.left = (left===undefined ? null : left)
    this.right = (right===undefined ? null : right)
}
// 你需要将他们合并为一个新的二叉树。合并的规则是如果两个节点重叠，
// 那么将他们的值相加作为节点合并后的新值，否则不为 NULL 的节点将直接作为新二叉树的节点
/*
 * @param {TreeNode} root1
 * @param {TreeNode} root2
 * @return {TreeNode}
*/
var mergeTrees = function(root1, root2) {
    var root;
    if(root1){
        if(root2){
            root = new TreeNode(root1.val+root2.val);
            root.left = mergeTrees(root1.left,root2.left);
            root.right = mergeTrees(root1.right,root2.right);
        }
        else{
            root = new TreeNode(root1.val);
            root.left = mergeTrees(root1.left,null);
            root.right = mergeTrees(root1.right,null);
        }
        return root;
    }
    else{
        if(root2){
            root = new TreeNode(root2.val);
            root.left = mergeTrees(null,root2.left);
            root.right = mergeTrees(null,root2.right);
            return root;
        }
        else
            return null;
    }
};
// 给定一个 完美二叉树 ，其所有叶子节点都在同一层，每个父节点都有两个子节点。二叉树定义如下：
function Node(val, left, right, next) {
    this.val = val === undefined ? null : val;
    this.left = left === undefined ? null : left;
    this.right = right === undefined ? null : right;
    this.next = next === undefined ? null : next;
};
// 填充它的每个next指针,让这个指针指向其下一个右侧节点。如果找不到下一个右侧节点，则将next指针设置为NULL。
/*
 * @param {Node} root
 * @return {Node}
*/
var connect = function(root) {
    if(!root)
        return null;
    var ret = root,p1;
    while(root.left){
        p1 = root;
        while(p1.next){
            p1.left.next = p1.right;
            p1.right.next = p1.next.left;
            p1 = p1.next;
        }
        p1.left.next = p1.right;
        root = root.left;
    }
    return ret;
};
// 字符串相乘
/*
 * @param {string} num1
 * @param {string} num2
 * @return {string}
*/
var multiply = function(num1, num2) {
    if(num1==="0"||num2==="0")
        return "0";
    const size1 = num1.length,size2 = num2.length,code_zero = '0'.charCodeAt(0);
    const res = new Array(size1+size2).fill(0);
    var back = size1+size2-1;
    for(let i =0;i<size1;++i)
        for(let j =0;j<size2;++j)
            res[i+j+1] += (num1.codePointAt(i) - code_zero)*(num2.codePointAt(j)-code_zero);
    for(let i =back;i>0;--i){
        res[i-1] += Math.floor(res[i]/10);
        res[i] %= 10;
    }
    if(res[0]===0)
        res[0] = '';
    return res.join('');
};
// 元素的 频数 是该元素在一个数组中出现的次数。
// 给你一个整数数组 nums 和一个整数 k 。在一步操作中，你可以选择 nums 的一个下标，并将该下标对应元素的值增加 1 。
// 执行最多 k 次操作后，返回数组中最高频元素的 最大可能频数 。
/*
 * @param {number[]} nums
 * @param {number} k
 * @return {number}
*/
var maxFrequency = function(nums, k) {
    nums.sort((a,b)=>a-b);
    const size = nums.length;
    var sum = 0,ret=1,left = 0;
    for(let right = 1;right<size;++right){
        sum += (nums[right]-nums[right-1])*(right - left);
        while(sum>k){
            sum -= nums[right] - nums[left];
            left += 1;
        }
        ret = Math.max(ret,right-left+1);
    }
    return ret;
};
// 给定一个数字，我们按照如下规则把它翻译为字符串：0 翻译成 “a” ，1 翻译成 “b”，……，11 翻译成 “l”，……，25 翻译成 “z”。
// 一个数字可能有多个翻译。请编程实现一个函数，用来计算一个数字有多少种不同的翻译方法。
/*
 * @param {number} num
 * @return {number}
*/
var translateNum = function(num) {
    if(num<10)
        return 1;
    var tmp = num%100;
    if(tmp>=10&&tmp<26)
        return translateNum(Math.floor(num/10))+translateNum(Math.floor(num/100));
    else
        return translateNum(Math.floor(num/10));
};
/*
 * @param {TreeNode} root
 * @return {string[]}
*/
var binaryTreePaths = function(root) {
    const paths = new Array();
    const construct_path = function(root,path){
        if(root){
            path += root.val.toString();
            if(!root.left&&!root.right)
                paths.push(path);
            else{
                path += "->";
                construct_path(root.left,path);
                construct_path(root.right,path);
            }
        }
    }
    construct_path(root,'');
    return paths;
};
// 给定一个由 0 和 1 组成的矩阵 mat ，请输出一个大小相同的矩阵，其中每一个格子是 mat 中对应位置元素到最近的 0 的距离。
// 两个相邻元素间的距离为 1 。
/*
 * @param {number[][]} mat
 * @return {number[][]}
*/
var updateMatrix = function(mat) {
    const queue = [];
    const n = mat.length,m = mat[0].length;
    const ans = new Array(n).fill(new Array(m).fill(-1));
    for(let i=0;i<n;++i){
        for(let j=0;j<m;++j){
            if(mat[i][j]===0){
                ans[i][j] = 0;
                queue.push([i,j]);
            }
        }
    }
    var x,y;
    --n;--m;
    while(queue.length){
        x = queue[0][0],y = queue[0][1];
        queue.shift();
        if(x>0&&ans[x-1][y]!==-1){
            ans[x-1][y] = ans[x][y] +1;
            queue.push([x-1,y]);
        }
        if(y>0&&ans[x][y-1]!==-1){
            ans[x][y-1] = ans[x][y] +1;
            queue.push([x,y-1]);
        }
        if(x<n&&ans[x+1][y]!==-1){
            ans[x+1][y] = ans[x][y] +1;
            queue.push([x+1,y]);
        }
        if(y<m&&ans[x][y]!==-1){
            ans[x][y+1] = ans[x][y] +1;
            queue.push([x,y+1]);
        }
    }
    return ans;
};
// 在给定的网格中，每个单元格可以有以下三个值之一：值 0 代表空单元格；值 1 代表新鲜橘子；值 2 代表腐烂的橘子。
// 每分钟，任何与腐烂的橘子（在 4 个正方向上）相邻的新鲜橘子都会腐烂。
// 返回直到单元格中没有新鲜橘子为止所必须经过的最小分钟数。如果不可能，返回 -1。
/*
 * @param {number[][]} grid
 * @return {number}
*/
var orangesRotting = function(grid) {
    const m =grid.length, n =grid[0].length;
    const queue = [],dis = new Array(m);
    for(let i=0;i<m;++i)
        dis[i] = new Array(n).fill(-1);
    var cnt,x,y,ans;
    for(let i=0;i<m;++i){
        for(let j=0;j<n;++j){
            if(grid[i][j]===2){
                queue.push([i,j]);
                dis[i][j] = 0;
            }
            if(grid[i][j]!==0)
                ++cnt;
        }
    }
    --m;--n;
    while(queue.length){
        x = queue[0][0],y = queue[0][1];
        queue.shift();
        --cnt;
        ans = dis[x][y];
        if(x>0&&dis[x-1][y]===-1&&grid[x-1][y]!==0){
            dis[x-1][y] = dis[x][y]+1;
            queue.push([x-1,y]);
        }
        if(y>0&&dis[x][y-1]===-1&&grid[x][y-1]!==0){
            dis[x][y-1] = dis[x][y]+1;
            queue.push([x,y-1]);
        }
        if(x<m&&dis[x+1][y]===-1&&grid[x+1][y]!==0){
            dis[x+1][y] = dis[x][y]+1;
            queue.push([x+1,y]);
        }
        if(y<n&&dis[x][y+1]===-1&&grid[x][y+1]!==0){
            dis[x][y+1] = dis[x][y]+1;
            queue.push([x,y+1]);
        }
    }
    return (cnt===0)? ans:-1;
};
// 所有 DNA 都由一系列缩写为 'A'，'C'，'G' 和 'T' 的核苷酸组成，例如："ACGAATTCCG"。
// 编写一个函数来找出所有目标子串，目标子串的长度为 10，且在 DNA 字符串 s 中出现次数超过一次。
/*
 * @param {string} s
 * @return {string[]}
*/
var findRepeatedDnaSequences = function(s) {
    const result = [];
    const map = new Map();
    for (let i = 0; i <= s.length - 10; i++) {
        const sub_str = s.slice(i, i + 10);
        if (map.get(sub_str) === 1) 
            result.push(sub_str);
        map.set(sub_str, map.has(sub_str) ? map.get(sub_str) + 1 : 1);
    }
    return result;
};
// 一个数对 (a,b) 的 数对和 等于 a + b 。最大数对和 是一个数对数组中最大的 数对和
// 给你一个长度为 偶数 n 的数组 nums ，请你将 nums 中的元素分成 n / 2 个数对nums 中每个元素 恰好 在 一个 数对中，且
// 最大数对和 的值 最小 。请你在最优数对划分的方案下，返回最小的 最大数对和 
/*
 * @param {number[]} nums
 * @return {number}
*/
var minPairSum = function(nums) {
    const size = nums.length,mid = Math.floor(size/2);
    let res = 0;
    nums.sort((a, b)=>a-b);
    for (let i = 0;i<mid; i++) 
        res = Math.max(res, nums[i] + nums[size-1-i]);
    return res;
};
/*
 * @param {ListNode} headA
 * @param {ListNode} headB
 * @return {ListNode}
*/
var getIntersectionNode = function(headA, headB) {
    if(!headB||!headA)
        return null;
    var p1 = headA,p2 = headB;
    while(p1!==p2){
        p1 = (p1)? p1.next:headB;
        p2 = (p2)? p2.next:headA;
    }
    return p1;
};
/*
 * @param {ListNode} head
 * @return {ListNode}
*/
var deleteDuplicates = function(head) {
    if(!head)
        return null;
    var root = new ListNode(-101,head),ret = root;
    var tmp_val,tmp_node;
    while(root.next){
        if(root.next.next&&root.next.val===root.next.next.val){
            tmp_node = root.next;tmp_val = root.next.val;
            while(tmp_node&&tmp_node.val===tmp_val)
                tmp_node = tmp_node.next;
            root.next = tmp_node;
        }
        else
            root = root.next;
    }
    return ret.next;
};
/*
 * @param {number[]} nums
 * @return {number[][]}
*/
var permute = function(nums) {
    const res = new Array();
    var permute_assist = function(first,len){
        if(first===len){
            res.push(Array.from(nums));
            return ;
        }
        var tmp;
        permute_assist(first+1,len);
        for(let i=first+1;i<len;++i){
            tmp = nums[i];
            nums[i] = nums[first];
            nums[first] = tmp;
            permute_assist(first+1,len);
            tmp = nums[i];
            nums[i] = nums[first];
            nums[first] = tmp;
        }
    }
    permute_assist(0,nums.length);
    return res;
};
/*
 * @param {string} s
 * @return {string[]}
*/
var letterCasePermutation = function(s) {
    const str_arr =s.split('');
    const size = str_arr.length;
    const ans = new Array();
    const permute_assist = function(i){
        if(i===size){
            ans.push(str_arr.join(''));
            return ;
        }
        permute_assist(i+1);
        if(isNaN(str_arr[i])){
            var tmp = str_arr[i].toLowerCase();
            if(tmp===str_arr[i]){
                str_arr[i] = str_arr[i].toUpperCase();
                permute_assist(i+1);
                str_arr[i] = str_arr[i].toLowerCase();
            }
            else{
                str_arr[i] = tmp;
                permute_assist(i+1);
                str_arr[i] = str_arr[i].toLowerCase();
            }
        } 
    }
    permute_assist(0);
    return ans;
};
// 给定两个整数n和k,返回范围[1,n]中所有可能的k个数的组合。你可以按任何顺序返回答案。
/*
 * @param {number} n
 * @param {number} k
 * @return {number[][]}
*/
var combine = function(n, k) {
    const tmp = new Array(),ans = new Array();
    const combine_assist = function(num,i){
        if(i===0){
            ans.push(Array.from(tmp));
            return ;
        }
        else if(num<i)
            return ;
        else{
            tmp.push(num);
            combine_assist(num-1,k-1);
            tmp.pop();
            combine_assist(num-1,k);
        }
    }
    combine_assist(n,k);
    return ans;
};
// 给定一个仅包含数字 2-9 的字符串，返回所有它能表示的字母组合。答案可以按 任意顺序 返回。
// 给出数字到字母的映射如下（与电话按键相同）。注意 1 不对应任何字母。
/*
 * @param {string} digits
 * @return {string[]}
*/
var letterCombinations = function(digits) {
    const size = digits.length;
    if(size===0)
        return [];
    const  letterMap = ["", "", "abc","def","ghi","jkl","mno","pqrs","tuv","wxyz"];
    const code_zero = '0'.codePointAt(0),tmp_arr = [],ans =[];
    var tmp;
    const assist = function(i){
        if(i===size){
            ans.push(tmp_arr.join(''));
            return ;
        }
        tmp = digits.codePointAt(i) -code_zero;
        for(const ch of letterMap[tmp]){
            tmp_arr.push(ch);
            assist(i+1);
            tmp_arr.pop();
        }
    }
    assist(0);
    return ans;
};
// 爬楼梯的方案数，一次可以爬一阶或者二阶
// @param {number} n @return {number}
var climbStairs = function(n) {
    let p = 0, q = 0, r = 1;
    for (let i = 1; i <= n; ++i) {
        p = q;
        q = r;
        r = p + q;
    }
    return r;
};
// 检查[left,right]中的整数是否都被ranges[i] = [start,end]覆盖
/*
 * @param {number[][]} ranges
 * @param {number} left
 * @param {number} right
 * @return {boolean}
*/
var isCovered = function(ranges, left, right) {
    const diff = new Array(52).fill(0);
    for(const [l,r] of ranges){
        ++diff[l];--diff[r+1];
    }
    var curr = 0;
    for(let i=0;i<51;++i){
        curr += diff[i];
        if(left<=i&&i<=right&&curr<=0)
            return false;
    }
    return true;
};
// 编写一个函数,输入是一个无符号整数(以二进制串的形式),返回其二进制表达式中数字位数为'1'的个数(也被称为汉明重量).
//  @param {number} n - a positive integer @return {number}
var hammingWeight = function(n) {
    let ret = 0;
    while (n) {
        n &= n - 1;
        ret++;
    }
    return ret;
};