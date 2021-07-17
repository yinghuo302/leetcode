/*
 * @Author: zanilia
 * @Date: 2021-07-14 20:31:29
 * @LastEditTime: 2021-07-17 19:19:39
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
/*
 * Definition for singly-linked list.
 * function ListNode(val, next) {
 *     this.val = (val===undefined ? 0 : val)
 *     this.next = (next===undefined ? null : next)
 * }
*/
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