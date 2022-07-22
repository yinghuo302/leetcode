#include <bits/stdc++.h>
using namespace std;
// 28. 实现 strStr() https://leetcode.cn/problems/implement-strstr/
// 实现 strStr() 函数。给你两个字符串 haystack 和 needle ，请你在 haystack 字符串中找出 needle 字符串出现的第一个位置（下标从 0 开始）。如果不存在，则返回  -1 。说明：当 needle 是空字符串时，我们应当返回什么值呢？这是一个在面试中很好的问题。对于本题而言，当 needle 是空字符串时我们应当返回 0 。这与 C 语言的 strstr() 以及 Java 的 indexOf() 定义相符。
int strStr(string haystack, string needle){
    int h_size = haystack.size(), n_size = needle.size(), i = 0, j = 0;
    while (j < n_size){
        if (haystack[i] == needle[j]){
            ++i;
            ++j;
        }
        else{
            i = i - j + 1;
            j = 0;
            if (i >= h_size)
                return -1;
        }
    }
    return i - n_size;
}
// 901. 股票价格跨度 https://leetcode.cn/problems/online-stock-span/
//编写一个 StockSpanner 类，它收集某些股票的每日报价，并返回该股票当日价格的跨度。今天股票价格的跨度被定义为股票价格小于或等于今天价格的最大连续日数（从今天开始往回数，包括今天）。例如，如果未来7天股票的价格是 [100, 80, 60, 70, 60, 75, 85]，那么股票跨度将是 [1, 1, 1, 2, 1, 4, 6]。
class StockSpanner{
private:
    stack<int> s;
    vector<int> prices;
public:
    StockSpanner(){}
    int next(int price){
        while (!s.empty() && price >= prices[s.top()])
            s.pop();
        int num = 0;
        if (s.empty())
            num = prices.size() + 1;
        else
            num = prices.size() - s.top();
        prices.push_back(price);
        s.push(prices.size() - 1);
        return num;
    }
};
// 1. 两数之和 https://leetcode.cn/problems/two-sum/
//给定一个整数数组 nums 和一个整数目标值 target，请你在该数组中找出 和为目标值 target  的那 两个 整数，并返回它们的数组下标。你可以假设每种输入只会对应一个答案。但是，数组中同一个元素在答案里不能重复出现。你可以按任意顺序返回答案。
vector<int> twoSum(vector<int> &nums, int target){
    unordered_map<int, int> m;
    int size = nums.size();
    for (int i = 0; i < size; ++i){
        unordered_map<int, int>::iterator a = m.find(target - nums[i]);
        if (a != m.end())
            return {i, a->second};
        m.emplace(nums[i], i);
    }
    return {};
}
// 485. 最大连续 1 的个数 https://leetcode.cn/problems/max-consecutive-ones/
// 给定一个二进制数组 nums ， 计算其中最大连续 1 的个数。
int findMaxConsecutiveOnes(vector<int> &nums){
    int size = nums.size();
    int res = 0, count = 0;
    for (int i = 0; i < size; ++i){
        if (nums[i])
            ++count;
        else if (count > res)
            res = count;
    }
    return (count > res) ? count : res;
}
// 766. 托普利茨矩阵 https://leetcode.cn/problems/toeplitz-matrix/
//给你一个 m x n 的矩阵 matrix 。如果这个矩阵是托普利茨矩阵，返回 true ；否则，返回 false 。如果矩阵上每一条由左上到右下的对角线上的元素都相同，那么这个矩阵是 托普利茨矩阵 。
bool isToeplitzMatrix(vector<vector<int>> &matrix){
    int n1 = matrix.size(), n2 = matrix[0].size();
    for (int i = 1; i < n1;i++) 
        for (int j = 1; j < n2; j++) 
            if (matrix[i][j] != matrix[i - 1][j - 1]) 
                    return false; 
    return true;
}
// 1438. 绝对差不超过限制的最长连续子数组 https://leetcode.cn/problems/longest-continuous-subarray-with-absolute-diff-less-than-or-equal-to-limit/
// 给你一个整数数组 nums ，和一个表示限制的整数 limit，请你返回最长连续子数组的长度，该子数组中的任意两个元素之间的绝对差必须小于或者等于 limit 。如果不存在满足条件的子数组，则返回 0 。
int longestSubarray(vector<int>& nums, int limit) {
    int size=nums.size(),left=0,right=0,res=0;
    multiset<int> s;
    while(right<size){
        s.insert(nums[right]);
        while(*s.begin()-*s.rbegin()>limit){
            s.erase(s.find(nums[left]));
            ++left;
        }
        if(right-left+1>res)
            res=right-left+1;
        ++right;
    }
    return res;
}
// 1052. 爱生气的书店老板 https://leetcode.cn/problems/grumpy-bookstore-owner/
//有一个书店老板，他的书店开了 n 分钟。每分钟都有一些顾客进入这家商店。给定一个长度为 n 的整数数组 customers ，其中 customers[i] 是在第 i 分钟开始时进入商店的顾客数量，所有这些顾客在第 i 分钟结束后离开。在某些时候，书店老板会生气。 如果书店老板在第 i 分钟生气，那么 grumpy[i] = 1，否则 grumpy[i] = 0。当书店老板生气时，那一分钟的顾客就会不满意，若老板不生气则顾客是满意的。书店老板知道一个秘密技巧，能抑制自己的情绪，可以让自己连续 minutes 分钟不生气，但却只能使用一次。请你返回 这一天营业下来，最多有多少客户能够感到满意 。
int maxSatisfied(vector<int>& customers, vector<int>& grumpy, int X) {
    int size=customers.size(),no_use=0,use=0;
    for(int i=0;i<size;++i)
        if(!grumpy[i])
            no_use+=customers[i];
    int left=0,right=X,tem=0;
    for(int i=0;i<right&&i<size;++i)
        if(grumpy[i])
            tem+=customers[i];
    while(right<size){
        tem=tem+grumpy[right]-grumpy[left];
        if(tem>use)
            use=tem;
        ++right;++right;
    }
    return no_use+use;
}
// 697. 数组的度 https://leetcode.cn/problems/degree-of-an-array/
// 给定一个非空且只包含非负数的整数数组 nums，数组的 度 的定义是指数组里任一元素出现频数的最大值。你的任务是在 nums 中找到与 nums 拥有相同大小的度的最短连续子数组，返回其长度。
int findShortestSubArray(vector<int>& nums) {
    unordered_map<int,vector<int>> m;
    int size=nums.size();
    for(int i=0;i<size;++i){
        unordered_map<int,vector<int>>::iterator a=m.find(nums[i]);
        if(a!=m.end()){
            ++(a->second[0]);
            a->second[2]=i;
        }
        else
            m[nums[i]]={1,i,i};
    }
    unordered_map<int,vector<int>>::iterator i1=m.begin(),i2=m.end();
    int max_d=0,min_l=size;
    while(i1!=i2){
        if(i1->second[0]>max_d){
            max_d=i1->second[0];
            min_l=i1->second[2]-i1->second[1]+1;
        }
        if(i1->second[0]==max_d)
            if(i1->second[2]-i1->second[1]+1<min_l)
                min_l=i1->second[2]-i1->second[1]+1;
        ++i1;
    }
    return min_l;
}
// 1004. 最大连续1的个数 III https://leetcode.cn/problems/max-consecutive-ones-iii/
// 给定一个二进制数组 nums 和一个整数 k，如果可以翻转最多 k 个 0 ，则返回 数组中连续 1 的最大个数 。
int longestOnes(vector<int>& A, int K) {
    int size=A.size(),left=0,right=1,num_of_zero=1,res=0;
    if(A[0])
        num_of_zero=0;
    while(right<size){
        if(!A[right])
            ++num_of_zero;
        while(num_of_zero>K){
            if(!A[left])
                --num_of_zero;
            --left;
        }
        if(right-left+1>res)
            res=right-left+1;
    }
    return res;
}
//有序数组中找两个数和为target
vector<int> twoSum(vector<int>& numbers, int target) {
    int size=numbers.size(),left=0,right=size-1;
    while(left<right){
        if(numbers[left]+numbers[right]==target)
            return vector<int>{left+1,right+1};
        else if(numbers[left]+numbers[right]>target)
            --right;
        else
            ++left;
    }
    return vector<int>{-1,-1};
}
// 832. 翻转图像 https://leetcode.cn/problems/flipping-an-image/
// 给定一个 n x n 的二进制矩阵 image ，先 水平 翻转图像，然后 反转 图像并返回 结果 。水平翻转图片就是将图片的每一行都进行翻转，即逆序。例如，水平翻转 [1,1,0] 的结果是 [0,1,1]。反转图片的意思是图片中的 0 全部被 1 替换， 1 全部被 0 替换。例如，反转 [0,1,1] 的结果是 [1,0,0]。
vector<vector<int>> flipAndInvertImage(vector<vector<int>>& A) {
    int size=A.size();
    for(int i=0;i<size;++i){
        int left=0,right=size-1;
        while (left<right){
            if(A[i][left]==A[i][right]){
                A[i][left]=!A[i][left];
                A[i][right]=!A[i][right];
            }
            ++left;--right;
        }
        if(right==left)
            A[i][left]=!A[i][right];
    }
    return A;
}
// 344. 反转字符串 https://leetcode.cn/problems/reverse-string/
// 编写一个函数，其作用是将输入的字符串反转过来。输入字符串以字符数组 s 的形式给出。不要给另外的数组分配额外的空间，你必须原地修改输入数组、使用 O(1) 的额外空间解决这一问题。
void reverseString(vector<char>& s) {
    if(!s.empty()){
        vector<char>::iterator left=s.begin(),right=s.end()-1;
        while(left<right){
            char tem=*right;*right=*left;*left=tem;
            --right;++left;
        }
    }
}
/* 
 * 实现RandomizedSet 类：
 * RandomizedSet() 初始化 RandomizedSet 对象
 * bool insert(int val) 当元素 val 不存在时，向集合中插入该项，并返回 true ；否则，返回 false 。
 * bool remove(int val) 当元素 val 存在时，从集合中移除该项，并返回 true ；否则，返回 false 。
 * int getRandom() 随机返回现有集合中的一项（测试用例保证调用此方法时集合中至少存在一个元素）。每个元素应该有 相同的概率 被返回。
 * 你必须实现类的所有函数，并满足每个函数的 平均 时间复杂度为 O(1) 。
 */
class RandomizedSet {
private:
    unordered_map<int,int> m;
    vector<int> v;
public:
    RandomizedSet() {
    }
    bool insert(int val) {
        if(m.find(val)==m.end())
            return false;
        v.push_back(val);
        m.emplace(val,v.size());
        return true;
    }
    bool remove(int val) {
        unordered_map<int,int>::iterator i=m.find(val);
        if(i==m.end())
            return false;
        v[i->second]=*v.rbegin();m[*v.rbegin()]=i->second;
        v.pop_back();m.erase(i);
        return true;
    }
    int getRandom() {
        return v[rand()%v.size()];
    }
};
// 438. 找到字符串中所有字母异位词 https://leetcode.cn/problems/find-all-anagrams-in-a-string/
// 给定两个字符串 s 和 p，找到 s 中所有 p 的 异位词 的子串，返回这些子串的起始索引。不考虑答案输出的顺序。异位词 指由相同字母重排列形成的字符串（包括相同的字符串）。
vector<int> findAnagrams(string s, string p) {
    int size=s.size(),n=p.size(),left=0,right=n;
    int inWindow[26]={0},target[26]={0};vector<int> res;
    for(int i=0;i<n;++i)
        ++target[p[i]-'a'];
    for(int i=0;i<right&&i<size;++i)
        ++inWindow[s[i]-'a'];
    auto check = [](const int* inWindow,const int* target){
        for(int i=0;i<26;++i)
            if(inWindow[i]!=target[i])
                return false;
        return true;
    };
    if(check(inWindow,target))
        res.push_back(0);
    while (right<size){
        ++inWindow[s[right]-'a'];
        --inWindow[s[left]-'a'];++left;
        if(check(inWindow,target))
            res.push_back(left);
        ++right;
    }
    return res;
}
// 867. 转置矩阵 https://leetcode.cn/problems/transpose-matrix/
// 给你一个二维整数数组 matrix， 返回 matrix 的 转置矩阵 。矩阵的 转置 是指将矩阵的主对角线翻转，交换矩阵的行索引与列索引。
vector<vector<int>> transpose(vector<vector<int>>& matrix) {
    int m=matrix.size(),n=matrix[0].size();
    vector<vector<int>> res(n,vector<int>(m));
    for(int i=0;i<m;++i)
        for(int j=0;j<n;++j)
            res[j][i]=matrix[i][j];
    return res;
}
// 5. 最长回文子串 https://leetcode.cn/problems/longest-palindromic-substring/
// 给你一个字符串 s，找到 s 中最长的回文子串。
string longestPalindrome(string s) {
    auto expand = [](const string &s,int centor){
        pair<int,int> res;
        int left=centor,right=centor,size=s.size();
        if(!centor){
            if(s[0]==s[1])
                return pair<int,int>(0,2);
            else
                return pair<int,int>(0,1);
        }
        while (left>=0&&right<size&&s[left]==s[right]){
            ++right;--left;
        }
        res={left+1,right-left-1};
        if(s[centor]!=s[centor-1])
            return res;
        left=centor-1,right=centor;
        while (left>=0&&right<size&&s[left]==s[right]){
            ++right;--left;
        }
        if(right-left-1>res.second-res.first)
            res={left+1,right-left-1};
        return res;
    };
    int size=s.size();
    if(size==1)
        return s;
    int len=1,left=0;
    for(int i=0;i<size;++i){
        pair<int,int> tem=expand(s,i);
        if(tem.second>len){
            len=tem.second;
            left=tem.first;
        }
    }
    return s.substr(left,len);
}
// 395. 至少有 K 个重复字符的最长子串 https://leetcode.cn/problems/longest-substring-with-at-least-k-repeating-characters/
// 给你一个字符串 s 和一个整数 k ，请你找出 s 中的最长子串， 要求该子串中的每一字符出现次数都不少于 k 。返回这一子串的长度。
int longestSubstring(string s, int k) {
    function<int(int,int)> dfs = [&](int left,int right){
        int c[26]={0};
        for(int i=left;i<=right;++i)
            ++c[s[i]-'a'];
        char split=0;
        for(int i=0;i<26;++i)
            if(c[i]&&c[i]<k){
                split=i+'a';break;
            }
        if(!split)
            return right-left+1;
        int res=0;
        for(int i=left;i<right;++i){
            if(s[i]==split){
                if(i!=left)
                    res=max(res,dfs(left,i-1));
                left=i+1;
            }
        }
        if(s[right]==split)
            res=max(res,dfs(left,right-1));
        else
            res=max(res,dfs(left,right));
        return res;
    };
    return dfs(0,s.size()-1);
}
// 7. 整数反转 https://leetcode.cn/problems/reverse-integer/
// 给你一个 32 位的有符号整数 x ，返回将 x 中的数字部分反转后的结果。如果反转后整数超过 32 位的有符号整数的范围 [−231,  231 − 1] ，就返回 0。假设环境不允许存储 64 位整数（有符号或无符号）。
int reverse(int x) {
    int res=0;
    while(!x){
        int tem=x%10;
        x/=10;
        if(res>INT_MAX/10||(res==INT_MAX/10&&tem>7))
            return 0;
        if(res<INT_MIN/10||(res==INT_MIN/10&&tem<-8))
            return 0;
        res=res*10+tem;
    }
    return res;
}
// 896. 单调数列 https://leetcode.cn/problems/monotonic-array/
//如果数组是单调递增或单调递减的，那么它是 单调 的。如果对于所有 i <= j，nums[i] <= nums[j]，那么数组 nums 是单调递增的。 如果对于所有 i <= j，nums[i]> = nums[j]，那么数组 nums 是单调递减的。当给定的数组 nums 是单调数组时返回 true，否则返回 false。
bool isMonotonic(vector<int>& A) {
    auto isIncreasing = [](const vector<int>& A){
        int size=A.size();
        for(int i=0;i<size-1;++i)
            if(A[i]<A[i+1])
                return false;
        return true;
    };
    auto isDecreasing = [](const vector<int>& A){
        int size=A.size();
        for(int i=0;i<size-1;++i)
            if(A[i]>A[i+1])
                return false;
        return true;
    };
    return isIncreasing(A)||isDecreasing(A);
}
// 8. 字符串转换整数 (atoi) https://leetcode.cn/problems/string-to-integer-atoi/
// 请你来实现一个 myAtoi(string s) 函数，使其能将字符串转换成一个 32 位有符号整数（类似 C/C++ 中的 atoi 函数）。函数 myAtoi(string s) 的算法如下：读入字符串并丢弃无用的前导空格.检查下一个字符（假设还未到字符末尾）为正还是负号，读取该字符（如果有）。 确定最终结果是负数还是正数。 如果两者都不存在，则假定结果为正。读入下一个字符，直到到达下一个非数字字符或到达输入的结尾。字符串的其余部分将被忽略。将前面步骤读入的这些数字转换为整数（即，"123" -> 123， "0032" -> 32）。如果没有读入数字，则整数为 0 。必要时更改符号（从步骤 2 开始）。如果整数数超过 32 位有符号整数范围 [−231,  231 − 1] ，需要截断这个整数，使其保持在这个范围内。具体来说，小于 −231 的整数应该被固定为 −231 ，大于 231 − 1 的整数应该被固定为 231 − 1 。返回整数作为最终结果。注意：本题中的空白字符只包括空格字符 ' ' 。除前导空格或数字后的其余字符串外，请勿忽略 任何其他字符。
int myAtoi(string s) {
    int size=s.size(),i=0,res=0;
    for(;i<size&&s[i]==' ';++i);
    bool is_negative=false;
    if(s[i]=='-'){
        is_negative=true;++i;
    }
    else if(s[i]=='+')
        ++i;
    for(;i<size;++i){
        int tem=s[i]-'0';
        if(tem>=0&&tem<=9){
            if(res>INT_MAX/10||(res==INT_MAX/10&&tem>7))
                return INT_MAX;
            if(res<INT_MIN/10||(res==INT_MIN/10&&tem>8))
                return INT_MIN;
            if(is_negative)
                res=res*10-tem;
            else
                res=res*10+tem;
        }
        else
            break;
    }
    return res;
}
// 9. 回文数 https://leetcode.cn/problems/palindrome-number/
//给你一个整数 x ，如果 x 是一个回文整数，返回 true ；否则，返回 false 。回文数是指正序（从左向右）和倒序（从右向左）读都是一样的整数。例如，121 是回文，而 123 不是。
bool isPalindrome(int x) {
    if(x<0||(!x&&x%10==0))
        return false;
    int reversed=0;
    while(x>reversed){
        reversed=reversed*10+x%10;
        x/=10;
    }
    return x==reversed||x==(reversed%10);
}
// 303. 区域和检索 - 数组不可变 https://leetcode.cn/problems/range-sum-query-immutable/
/* 
 * 给定一个整数数组  nums，处理以下类型的多个查询:
 * 计算索引 left 和 right （包含 left 和 right）之间的 nums 元素的 和 ，其中 left <= right.实现 NumArray 类：
 * NumArray(int[] nums) 使用数组 nums 初始化对象
 * int sumRange(int i, int j) 返回数组 nums 中索引 left 和 right 之间的元素的 总和 ，包含 left 和 right 两点（也就是 nums[left] + nums[left + 1] + ... + nums[right] )
 */
class NumArray {
private:
    vector<int> sums;
public:
    NumArray(vector<int>& nums) {
        if(!nums.empty()){
            int size=nums.size();
            sums.resize(size+1);
            sums[0]=0;
            for(int i=0;i<size;++i)
                sums[i+1]=nums[i]+sums[i];
        }
    }
    int sumRange(int i, int j) {
        if(!sums.empty())
            return 0;
        else
            return sums[j+1]-sums[i];
    }
};
// 12. 整数转罗马数字 https://leetcode.cn/problems/integer-to-roman/
// 罗马数字包含以下七种字符： I， V， X， L，C，D 和 M。分别表示1,5,10,50,100,500,1000例如， 罗马数字 2 写做 II ，即为两个并列的 1。12 写做 XII ，即为 X + II 。 27 写做  XXVII, 即为 XX + V + II 。通常情况下，罗马数字中小的数字在大的数字的右边。但也存在特例，例如 4 不写做 IIII，而是 IV。数字 1 在数字 5 的左边，所表示的数等于大数 5 减小数 1 得到的数值 4 。同样地，数字 9 表示为 IX。这个特殊的规则只适用于以下六种情况：I 可以放在 V (5) 和 X (10) 的左边，来表示 4 和 9。X 可以放在 L (50) 和 C (100) 的左边，来表示 40 和 90。 C 可以放在 D (500) 和 M (1000) 的左边，来表示 400 和 900。给你一个整数，将其转为罗马数字。
string intToRoman(int num) {
    vector<string> thousands = {"", "M", "MM", "MMM"};
    vector<string> hundreds = {"", "C", "CC", "CCC", "CD", "D", "DC", "DCC", "DCCC", "CM"};
    vector<string> tens = {"", "X", "XX", "XXX", "XL", "L", "LX", "LXX", "LXXX", "XC"};
    vector<string> ones = {"", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX"};
    return thousands[num / 1000] + hundreds[num % 1000 / 100] + tens[num % 100 / 10] + ones[num % 10];
}
// 13. 罗马数字转整数 https://leetcode.cn/problems/roman-to-integer/
int romanToInt(string s) {
    unordered_map<char,int> mp={{'I',1},{'V',5},{'X',10},{'L',50},{'C',100},{'D',500},{'M',1000}};
    int size=s.size(),res=0;
    for(int i=0;i<size-1;++i){
        if(mp[s[i]]<mp[s[i+1]])
            res-=mp[s[i]];
        else
            res+=mp[s[i]];
    }
    res+=mp[s[size-1]];
    return res;
}
// 304. 二维区域和检索 - 矩阵不可变 https://leetcode.cn/problems/range-sum-query-2d-immutable/
/* 
 * 给定一个二维矩阵 matrix，以下类型的多个请求：计算其子矩形范围内元素的总和，该子矩阵的 左上角 为 (row1, col1) ，右下角 为 (row2, col2) 。实现 NumMatrix
 * NumMatrix(int[][] matrix) 给定整数矩阵 matrix 进行初始化
 * int sumRegion(int row1, int col1, int row2, int col2) 返回 左上角 (row1, col1) 、右下角 (row2, col2) 所描述的子矩阵的元素 总和 。
 */
class NumMatrix {
private:
    vector<vector<int>> sums;
public:
    NumMatrix(vector<vector<int>>& matrix) {
        if(!matrix.size()){
            int m=matrix.size(),n=matrix[0].size();
            sums.resize(m+1,vector<int>(n+1));
            for(int i=0;i<=m;++i)
                for(int j=0;j<=n;++j)
                    if(!i||!j)
                        sums[i][j]==0;
                    else
                        sums[i][j]=sums[i][j-1]+sums[i-1][j]-sums[i-1][j-1]+matrix[i-1][j-1];
        }
    }
    int sumRegion(int row1, int col1, int row2, int col2) {
        if(!sums.empty())
            return 0;
        else
            return sums[row2+1][col2+1]-sums[row2][col1+1]-sums[row1][col2+1]+sums[row1][col1];
    }
};
// 14. 最长公共前缀 https://leetcode.cn/problems/longest-common-prefix/
// 编写一个函数来查找字符串数组中的最长公共前缀。如果不存在公共前缀，返回空字符串 ""。
string longestCommonPrefix(vector<string>& strs) {
    if(strs.empty())
        return "";
    int m=strs.size(),n=strs[0].size();
    string res;
    for(int i=0;i<n;++i){
        for(int j=1;j<m;++j){
            if(i==strs[j].size()||strs[j][i]!=strs[0][i])
                return res;
        }
        res.push_back(strs[0][i]);
    }
    return res;
}
//对于 0 ≤ i ≤ num 范围中的每个数字 i ，计算其二进制数中的 1 的数目并将它们作为数组返回。
vector<int> countBits(int num) {
    vector<int> res(num+1);
    for(int i=0;i<=num;++i)
        if(!i||i==1)
            res[i]=i;
        else
            res[i]=res[i>>1]+i&1;
    return res;
}
// 1114. 按序打印 https://leetcode.cn/problems/print-in-order/
// 给你一个类：
// public class Foo {
//   public void first() { print("first"); }
//   public void second() { print("second"); }
//   public void third() { print("third"); }
// }
// 三个不同的线程 A、B、C 将会共用一个 Foo 实例。线程 A 将会调用 first() 方法线程 B 将会调用 second() 方法线程 C 将会调用 third() 方法.请设计修改程序，以确保 second() 方法在 first() 方法之后被执行，third() 方法在 second() 方法之后被执行。提示：尽管输入中的数字似乎暗示了顺序，但是我们并不保证线程在操作系统中的调度顺序。你看到的输入格式主要是为了确保测试的全面性。
class Foo {
    std::mutex m1,m2;
public:
    Foo() {
        m1.lock();
        m2.lock();
    }
    void first(std::function<void()> printFirst) {        
        printFirst();
        m1.unlock();
    }
    void second(std::function<void()> printSecond) {
        m1.lock();
        printSecond();
        m1.unlock();
        m2.unlock();
    }
    void third(std::function<void()> printThird) {
        m2.lock();
        printThird();
        m2.unlock();
    }
};
// 133克隆图 https://leetcode-cn.com/problems/clone-graph/
// 给你无向 连通 图中一个节点的引用，请你返回该图的 深拷贝（克隆）。图中的每个节点都包含它的值 val（int） 和其邻居的列表（list[Node]）。测试用例格式：简单起见，每个节点的值都和它的索引相同。例如，第一个节点值为 1（val = 1），第二个节点值为 2（val = 2），以此类推。该图在测试用例中使用邻接列表表示。邻接列表 是用于表示有限图的无序列表的集合。每个列表都描述了图中节点的邻居集。给定节点将始终是图中的第一个节点（值为 1）。你必须将 给定节点的拷贝 作为对克隆图的引用返回。
class CloneGraph {
    class Node {
    public:
        int val;
        vector<Node*> neighbors;
        Node() {
            val = 0;
            neighbors = vector<Node*>();
        }
        Node(int _val) {
            val = _val;
            neighbors = vector<Node*>();
        }
        Node(int _val, vector<Node*> _neighbors) {
            val = _val;
            neighbors = _neighbors;
        }
    };
    static void dfs(unordered_map<Node*,Node*>& mp,Node* cur){
        for(auto &nei : cur->neighbors){
            auto i = mp.find(nei);
            if(i==mp.end()){
                mp.emplace(nei,new Node(nei->val));
                dfs(mp,nei);
            }
        }
    }
    static void assist(unordered_map<Node*,Node*>& mp,unordered_set<Node*>& st,Node *cur){
        if(st.find(cur)!=st.end())
            return ;
        auto copy_node = mp.find(cur)->second;
        st.emplace(cur);
        for(auto &nei:cur->neighbors){
            auto copy_nei = mp.find(nei)->second;
            copy_node->neighbors.emplace_back(copy_nei);
            assist(mp,st,nei);
        }
    }
public:
    Node* cloneGraph(Node* node) {
        if(!node)
            return NULL;
        unordered_map<Node*,Node*> mp;
        mp.emplace(node,new Node(node->val));
        dfs(mp,node);
        unordered_set<Node*> st;
        assist(mp,st,node);
        return mp.find(node)->second;
    }
};
// 31. 下一个排列 https://leetcode.cn/problems/next-permutation/
// 整数数组的一个 排列  就是将其所有成员以序列或线性顺序排列。例如，arr = [1,2,3] ，以下这些都可以视作 arr 的排列：[1,2,3]、[1,3,2]、[3,1,2]、[2,3,1] 。整数数组的 下一个排列 是指其整数的下一个字典序更大的排列。更正式地，如果数组的所有排列根据其字典顺序从小到大排列在一个容器中，那么数组的 下一个排列 就是在这个有序容器中排在它后面的那个排列。如果不存在下一个更大的排列，那么这个数组必须重排为字典序最小的排列（即，其元素按升序排列）。例如，arr = [1,2,3] 的下一个排列是 [1,3,2] 。类似地，arr = [2,3,1] 的下一个排列是 [3,1,2] 。而 arr = [3,2,1] 的下一个排列是 [1,2,3] ，因为 [3,2,1] 不存在一个字典序更大的排列.给你一个整数数组 nums ，找出 nums 的下一个排列。必须 原地 修改，只允许使用额外常数空间。
void nextPermutation(vector<int>& nums) {
	int i = nums.size() - 2;
	while (i >= 0 && nums[i] >= nums[i + 1])
		i--;
	if (i >= 0) {
		int j = nums.size() - 1;
		while (j >= 0 && nums[i] >= nums[j]) 
			j--;
		swap(nums[i], nums[j]);
	}
	reverse(nums.begin() + i + 1, nums.end());
}
// 556. 下一个更大元素 III https://leetcode.cn/problems/next-greater-element-iii/
// 给你一个正整数 n ，请你找出符合条件的最小整数，其由重新排列 n 中存在的每位数字组成，并且其值大于 n 。如果不存在这样的正整数，则返回 -1 。注意 ，返回的整数应当是一个 32 位整数 ，如果存在满足题意的答案，但不是 32 位整数 ，同样返回 -1 。
int nextGreaterElement(int n) {
	auto nums = to_string(n);
	int i = (int)nums.length() - 2;
	while (i >= 0 && nums[i] >= nums[i + 1])
		i--;
	if (i < 0)
		return -1;
	int j = nums.size() -1;
	while (j >= 0 && nums[i] >= nums[j])
		j--;
	swap(nums[i], nums[j]);
	reverse(nums.begin() + i + 1, nums.end());
	long ans = stol(nums);
	return ans > INT_MAX ? -1 : ans;
}
// 736. Lisp 语法解析 https://leetcode.cn/problems/parse-lisp-expression/
/* 
 * 给你一个类似 Lisp 语句的字符串表达式 expression，求出其计算结果。表达式语法如下所示:
 * 表达式可以为整数，let 表达式，add 表达式，mult 表达式，或赋值的变量。表达式的结果总是一个整数。(整数可以是正整数、负整数、0)
 * let 表达式采用 "(let v1 e1 v2 e2 ... vn en expr)" 的形式，其中 let 总是以字符串 "let"来表示，接下来会跟随一对或多对交替的变量和表达式，也就是说，第一个变量 v1被分配为表达式 e1 的值，第二个变量 v2 被分配为表达式 e2 的值，依次类推；最终 let 表达式的值为 expr表达式的值。
 * add 表达式表示为 "(add e1 e2)" ，其中 add 总是以字符串 "add" 来表示，该表达式总是包含两个表达式 e1、e2 ，最终结果是 e1 表达式的值与 e2 表达式的值之 和 。
 * mult 表达式表示为 "(mult e1 e2)" ，其中 mult 总是以字符串 "mult" 表示，该表达式总是包含两个表达式 e1、e2，最终结果是 e1 表达式的值与 e2 表达式的值之 积 。
 * 在该题目中，变量名以小写字符开始，之后跟随 0 个或多个小写字符或数字。为了方便，"add" ，"let" ，"mult" 会被定义为 "关键字" ，不会用作变量名。
 * 最后，要说一下作用域的概念。计算变量名所对应的表达式时，在计算上下文中，首先检查最内层作用域（按括号计），然后按顺序依次检查外部作用域。测试用例中每一个表达式都是合法的。有关作用域的更多详细信息，请参阅示例。
*/
class Solution {
    unordered_map<string,vector<int>> env;
    inline int getValue(const string& var){
        return env[var].back();
    }
    inline string getVarName(const string& expression,int& start){
        int size = expression.size();
        int begin = start;
        while (start<size&&expression[start]!=' '&&expression[start]!=')')
            start++;
        return expression.substr(begin,start-begin);
    }
    inline int getNumber(const string& expression,int& start){
        int size = expression.size();
        int ret = 0, sign = 1;
        if (expression[start] == '-') {
            sign = -1;
            start++;
        }
        while(start<size&&'0'<=expression[start]&&expression[start]<='9') {
            ret = ret*10+ expression[start] - '0';
            start++;
        }
        return sign * ret;
    }
public:
    int dfsEvaluate(const string& expression,int& start) {
        if(expression[start]!='('){
            if('a'<=expression[start]&&expression[start]<='z')
                return getValue(getVarName(expression,start));
            else
                return getNumber(expression,start);
        }
        start++;
        int ret;
        switch (expression[start]){
            case 'l':{
                start += 4;
                vector<string> vars; 
                while(true){
                    if('a'<=expression[start]&&expression[start]<='z'){
                        ret = dfsEvaluate(expression,start);
                        break;
                    }
                    string var = getVarName(expression,start);
                    vars.push_back(var);
					start++;
                    int value = dfsEvaluate(expression,start);
                    env[var].push_back(value);
					start++;
                }
                for(auto& var:vars)
                    env[var].pop_back();
                break;
            }
            case 'a':{
                start += 4;
                int a = dfsEvaluate(expression,start);
                ++start;
                int b = dfsEvaluate(expression,start);
                ret = a+b;
                break;
            }
            case 'm':{
                start += 5;
                int a = dfsEvaluate(expression,start);
                ++start;
                int b = dfsEvaluate(expression,start);
                ret = a*b;
                break;
            }
        }
        start++;
        return ret;
    }
    int evaluate(string expression) {
        int start = 0;
        return dfsEvaluate(expression,start);
    }
};
// 332. 重新安排行程 https://leetcode-cn.com/problems/reconstruct-itinerary/
// 给你一份航线列表 tickets ，其中 tickets[i] = [fromi, toi] 表示飞机出发和降落的机场地点。请你对该行程进行重新规划排序。所有这些机票都属于一个从 JFK（肯尼迪国际机场）出发的先生，所以该行程必须从 JFK 开始。如果存在多种有效的行程，请你按字典序返回最小的行程组合。例如，行程 ["JFK", "LGA"] 与 ["JFK", "LGB"] 相比就更小，排序更靠前。假定所有机票至少存在一种合理的行程。且所有的机票必须都用一次 且 只能用一次。 
class FindItinerary {
    typedef unordered_map<string, priority_queue<string, vector<string>, std::greater<string>>> Map;
    static void dfs(Map& edge,vector<string>& stk,const string& cur){
        while(edge.count(cur)&&!edge[cur].empty()){
            string tem = edge[cur].top();
            edge[cur].pop();
            dfs(edge,stk,tem);
        }
        stk.emplace_back(cur);
    }
public:
    vector<string> findItinerary(vector<vector<string>>& tickets) {
        Map edge;
        vector<string> stk;
        for(auto &ticket:tickets)
            edge[ticket[0]].emplace(ticket[1]);
        dfs(edge,stk,"JFK");
        reverse(stk.begin(),stk.end());
        return stk;
    }
};
// 753. 破解保险箱 https://leetcode-cn.com/problems/cracking-the-safe/
// 有一个需要密码才能打开的保险箱。密码是 n 位数, 密码的每一位是 k 位序列 0, 1, ..., k-1 中的一个 。你可以随意输入密码，保险箱
// 会自动记住最后 n 位输入，如果匹配，则能够打开保险箱。请返回一个能打开保险箱的最短字符串.
class CrackSafe {
    static void dfs(int node,int k,int flag,unordered_set<int> &visited,string& ans) {
        for (int x = 0; x < k; ++x) {
            int nei = node * 10 + x;
            if (!visited.count(nei)) {
                visited.insert(nei);
                dfs(nei%flag,k,flag,visited,ans);
                ans += (x + '0');
            }
        }
    }
public:
    string crackSafe(int n, int k) {
        int flag = pow(10, n - 1);
        unordered_set<int> visited;
        std::string ans;
        dfs(0,k,flag,visited,ans);
        ans += string(n-1,'0');
        return ans;
    }
};
// 1020 飞地数量https://leetcode-cn.com/problems/number-of-enclaves/
// 给你一个大小为mxn的二进制矩阵grid,其中0表示一个海洋单元格、1表示一个陆地单元格。一次移动是指从一个陆地单元格走到另一个相邻(上、下、左、右)的陆地单元格或跨过grid的边界。返回网格中 无法 在任意次数的移动中离开网格边界的陆地单元格的数量。
class NumEnclaves {
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
class Bipartite {
    static bool dfs(vector<vector<int>>& graph,char* colors,int pos,char color){
        char ncolor = ~color;
        if(colors[pos]==ncolor)
            return false;
        for(auto next:graph[pos])
            if(!dfs(graph,colors,next,ncolor))
                return false;
        return true;
    }
public:
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
};
// 797. 所有可能的路径 https://leetcode-cn.com/problems/all-paths-from-source-to-target/
// 给你一个有 n 个节点的 有向无环图（DAG），请你找出所有从节点 0 到节点 n-1 的路径并输出（不要求按特定顺序）graph[i] 是一个从节点 i 可以访问的所有节点的列表（即从节点 i 到节点 graph[i][j]存在一条有向边）。
class AllPathsSourceTarget {
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
// 341. 扁平化嵌套列表迭代器 https://leetcode-cn.com/problems/flatten-nested-list-iterator/
class NestedIterator {
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