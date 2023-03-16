/*
 * @Author: zanilia
 * @Date: 2022-02-18 16:59:43
 * @LastEditTime: 2022-02-21 23:30:18
 * @Descripttion: 
 */
#include <bits/stdc++.h>
using namespace std;
// 464. 我能赢吗 https://leetcode-cn.com/problems/can-i-win/
// 在 "100 game" 这个游戏中，两名玩家轮流选择从 1 到 10 的任意整数，累计整数和，先使得累计整数和 达到或超过  100 的玩家，即为胜者。如果我们将游戏规则改为 “玩家 不能 重复使用整数” 呢？例如，两个玩家可以轮流从公共整数池中抽取从 1 到 15 的整数（不放回），直到累计整数和 >= 100。给定两个整数 maxChoosableInteger （整数池中可选择的最大数）和 desiredTotal（累计和），若先出手的玩家是否能稳赢则返回 true ，否则返回 false 。假设两位玩家游戏时都表现 最佳 。
class Solution {
    char* dp;
    int choosable;
    bool dfs(int state,int desired){
        if(dp[state])
            return (dp[state]==1);
        for(int i=1;i<choosable;++i){
            int cur = 1 << (i-1);
            if(cur&state)
                continue;
            if(i>=desired||!dfs(state|cur,desired-i)){
                dp[state] = 1;
                return true;
            }
        }
        dp[state] = 2;
        return false;
    }
public:
    bool canIWin(int maxChoosableInteger, int desiredTotal) {
        if (maxChoosableInteger>=desiredTotal)
            return true;
        if ((1+maxChoosableInteger)*maxChoosableInteger/2 < desiredTotal)
            return false;
        choosable = maxChoosableInteger;
        dp = new char[1<<maxChoosableInteger];
        memset(dp,0,1<<maxChoosableInteger);
        return dfs(0,desiredTotal);
    }
};
// 486. 预测赢家 https://leetcode-cn.com/problems/predict-the-winner/
// 给你一个整数数组 nums 。玩家 1 和玩家 2 基于这个数组设计了一个游戏。玩家 1 和玩家 2 轮流进行自己的回合，玩家 1 先手。开始时，两个玩家的初始分值都是 0 。每一回合，玩家从数组的任意一端取一个数字（即，nums[0] 或 nums[nums.length - 1]），取到的数字将会从数组中移除（数组长度减 1 ）。玩家选中的数字将会加到他的得分上。当数组中没有剩余数字可取时，游戏结束。如果玩家 1 能成为赢家，返回 true 。如果两个玩家得分相等，同样认为玩家 1 是游戏的赢家，也返回 true 。你可以假设每个玩家的玩法都会使他的分数最大化。
bool PredictTheWinner(vector<int>& nums) {
	int size = nums.size();
	if(!(size&1))
		return true;
	int dp[size];
	copy(nums.begin(),nums.end(),dp);
	for(int i=size-2;i>=0;--i)
		for(int j=i+1;j<size;++j)
			dp[j] = max(nums[i]-dp[j],nums[j]-dp[j-1]);
	return (dp[size-1]>=0);
}
// 810. 黑板异或游戏 https://leetcode-cn.com/problems/chalkboard-xor-game/
// 黑板上写着一个非负整数数组 nums[i] 。Alice 和 Bob 轮流从黑板上擦掉一个数字，Alice 先手。如果擦除一个数字后，剩余的所有数字按位异或运算得出的结果等于 0 的话，当前玩家游戏失败。 另外，如果只剩一个数字，按位异或运算得到它本身；如果无数字剩余，按位异或运算结果为 0。并且，轮到某个玩家时，如果当前黑板上所有数字按位异或运算结果等于 0 ，这个玩家获胜。假设两个玩家每步都使用最优解，当且仅当 Alice 获胜时返回 true。
bool xorGame(vector<int>& nums) {
	if(!(nums.size()&1))
		return true;
	int sum =0; 
	for(auto &num:nums)
		sum ^= num;
	return sum==0;
}
// 843. 猜猜这个单词 https://leetcode-cn.com/problems/guess-the-word/
// 这是一个 交互式问题 。我们给出了一个由一些 不同的 单词组成的列表 wordlist ，对于每个 wordlist[i] 长度均为 6 ，这个列表中的一个单词将被选作 secret 。你可以调用 Master.guess(word) 来猜单词。你所猜的单词应当是存在于原列表并且由 6 个小写字母组成的类型 string 。此函数将会返回一个 integer ，表示你的猜测与秘密单词 secret 的准确匹配（值和位置同时匹配）的数目。此外，如果你的猜测不在给定的单词列表中，它将返回 -1。对于每个测试用例，你有 10 次机会来猜出这个单词。当所有调用都结束时，如果您对 Master.guess 的调用在 10 次以内，并且至少有一次猜到 secret ，将判定为通过该用例。
class Master {
public:
	int guess(string word);
};
void findSecretWord(vector<string>& wordlist, Master& master) {

}
// 1140. 石子游戏 II https://leetcode-cn.com/problems/stone-game-ii/
// 爱丽丝和鲍勃继续他们的石子游戏。许多堆石子 排成一行，每堆都有正整数颗石子 piles[i]。游戏以谁手中的石子最多来决出胜负。爱丽丝和鲍勃轮流进行，爱丽丝先开始。最初，M = 1。在每个玩家的回合中，该玩家可以拿走剩下的 前 X 堆的所有石子，其中 1 <= X <= 2M。然后，令 M = max(M, X)。游戏一直持续到所有石子都被拿走。假设爱丽丝和鲍勃都发挥出最佳水平，返回爱丽丝可以得到的最大数量的石头。
class Solution {
public:
	int dp[105][105];
    int stoneGameII(vector<int>& piles) {
        for(int i=piles.size()-1;i>=0;i--){
            for(int j=1;j<=50;j++){
                dp[i][j]=-9999999;
            }
        }
        int z=0;
        for(int i=piles.size()-1;i>=0;i--){
            z+=piles[i];
            for(int j=1;j<=50;j++){
                int sum=piles[i];
                int temp=piles.size()-i;
                for(int k=1;k<=(min(2*j,temp));k++){
                    dp[i][j]=max(dp[i][j],sum-dp[i+k][min(50,max(j,k))]);
                    if((i+k)<piles.size())
                        sum+=piles[i+k];
                }
            }
        }
        return dp[0][1]+(z-dp[0][1])/2;
    }
};
// 1561. 你可以获得的最大硬币数目 https://leetcode-cn.com/problems/maximum-number-of-coins-you-can-get/
// 有 3n 堆数目不一的硬币，你和你的朋友们打算按以下方式分硬币：每一轮中，你将会选出 任意 3 堆硬币（不一定连续）。Alice 将会取走硬币数量最多的那一堆。你将会取走硬币数量第二多的那一堆。Bob 将会取走最后一堆。重复这个过程，直到没有更多硬币。给你一个整数数组 piles ，其中 piles[i] 是第 i 堆中硬币的数目。返回你可以获得的最大硬币数目。
int maxCoins(vector<int>& piles) {
    int size = piles.size();
    sort(piles.begin(),piles.end());
    int ans = 0;
    for(int i=size/3;i<size;i+=2)
        ans += piles[i];
    return ans;
}
