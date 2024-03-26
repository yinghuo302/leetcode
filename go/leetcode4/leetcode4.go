package leetcode4

import (
	"container/heap"
	"container/list"
	"sort"
	"strings"
)

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func max(a, b int) int {
	if a < b {
		return b
	}
	return a
}

// 2944. 购买水果需要的最少金币数 https://leetcode.cn/problems/minimum-number-of-coins-for-fruits
// 你在一个水果超市里，货架上摆满了玲琅满目的奇珍异果。给你一个下标从 1 开始的数组 prices ，其中 prices[i] 表示你购买第 i 个水果需要花费的金币数目。水果超市有如下促销活动：如果你花费 price[i] 购买了水果 i ，那么接下来的 i 个水果你都可以免费获得。注意 ，即使你 可以 免费获得水果 j ，你仍然可以花费 prices[j] 个金币去购买它以便能免费获得接下来的 j 个水果。请你返回获得所有水果所需要的 最少 金币数。
func minimumCoins(prices []int) int {
	n := len(prices)
	dp := make([]int, n+1)
	dp[0] = 0
	for i := 1; i <= n; i++ {
		tem := 1 << 30
		for j := i; j >= (i+1)/2; j-- {
			tem = min(dp[j-1]+prices[j-1], tem)
		}
		dp[i] = tem
	}
	return dp[n]
}

// 2945. 找到最大非递减数组的长度 https://leetcode.cn/problems/find-maximum-non-decreasing-array-length
// 给你一个下标从 0 开始的整数数组 nums 。你可以执行任意次操作。每次操作中，你需要选择一个 子数组 ，并将这个子数组用它所包含元素的 和 替换。比方说，给定数组是 [1,3,5,6] ，你可以选择子数组 [3,5] ，用子数组的和 8 替换掉子数组，然后数组会变为 [1,8,6] 。请你返回执行任意次操作以后，可以得到的 最长非递减 数组的长度。子数组 指的是一个数组中一段连续 非空 的元素序列。
func findMaximumLength(nums []int) int {
	n := len(nums)
	sum, dp, last := make([]int, n+1), make([]int, n+1), make([]int, n+1)
	q, head, tail := make([]int, n+1), 0, 1
	q[0] = 0
	for i := 1; i <= n; i++ {
		sum[i] = nums[i-1] + sum[i-1]
		for tail-head > 1 && sum[q[head+1]]+last[q[head+1]] <= sum[i] {
			head++
		}
		dp[i] = dp[q[head]] + 1
		last[i] = sum[i] - sum[q[head]]
		for tail-head > 0 && sum[q[tail-1]]+last[q[tail-1]] >= sum[i]+last[i] {
			tail--
		}
		q[tail] = i
		tail++
	}
	return dp[n]
}

// 2949. 统计美丽子字符串 II https://leetcode.cn/problems/count-beautiful-substrings-ii/
// 给你一个字符串 s 和一个正整数 k 。用 vowels 和 consonants 分别表示字符串中元音字母和辅音字母的数量。如果某个字符串满足以下条件，则称其为 美丽字符串 ：vowels == consonants，即元音字母和辅音字母的数量相等。(vowels * consonants) % k == 0，即元音字母和辅音字母的数量的乘积能被 k 整除。返回字符串 s 中 非空美丽子字符串 的数量。子字符串是字符串中的一个连续字符序列。英语中的 元音字母 为 'a'、'e'、'i'、'o' 和 'u' 。英语中的 辅音字母 为除了元音字母之外的所有字母。
func beautifulSubstrings(s string, k int) int64 {
	k = func(n int) int {
		res := 1
		for i := 2; i*i <= n; i++ {
			i2 := i * i
			for n%i2 == 0 {
				res *= i
				n /= i2
			}
			if n%i == 0 {
				res *= i
				n /= i
			}
		}
		if n > 1 {
			res *= n
		}
		return res
	}(4 * k)
	str := "aeiou"
	type pair struct{ len, sum int }
	ans, sum, cnt := int64(0), 0, map[pair]int{{k - 1, 0}: 1}
	for i, c := range s {
		sum += 1
		if strings.ContainsRune(str, c) {
			sum -= 2
		}
		p := pair{i % k, sum}
		ans += int64(cnt[p])
		cnt[p]++
	}
	return ans

}

// 1657. 确定两个字符串是否接近 https://leetcode.cn/problems/determine-if-two-strings-are-close/
// 如果可以使用以下操作从一个字符串得到另一个字符串，则认为两个字符串 接近 ：操作 1：交换任意两个 现有 字符。例如，abcde -> aecdb.操作 2：将一个 现有 字符的每次出现转换为另一个 现有 字符，并对另一个字符执行相同的操作。例如，aacabb -> bbcbaa（所有 a 转化为 b ，而所有的 b 转换为 a ）你可以根据需要对任意一个字符串多次使用这两种操作。给你两个字符串，word1 和 word2 。如果 word1 和 word2 接近 ，就返回 true ；否则，返回 false 。
func closeStrings(word1 string, word2 string) bool {
	cnt1, cnt2 := make([]int, 26), make([]int, 26)
	for _, c := range word1 {
		cnt1[c-'a']++
	}
	for _, c := range word2 {
		cnt2[c-'a']++
	}
	for i := 0; i < 26; i++ {
		if (cnt1[i] == 0 && cnt2[i] != 0) || (cnt1[i] != 0 && cnt2[i] == 0) {
			return false
		}
	}
	sort.Ints(cnt1)
	sort.Ints(cnt2)
	for i := 0; i < 26; i++ {
		if cnt1[i] != cnt2[i] {
			return false
		}
	}
	return true
}

// 2661. 找出叠涂元素 https://leetcode.cn/problems/first-completely-painted-row-or-column
// 给你一个下标从 0 开始的整数数组 arr 和一个 m x n 的整数 矩阵 mat 。arr 和 mat 都包含范围 [1，m * n] 内的 所有 整数。从下标 0 开始遍历 arr 中的每个下标 i ，并将包含整数 arr[i] 的 mat 单元格涂色。请你找出 arr 中在 mat 的某一行或某一列上都被涂色且下标最小的元素，并返回其下标 i 。
func firstCompleteIndex(arr []int, mat [][]int) int {
	type pair struct{ i, j int }
	mp := make(map[int]pair)
	m, n := len(mat), len(mat[0])
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			mp[mat[i][j]] = pair{i, j}
		}
	}
	rcnt, ccnt := make([]int, m), make([]int, n)
	for idx, val := range arr {
		p, ok := mp[val]
		if !ok {
			continue
		}
		rcnt[p.i]++
		ccnt[p.j]++
		if rcnt[p.i] == n || ccnt[p.j] == m {
			return idx
		}
	}
	return -1
}

// 1094. 拼车 https://leetcode.cn/problems/car-pooling
// 车上最初有 capacity 个空座位。车 只能 向一个方向行驶（也就是说，不允许掉头或改变方向）给定整数 capacity 和一个数组 trips ,  trip[i] = [numPassengersi, fromi, toi] 表示第 i 次旅行有 numPassengersi 乘客，接他们和放他们的位置分别是 fromi 和 toi 。这些位置是从汽车的初始位置向东的公里数。当且仅当你可以在所有给定的行程中接送所有乘客时，返回 true，否则请返回 false。
func carPooling(trips [][]int, capacity int) bool {
	diff := make([]int, 1004)
	for _, trip := range trips {
		diff[trip[1]] += trip[0]
		diff[trip[2]] -= trip[0]
	}
	passengers := 0
	for _, d := range diff {
		passengers += d
		if passengers > capacity {
			return false
		}
	}
	return true
}

// 1423. 可获得的最大点数 https://leetcode.cn/problems/maximum-points-you-can-obtain-from-cards
// 几张卡牌 排成一行，每张卡牌都有一个对应的点数。点数由整数数组 cardPoints 给出。每次行动，你可以从行的开头或者末尾拿一张卡牌，最终你必须正好拿 k 张卡牌。你的点数就是你拿到手中的所有卡牌的点数之和。给你一个整数数组 cardPoints 和整数 k，请你返回可以获得的最大点数。
func maxScore(cardPoints []int, k int) int {
	presum := make([]int, len(cardPoints)+1)
	presum[0] = 0
	for idx, val := range cardPoints {
		presum[idx+1] = presum[idx] + val
	}
	ans, n := 0, len(cardPoints)
	for i := 0; i <= k; i++ {
		ans = max(ans, presum[i]-presum[0]+presum[n]-presum[n-k+i])
	}
	return ans
}

// 1038. 从二叉搜索树到更大和树 https://leetcode.cn/problems/binary-search-tree-to-greater-sum-tree
// 给定一个二叉搜索树 root (BST)，请将它的每个节点的值替换成树中大于或者等于该节点值的所有节点值之和。提醒一下， 二叉搜索树 满足下列约束条件：节点的左子树仅包含键 小于 节点键的节点。节点的右子树仅包含键 大于 节点键的节点。左右子树也必须是二叉搜索树。
func bstToGst(root *TreeNode) *TreeNode {
	val := 0
	var assist func(*TreeNode) *TreeNode
	assist = func(node *TreeNode) *TreeNode {
		if node == nil {
			return nil
		}
		ret := &TreeNode{}
		ret.Right = assist(node.Right)
		val += node.Val
		ret.Val = val
		ret.Left = assist(node.Left)
		return ret
	}
	return assist(root)
}

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

// 2477. 到达首都的最少油耗 https://leetcode.cn/problems/minimum-fuel-cost-to-report-to-the-capital
// 给你一棵 n 个节点的树（一个无向、连通、无环图），每个节点表示一个城市，编号从 0 到 n - 1 ，且恰好有 n - 1 条路。0 是首都。给你一个二维整数数组 roads ，其中 roads[i] = [ai, bi] ，表示城市 ai 和 bi 之间有一条 双向路 。每个城市里有一个代表，他们都要去首都参加一个会议。每座城市里有一辆车。给你一个整数 seats 表示每辆车里面座位的数目。城市里的代表可以选择乘坐所在城市的车，或者乘坐其他城市的车。相邻城市之间一辆车的油耗是一升汽油。请你返回到达首都最少需要多少升汽油。
func minimumFuelCost(roads [][]int, seats int) int64 {
	n := len(roads) + 1
	edges := make([][]int, n)
	for _, road := range roads {
		edges[road[0]] = append(edges[road[0]], road[1])
		edges[road[1]] = append(edges[road[1]], road[0])
	}
	ans := int64(0)
	var dfs func(int, int) int
	dfs = func(node, prev int) int {
		personSum := 1
		for _, next := range edges[node] {
			if next != prev {
				nextPerson := dfs(next, node)
				ans += int64((nextPerson + seats - 1) / seats)
				personSum += nextPerson
			}
		}
		return personSum
	}
	dfs(0, -1)
	return ans
}

// 2646. 最小化旅行的价格总和 https://leetcode.cn/problems/minimize-the-total-price-of-the-trips
// 现有一棵无向、无根的树，树中有 n 个节点，按从 0 到 n - 1 编号。给你一个整数 n 和一个长度为 n - 1 的二维整数数组 edges ，其中 edges[i] = [ai, bi] 表示树中节点 ai 和 bi 之间存在一条边每个节点都关联一个价格。给你一个整数数组 price ，其中 price[i] 是第 i 个节点的价格。给定路径的 价格总和 是该路径上所有节点的价格之和。另给你一个二维整数数组 trips ，其中 trips[i] = [starti, endi] 表示您从节点 starti 开始第 i 次旅行，并通过任何你喜欢的路径前往节点 endi 。在执行第一次旅行之前，你可以选择一些 非相邻节点 并将价格减半。返回执行所有旅行的最小价格总和。
func minimumTotalPrice(n int, edges [][]int, price []int, trips [][]int) int {
	g := make([][]int, n)
	for _, edge := range edges {
		g[edge[0]] = append(g[edge[0]], edge[1])
		g[edge[1]] = append(g[edge[1]], edge[0])
	}
	uf := make([]int, n)
	var unionfind func(int) int
	unionfind = func(x int) int {
		if uf[x] == x {
			return x
		}
		uf[x] = unionfind(uf[x])
		return uf[x]
	}
	query := make([][]int, n)
	for _, trip := range trips {
		query[trip[0]] = append(query[trip[0]], trip[1])
		if trip[0] != trip[1] {
			query[trip[1]] = append(query[trip[1]], trip[0])
		}
	}
	visited, diff, parent := make([]bool, n), make([]int, n), make([]int, n)
	var tarjan func(int, int)
	tarjan = func(node, pa int) {
		parent[node], uf[node] = pa, node
		for _, child := range g[node] {
			if child != pa {
				tarjan(child, node)
				uf[child] = node
			}
		}
		for _, q_node := range query[node] {
			if q_node != node && !visited[q_node] {
				continue
			}
			lca := unionfind(q_node)
			diff[node]++
			diff[q_node]++
			diff[lca]--
			if parent[lca] >= 0 {
				diff[parent[lca]]--
			}
		}
		visited[node] = true
	}
	tarjan(0, -1)
	cnt := make([]int, n)
	var getCount func(int, int) int
	getCount = func(node, pa int) int {
		cnt[node] = diff[node]
		for _, child := range g[node] {
			if child != pa {
				cnt[node] += getCount(child, node)
			}
		}
		return cnt[node]
	}
	getCount(0, -1)
	var dp func(int, int) (int, int)
	dp = func(node, pa int) (int, int) {
		cur1, cur2 := price[node]*cnt[node], price[node]*cnt[node]/2
		for _, child := range g[node] {
			if child != pa {
				child_dp1, child_dp2 := dp(child, node)
				cur1, cur2 = cur1+min(child_dp1, child_dp2), cur2+child_dp1
			}
		}
		return cur1, cur2
	}
	return min(dp(0, -1))
}

// 1466. 重新规划路线 https://leetcode.cn/problems/reorder-routes-to-make-all-paths-lead-to-the-city-zero
// n 座城市，从 0 到 n-1 编号，其间共有 n-1 条路线。因此，要想在两座不同城市之间旅行只有唯一一条路线可供选择（路线网形成一颗树）。去年，交通运输部决定重新规划路线，以改变交通拥堵的状况。路线用 connections 表示，其中 connections[i] = [a, b] 表示从城市 a 到 b 的一条有向路线。今年，城市 0 将会举办一场大型比赛，很多游客都想前往城市 0 。请你帮助重新规划路线方向，使每个城市都可以访问城市 0 。返回需要变更方向的最小路线数。题目数据 保证 每个城市在重新规划路线方向后都能到达城市 0 。
func minReorder(n int, connections [][]int) int {
	st, g := make(map[int64]struct{}), make([][]int, n)
	for _, connection := range connections {
		g[connection[0]] = append(g[connection[0]], connection[1])
		g[connection[1]] = append(g[connection[1]], connection[0])
		st[(int64(connection[0])<<32)+int64(connection[1])] = struct{}{}
	}
	ans := 0
	var dfs func(int, int)
	dfs = func(node, pa int) {
		if _, ok := st[(int64(pa)<<32)+int64(node)]; ok {
			ans++
		}
		for _, child := range g[node] {
			if pa != child {
				dfs(child, node)
			}
		}
	}
	dfs(0, -1)
	return ans
}

// 2008. 出租车的最大盈利 https://leetcode.cn/problems/maximum-earnings-from-taxi
// 你驾驶出租车行驶在一条有 n 个地点的路上。这 n 个地点从近到远编号为 1 到 n ，你想要从 1 开到 n ，通过接乘客订单盈利。你只能沿着编号递增的方向前进，不能改变方向。乘客信息用一个下标从 0 开始的二维数组 rides 表示，其中 rides[i] = [starti, endi, tipi] 表示第 i 位乘客需要从地点 starti 前往 endi ，愿意支付 tipi 元的小费。每一位 你选择接单的乘客 i ，你可以 盈利 endi - starti + tipi 元。你同时 最多 只能接一个订单。给你 n 和 rides ，请你返回在最优接单方案下，你能盈利 最多 多少元。注意：你可以在一个地点放下一位乘客，并在同一个地点接上另一位乘客。
func maxTaxiEarnings(n int, rides [][]int) int64 {
	sort.Slice(rides, func(i, j int) bool {
		return rides[i][1] < rides[j][1]
	})
	type info struct{ end, earn int64 }
	dp := make([]info, 0, n)
	dp = append(dp, info{end: -1, earn: 0})
	maxEarn := int64(0)
	for _, ride := range rides {
		idx := sort.Search(len(dp)+1, func(i int) bool {
			return i == len(dp) || dp[i].end > int64(ride[0])
		}) - 1
		maxEarn = max_int64(int64(ride[1]-ride[0]+ride[2])+dp[idx].earn, maxEarn)
		dp = append(dp, info{end: int64(ride[1]), earn: maxEarn})
	}
	return dp[len(dp)-1].earn
}

func max_int64(a, b int64) int64 {
	if a < b {
		return b
	}
	return a
}

// 100136. 统计好分割方案的数目 https://leetcode.cn/problems/count-the-number-of-good-partitions/
// 给你一个下标从 0 开始、由 正整数 组成的数组 nums。将数组分割成一个或多个 连续 子数组，如果不存在包含了相同数字的两个子数组，则认为是一种 好分割方案 。返回 nums 的 好分割方案 的 数目。由于答案可能很大，请返回答案对 109 + 7 取余 的结果。
func numberOfGoodPartitions(nums []int) int {
	const mod int = 1e9 + 7
	mp := make(map[int]int)
	for idx, num := range nums {
		mp[num] = idx
	}
	maxR, ans := 0, 1
	for idx, num := range nums[:len(nums)-1] {
		maxR = max(mp[num], maxR)
		if maxR == idx {
			ans = ans * 2 % mod
		}
	}
	return ans
}

// 1631. 最小体力消耗路径 https://leetcode.cn/problems/path-with-minimum-effort
// 你准备参加一场远足活动。给你一个二维 rows x columns 的地图 heights ，其中 heights[row][col] 表示格子 (row, col) 的高度。一开始你在最左上角的格子 (0, 0) ，且你希望去最右下角的格子 (rows-1, columns-1) （注意下标从 0 开始编号）。你每次可以往 上，下，左，右 四个方向之一移动，你想要找到耗费 体力 最小的一条路径。一条路径耗费的 体力值 是路径上相邻格子之间 高度差绝对值 的 最大值 决定的。请你返回从左上角走到右下角的最小 体力消耗值 。
func minimumEffortPath(heights [][]int) int {
	return sort.Search(1e6, func(k int) bool {
		m, n, head, tail := len(heights), len(heights[0]), 0, 1
		dirs, end := [4][3]int{{0, 1, 1}, {0, -1, -1}, {1, 0, n}, {-1, 0, -n}}, m*n-1
		q := make([]int, m*n)
		visit := make([]bool, m*n)
		visit[0] = true
		q[0] = 0
		dis := func(ma, na, mb, nb int) int {
			if mb < 0 || mb >= m || nb < 0 || nb >= n {
				return 1e9
			}
			d := heights[ma][na] - heights[mb][nb]
			if d < 0 {
				d = -d
			}
			return d
		}
		for head < tail {
			if q[head] == end {
				return true
			}
			ma, na := q[head]/n, q[head]%n
			for _, dir := range dirs {
				mb, nb, next := ma+dir[0], na+dir[1], q[head]+dir[2]
				if dis(ma, na, mb, nb) > k || visit[next] {
					continue
				}
				q[tail] = next
				visit[next] = true
				tail++
			}
			head++
		}
		return false
	})
}

// 2132. 用邮票贴满网格图 https://leetcode.cn/problems/stamping-the-grid
// 给你一个 m x n 的二进制矩阵 grid ，每个格子要么为 0 （空）要么为 1 （被占据）。给你邮票的尺寸为 stampHeight x stampWidth 。我们想将邮票贴进二进制矩阵中，且满足以下 限制 和 要求 ：覆盖所有 空 格子。不覆盖任何 被占据 的格子。我们可以放入任意数目的邮票。邮票可以相互有 重叠 部分。邮票不允许 旋转 。邮票必须完全在矩阵 内 。如果在满足上述要求的前提下，可以放入邮票，请返回 true ，否则返回 false 。
func possibleToStamp(grid [][]int, stampHeight int, stampWidth int) bool {
	m, n := len(grid), len(grid[0])
	presum, diff := make([][]int, m+2), make([][]int, m+2)
	for i := 0; i <= m+1; i++ {
		presum[i] = make([]int, n+2)
		diff[i] = make([]int, n+2)
	}
	for i := 1; i <= m; i++ {
		for j := 1; j <= n; j++ {
			presum[i][j] = presum[i-1][j] + presum[i][j-1] - presum[i-1][j-1] + grid[i-1][j-1]
		}
	}
	for i := 1; i+stampHeight-1 <= m; i++ {
		for j := 1; j+stampWidth-1 <= n; j++ {
			x, y := i+stampHeight-1, j+stampWidth-1
			if presum[x][y]-presum[x][j-1]-presum[i-1][y]+presum[i-1][j-1] == 0 {
				diff[i][j]++
				diff[i][y+1]--
				diff[x+1][j]--
				diff[x+1][y+1]++
			}
		}
	}
	for i := 1; i <= m; i++ {
		for j := 1; j <= n; j++ {
			diff[i][j] += diff[i-1][j] + diff[i][j-1] - diff[i-1][j-1]
			if diff[i][j] == 0 && grid[i-1][j-1] == 0 {
				return false
			}
		}
	}
	return true
}

// 2415. 反转二叉树的奇数层 https://leetcode.cn/problems/reverse-odd-levels-of-binary-tree/
// 给你一棵 完美 二叉树的根节点 root ，请你反转这棵树中每个 奇数 层的节点值。例如，假设第 3 层的节点值是 [2,1,3,4,7,11,29,18] ，那么反转后它应该变成 [18,29,11,7,4,3,1,2] 。反转后，返回树的根节点。完美 二叉树需满足：二叉树的所有父节点都有两个子节点，且所有叶子节点都在同一层。节点的 层数 等于该节点到根节点之间的边数。
func reverseOddLevels(root *TreeNode) *TreeNode {
	height, node := 2, root.Left // nil节点也算一层
	for node != nil {
		height++
		node = node.Left
	}
	arr := make([][]*TreeNode, 2)
	arr[0] = make([]*TreeNode, 1<<height)
	arr[1] = make([]*TreeNode, 1<<height)
	arr[0][0], arr[1][0], arr[1][1] = root, root.Left, root.Right
	curr, prev, size1, size2 := 1, 0, 1, 2
	for i := 2; i < height; i++ {
		if curr != 0 {
			for j := 0; j < size1; j++ {
				arr[prev][j].Left = arr[curr][size2-j*2-1]
				arr[prev][j].Right = arr[curr][size2-j*2-2]
			}
		} else {
			for j := 0; j < size1; j++ {
				arr[prev][size1-j-1].Left = arr[curr][j*2]
				arr[prev][size1-j-1].Right = arr[curr][j*2+1]
			}
		}
		curr, prev, size1, size2 = curr^1, prev^1, size1*2, size2*2
		for j := 0; j < size1; j++ {
			arr[curr][j*2] = arr[prev][j].Left
			arr[curr][j*2+1] = arr[prev][j].Right
		}
	}
	return root
}

// 1962. 移除石子使总数最小 https://leetcode.cn/problems/remove-stones-to-minimize-the-total
// 给你一个整数数组 piles ，数组 下标从 0 开始 ，其中 piles[i] 表示第 i 堆石子中的石子数量。另给你一个整数 k ，请你执行下述操作 恰好 k 次：选出任一石子堆 piles[i] ，并从中 移除 floor(piles[i] / 2) 颗石子。注意：你可以对 同一堆 石子多次执行此操作。返回执行 k 次操作后，剩下石子的 最小 总数。floor(x) 为 小于 或 等于 x 的 最大 整数。（即，对 x 向下取整）。
func minStoneSum(piles []int, k int) int {
	hp, sum := &intHp{arr: piles}, 0
	for _, pile := range piles {
		sum += pile
	}
	heap.Init(hp)
	for ; k != 0; k-- {
		val := heap.Pop(hp).(int)
		sum, val = sum-val/2, val-val/2
		heap.Push(hp, val)
	}
	return sum
}

type intHp struct {
	arr []int
}

func (hp *intHp) Less(i, j int) bool {
	return hp.arr[i] > hp.arr[j]
}

func (hp *intHp) Swap(i, j int) {
	hp.arr[i], hp.arr[j] = hp.arr[j], hp.arr[i]
}

func (hp *intHp) Len() int {
	return len(hp.arr)
}

func (hp *intHp) Push(v interface{}) {
	hp.arr = append(hp.arr, v.(int))
}

func (hp *intHp) Pop() interface{} {
	ret := hp.arr[len(hp.arr)-1]
	hp.arr = hp.arr[:len(hp.arr)-1]
	return ret
}

// 1671. 得到山形数组的最少删除次数 https://leetcode.cn/problems/minimum-number-of-removals-to-make-mountain-array
// 我们定义 arr 是 山形数组 当且仅当它满足：arr.length >= 3,存在某个下标 i （从 0 开始） 满足 0 < i < arr.length - 1 且：arr[0] < arr[1] < ... < arr[i - 1] < arr[i] , arr[i] > arr[i + 1] > ... > arr[arr.length - 1]。给你整数数组 nums​ ，请你返回将 nums 变成 山形状数组 的​ 最少 删除次数。
func minimumMountainRemovals(nums []int) int {
	n := len(nums)
	pre := getLISArray(nums)
	suf := getLISArray(reversed(nums))
	suf = reversed(suf)

	ans := 0
	for i := 0; i < n; i++ {
		if pre[i] > 1 && suf[i] > 1 {
			ans = max(ans, pre[i]+suf[i]-1)
		}
	}
	return n - ans
}

func getLISArray(nums []int) []int {
	n := len(nums)
	dp := make([]int, n)
	for i := 0; i < n; i++ {
		dp[i] = 1
		for j := 0; j < i; j++ {
			if nums[j] < nums[i] {
				dp[i] = max(dp[i], dp[j]+1)
			}
		}
	}
	return dp
}

func reversed(nums []int) []int {
	n := len(nums)
	ans := make([]int, n)
	for i := 0; i < n; i++ {
		ans[i] = nums[n-1-i]
	}
	return ans
}

// 1954. 收集足够苹果的最小花园周长 https://leetcode.cn/problems/minimum-garden-perimeter-to-collect-enough-apples
// 给你一个用无限二维网格表示的花园，每一个 整数坐标处都有一棵苹果树。整数坐标 (i, j) 处的苹果树有 |i| + |j| 个苹果。你将会买下正中心坐标是 (0, 0) 的一块 正方形土地 ，且每条边都与两条坐标轴之一平行。给你一个整数 neededApples ，请你返回土地的 最小周长 ，使得 至少 有 neededApples 个苹果在土地 里面或者边缘上。|x| 的值定义为：如果 x >= 0 ，那么值为 x.如果 x < 0 ，那么值为 -x
func minimumPerimeter(neededApples int64) int64 {
	return int64(sort.Search(100001, func(mid int) bool {
		return 2*int64(mid)*int64(mid+1)*int64(mid*2+1) >= neededApples
	})) * 8
}

// 2735. 收集巧克力 https://leetcode.cn/problems/collecting-chocolates
// 给你一个长度为 n 、下标从 0 开始的整数数组 nums ，表示收集不同巧克力的成本。每个巧克力都对应一个不同的类型，最初，位于下标 i 的巧克力就对应第 i 个类型。在一步操作中，你可以用成本 x 执行下述行为：同时修改所有巧克力的类型，将巧克力的类型 ith 修改为类型 ((i + 1) mod n)th。假设你可以执行任意次操作，请返回收集所有类型巧克力所需的最小成本。
func minCost(nums []int, x int) int64 {
	ans, size, dp := int64(1)<<62, len(nums), make([]int, len(nums))
	copy(dp, nums)
	for k := 0; k < size; k++ {
		tem := int64(k * x)
		for i := 0; i < size; i++ {
			dp[i] = min(dp[i], nums[(i+k)%size])
			tem += int64(dp[i])
		}
		if tem < ans {
			ans = tem
		}
	}
	return ans
}

// 2487. 从链表中移除节点 https://leetcode.cn/problems/remove-nodes-from-linked-list
// 给你一个链表的头节点 head 。移除每个右侧有一个更大数值的节点。返回修改后链表的头节点 head 。
func removeNodes(head *ListNode) *ListNode {
	stk := make([]*ListNode, 1)
	stk[0] = &ListNode{Val: 0x3f3f3f3f}
	for head != nil {
		for head.Val > stk[len(stk)-1].Val {
			stk = stk[:len(stk)-1]
		}
		stk = append(stk, head)
		head = head.Next
	}
	if len(stk) == 1 {
		return nil
	}
	for idx := 2; idx < len(stk); idx++ {
		stk[idx-1].Next = stk[idx]
	}
	return stk[1]
}

type ListNode struct {
	Val  int
	Next *ListNode
}

// 2501. 数组中最长的方波 https://leetcode.cn/problems/longest-square-streak-in-an-array
// 给你一个整数数组 nums 。如果 nums 的子序列满足下述条件，则认为该子序列是一个 方波 ：子序列的长度至少为 2 ，并且将子序列从小到大排序 之后 ，除第一个元素外，每个元素都是前一个元素的 平方 。返回 nums 中 最长方波 的长度，如果不存在 方波 则返回 -1 。子序列 也是一个数组，可以由另一个数组删除一些或不删除元素且不改变剩余元素的顺序得到。
func longestSquareStreak(nums []int) int {
	st, ans := make(map[int]struct{}), 1
	for _, num := range nums {
		st[num] = struct{}{}
	}
	for _, num := range nums {
		l, cur := 1, num
		_, ok := st[cur*cur]
		for ok {
			cur = cur * cur
			l++
			_, ok = st[cur*cur]
		}
		ans = max(ans, l)
	}
	if ans == 1 {
		return -1
	}
	return ans
}

// 2502. 设计内存分配器 https://leetcode.cn/problems/design-memory-allocator/description/
// 给你一个整数 n ，表示下标从 0 开始的内存数组的大小。所有内存单元开始都是空闲的。请你设计一个具备以下功能的内存分配器：分配 一块大小为 size 的连续空闲内存单元并赋 id mID 。释放 给定 id mID 对应的所有内存单元。注意：多个块可以被分配到同一个 mID 。你必须释放 mID 对应的所有内存单元，即便这些内存单元被分配在不同的块中。实现 Allocator 类：Allocator(int n) 使用一个大小为 n 的内存数组初始化 Allocator 对象。int allocate(int size, int mID) 找出大小为 size 个连续空闲内存单元且位于  最左侧 的块，分配并赋 id mID 。返回块的第一个下标。如果不存在这样的块，返回 -1 。int free(int mID) 释放 id mID 对应的所有内存单元。返回释放的内存单元数目。
type Allocator struct {
	allocatedBlockId map[int]struct{}
	blocks           list.List
	blockId          int
	mId2Id           map[int][]int
	// blocks map[int][]int
}

type BlockInfo struct {
	begin, end, id int // [begin,end)
}

func Constructor(n int) Allocator {
	return Allocator{}
}

func (this *Allocator) Allocate(size int, mID int) int {
	blockSize, arr := 0, make([]*list.Element, 0)
	for i := this.blocks.Front(); i != nil; i = i.Next() {
		info := i.Value.(BlockInfo)
		arr = append(arr, i)
		if _, ok := this.allocatedBlockId[info.id]; ok {
			blockSize = 0
			arr = make([]*list.Element, 0)
		} else {
			blockSize += info.end - info.begin
			if blockSize >= size {
				this.blockId++
				begin, end := arr[0].Value.(BlockInfo).begin, info.end
				ele := this.blocks.InsertAfter(BlockInfo{begin: begin, end: begin + size, id: this.blockId}, arr[0])
				if blockSize > size {
					this.blocks.InsertAfter(BlockInfo{begin: begin + size, end: end, id: 0}, ele)
				}
				for _, ele := range arr {
					this.blocks.Remove(ele)
				}
				return begin
			}
		}
	}
	return -1
}

func (this *Allocator) Free(mID int) int {
	ids, ok := this.mId2Id[mID]
	if !ok {
		return 0
	}
	ret := len(ids)
	for _, id := range ids {
		delete(this.allocatedBlockId, id)
	}
	return ret
}

// 1944. 队列中可以看到的人数 https://leetcode.cn/problems/number-of-visible-people-in-a-queue
// 有 n 个人排成一个队列，从左到右 编号为 0 到 n - 1 。给你以一个整数数组 heights ，每个整数 互不相同，heights[i] 表示第 i 个人的高度。一个人能 看到 他右边另一个人的条件是这两人之间的所有人都比他们两人 矮 。更正式的，第 i 个人能看到第 j 个人的条件是 i < j 且 min(heights[i], heights[j]) > max(heights[i+1], heights[i+2], ..., heights[j-1]) 。请你返回一个长度为 n 的数组 answer ，其中 answer[i] 是第 i 个人在他右侧队列中能 看到 的 人数 。
func canSeePersonsCount(heights []int) []int {
	size := len(heights)
	stk, ans := make([]int, 1, size+1), make([]int, size)
	stk[0] = 0x3f3f3f3f
	for i := size - 1; i >= 0; i-- {
		h := heights[i]
		for h < stk[len(stk)-1] {
			ans[i]++
			stk = stk[:len(stk)-1]
		}
		if len(stk) > 1 {
			ans[i]++
		}
		stk = append(stk, h)
	}
	return ans
}

// 2807. 在链表中插入最大公约数 https://leetcode.cn/problems/insert-greatest-common-divisors-in-linked-list/description/
// 给你一个链表的头 head ，每个结点包含一个整数值。在相邻结点之间，请你插入一个新的结点，结点值为这两个相邻结点值的 最大公约数 。请你返回插入之后的链表。两个数的 最大公约数 是可以被两个数字整除的最大正整数。
func insertGreatestCommonDivisors(head *ListNode) *ListNode {
	for p := head; p.Next != nil; {
		p.Next = &ListNode{Val: gcd(p.Val, p.Next.Val), Next: p.Next}
		p = p.Next.Next
	}
	return head
}

func gcd(a, b int) int {
	if a < b {
		return gcd(b, a)
	}
	if b == 0 {
		return a
	}
	return gcd(b, a%b)
}

// 2171. 拿出最少数目的魔法豆 https://leetcode.cn/problems/removing-minimum-number-of-magic-beans
// 给定一个 正整数 数组 beans ，其中每个整数表示一个袋子里装的魔法豆的数目。请你从每个袋子中 拿出 一些豆子（也可以 不拿出），使得剩下的 非空 袋子中（即 至少还有一颗 魔法豆的袋子）魔法豆的数目 相等。一旦把魔法豆从袋子中取出，你不能再将它放到任何袋子中。请返回你需要拿出魔法豆的 最少数目。
func minimumRemoval(beans []int) int64 {
	sort.Ints(beans)
	sum, ans, n := int64(0), int64(1<<62), len(beans)
	for _, bean := range beans {
		sum += int64(bean)
	}
	for i, bean := range beans {
		ans = min_int64(ans, sum-int64(bean)*int64(n-i))
	}
	return ans
}

func min_int64(a, b int64) int64 {
	if a < b {
		return a
	}
	return b
}

// 2671. 频率跟踪器 https://leetcode.cn/problems/frequency-tracker
// 请你设计并实现一个能够对其中的值进行跟踪的数据结构，并支持对频率相关查询进行应答。实现 FrequencyTracker 类：FrequencyTracker()：使用一个空数组初始化 FrequencyTracker 对象。void add(int number)：添加一个 number 到数据结构中。void deleteOne(int number)：从数据结构中删除一个 number 。数据结构 可能不包含 number ，在这种情况下不删除任何内容。bool hasFrequency(int frequency): 如果数据结构中存在出现 frequency 次的数字，则返回 true，否则返回 false。
type FrequencyTracker struct {
	Num2Freq map[int]int
	FreqCnt  map[int]int
}

func FrequencyTrackerConstructor() FrequencyTracker {
	return FrequencyTracker{
		Num2Freq: make(map[int]int),
		FreqCnt:  make(map[int]int),
	}
}

func (this *FrequencyTracker) Add(number int) {
	oldFreq := this.Num2Freq[number]
	this.Num2Freq[number] = oldFreq + 1
	this.FreqCnt[oldFreq]--
	this.FreqCnt[oldFreq+1]++
}

func (this *FrequencyTracker) DeleteOne(number int) {
	oldFreq := this.Num2Freq[number]
	if oldFreq != 0 {
		this.Num2Freq[number] = oldFreq - 1
	}
	this.FreqCnt[oldFreq]--
	this.FreqCnt[oldFreq-1]++
}

func (this *FrequencyTracker) HasFrequency(frequency int) bool {
	return this.FreqCnt[frequency] != 0
}

// 432. 全 O(1) 的数据结构 https://leetcode.cn/problems/all-oone-data-structure/
// 请你设计一个用于存储字符串计数的数据结构，并能够返回计数最小和最大的字符串。实现 AllOne 类：AllOne() 初始化数据结构的对象。inc(String key) 字符串 key 的计数增加 1 。如果数据结构中尚不存在 key ，那么插入计数为 1 的 key 。dec(String key) 字符串 key 的计数减少 1 。如果 key 的计数在减少后为 0 ，那么需要将这个 key 从数据结构中删除。测试用例保证：在减少计数前，key 存在于数据结构中。getMaxKey() 返回任意一个计数最大的字符串。如果没有元素存在，返回一个空字符串 "" 。getMinKey() 返回任意一个计数最小的字符串。如果没有元素存在，返回一个空字符串 "" 。 注意：每个函数都应当满足 O(1) 平均时间复杂度。
type AllOne struct {
	Head *Node
	Tail *Node
	mp   map[string]*Node
}

type Node struct {
	Count int
	Prev  *Node
	Next  *Node
	st    map[string]struct{}
}

func AllOneConstructor() AllOne {
	head := &Node{Count: 0}
	tail := &Node{Count: 1 << 30, Prev: head}
	head.Next = tail
	return AllOne{
		Head: head,
		Tail: tail,
		mp:   make(map[string]*Node),
	}
}

func (this *AllOne) Inc(key string) {
	node, ok := this.mp[key]
	if !ok {
		node = this.Head
	}
	if node.Next.Count != node.Count+1 {
		new_node := &Node{Count: node.Count + 1, st: make(map[string]struct{})}
		InsertAfter(node, new_node)
	}
	this.mp[key] = node.Next
	node.Next.st[key] = struct{}{}
	delete(node.st, key)
	if node.Count != 0 && len(node.st) == 0 {
		DeleteNode(node)
	}
}

func InsertAfter(node *Node, new_node *Node) {
	new_node.Next = node.Next
	new_node.Prev = node
	node.Next.Prev = new_node
	node.Next = new_node
}

func DeleteNode(node *Node) {
	node.Prev.Next = node.Next
	node.Next.Prev = node.Prev
}

func (this *AllOne) Dec(key string) {
	node, ok := this.mp[key]
	if !ok {
		return
	}
	if node.Count != 1 {
		if node.Prev.Count != node.Count-1 {
			InsertAfter(node.Prev, &Node{Count: node.Count - 1, st: make(map[string]struct{})})
		}
		node.Prev.st[key] = struct{}{}
	}
	if node.Count == 1 {
		delete(this.mp, key)
	} else {
		this.mp[key] = node.Prev
	}
	delete(node.st, key)
	if len(node.st) == 0 {
		DeleteNode(node)
	}
}

func (this *AllOne) GetMaxKey() string {
	for k, _ := range this.Tail.Prev.st {
		return k
	}
	return ""
}

func (this *AllOne) GetMinKey() string {
	for k, _ := range this.Head.Next.st {
		return k
	}
	return ""
}

// 2617. 网格图中最少访问的格子数 https://leetcode.cn/problems/minimum-number-of-visited-cells-in-a-grid
// 给你一个下标从 0 开始的 m x n 整数矩阵 grid 。你一开始的位置在 左上角 格子 (0, 0) 。当你在格子 (i, j) 的时候，你可以移动到以下格子之一：满足 j < k <= grid[i][j] + j 的格子 (i, k) （向右移动），或者满足 i < k <= grid[i][j] + i 的格子 (k, j) （向下移动）。请你返回到达 右下角 格子 (m - 1, n - 1) 需要经过的最少移动格子数，如果无法到达右下角格子，请你返回 -1 。
func minimumVisitedCells(grid [][]int) int {
	m, n, max_i, step := len(grid), len(grid[0]), 0x3f3f3f3f, 0
	type State struct {
		step int
		next int
	}
	colStks := make([][]State, n)
	for i := 0; i < n; i++ {
		colStks[i] = append([]State{}, State{next: max_i, step: max_i})
	}
	colStks[0] = append(colStks[0], State{next: 0, step: 0})
	updateStk := func(stk *[]State, step, next int) {
		if step < (*stk)[len(*stk)-1].step {
			(*stk) = append((*stk), State{next: next, step: step})
		} else if step == (*stk)[len(*stk)-1].step {
			(*stk)[len(*stk)-1].next = max((*stk)[len(*stk)-1].next, next)
		} else {
			if next < (*stk)[len(*stk)-2].step {
				(*stk) = append((*stk), (*stk)[len(*stk)-1])
				(*stk)[len(*stk)-2] = State{next: next, step: step}
			} else if next > (*stk)[len(*stk)-2].next {
				(*stk)[len(*stk)-2] = State{next: next, step: step}
			}
		}
	}
	for i := 0; i < m; i++ {
		rowStk := append([]State{}, State{next: max_i, step: max_i})
		for j := 0; j < n; j++ {
			for colStks[j][len(colStks[j])-1].next < i {
				colStks[j] = colStks[j][:len(colStks[j])-1]
			}
			for rowStk[len(rowStk)-1].next < j {
				rowStk = rowStk[:len(rowStk)-1]
			}
			step = min(rowStk[len(rowStk)-1].step, colStks[j][len(colStks[j])-1].step) + 1
			if step >= max_i { // 无法到达
				continue
			}
			updateStk(&rowStk, step, j+grid[i][j])
			updateStk(&colStks[j], step, i+grid[i][j])
		}
	}
	if step >= max_i {
		return -1
	}
	return step
}

// 2642. 设计可以求最短路径的图类 https://leetcode.cn/problems/design-graph-with-shortest-path-calculator
// 给你一个有 n 个节点的 有向带权 图，节点编号为 0 到 n - 1 。图中的初始边用数组 edges 表示，其中 edges[i] = [fromi, toi, edgeCosti] 表示从 fromi 到 toi 有一条代价为 edgeCosti 的边。请你实现一个 Graph 类：Graph(int n, int[][] edges) 初始化图有 n 个节点，并输入初始边。addEdge(int[] edge) 向边集中添加一条边，其中 edge = [from, to, edgeCost] 。数据保证添加这条边之前对应的两个节点之间没有有向边。int shortestPath(int node1, int node2) 返回从节点 node1 到 node2 的路径 最小 代价。如果路径不存在，返回 -1 。一条路径的代价是路径中所有边代价之和。
type Graph struct {
	g [][]GraphNode
}

type GraphNode struct {
	next, cost int
}

func GraphConstructor(n int, edges [][]int) Graph {
	g := make([][]GraphNode, n)
	for i := 0; i < n; i++ {
		g[i] = make([]GraphNode, 0)
	}
	for _, edge := range edges {
		g[edge[0]] = append(g[edge[0]], GraphNode{next: edge[1], cost: edge[2]})
	}
	return Graph{g: g}
}

func (this *Graph) AddEdge(edge []int) {
	this.g[edge[0]] = append(this.g[edge[0]], GraphNode{next: edge[1], cost: edge[2]})
}

func (this *Graph) ShortestPath(node1 int, node2 int) int {
	n, max_i := len(this.g), 0x3f3f3f3f
	dist := make([]int, n)
	for i := 0; i < n; i++ {
		dist[i] = max_i
	}
	dist[node1] = 0
	pq := &NodePriorityQueue{}
	heap.Push(pq, GraphNode{cost: 0, next: node1})
	for pq.Len() > 0 {
		node := heap.Pop(pq).(GraphNode)
		if node.next == node2 {
			return node.cost
		}
		if node.cost > dist[node.next] {
			continue
		}
		for _, edge := range this.g[node.next] {
			if node.cost+edge.cost < dist[edge.next] {
				dist[edge.next] = node.cost + edge.cost
				heap.Push(pq, GraphNode{cost: dist[edge.next], next: edge.next})
			}
		}
	}
	return -1
}

type NodePriorityQueue []GraphNode

func (pq NodePriorityQueue) Swap(i, j int) {
	pq[i], pq[j] = pq[j], pq[i]
}

func (pq NodePriorityQueue) Len() int {
	return len(pq)
}

func (pq NodePriorityQueue) Less(i, j int) bool {
	return pq[i].cost < pq[j].cost
}

func (pq *NodePriorityQueue) Push(x any) {
	*pq = append(*pq, x.(GraphNode))
}

func (pq *NodePriorityQueue) Pop() any {
	n := len(*pq)
	x := (*pq)[n-1]
	*pq = (*pq)[:n-1]
	return x
}
