/*
 * @Author: zanilia
 * @Date: 2022-10-16 16:02:54
 * @LastEditTime: 2023-02-27 16:24:25
 * @Descripttion:
 */
package leetcode3

import (
	"sort"
)

// 6211. 创建价值相同的连通块 https://leetcode.cn/problems/create-components-with-same-value/
// 有一棵 n 个节点的无向树，节点编号为 0 到 n - 1 。给你一个长度为 n 下标从 0 开始的整数数组 nums ，其中 nums[i] 表示第 i 个节点的值。同时给你一个长度为 n - 1 的二维整数数组 edges ，其中 edges[i] = [ai, bi] 表示节点 ai 与 bi 之间有一条边。你可以 删除 一些边，将这棵树分成几个连通块。一个连通块的 价值 定义为这个连通块中 所有 节点 i 对应的 nums[i] 之和。你需要删除一些边，删除后得到的各个连通块的价值都相等。请返回你可以删除的边数 最多 为多少。
func componentValue(nums []int, edges [][]int) int {
	n := len(nums)
	graph := make([][]int, n)
	for i := 0; i < n-1; i++ {
		graph[edges[i][0]] = append(graph[edges[i][0]], edges[i][1])
		graph[edges[i][1]] = append(graph[edges[i][1]], edges[i][0])
	}
	var dfs func(node, pa, target int) int
	dfs = func(node, pa, target int) int {
		ret := nums[node]
		for _, next := range graph[node] {
			if next == pa {
				continue
			}
			t := dfs(next, node, target)
			if t < 0 {
				return -1
			}
			ret += t
		}
		if ret > target {
			return -1
		}
		return ret % target
	}
	sum := 0
	for _, num := range nums {
		sum += num
	}
	for i := n; i > 0; i-- {
		if sum%i == 0 {
			target := sum / i
			if dfs(0, -1, target) == 0 {
				return i - 1
			}
		}
	}
	return 0
}

// 6214. 判断两个事件是否存在冲突 https://leetcode.cn/problems/determine-if-two-events-have-conflict/
// 给你两个字符串数组 event1 和 event2 ，表示发生在同一天的两个闭区间时间段事件，其中：event1 = [startTime1, endTime1] 且event2 = [startTime2, endTime2].事件的时间为有效的 24 小时制且按 HH:MM 格式给出。当两个事件存在某个非空的交集时（即，某些时刻是两个事件都包含的），则认为出现 冲突 。如果两个事件之间存在冲突，返回 true ；否则，返回 false 。
func haveConflict(event1 []string, event2 []string) bool {
	if event1[0] < event2[0] {
		return event1[1] >= event2[0]
	} else {
		return event2[1] >= event1[0]
	}
}

// 6224. 最大公因数等于 K 的子数组数目 https://leetcode.cn/contest/weekly-contest-316/problems/number-of-subarrays-with-gcd-equal-to-k/ 枚举遍历
// 给你一个整数数组 nums 和一个整数 k ，请你统计并返回 nums 的子数组中元素的最大公因数等于 k 的子数组数目。子数组 是数组中一个连续的非空序列。数组的最大公因数 是能整除数组中所有元素的最大整数。
func subarrayGCD(nums []int, k int) int {
	size, ans := len(nums), 0
	for i := 0; i < size; i++ {
		f := nums[i]
		if f == k {
			ans++
		}
		for j := i + 1; j < size; j++ {
			f = gcd(f, nums[i])
			if f == k {
				ans++
			} else if f%k != 0 {
				break
			}
		}
	}
	return ans
}

func gcd(a, b int) int {
	for a != 0 {
		a, b = b%a, a
	}
	return b
}

// 6216. 使数组相等的最小开销 https://leetcode.cn/contest/weekly-contest-316/problems/minimum-cost-to-make-array-equal/ 二分查找
// 给你两个下标从 0 开始的数组 nums 和 cost ，分别包含 n 个 正 整数。你可以执行下面操作 任意 次：将 nums 中 任意 元素增加或者减小 1 。对第 i 个元素执行一次操作的开销是 cost[i] 。请你返回使 nums 中所有元素 相等 的 最少 总开销。
func minCost(nums []int, cost []int) int64 {
	mi, ma := nums[0], nums[0]
	for _, num := range nums {
		mi = min(mi, num)
		ma = max(ma, num)
	}
	if mi == ma {
		return 0
	}
	mid := sort.Search(ma-mi+1, func(mid int) bool {
		mid += mi
		diff := int64(0)
		for i, num := range nums {
			if num <= mid {
				diff += int64(cost[i])
			} else {
				diff -= int64(cost[i])
			}
		}
		return diff > 0
	})
	ans := int64(0)
	for i, num := range nums {
		ans += int64(abs(mid-num)) * int64(cost[i])
	}
	return ans
}

func abs(a int) int {
	if a >= 0 {
		return a
	}
	return -a
}

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

// 6217. 使数组相似的最少操作次数 https://leetcode.cn/contest/weekly-contest-316/problems/minimum-number-of-operations-to-make-arrays-similar/ 排序
// 给你两个正整数数组 nums 和 target ，两个数组长度相等。在一次操作中，你可以选择两个 不同 的下标 i 和 j ，其中 0 <= i, j < nums.length ，并且：令 nums[i] = nums[i] + 2 且令 nums[j] = nums[j] - 2 。如果两个数组中每个元素出现的频率相等，我们称两个数组是 相似 的。请你返回将 nums 变得与 target 相似的最少操作次数。测试数据保证 nums 一定能变得与 target 相似。
func makeSimilar(nums []int, target []int) int64 {
	a1, a2, b1, b2 := make([]int, 0), make([]int, 0), make([]int, 0), make([]int, 0)
	for _, num := range nums {
		if num&1 != 0 {
			a1 = append(a1, num)
		} else {
			a2 = append(a2, num)
		}
	}
	for _, num := range target {
		if num&1 != 0 {
			b1 = append(a1, num)
		} else {
			b2 = append(a2, num)
		}
	}
	sort.Ints(a1)
	sort.Ints(a2)
	sort.Ints(b1)
	sort.Ints(b2)
	ans := int64(0)
	for i, num := range a1 {
		ans += int64(abs(num - b1[i]))
	}
	for i, num := range a2 {
		ans += int64(abs(num - b2[i]))
	}
	return ans / 4
}

// 6225. 差值数组不同的字符串 https://leetcode.cn/problems/odd-string-difference/
// 给你一个字符串数组 words ，每一个字符串长度都相同，令所有字符串的长度都为 n 。每个字符串 words[i] 可以被转化为一个长度为 n - 1 的 差值整数数组 difference[i] ，其中对于 0 <= j <= n - 2 有 difference[i][j] = words[i][j+1] - words[i][j] 。注意两个字母的差值定义为它们在字母表中 位置 之差，也就是说 'a' 的位置是 0 ，'b' 的位置是 1 ，'z' 的位置是 25 。比方说，字符串 "acb" 的差值整数数组是 [2 - 0, 1 - 2] = [2, -1] 。words 中所有字符串 除了一个字符串以外 ，其他字符串的差值整数数组都相同。你需要找到那个不同的字符串。请你返回 words中 差值整数数组 不同的字符串。
func oddString(words []string) string {
	type MyArr [20]int
	size, n := len(words), len(words[0])
	diff := make([]MyArr, size)
	for i := 0; i < size; i++ {
		for j := 0; j < 20; j++ {
			diff[i][j] = 0
		}
	}
	for i := 0; i < size; i++ {
		for j := 0; j < n-2; j++ {
			diff[i][j] = int(words[i][j+1] - words[i][j])
		}
	}
	for i := 1; i < size; i++ {
		if diff[i] != diff[0] {
			if i == 1 {
				if diff[i] != diff[i+1] {
					return words[i]
				} else {
					return words[0]
				}
			} else {
				if diff[i] != diff[i-1] {
					return words[i]
				} else {
					return words[0]
				}
			}
		}
	}
	return words[0]
}

// 6228. 距离字典两次编辑以内的单词 https://leetcode.cn/problems/words-within-two-edits-of-dictionary/
// 给你两个字符串数组 queries 和 dictionary 。数组中所有单词都只包含小写英文字母，且长度都相同。一次 编辑 中，你可以从 queries 中选择一个单词，将任意一个字母修改成任何其他字母。从 queries 中找到所有满足以下条件的字符串：不超过 两次编辑内，字符串与 dictionary 中某个字符串相同。请你返回 queries 中的单词列表，这些单词距离 dictionary 中的单词 编辑次数 不超过 两次 。单词返回的顺序需要与 queries 中原本顺序相同。
func twoEditWords(queries []string, dictionary []string) []string {
	getEditDistance := func(from string, to string) int {
		size := len(from)
		if size != len(to) {
			return 1 << 30
		}
		cnt := 0
		for idx, ch := range from {
			if ch != rune(to[idx]) {
				cnt++
			}
		}
		return cnt
	}
	ans := make([]string, 0, len(queries))
	for _, query := range queries {
		for _, dict := range dictionary {
			if getEditDistance(query, dict) <= 2 {
				ans = append(ans, query)
				break
			}
		}
	}
	return ans
}

// 6226. 摧毁一系列目标 https://leetcode.cn/problems/destroy-sequential-targets/
// 给你一个下标从 0 开始的数组 nums ，它包含若干正整数，表示数轴上你需要摧毁的目标所在的位置。同时给你一个整数 space 。你有一台机器可以摧毁目标。给机器 输入 nums[i] ，这台机器会摧毁所有位置在 nums[i] + c * space 的目标，其中 c 是任意非负整数。你想摧毁 nums 中 尽可能多 的目标。请你返回在摧毁数目最多的前提下，nums[i] 的 最小值 。
func destroyTargets(nums []int, space int) int {
	mp := make(map[int]int)
	for _, num := range nums {
		mp[num%space]++
	}
	min_mod, max_cnt := make(map[int]struct{}), -1
	for num, cnt := range mp {
		if cnt > max_cnt {
			max_cnt = cnt
			min_mod = make(map[int]struct{})
			min_mod[num] = struct{}{}
		} else if cnt == max_cnt {
			min_mod[num] = struct{}{}
		}
	}
	ans := (1 << 30)
	for _, num := range nums {
		_, ok := mp[num%space]
		if ok {
			ans = min(ans, num)
		}
	}
	return ans
}

// 6227. 下一个更大元素 IV https://leetcode.cn/problems/next-greater-element-iv/
// 给你一个下标从 0 开始的非负整数数组 nums 。对于 nums 中每一个整数，你必须找到对应元素的 第二大 整数。如果 nums[j] 满足以下条件，那么我们称它为 nums[i] 的 第二大 整数：j > i，nums[j] > nums[i]，恰好存在 一个 k 满足 i < k < j 且 nums[k] > nums[i] 。如果不存在 nums[j] ，那么第二大整数为 -1 。比方说，数组 [1, 2, 4, 3] 中，1 的第二大整数是 4 ，2 的第二大整数是 3 ，3 和 4 的第二大整数是 -1 。请你返回一个整数数组 answer ，其中 answer[i]是 nums[i] 的第二大整数。
func secondGreaterElement(nums []int) []int {
	n := len(nums)
	ans := make([]int, n)
	for i := 0; i < n; i++ {
		ans[i] = -1
	}
	s, t := make([]int, 0, n), make([]int, 0, n)
	for i := 0; i < n; i++ {
		for len(t) > 0 && nums[t[len(t)-1]] < nums[i] {
			ans[t[len(t)-1]] = nums[i]
			t = t[:len(t)-1]
		}
		j := len(s) - 1
		for j >= 0 && nums[s[j]] < nums[i] {
			j--
		}
		t = append(t, s[j+1:]...)
		s = append(s[:j+1], i)
	}
	return ans
}

// 6222. 美丽整数的最小增量 https://leetcode.cn/problems/minimum-addition-to-make-integer-beautiful/
// 给你两个正整数 n 和 target 。如果某个整数每一位上的数字相加小于或等于 target ，则认为这个整数是一个 美丽整数 。找出并返回满足 n + x 是 美丽整数 的最小非负整数 x 。生成的输入保证总可以使 n 变成一个美丽整数。
func makeIntegerBeautiful(n int64, target int) int64 {
	return 0
}

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

// 6223. 移除子树后的二叉树高度 https://leetcode.cn/problems/height-of-binary-tree-after-subtree-removal-queries/
// 给你一棵 二叉树 的根节点 root ，树中有 n 个节点。每个节点都可以被分配一个从 1 到 n 且互不相同的值。另给你一个长度为 m 的数组 queries 。你必须在树上执行 m 个 独立 的查询，其中第 i 个查询你需要执行以下操作：从树中 移除 以 queries[i] 的值作为根节点的子树。题目所用测试用例保证 queries[i] 不 等于根节点的值。返回一个长度为 m 的数组 answer ，其中 answer[i] 是执行第 i 个查询后树的高度。注意：查询之间是独立的，所以在每个查询执行后，树会回到其 初始 状态。树的高度是从根到树中某个节点的 最长简单路径中的边数 。
func treeQueries(root *TreeNode, queries []int) []int {
	return make([]int, 0)
}

func distinctAverages(nums []int) int {
	sort.Ints(nums)
	sum := 0
	for _, num := range nums {
		sum += num
	}
	mp := make(map[int]struct{})
	size := len(nums)
	for i := 0; i < size/2; i++ {
		mp[nums[i]+nums[size-1-i]] = struct{}{}
	}
	return len(mp)
}

// 6249. 分割圆的最少切割次数 https://leetcode.cn/problems/minimum-cuts-to-divide-a-circle
// 圆内一个 有效切割 ，符合以下二者之一：该切割是两个端点在圆上的线段，且该线段经过圆心。该切割是一端在圆心另一端在圆上的线段。给你一个整数 n ，请你返回将圆切割成相等的 n 等分的 最少 切割次数。
func numberOfCuts(n int) int {
	if n == 0 {
		return 0
	} else if n%2 == 0 {
		return n / 2
	} else {
		return n
	}
}

// 6277. 行和列中一和零的差值 https://leetcode.cn/problems/difference-between-ones-and-zeros-in-row-and-column/
// 给你一个下标从 0 开始的 m x n 二进制矩阵 grid 。我们按照如下过程，定义一个下标从 0 开始的 m x n 差值矩阵 diff ：令第 i 行一的数目为 onesRowi 。令第 j 列一的数目为 onesColj 。令第 i 行零的数目为 zerosRowi 。令第 j 列零的数目为 zerosColj 。diff[i][j] = onesRowi + onesColj - zerosRowi - zerosColj。请你返回差值矩阵 diff 。
func onesMinusZeros(grid [][]int) [][]int {
	m, n := len(grid), len(grid[0])
	mp := make(map[int]int)
	getCnt := func(pos, val, dir int) int {
		tem := pos*4 + val*2 + dir
		cnt, flag := mp[tem]
		if flag {
			return cnt
		} else {
			cnt = 0
		}
		if dir == 1 {
			for i := 0; i < m; i++ {
				if grid[i][pos] == val {
					cnt++
				}
			}
		} else {
			for i := 0; i < n; i++ {
				if grid[pos][i] == val {
					cnt++
				}
			}
		}
		mp[tem] = cnt
		return cnt
	}
	res := make([][]int, m)
	for i := 0; i < m; i++ {
		res[i] = make([]int, n)
		for j := 0; j < n; j++ {
			res[i][j] = getCnt(i, 1, 0) + getCnt(j, 1, 1) - getCnt(i, 0, 0) - getCnt(j, 0, 1)
		}
	}
	return res
}

// 6250. 商店的最少代价 https://leetcode.cn/problems/minimum-penalty-for-a-shop/
// 给你一个顾客访问商店的日志，用一个下标从 0 开始且只包含字符 'N' 和 'Y' 的字符串 customers 表示：如果第 i 个字符是 'Y' ，它表示第 i 小时有顾客到达。如果第 i 个字符是 'N' ，它表示第 i 小时没有顾客到达。如果商店在第 j 小时关门（0 <= j <= n），代价按如下方式计算：在开门期间，如果某一个小时没有顾客到达，代价增加 1 。在关门期间，如果某一个小时有顾客到达，代价增加 1 。请你返回在确保代价 最小 的前提下，商店的 最早 关门时间。注意，商店在第 j 小时关门表示在第 j 小时以及之后商店处于关门状态。
func bestClosingTime(customers string) int {
	n := len(customers)
	n_cnt, y_cnt := make([]int, n+1), make([]int, n+1)
	n_cnt[n], y_cnt[n] = 0, 0
	for i := n - 1; i >= 0; i-- {
		y_cnt[i] = y_cnt[i+1]
		if customers[i] == 'Y' {
			y_cnt[i]++
		}
	}
	for i := 1; i <= n; i++ {
		n_cnt[i] = n_cnt[i-1]
		if customers[i-1] == 'N' {
			n_cnt[i]++
		}
	}
	min_cost, min_pos := 1<<30, 0
	for i := 0; i <= n; i++ {
		tem_cost := y_cnt[i] + n_cnt[i]
		if tem_cost < min_cost {
			min_pos = i
			min_cost = tem_cost
		}
	}
	return min_pos
}

// 6251. 统计回文子序列数目 https://leetcode.cn/problems/count-palindromic-subsequences/
// 给你数字字符串 s ，请你返回 s 中长度为 5 的 回文子序列 数目。由于答案可能很大，请你将答案对 109 + 7 取余 后返回。提示：如果一个字符串从前往后和从后往前读相同，那么它是 回文字符串 。子序列是一个字符串中删除若干个字符后，不改变字符顺序，剩余字符构成的字符串。
func countPalindromes(s string) int {
	mod, n, ans := int(1e9+7), len(s), 0
	suf, suf2 := [10]int{}, [10][10]int{}
	pre, pre2 := [10]int{}, [10][10]int{}
	for i := n - 1; i >= 0; i-- {
		tem := s[i] - '0'
		for ch, cnt := range suf {
			suf2[tem][ch] += cnt
		}
		suf[tem]++
	}
	for _, ch := range s[:n-1] {
		ch -= '0'
		suf[ch]--
		for j, c := range suf {
			suf2[ch][j] -= c
		}
		for i := 0; i < 10; i++ {
			for j := 0; j < 10; j++ {
				ans = (ans + pre2[i][j]*suf2[i][j]) % mod
			}
		}
		for idx, cnt := range pre {
			pre2[ch][idx] += cnt
		}
		pre[ch]++
	}
	return ans
}

// 1930. 长度为 3 的不同回文子序列 https://leetcode.cn/problems/unique-length-3-palindromic-subsequences/
// 给你一个字符串 s ，返回 s 中 长度为 3 的不同回文子序列 的个数。即便存在多种方法来构建相同的子序列，但相同的子序列只计数一次。回文 是正着读和反着读一样的字符串。子序列 是由原字符串删除其中部分字符（也可以不删除）且不改变剩余字符之间相对顺序形成的一个新字符串。例如，"ace" 是 "abcde" 的一个子序列。
func countPalindromicSubsequence(s string) int {
	size := len(s)
	pre, suf := make([]int, 26), make([]int, 26)
	var flag [26][26]bool
	for i := 1; i < size; i++ {
		suf[s[i]-'a']++
	}
	for i := 1; i < size; i++ {
		pre[s[i-1]-'a']++
		suf[s[i]-'a']--
		for j := 0; j < 26; j++ {
			if pre[j] >= 1 && suf[j] >= 1 {
				flag[j][s[i]-'a'] = true
			}
		}
	}
	ans := 0
	for i := 0; i < 26; i++ {
		for j := 0; j < 26; j++ {
			if flag[i][j] {
				ans++
			}
		}
	}
	return ans
}

// 1144. 递减元素使数组呈锯齿状 https://leetcode.cn/problems/decrease-elements-to-make-array-zigzag/
// 给你一个整数数组 nums，每次 操作 会从中选择一个元素并 将该元素的值减少 1。如果符合下列情况之一，则数组 A 就是 锯齿数组：每个偶数索引对应的元素都大于相邻的元素，即 A[0] > A[1] < A[2] > A[3] < A[4] > ...或者，每个奇数索引对应的元素都大于相邻的元素，即 A[0] < A[1] > A[2] < A[3] > A[4] < ... 返回将数组 nums 转换为锯齿数组所需的最小操作次数。
func movesToMakeZigzag(nums []int) int {
	size := len(nums)
	f := func(start int) int {
		res := 0
		for i := start; i < size; i += 2 {
			diff := 0
			if i-1 >= 0 {
				diff = max(diff, nums[i]-nums[i-1]+1)
			}
			if i+1 < size {
				diff = max(diff, nums[i]-nums[i+1]+1)
			}
			res += diff
		}
		return res
	}
	return min(f(0), f(1))
}
