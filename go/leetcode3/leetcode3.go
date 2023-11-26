package leetcode3

import (
	"container/heap"
	"sort"
	"strings"
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

// 2208. 将数组和减半的最少操作次数 https://leetcode.cn/problems/minimum-operations-to-halve-array-sum/
// 给你一个正整数数组 nums 。每一次操作中，你可以从 nums 中选择 任意 一个数并将它减小到 恰好 一半。（注意，在后续操作中你可以对减半过的数继续执行操作） 请你返回将 nums 数组和 至少 减少一半的 最少 操作数。
func halveArray(nums []int) int {
	size := len(nums)
	hp := &doubleHp{arr: make([]float64, size)}
	sum, ans := float64(0), 0
	for i, num := range nums {
		hp.arr[i] = float64(num)
		sum += float64(num)
	}
	sum /= 2
	heap.Init(hp)
	for sum > 0 {
		top := heap.Pop(hp).(float64)
		sum -= top / 2
		heap.Push(hp, top/2)
		ans++
	}
	return ans
}

type doubleHp struct{ arr []float64 }

func (hp *doubleHp) Less(i, j int) bool {
	return hp.arr[i] > hp.arr[j]
}

func (hp *doubleHp) Swap(i, j int) {
	hp.arr[i], hp.arr[j] = hp.arr[j], hp.arr[i]
}

func (hp *doubleHp) Len() int {
	return len(hp.arr)
}

func (hp *doubleHp) Push(v interface{}) {
	hp.arr = append(hp.arr, v.(float64))
}

func (hp *doubleHp) Pop() interface{} {
	size := len(hp.arr)
	ret := hp.arr[size-1]
	hp.arr = hp.arr[:len(hp.arr)-1]
	return ret
}

// 2734. 执行子串操作后的字典序最小字符串 https://leetcode.cn/problems/lexicographically-smallest-string-after-substring-operation/
// 给你一个仅由小写英文字母组成的字符串 s 。在一步操作中，你可以完成以下行为：选则 s 的任一非空子字符串，可能是整个字符串，接着将字符串中的每一个字符替换为英文字母表中的前一个字符。例如，'b' 用 'a' 替换，'a' 用 'z' 替换。返回执行上述操作 恰好一次 后可以获得的 字典序最小 的字符串。子字符串 是字符串中的一个连续字符序列。现有长度相同的两个字符串 x 和 字符串 y ，在满足 x[i] != y[i] 的第一个位置 i 上，如果  x[i] 在字母表中先于 y[i] 出现，则认为字符串 x 比字符串 y 字典序更小 。
func smallestString(s string) string {
	byteArr, size := []byte(s), len(s)
	for i := 0; i < size; i++ {
		if byteArr[i] != 'a' {
			for ; i < size && byteArr[i] != 'a'; i++ {
				byteArr[i]--
			}
			return string(byteArr)
		}
	}
	byteArr[size-1] = 'z'
	return string(byteArr)
}

// 1529. 最少的后缀翻转次数 https://leetcode.cn/problems/minimum-suffix-flips/
// 给你一个长度为 n 、下标从 0 开始的二进制字符串 target 。你自己有另一个长度为 n 的二进制字符串 s ，最初每一位上都是 0 。你想要让 s 和 target 相等。在一步操作，你可以选择下标 i（0 <= i < n）并翻转在 闭区间 [i, n - 1] 内的所有位。翻转意味着 '0' 变为 '1' ，而 '1' 变为 '0' 。返回使 s 与 target 相等需要的最少翻转次数。
func minFlips(target string) int {
	ans, prev := 0, '0'
	for _, ch := range target {
		if ch != prev {
			ans++
		}
		ch = prev
	}
	return ans
}

// 1289. 下降路径最小和 II https://leetcode.cn/problems/minimum-falling-path-sum-ii/
// 给你一个 n x n 整数矩阵 grid ，请你返回 非零偏移下降路径 数字和的最小值。非零偏移下降路径 定义为：从 grid 数组中的每一行选择一个数字，且按顺序选出来的数字中，相邻数字不在原数组的同一列。
func minFallingPathSum(grid [][]int) int {
	INT32_MAX, n := 1<<30, len(grid)
	first_min, second_min, first_idx := 0, 0, -1
	for i := 0; i < n; i++ {
		curr_first, curr_second, curr_idx, curr_sum := INT32_MAX, INT32_MAX, -1, 0
		for j := 0; j < n; j++ {
			if j == first_idx {
				curr_sum = grid[i][j] + second_min
			} else {
				curr_sum = grid[i][j] + first_min
			}
			if curr_sum < curr_first {
				curr_second = curr_first
				curr_first = curr_sum
				curr_idx = j
			} else if curr_sum < curr_second {
				curr_second = curr_sum
			}
		}
		first_min, second_min, first_idx = curr_first, curr_second, curr_idx
	}
	return first_min
}

// 1253. 重构 2 行二进制矩阵 https://leetcode.cn/problems/reconstruct-a-2-row-binary-matrix/
// 给你一个 2 行 n 列的二进制数组：矩阵是一个二进制矩阵，这意味着矩阵中的每个元素不是 0 就是 1。第 0 行的元素之和为 upper。第 1 行的元素之和为 lower。第 i 列（从 0 开始编号）的元素之和为 colsum[i]，colsum 是一个长度为 n 的整数数组。你需要利用 upper，lower 和 colsum 来重构这个矩阵，并以二维整数数组的形式返回它。如果有多个不同的答案，那么任意一个都可以通过本题。如果不存在符合要求的答案，就请返回一个空的二维数组。
func reconstructMatrix(upper int, lower int, colsum []int) [][]int {
	n := len(colsum)
	ans := make([][]int, 2)
	ans[0], ans[1] = make([]int, n), make([]int, n)
	for idx, val := range colsum {
		ans[0][idx], ans[1][idx] = 0, 0
		if val == 1 {
			if upper > lower {
				ans[0][idx] = 1
				upper--
			} else {
				ans[1][idx] = 1
				lower--
			}
		} else if val == 2 {
			ans[0][idx], ans[1][idx] = 1, 1
			upper--
			lower--
		}
		if upper < 0 || lower < 0 {
			return make([][]int, 0)
		}
	}
	if upper > 0 || lower > 0 {
		return make([][]int, 0)
	}
	return ans
}

type ListNode struct {
	Val  int
	Next *ListNode
}

// 23. 合并 K 个升序链表 https://leetcode.cn/problems/merge-k-sorted-lists/
// 给你一个链表数组，每个链表都已经按升序排列。请你将所有链表合并到一个升序链表中，返回合并后的链表。
func mergeKLists(lists []*ListNode) *ListNode {
	size := len(lists)
	if size != 0 {
		return nil
	}
	for size != 1 {
		for i := 0; i < size; i++ {
			lists[i] = mergeTwoList(lists[2*i], lists[2*i+1])
		}
		size /= 2
	}
	return lists[0]
}

func mergeTwoList(l1 *ListNode, l2 *ListNode) *ListNode {
	head := &ListNode{Val: 0, Next: nil}
	p := head
	for l1 != nil && l2 != nil {
		if l1.Val < l2.Val {
			p.Next = l1
			l1 = l1.Next
		} else {
			p.Next = l2
			l2.Next = l2
		}
		p = p.Next
	}
	if l1 != nil {
		p.Next = l1
	}
	if l2 != nil {
		p.Next = l2
	}
	return head.Next
}

// 6201. 找出前缀异或的原始数组 https://leetcode.cn/problems/find-the-original-array-of-prefix-xor/
// 给你一个长度为 n 的 整数 数组 pref 。找出并返回满足下述条件且长度为 n 的数组 arr ：pref[i] = arr[0] ^ arr[1] ^ ... ^ arr[i].注意 ^ 表示 按位异或（bitwise-xor）运算。可以证明答案是 唯一 的。
func findArray(pref []int) []int {
	size := len(pref)
	ans := make([]int, size)
	ans[0] = pref[0]
	for i := 1; i < size; i++ {
		ans[i] = pref[i] ^ pref[i-1]
	}
	return ans
}

// 617. 合并二叉树 https://leetcode.cn/problems/merge-two-binary-trees/
// 给你两棵二叉树： root1 和 root2 。想象一下，当你将其中一棵覆盖到另一棵之上时，两棵树上的一些节点将会重叠（而另一些不会）。你需要将这两棵树合并成一棵新二叉树。合并的规则是：如果两个节点重叠，那么将这两个节点的值相加作为合并后节点的新值；否则，不为 null 的节点将直接作为新二叉树的节点。返回合并后的二叉树。注意: 合并过程必须从两个树的根节点开始。
func mergeTrees(root1 *TreeNode, root2 *TreeNode) *TreeNode {
	if root1 == nil && root2 == nil {
		return nil
	}
	var left1, right1, left2, right2 *TreeNode
	left1, right1, left2, right2 = nil, nil, nil, nil
	val := 0
	if root1 != nil {
		val += root1.Val
		left1, right1 = root1.Left, root1.Right
	}
	if root2 != nil {
		val += root2.Val
		left1, right2 = root2.Left, root2.Right
	}
	root := &TreeNode{Val: val}
	root.Left = mergeTrees(left1, left2)
	root.Right = mergeTrees(right1, right2)
	return root
}

// 833. 字符串中的查找与替换 https://leetcode.cn/problems/find-and-replace-in-string/
// 你会得到一个字符串 s (索引从 0 开始)，你必须对它执行 k 个替换操作。替换操作以三个长度均为 k 的并行数组给出：indices, sources,  targets。要完成第 i 个替换操作:检查 子字符串  sources[i] 是否出现在 原字符串 s 的索引 indices[i] 处。如果没有出现， 什么也不做 。如果出现，则用 targets[i] 替换 该子字符串。例如，如果 s = "abcd" ， indices[i] = 0 , sources[i] = "ab"， targets[i] = "eee" ，那么替换的结果将是 "eeecd" 。所有替换操作必须 同时 发生，这意味着替换操作不应该影响彼此的索引。测试用例保证元素间不会重叠 。例如，一个 s = "abc" ，  indices = [0,1] ， sources = ["ab"，"bc"] 的测试用例将不会生成，因为 "ab" 和 "bc" 替换重叠。在对 s 执行所有替换操作后返回 结果字符串 。子字符串 是字符串中连续的字符序列。
func findReplaceString(s string, indices []int, sources []string, targets []string) string {
	builder := new(strings.Builder)
	size, size1 := len(indices), len(s)
	idx := make([]int, size)
	for i := 0; i < size; i++ {
		idx[i] = i
	}
	sort.Slice(idx, func(i, j int) bool {
		return indices[idx[i]] < indices[idx[j]]
	})
	prev := 0
	for _, i := range idx {
		builder.WriteString(s[prev:indices[i]])
		end := indices[i] + len(sources[i])
		if end <= size1 && s[indices[i]:end] == sources[i] {
			builder.WriteString(targets[i])
			prev = end
		} else {
			prev = indices[i]
		}
	}
	builder.WriteString(s[prev:size1])
	return builder.String()
}

// 1333. 餐厅过滤器 https://leetcode.cn/problems/filter-restaurants-by-vegan-friendly-price-and-distance
// 给你一个餐馆信息数组 restaurants，其中  restaurants[i] = [idi, ratingi, veganFriendlyi, pricei, distancei]。你必须使用以下三个过滤器来过滤这些餐馆信息。其中素食者友好过滤器 veganFriendly 的值可以为 true 或者 false，如果为 true 就意味着你应该只包括 veganFriendlyi 为 true 的餐馆，为 false 则意味着可以包括任何餐馆。此外，我们还有最大价格 maxPrice 和最大距离 maxDistance 两个过滤器，它们分别考虑餐厅的价格因素和距离因素的最大值。过滤后返回餐馆的 id，按照 rating 从高到低排序。如果 rating 相同，那么按 id 从高到低排序。简单起见， veganFriendlyi 和 veganFriendly 为 true 时取值为 1，为 false 时，取值为 0 。
func filterRestaurants(restaurants [][]int, veganFriendly int, maxPrice int, maxDistance int) []int {
	filter := func(i int) bool {
		if veganFriendly != 0 && restaurants[i][2] == 0 {
			return true
		}
		if restaurants[i][3] > maxPrice || restaurants[i][4] > maxDistance {
			return true
		}
		return false
	}
	sort.Slice(restaurants, func(i, j int) bool {
		if filter(i) {
			return false
		}
		if filter(j) {
			return true
		}
		if restaurants[i][1] == restaurants[j][1] {
			return restaurants[i][0] > restaurants[j][0]
		}
		return restaurants[i][1] > restaurants[j][1]
	})
	n := len(restaurants)
	ret := make([]int, 0, n)
	for i := 0; i < n; i++ {
		if filter(i) {
			break
		}
		ret = append(ret, restaurants[i][0])
	}
	return ret
}

// 1718. 构建字典序最大的可行序列 https://leetcode.cn/problems/construct-the-lexicographically-largest-valid-sequence/
// 给你一个整数 n ，请你找到满足下面条件的一个序列：整数 1 在序列中只出现一次。2 到 n 之间每个整数都恰好出现两次。对于每个 2 到 n 之间的整数 i ，两个 i 之间出现的距离恰好为 i 。序列里面两个数 a[i] 和 a[j] 之间的 距离 ，我们定义为它们下标绝对值之差 |j - i| 。请你返回满足上述条件中 字典序最大 的序列。题目保证在给定限制条件下，一定存在解。一个序列 a 被认为比序列 b （两者长度相同）字典序更大的条件是： a 和 b 中第一个不一样的数字处，a 序列的数字比 b 序列的数字大。比方说，[0,1,9,0] 比 [0,1,5,6] 字典序更大，因为第一个不同的位置是第三个数字，且 9 比 5 大。
func constructDistancedSequence(n int) []int {
	size := 2*n - 1
	ret := make([]int, size)
	used := make([]bool, n+1)
	var dfs func(int) bool
	dfs = func(i int) bool {
		if i == size {
			return true
		}
		if ret[i] != 0 {
			return dfs(i + 1)
		}
		for j := n; j > 0; j-- {
			if used[j] {
				continue
			}
			if j == 1 {
				used[j] = true
				ret[i] = 1
				if dfs(i + 1) {
					return true
				}
				ret[i] = 0
				used[j] = false
			}
			if i+j < size && ret[i+j] == 0 {
				used[j] = true
				ret[i] = j
				ret[i+j] = j
				if dfs(i + 1) {
					return true
				}
				used[j] = false
				ret[i] = 0
				ret[i+j] = 0
			}
		}
		return false
	}
	dfs(0)
	return ret
}
