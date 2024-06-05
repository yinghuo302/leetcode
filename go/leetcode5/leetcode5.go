package leetcode5

import (
	"math"
	"sort"
	"strconv"
)

func max(a, b int) int {
	if a < b {
		return b
	}
	return a
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// 467. 环绕字符串中唯一的子字符串 https://leetcode.cn/problems/unique-substrings-in-wraparound-string
// 定义字符串 base 为一个 "abcdefghijklmnopqrstuvwxyz" 无限环绕的字符串，所以 base 看起来是这样的："...zabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcd....".给你一个字符串 s ，请你统计并返回 s 中有多少 不同非空子串 也在 base 中出现。
func findSubstringInWraproundString(s string) int {
	dp, k, prev, ans := make([]int, 26), 0, byte('0'), 0
	for i := 0; i < len(s); i++ {
		if (s[i]-prev+26)%26 == 1 {
			k++
		} else {
			k = 1
		}
		prev = s[i]
		dp[s[i]-'a'] = max(dp[s[i]-'a'], k)
	}
	for _, l := range dp {
		ans += l
	}
	return ans
}

// 2580. 统计将重叠区间合并成组的方案数 https://leetcode.cn/problems/count-ways-to-group-overlapping-ranges
// 给你一个二维整数数组 ranges ，其中 ranges[i] = [starti, endi] 表示 starti 到 endi 之间（包括二者）的所有整数都包含在第 i 个区间中。你需要将 ranges 分成 两个 组（可以为空），满足：每个区间只属于一个组。两个有 交集 的区间必须在 同一个 组内。如果两个区间有至少 一个 公共整数，那么这两个区间是 有交集 的。比方说，区间 [1, 3] 和 [2, 5] 有交集，因为 2 和 3 在两个区间中都被包含。请你返回将 ranges 划分成两个组的 总方案数 。由于答案可能很大，将它对 109 + 7 取余 后返回。
func countWays(ranges [][]int) int {
	mod, n, ans := int64(1e9+7), len(ranges), int64(1)
	sort.Slice(ranges, func(i, j int) bool {
		return ranges[i][0] < ranges[j][0]
	})
	for i := 0; i < n; {
		right := ranges[i][1]
		for i < n && ranges[i][0] <= right {
			right = max(ranges[i][1], right)
			i++
		}
		ans = ans * 2 % mod
	}
	return int(ans)
}

// 1997. 访问完所有房间的第一天 https://leetcode.cn/problems/first-day-where-you-have-been-in-all-the-rooms
// 你需要访问 n 个房间，房间从 0 到 n - 1 编号。同时，每一天都有一个日期编号，从 0 开始，依天数递增。你每天都会访问一个房间。最开始的第 0 天，你访问 0 号房间。给你一个长度为 n 且 下标从 0 开始 的数组 nextVisit 。在接下来的几天中，你访问房间的 次序 将根据下面的 规则 决定：假设某一天，你访问 i 号房间。如果算上本次访问，访问 i 号房间的次数为 奇数 ，那么 第二天 需要访问 nextVisit[i] 所指定的房间，其中 0 <= nextVisit[i] <= i 。如果算上本次访问，访问 i 号房间的次数为 偶数 ，那么 第二天 需要访问 (i + 1) mod n 号房间。请返回你访问完所有房间的第一天的日期编号。题目数据保证总是存在这样的一天。由于答案可能很大，返回对 109 + 7 取余后的结果。
func firstDayBeenInAllRooms(nextVisit []int) int {
	size, mod := len(nextVisit), int(1e9+7)
	dp := make([]int, size)
	dp[0] = 2
	for i := 1; i < size; i++ {
		dp[i] = dp[i-1] + 2
		if nextVisit[i] != 0 {
			dp[i] = (dp[i] - dp[nextVisit[i]-1] + mod) % mod
		}
		dp[i] = (dp[i] + dp[i-1]) % mod
	}
	return dp[size-2]
}

// 2908. 元素和最小的山形三元组 I https://leetcode.cn/problems/minimum-sum-of-mountain-triplets-i
// 给你一个下标从 0 开始的整数数组 nums 。如果下标三元组 (i, j, k) 满足下述全部条件，则认为它是一个 山形三元组 ：i < j < k。nums[i] < nums[j] 且 nums[k] < nums[j]。请你找出 nums 中 元素和最小 的山形三元组，并返回其 元素和 。如果不存在满足条件的三元组，返回 -1 。
func minimumSum(nums []int) int {
	size, ans := len(nums), 0x3f3f3f3f
	left, right := make([]int, size), nums[size-1]
	left[0] = nums[0]
	for i := 1; i < size; i++ {
		left[i] = min(left[i-1], nums[i])
	}
	for i := size - 2; i >= 0; i-- {
		if nums[i] > left[i] && nums[i] > right {
			ans = min(ans, nums[i]+left[i]+right)
		}
		right = min(right, nums[i])
	}
	if ans == 0x3f3f3f3f {
		return -1
	}
	return ans
}

// 2952. 需要添加的硬币的最小数量 https://leetcode.cn/problems/minimum-number-of-coins-to-be-added
// 给你一个下标从 0 开始的整数数组 coins，表示可用的硬币的面值，以及一个整数 target 。如果存在某个 coins 的子序列总和为 x，那么整数 x 就是一个 可取得的金额 。返回需要添加到数组中的 任意面值 硬币的 最小数量 ，使范围 [1, target] 内的每个整数都属于 可取得的金额 。数组的 子序列 是通过删除原始数组的一些（可能不删除）元素而形成的新的 非空 数组，删除过程不会改变剩余元素的相对位置。
func minimumAddedCoins(coins []int, target int) int {
	sort.Ints(coins)
	ans, x, size, idx := 0, 1, len(coins), 0
	for x <= target {
		if idx <= size && coins[idx] <= x {
			x += coins[idx]
			idx++
		} else {
			x = x * 2
			ans++
		}
	}
	return ans
}

// 331. 验证二叉树的前序序列化 https://leetcode.cn/problems/verify-preorder-serialization-of-a-binary-tree/
// 序列化二叉树的一种方法是使用 前序遍历 。当我们遇到一个非空节点时，我们可以记录下这个节点的值。如果它是一个空节点，我们可以使用一个标记值记录，例如 #。例如，上面的二叉树可以被序列化为字符串 "9,3,4,#,#,1,#,#,2,#,6,#,#"，其中 # 代表一个空节点。给定一串以逗号分隔的序列，验证它是否是正确的二叉树的前序序列化。编写一个在不重构树的条件下的可行算法。保证 每个以逗号分隔的字符或为一个整数或为一个表示 null 指针的 '#' 。你可以认为输入格式总是有效的。例如它永远不会包含两个连续的逗号，比如 "1,,3" 。注意：不允许重建树。
func isValidSerialization(preorder string) bool {
	idx, cnt, size := 0, 1, len(preorder)
	for idx < size {
		if cnt == 0 {
			return false
		}
		if preorder[idx] == ',' {
			idx++
		} else if preorder[idx] == '#' {
			idx++
			cnt--
		} else {
			for idx < size && preorder[idx] != ',' {
				idx++
			}
			cnt++
		}
	}
	return cnt == 0
}

// 2810. 故障键盘 https://leetcode.cn/problems/faulty-keyboard
// 你的笔记本键盘存在故障，每当你在上面输入字符 'i' 时，它会反转你所写的字符串。而输入其他字符则可以正常工作。给你一个下标从 0 开始的字符串 s ，请你用故障键盘依次输入每个字符。返回最终笔记本屏幕上输出的字符串。
func finalString(s string) string {
	q, head, size := make([]byte, 0), false, len(s)
	for i := 0; i < size; i++ {
		if s[i] != 'i' {
			if head {
				q = append([]byte{s[i]}, q...)
			} else {
				q = append(q, s[i])
			}
		} else {
			head = !head
		}
	}
	if head {
		for i, j := 0, len(q)-1; i < j; i, j = i+1, j-1 {
			q[i], q[j] = q[j], q[i]
		}
	}
	return string(q)
}

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

// 894. 所有可能的真二叉树 https://leetcode.cn/problems/all-possible-full-binary-trees/
// 给你一个整数 n ，请你找出所有可能含 n 个节点的 真二叉树 ，并以列表形式返回。答案中每棵树的每个节点都必须符合 Node.val == 0 。答案的每个元素都是一棵真二叉树的根节点。你可以按 任意顺序 返回最终的真二叉树列表。真二叉树 是一类二叉树，树中每个节点恰好有 0 或 2 个子节点。
func allPossibleFBT(n int) []*TreeNode {
	if n == 1 {
		return []*TreeNode{{Val: 0}}
	}
	n, ans := n-1, make([]*TreeNode, 0)
	for i := 1; i < n; i++ {
		left, right := allPossibleFBT(i), allPossibleFBT(n-i)
		for _, l := range left {
			for _, r := range right {
				ans = append(ans, &TreeNode{Val: 0, Left: l, Right: r})
			}
		}
	}
	return ans
}

// 47. 全排列 II https://leetcode.cn/problems/permutations-ii/
// 给定一个可包含重复数字的序列 nums ，按任意顺序 返回所有不重复的全排列。
func permuteUnique(nums []int) [][]int {
	sort.Ints(nums)
	size := len(nums)
	ans, perm, vis := make([][]int, 0), make([]int, size), make([]bool, size)
	var backtrace func(int)
	backtrace = func(idx int) {
		if idx == size {
			ans = append(ans, append([]int{}, perm...))
			return
		}
		for i := 0; i < size; i++ {
			if vis[i] || (i > 0 && nums[i] == nums[i-1] && !vis[i-1]) {
				continue
			}
			perm[idx] = nums[i]
			vis[i] = true
			backtrace(idx + 1)
			vis[i] = false
		}
	}
	backtrace(0)
	return ans
}

// 40. 组合总和 II https://leetcode.cn/problems/combination-sum-ii
// 给定一个候选人编号的集合 candidates 和一个目标数 target ，找出 candidates 中所有可以使数字和为 target 的组合。candidates 中的每个数字在每个组合中只能使用 一次 。注意：解集不能包含重复的组合。
func combinationSum2(candidates []int, target int) [][]int {
	type Freq struct{ val, cnt int }
	sort.Ints(candidates)
	freq, prev, cnt := make([]Freq, 0), -1, 0
	for _, val := range candidates {
		if prev == -1 {
			prev = val
		}
		if prev != val {
			freq = append(freq, Freq{val: prev, cnt: cnt})
			prev, cnt = val, 0
		}
		cnt++
	}
	freq = append(freq, Freq{val: prev, cnt: cnt})
	ans, arr, rest := make([][]int, 0), make([]int, 0), target
	var backtrace func(int)
	backtrace = func(i int) {
		if rest == 0 {
			ans = append(ans, append([]int{}, arr...))
			return
		}
		if i == len(freq) || rest < 0 {
			return
		}
		mx := min(rest/freq[i].val, freq[i].cnt)
		for j := 0; j <= mx; j++ {
			backtrace(i + 1)
			arr = append(arr, freq[i].val)
			rest -= freq[i].val
		}
		arr = arr[:len(arr)-mx-1]
		rest += (mx + 1) * freq[i].val
	}
	backtrace(0)
	return ans
}

// 2841. 几乎唯一子数组的最大和 https://leetcode.cn/problems/maximum-sum-of-almost-unique-subarray
// 给你一个整数数组 nums 和两个正整数 m 和 k 。请你返回 nums 中长度为 k 的 几乎唯一 子数组的 最大和 ，如果不存在几乎唯一子数组，请你返回 0 。如果 nums 的一个子数组有至少 m 个互不相同的元素，我们称它是 几乎唯一 子数组。子数组指的是一个数组中一段连续 非空 的元素序列。
func maxSum(nums []int, m int, k int) int64 {
	cnt := make(map[int]int)
	sum, ans := int64(0), int64(0)
	for i := 0; i < k; i++ {
		cnt[nums[i]]++
		sum += int64(nums[i])
	}
	left, right, size := 0, k-1, len(nums)
	for {
		if len(cnt) >= m && sum >= ans {
			ans = sum
		}
		if right++; right >= size {
			break
		}
		sum += int64(nums[right] - nums[left])
		cnt[nums[left]]--
		cnt[nums[right]]++
		if cnt[nums[left]] == 0 {
			delete(cnt, nums[left])
		}
		left++
	}
	return ans
}

// 2192. 有向无环图中一个节点的所有祖先 https://leetcode.cn/problems/all-ancestors-of-a-node-in-a-directed-acyclic-graph
// 给你一个正整数 n ，它表示一个 有向无环图 中节点的数目，节点编号为 0 到 n - 1 （包括两者）。给你一个二维整数数组 edges ，其中 edges[i] = [fromi, toi] 表示图中一条从 fromi 到 toi 的单向边。请你返回一个数组 answer，其中 answer[i]是第 i 个节点的所有 祖先 ，这些祖先节点 升序 排序。如果 u 通过一系列边，能够到达 v ，那么我们称节点 u 是节点 v 的 祖先 节点。
func getAncestors(n int, edges [][]int) [][]int {
	g := make([][]int, n)
	for _, edge := range edges {
		g[edge[1]] = append(g[edge[1]], edge[0])
	}
	vis, ans := make([]bool, n), make([][]int, n)
	var dfs func(int)
	dfs = func(i int) {
		if vis[i] {
			return
		}
		vis[i] = true
		for _, next := range g[i] {
			dfs(next)
			ans[i] = append(ans[i], ans[next]...)
		}
		ans[i] = append(ans[i], g[i]...)
		sort.Ints(ans[i])
		ans[i] = removeDuplicates(ans[i])
	}
	for i := 0; i < n; i++ {
		dfs(i)
	}
	return ans
}

func removeDuplicates(arr []int) []int {
	if len(arr) == 0 {
		return arr
	}
	j := 0
	for i := 1; i < len(arr); i++ {
		if arr[j] != arr[i] {
			j++
			arr[j] = arr[i]
		}
	}
	return arr[:j+1]
}

// 1026. 节点与其祖先之间的最大差值 https://leetcode.cn/problems/maximum-difference-between-node-and-ancestor/
// 给定二叉树的根节点 root，找出存在于 不同 节点 A 和 B 之间的最大值 V，其中 V = |A.val - B.val|，且 A 是 B 的祖先。（如果 A 的任何子节点之一为 B，或者 A 的任何子节点是 B 的祖先，那么我们认为 A 是 B 的祖先）
func maxAncestorDiff(root *TreeNode) int {
	var dfs func(*TreeNode, int, int)
	ans := 0
	dfs = func(node *TreeNode, mx, mn int) {
		if node == nil {
			return
		}
		mn, mx = min(mn, node.Val), max(mx, node.Val)
		dfs(node.Left, mx, mn)
		dfs(node.Right, mx, mn)
		ans = max(ans, mx-mn)
	}
	dfs(root, 0, 1<<30)
	return ans
}

// 996. 正方形数组的数目 https://leetcode.cn/problems/number-of-squareful-arrays
// 给定一个非负整数数组 A，如果该数组每对相邻元素之和是一个完全平方数，则称这一数组为正方形数组。返回 A 的正方形排列的数目。两个排列 A1 和 A2 不同的充要条件是存在某个索引 i，使得 A1[i] != A2[i]。
func numSquarefulPerms(nums []int) int {
	n := len(nums)
	g, mp := make([][]int, n), make([]map[int]int, n)
	for i := 0; i < n; i++ {
		g[i] = make([]int, 0)
		mp[i] = make(map[int]int)
	}
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			r := int(math.Sqrt(float64(nums[i] + nums[j])))
			if r*r == nums[i]+nums[j] {
				g[i] = append(g[i], j)
				g[j] = append(g[j], i)
			}
		}
	}
	var dfs func(int, int) int
	dfs = func(node, vis int) int {
		if vis == (1<<n)-1 {
			return 1
		}
		if _, ok := mp[node][vis]; ok {
			return mp[node][vis]
		}
		ans := 0
		for _, val := range g[node] {
			if vis&(1<<val) == 0 {
				ans += dfs(val, vis|(1<<val))
			}
		}
		mp[node][vis] = ans
		return ans
	}
	ans, cnt, fact := 0, make(map[int]int), make([]int, n+1)
	fact[0] = 1
	for i := 0; i < n; i++ {
		ans += dfs(i, 1<<i)
		cnt[nums[i]]++
		fact[i+1] = (i + 1) * fact[i]
	}
	for _, v := range cnt {
		ans /= fact[v]
	}
	return ans
}

// 1483. 树节点的第 K 个祖先 https://leetcode.cn/problems/kth-ancestor-of-a-tree-node
// 给你一棵树，树上有 n 个节点，按从 0 到 n-1 编号。树以父节点数组的形式给出，其中 parent[i] 是节点 i 的父节点。树的根节点是编号为 0 的节点。树节点的第 k 个祖先节点是从该节点到根节点路径上的第 k 个节点。实现 TreeAncestor 类：TreeAncestor（int n， int[] parent） 对树和父数组中的节点数初始化对象。getKthAncestor(int node, int k) 返回节点 node 的第 k 个祖先节点。如果不存在这样的祖先节点，返回 -1 。
type TreeAncestor struct {
	Ancestor [][]int
}

const Depth int = 16

func TreeAncestorConstructor(n int, parent []int) TreeAncestor {
	parent = append(parent, n)
	parent[0] = n
	ancestor := make([][]int, n+1)
	for i := 0; i <= n; i++ {
		ancestor[i] = make([]int, Depth)
		for j := 0; j < Depth; j++ {
			ancestor[i][j] = n
		}
		ancestor[i][0] = parent[i]
	}
	for i := 1; i < Depth; i++ {
		for node := 0; node < n; node++ {
			ancestor[node][i] = ancestor[ancestor[node][i-1]][i-1]
		}
	}
	return TreeAncestor{Ancestor: ancestor}
}

func (this *TreeAncestor) GetKthAncestor(node int, k int) int {
	for i := 0; i < Depth; i++ {
		if ((k >> i) & 1) != 0 {
			node = this.Ancestor[node][i]
		}
	}
	if node+1 == len(this.Ancestor) {
		return -1
	}
	return node
}

// 1600. 王位继承顺序 https://leetcode.cn/problems/throne-inheritance
// 一个王国里住着国王、他的孩子们、他的孙子们等等。每一个时间点，这个家庭里有人出生也有人死亡。这个王国有一个明确规定的王位继承顺序，第一继承人总是国王自己。我们定义递归函数 Successor(x, curOrder) ，给定一个人 x 和当前的继承顺序，该函数返回 x 的下一继承人。Successor(x, curOrder):如果 x 没有孩子或者所有 x 的孩子都在 curOrder 中：如果 x 是国王，那么返回 null。否则，返回 Successor(x 的父亲, curOrder)。否则，返回 x 不在 curOrder 中最年长的孩子。比方说，假设王国由国王，他的孩子 Alice 和 Bob （Alice 比 Bob 年长）和 Alice 的孩子 Jack 组成。一开始， curOrder 为 ["king"].调用 Successor(king, curOrder) ，返回 Alice ，所以我们将 Alice 放入 curOrder 中，得到 ["king", "Alice"] 。调用 Successor(Alice, curOrder) ，返回 Jack ，所以我们将 Jack 放入 curOrder 中，得到 ["king", "Alice", "Jack"] 。调用 Successor(Jack, curOrder) ，返回 Bob ，所以我们将 Bob 放入 curOrder 中，得到 ["king", "Alice", "Jack", "Bob"] 。调用 Successor(Bob, curOrder) ，返回 null 。最终得到继承顺序为 ["king", "Alice", "Jack", "Bob"] 。通过以上的函数，我们总是能得到一个唯一的继承顺序。请你实现 ThroneInheritance 类：ThroneInheritance(string kingName) 初始化一个 ThroneInheritance 类的对象。国王的名字作为构造函数的参数传入。void birth(string parentName, string childName) 表示 parentName 新拥有了一个名为 childName 的孩子。void death(string name) 表示名为 name 的人死亡。一个人的死亡不会影响 Successor 函数，也不会影响当前的继承顺序。你可以只将这个人标记为死亡状态。string[] getInheritanceOrder() 返回 除去 死亡人员的当前继承顺序列表。

type ThroneInheritance struct {
	childs map[string][]string
	dead   map[string]struct{}
	king   string
}

func ThroneInheritanceConstructor(kingName string) ThroneInheritance {
	return ThroneInheritance{childs: make(map[string][]string), dead: make(map[string]struct{}), king: kingName}
}

func (this *ThroneInheritance) Birth(parentName string, childName string) {
	this.childs[parentName] = append(this.childs[parentName], childName)
}

func (this *ThroneInheritance) Death(name string) {
	this.dead[name] = struct{}{}
}

func (this *ThroneInheritance) GetInheritanceOrder() []string {
	ans := make([]string, 0)
	var dfs func(string)
	dfs = func(s string) {
		if _, ok := this.dead[s]; !ok {
			ans = append(ans, s)
		}
		for _, child := range this.childs[s] {
			dfs(child)
		}
	}
	dfs(this.king)
	return ans
}

// 2009. 使数组连续的最少操作数 https://leetcode.cn/problems/minimum-number-of-operations-to-make-array-continuous
// 给你一个整数数组 nums 。每一次操作中，你可以将 nums 中 任意 一个元素替换成 任意 整数。如果 nums 满足以下条件，那么它是 连续的 ：nums 中所有元素都是 互不相同 的。nums 中 最大 元素与 最小 元素的差等于 nums.length - 1 。比方说，nums = [4, 2, 5, 3] 是 连续的 ，但是 nums = [1, 2, 3, 5, 6] 不是连续的 。请你返回使 nums 连续 的 最少 操作次数。
func minOperations(nums []int) int {
	sort.Ints(nums)
	left, right, size, ans := 0, 0, len(nums), len(nums)
	cnt := make(map[int]int)
	for ; left < size; left++ {
		for right < size && nums[right] < nums[left]+size {
			cnt[nums[right]]++
			right++
		}
		ans = min(ans, size-len(cnt))
		if right == size {
			break
		}
		cnt[nums[left]]--
		if cnt[nums[left]] == 0 {
			delete(cnt, nums[left])
		}
	}
	return ans
}

// 2529. 正整数和负整数的最大计数 https://leetcode.cn/problems/maximum-count-of-positive-integer-and-negative-integer/
// 给你一个按 非递减顺序 排列的数组 nums ，返回正整数数目和负整数数目中的最大值。换句话讲，如果 nums 中正整数的数目是 pos ，而负整数的数目是 neg ，返回 pos 和 neg二者中的最大值。注意：0 既不是正整数也不是负整数。
func maximumCount(nums []int) int {
	first_zero := sort.Search(len(nums), func(i int) bool {
		return nums[i] >= 0
	})
	first_pos := sort.Search(len(nums), func(i int) bool {
		return nums[i] > 0
	})
	return max(first_zero, len(nums)-first_pos)
}

// 2781. 最长合法子字符串的长度 https://leetcode.cn/problems/length-of-the-longest-valid-substring/
// 给你一个字符串 word 和一个字符串数组 forbidden 。如果一个字符串不包含 forbidden 中的任何字符串，我们称这个字符串是 合法 的。请你返回字符串 word 的一个 最长合法子字符串 的长度。子字符串 指的是一个字符串中一段连续的字符，它可以为空。
func longestValidSubstring(word string, forbidden []string) int {
	mp := make(map[string]struct{})
	for _, forb := range forbidden {
		mp[forb] = struct{}{}
	}
	size, prev, ans := len(word), 0, 0
	for right := 0; right < size; right++ {
		for left, mn := right, max(right-9, prev); left >= mn; left-- {
			if _, ok := mp[word[left:right+1]]; ok {
				prev = left + 1
				break
			}
		}
		ans = max(ans, right-prev+1)
	}
	return ans
}

// 241. 为运算表达式设计优先级 https://leetcode.cn/problems/different-ways-to-add-parentheses/
// 给你一个由数字和运算符组成的字符串 expression ，按不同优先级组合数字和运算符，计算并返回所有可能组合的结果。你可以 按任意顺序 返回答案。生成的测试用例满足其对应输出值符合 32 位整数范围，不同结果的数量不超过 104 。
func diffWaysToCompute(expression string) []int {
	mp := make(map[int][]int)
	var dfs func(int) []int
	dfs = func(i int) []int {
		if res, ok := mp[i]; ok {
			return res
		}
		l, r := i>>8, i&0xff
		if val, err := strconv.Atoi(expression[l:r]); err == nil {
			mp[i] = []int{val}
			return mp[i]
		}
		res := make([]int, 0)
		for mid := l; mid < r; mid++ {
			if expression[mid] == '+' || expression[mid] == '-' || expression[mid] == '*' {
				left, right := dfs(l*256+mid), dfs((mid+1)*256+r)
				for _, l := range left {
					for _, r := range right {
						switch expression[mid] {
						case '+':
							res = append(res, l+r)
						case '-':
							res = append(res, l-r)
						case '*':
							res = append(res, l*r)
						}
					}
				}
			}
		}
		mp[i] = res
		return res
	}
	return dfs(len(expression))
}

// 2982. 找出出现至少三次的最长特殊子字符串 II https://leetcode.cn/problems/find-longest-special-substring-that-occurs-thrice-ii/
// 给你一个仅由小写英文字母组成的字符串 s 。如果一个字符串仅由单一字符组成，那么它被称为 特殊 字符串。例如，字符串 "abc" 不是特殊字符串，而字符串 "ddd"、"zz" 和 "f" 是特殊字符串。返回在 s 中出现 至少三次 的 最长特殊子字符串 的长度，如果不存在出现至少三次的特殊子字符串，则返回 -1 。子字符串 是字符串中的一个连续 非空 字符序列。
func maximumLength(s string) int {
	prev, cnt, maxCnt := '-', 0, 0
	arr := make([][]int, 26)
	for _, ch := range s {
		if prev == ch {
			cnt++
		} else {
			if prev != '-' {
				maxCnt = max(cnt, maxCnt)
				arr[prev-'a'] = append(arr[prev-'a'], cnt)
			}
			cnt = 1
			prev = ch
		}
	}
	maxCnt = max(cnt, maxCnt)
	arr[prev-'a'] = append(arr[prev-'a'], cnt)
	ret := sort.Search(maxCnt+1, func(target int) bool {
		if target == 0 {
			return false
		}
		for i := 0; i < 26; i++ {
			sum := 0
			for _, cnt := range arr[i] {
				if cnt >= target {
					sum += cnt - target + 1
				}
			}
			if sum >= 3 {
				return false
			}
		}
		return true
	}) - 1
	if ret == 0 {
		return -1
	}
	return ret
}

type BIT struct {
	arr []int
}

func lowbit(x int) int {
	return x & (-x)
}

func BITConstructor(n int) BIT {
	return BIT{
		arr: make([]int, n+1),
	}
}

func (this *BIT) Update(pos, inc int) {
	for pos < len(this.arr) {
		this.arr[pos] += inc
		pos += lowbit(pos)
	}
}

func (this *BIT) GetSum(pos int) int {
	sum := 0
	for pos > 0 {
		sum += this.arr[pos]
		pos -= lowbit(pos)
	}
	return sum
}

// 3072. 将元素分配到两个数组中 II https://leetcode.cn/problems/distribute-elements-into-two-arrays-ii
// 给你一个下标从 1 开始、长度为 n 的整数数组 nums 。现定义函数 greaterCount ，使得 greaterCount(arr, val) 返回数组 arr 中 严格大于 val 的元素数量。你需要使用 n 次操作，将 nums 的所有元素分配到两个数组 arr1 和 arr2 中。在第一次操作中，将 nums[1] 追加到 arr1 。在第二次操作中，将 nums[2] 追加到 arr2 。之后，在第 i 次操作中：如果 greaterCount(arr1, nums[i]) > greaterCount(arr2, nums[i]) ，将 nums[i] 追加到 arr1 。如果 greaterCount(arr1, nums[i]) < greaterCount(arr2, nums[i]) ，将 nums[i] 追加到 arr2 。如果 greaterCount(arr1, nums[i]) == greaterCount(arr2, nums[i]) ，将 nums[i] 追加到元素数量较少的数组中。如果仍然相等，那么将 nums[i] 追加到 arr1 。连接数组 arr1 和 arr2 形成数组 result 。例如，如果 arr1 == [1,2,3] 且 arr2 == [4,5,6] ，那么 result = [1,2,3,4,5,6] 。返回整数数组 result 。
func resultArray(nums []int) []int {
	n := len(nums)
	numsCopy := append([]int{}, nums...)
	sort.Ints(numsCopy)
	mp := make(map[int]int)
	prev, cnt := -1, n
	for _, num := range numsCopy {
		if prev != num {
			prev = num
			mp[num] = cnt
			cnt--
		}
	}
	lBIT, rBIT := BITConstructor(n+4), BITConstructor(n+4)
	larr, rarr := make([]int, 0), make([]int, 0)
	for idx, num := range nums {
		l, r := lBIT.GetSum(mp[num]-1), rBIT.GetSum(mp[num]-1)
		left := true
		if idx == 1 || l < r || (l == r && len(larr) > len(rarr)) {
			left = false
		}
		if left {
			lBIT.Update(mp[num], 1)
			larr = append(larr, num)
		} else {
			rBIT.Update(mp[num], 1)
			rarr = append(rarr, num)
		}

	}
	return append(larr, rarr...)
}
