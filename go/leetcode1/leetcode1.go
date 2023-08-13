package leetcode1

import (
	"math"
	"math/rand"
	"sort"
)

func upperBound(begin int, end int, f func(int) int) int {
	ret := end
	end--
	if f(end) <= 0 {
		return ret
	}
	for begin < end {
		mid := (end-begin)/2 + begin
		t := f(mid)
		if t > 0 {
			end = mid
		} else {
			begin = mid + 1
		}
	}
	return begin
}

// 240. 搜索二维矩阵 II https://leetcode.cn/problems/search-a-2d-matrix-ii/
// 编写一个高效的算法来搜索 m x n 矩阵 matrix 中的一个目标值 target 。该矩阵具有以下特性：每行的元素从左到右升序排列。每列的元素从上到下升序排列。
func searchMatrix(matrix [][]int, target int) bool {
	m, n := len(matrix), len(matrix[0])
	x, y := 0, n-1
	for x < m && y >= 0 {
		if matrix[x][y] < target {
			x++
		} else if matrix[x][y] == target {
			return true
		} else {
			y--
		}
	}
	return false
}

// 275. H 指数 II https://leetcode.cn/problems/h-index-ii/
// 给你一个整数数组 citations ，其中 citations[i] 表示研究者的第 i 篇论文被引用的次数，citations 已经按照 升序排列 。计算并返回该研究者的 h 指数。h 指数的定义：h 代表“高引用次数”（high citations），一名科研人员的 h 指数是指他（她）的 （n 篇论文中）总共有 h 篇论文分别被引用了至少 h 次。且其余的 n - h 篇论文每篇被引用次数 不超过 h 次。提示：如果 h 有多种可能的值，h 指数 是其中最大的那个。请你设计并实现对数时间复杂度的算法解决此问题。
func hIndex(citations []int) int {
	n := len(citations)
	check := func(mid int) int {
		if citations[mid] >= n-mid {
			return 1
		}
		return -1
	}
	ans := n - upperBound(0, n, check)
	return ans
}

// 540. 有序数组中的单一元素 https://leetcode.cn/problems/single-element-in-a-sorted-array/
// 给你一个仅由整数组成的有序数组，其中每个元素都会出现两次，唯有一个数只会出现一次。请你找出并返回只出现一次的那个数。你设计的解决方案必须满足 O(log n) 时间复杂度和 O(1) 空间复杂度。
func singleNonDuplicate(nums []int) int {
	check := func(mid int) int {
		if nums[mid] != nums[mid^1] {
			return 1
		}
		return -1
	}
	right := len(nums)
	if right == 1 {
		return nums[0]
	}
	return nums[upperBound(0, right-1, check)]
}

// 1838. 最高频元素的频数 https://leetcode.cn/problems/frequency-of-the-most-frequent-element/
// 元素的 频数 是该元素在一个数组中出现的次数。给你一个整数数组 nums 和一个整数 k 。在一步操作中，你可以选择 nums 的一个下标，并将该下标对应元素的值增加 1 。执行最多 k 次操作后，返回数组中最高频元素的 最大可能频数 。
func maxFrequency(nums []int, k int) int {
	sort.Ints(nums)
	ans := 1
	size := len(nums)
	for left, right, cnt := 0, 1, 0; right < size; right++ {
		cnt += (nums[right] - nums[right-1]) * (right - left)
		for cnt > k {
			cnt -= nums[right] - nums[left]
			left++
		}
		ans = max(ans, right-left+1)
	}
	return ans
}
func max(a int, b int) int {
	if a > b {
		return a
	}
	return b
}

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

// 222. 完全二叉树的节点个数 https://leetcode.cn/problems/count-complete-tree-nodes/
// 给你一棵 完全二叉树 的根节点 root ，求出该树的节点个数。完全二叉树 的定义如下：在完全二叉树中，除了最底层节点可能没填满外，其余每层节点数都达到最大值，并且最下面一层的节点都集中在该层最左边的若干位置。若最底层为第 h 层，则该层包含 1~ 2h 个节点。
func countNodes(root *TreeNode) int {
	depth := 0
	node := root
	for node != nil {
		node = node.Left
		depth++
	}
	begin, end := 1<<(depth-1), 1<<depth
	check := func(mid int) int {
		node = root
		k := 1 << (depth - 2)
		for k != 0 {
			if mid&k == 0 {
				node = node.Left
			} else {
				node = node.Right
			}
			k = k >> 1
		}
		if node != nil {
			return -1
		}
		return 1
	}
	return upperBound(begin, end, check) - 1
}

// 1712. 将数组分成三个子数组的方案数 https://leetcode.cn/problems/ways-to-split-array-into-three-subarrays/
// 我们称一个分割整数数组的方案是 好的 ，当它满足：数组被分成三个 非空 连续子数组，从左至右分别命名为 left ， mid ， right 。left 中元素和小于等于 mid 中元素和，mid 中元素和小于等于 right 中元素和。给你一个 非负 整数数组 nums ，请你返回 好的 分割 nums 方案数目。由于答案可能会很大，请你将结果对 109 + 7 取余后返回。
func waysToSplit(nums []int) int {
	size := len(nums)
	presum := make([]int, size+1)
	presum[0] = 0
	for i := 0; i < size; i++ {
		presum[i+1] = presum[i] + nums[i]
	}
	MOD := 1000000000 + 7
	left, right, ans := 1, 2, 0
	for i := 1; i < size; i++ {
		if left < i+1 {
			left = i + 1
		}
		for left < size && presum[left]-presum[i] < presum[i] {
			left++
		}
		if right < left {
			right = left
		}
		for right < size && presum[size]-presum[right] >= presum[right]-presum[i] {
			right++
		}
		ans = (ans + right - left) % MOD
	}
	return ans
}

// 436. 寻找右区间 https://leetcode.cn/problems/find-right-interval/
// 给你一个区间数组 intervals ，其中 intervals[i] = [starti, endi] ，且每个 starti 都 不同 。区间 i 的 右侧区间 可以记作区间 j ，并满足 startj >= endi ，且 startj 最小化 。返回一个由每个区间 i 的 右侧区间 在 intervals 中对应下标组成的数组。如果某个区间 i 不存在对应的 右侧区间 ，则下标 i 处的值设为 -1 。
func findRightInterval(intervals [][]int) []int {
	size := len(intervals)
	ans := make([]int, size)
	for i := 0; i < size; i++ {
		intervals[i] = append(intervals[i], i)
	}
	sort.Slice(intervals, func(i, j int) bool {
		return intervals[i][0] < intervals[j][0]
	})
	for i := 0; i < size; i++ {
		pos := sort.Search(size, func(j int) bool {
			return intervals[j][0] >= intervals[i][1]
		})
		if pos < size {
			ans[intervals[i][2]] = intervals[pos][2]
		} else {
			ans[intervals[i][2]] = -1
		}
	}
	return ans
}

// 826. 安排工作以达到最大收益 https://leetcode.cn/problems/most-profit-assigning-work/
// 你有 n 个工作和 m 个工人。给定三个数组： difficulty, profit 和 worker ，其中:difficulty[i] 表示第 i 个工作的难度，profit[i] 表示第 i 个工作的收益。worker[i] 是第 i 个工人的能力，即该工人只能完成难度小于等于 worker[i] 的工作。每个工人 最多 只能安排 一个 工作，但是一个工作可以 完成多次 。举个例子，如果 3 个工人都尝试完成一份报酬为 $1 的同样工作，那么总收益为 $3 。如果一个工人不能完成任何工作，他的收益为 $0 .返回 在把工人分配到工作岗位后，我们所能获得的最大利润 。
func maxProfitAssignment(difficulty []int, profit []int, worker []int) int {
	size := len(difficulty)
	type pair struct{ x, y int }
	jobs := make([]pair, size)
	for i := 0; i < size; i++ {
		jobs[i].x, jobs[i].y = difficulty[i], profit[i]
	}
	sort.Slice(jobs, func(i, j int) bool {
		return jobs[i].x < jobs[j].x
	})
	sort.Ints(worker)
	ans, i, best := 0, 0, 0
	for _, skill := range worker {
		for i < size && skill >= jobs[i].x {
			best = max(best, jobs[i].y)
			i++
		}
		ans += best
	}
	return ans
}

// 81. 搜索旋转排序数组 II https://leetcode.cn/problems/search-in-rotated-sorted-array-ii/
// 已知存在一个按非降序排列的整数数组 nums ，数组中的值不必互不相同。在传递给函数之前，nums 在预先未知的某个下标 k（0 <= k < nums.length）上进行了 旋转 ，使数组变为 [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]（下标 从 0 开始 计数）。例如， [0,1,2,4,4,4,5,6,6,7] 在下标 5 处经旋转后可能变为 [4,5,6,6,7,0,1,2,4,4] 。给你 旋转后 的数组 nums 和一个整数 target ，请你编写一个函数来判断给定的目标值是否存在于数组中。如果 nums 中存在这个目标值 target ，则返回 true ，否则返回 false 。你必须尽可能减少整个操作步骤。
func search(nums []int, target int) bool {
	n := len(nums)
	if n == 0 {
		return false
	}
	if n == 1 {
		return nums[0] == target
	}
	l, r := 0, n-1
	for l <= r {
		mid := (l + r) / 2
		if nums[mid] == target {
			return true
		}
		if nums[l] == nums[mid] && nums[mid] == nums[r] {
			l++
			r--
		} else if nums[l] <= nums[mid] {
			if nums[l] <= target && target < nums[mid] {
				r = mid - 1
			} else {
				l = mid + 1
			}
		} else {
			if nums[mid] < target && target <= nums[n-1] {
				l = mid + 1
			} else {
				r = mid - 1
			}
		}
	}
	return false
}

// 162. 寻找峰值 https://leetcode.cn/problems/find-peak-element/
// 峰值元素是指其值严格大于左右相邻值的元素。给你一个整数数组 nums，找到峰值元素并返回其索引。数组可能包含多个峰值，在这种情况下，返回 任何一个峰值 所在位置即可。你可以假设 nums[-1] = nums[n] = -∞ 。你必须实现时间复杂度为 O(log n) 的算法来解决此问题。
func findPeakElement(nums []int) int {
	size := len(nums)
	return sort.Search(size, func(i int) bool {
		return i == size-1 || nums[i] > nums[i+1]
	})
}

// 154. 寻找旋转排序数组中的最小值 II https://leetcode.cn/problems/find-minimum-in-rotated-sorted-array-ii/
// 已知一个长度为 n 的数组，预先按照升序排列，经由 1 到 n 次 旋转 后，得到输入数组。例如，原数组 nums = [0,1,4,4,5,6,7] 在变化后可能得到：若旋转 4 次，则可以得到 [4,5,6,7,0,1,4],若旋转 7 次，则可以得到 [0,1,4,4,5,6,7]。注意，数组 [a[0], a[1], a[2], ..., a[n-1]] 旋转一次 的结果为数组 [a[n-1], a[0], a[1], a[2], ..., a[n-2]] 。给你一个可能存在 重复 元素值的数组 nums ，它原来是一个升序排列的数组，并按上述情形进行了多次旋转。请你找出并返回数组中的 最小元素 。你必须尽可能减少整个过程的操作步骤。
func findMin(nums []int) int {
	size := len(nums)
	left, right := 0, size-1
	for left < right {
		mid := (right-left)/2 + left
		if nums[mid] < nums[right] {
			right = mid
		} else if nums[mid] > nums[right] {
			left = mid + 1
		} else {
			right--
		}
	}
	return nums[left]
}

// 528. 按权重随机选择 https://leetcode.cn/problems/random-pick-with-weight/
// 给你一个 下标从 0 开始 的正整数数组 w ，其中 w[i] 代表第 i 个下标的权重。请你实现一个函数 pickIndex ，它可以 随机地 从范围 [0, w.length - 1] 内（含 0 和 w.length - 1）选出并返回一个下标。选取下标 i 的 概率 为 w[i] / sum(w) 。例如，对于 w = [1, 3]，挑选下标 0 的概率为 1 / (1 + 3) = 0.25 （即，25%），而选取下标 1 的概率为 3 / (1 + 3) = 0.75（即，75%）。
type Solution struct {
	pre_sum []int
}

func Constructor(w []int) Solution {
	size := len(w)
	var s Solution
	s.pre_sum = make([]int, size)
	s.pre_sum[0] = w[0]
	for i := 1; i < size; i++ {
		s.pre_sum[i] = s.pre_sum[i-1] + w[i]
	}
	return s
}

func (this *Solution) PickIndex() int {
	size := len(this.pre_sum)
	m := this.pre_sum[size-1]
	x := rand.Intn(m)
	return sort.Search(size, func(i int) bool {
		return this.pre_sum[i] > x
	})
}

// 1508. 子数组和排序后的区间和 https://leetcode.cn/problems/range-sum-of-sorted-subarray-sums/
// 给你一个数组 nums ，它包含 n 个正整数。你需要计算所有非空连续子数组的和，并将它们按升序排序，得到一个新的包含 n * (n + 1) / 2 个数字的数组。请你返回在新数组中下标为 left 到 right （下标从 1 开始）的所有数字和（包括左右端点）。由于答案可能很大，请你将它对 10^9 + 7 取模后返回。
func rangeSum(nums []int, n int, left int, right int) int {
	sum, mod := 0, int64(1e9+7)
	for num := range nums {
		sum += num
	}
	check := func(target int) func(int) bool {
		return func(i int) bool {
			return getCount(nums, i) >= target
		}
	}
	l := sort.Search(sum, check(left))
	r := sort.Search(sum, check(right))
	if l == r {
		return int(int64(1+right-left) * int64(l) % mod)
	}
	ans := int64(right-getCount(nums, r-1)) * int64(r)
	ans -= int64(left-getCount(nums, l-1)) * int64(l)
	sum = 0
	for i, pre := 0, 0; i < n; i++ {
		sum += nums[i]
		for pre <= i && sum >= r {
			sum -= nums[pre]
			pre++
		}
		tsum, tpre := sum, pre
		for tpre <= i && tsum >= l {
			ans += int64(tsum)
			tsum -= nums[tpre]
			tpre++
		}
	}
	return int(ans % mod)
}

func getCount(nums []int, target int) int {
	size := len(nums)
	l, r, cnt, tem_sum := 0, 0, 0, 0
	for r < size {
		for r < size && tem_sum < target {
			tem_sum += nums[r]
			r++
		}
		if tem_sum >= target {
			cnt += size - r + 1
		}
		tem_sum -= nums[l]
		l++
	}
	return cnt
}

// 1574. 删除最短的子数组使剩余数组有序 https://leetcode.cn/problems/shortest-subarray-to-be-removed-to-make-array-sorted/
// 给你一个整数数组 arr ，请你删除一个子数组（可以为空），使得 arr 中剩下的元素是 非递减 的。一个子数组指的是原数组中连续的一个子序列。请你返回满足题目要求的最短子数组的长度。
func findLengthOfShortestSubarray(arr []int) int {
	size := len(arr)
	if size == 1 {
		return 0
	}
	right := size - 1
	for right > 0 && arr[right-1] <= arr[right] {
		right--
	}
	ans := right
	for left := 0; left <= right; left++ {
		if left != 0 && arr[left] < arr[left-1] {
			break
		}
		for right < size && arr[right] < arr[left] {
			right++
		}
		ans = min(max(right-left-1, 0), ans)
	}
	return ans
}

func min(a, b int) int {
	if a > b {
		return b
	}
	return a
}

// 1498. 满足条件的子序列数目 https://leetcode.cn/problems/number-of-subsequences-that-satisfy-the-given-sum-condition/
// 给你一个整数数组 nums 和一个整数 target 。请你统计并返回 nums 中能满足其最小元素与最大元素的 和 小于或等于 target 的 非空 子序列的数目。由于答案可能很大，请将结果对 109 + 7 取余后返回。
func numSubseq(nums []int, target int) int {
	sort.Ints(nums)
	size, mod := len(nums), int(1e9+7)
	f := make([]int, size)
	f[0] = 1
	for i := 1; i < size; i++ {
		f[i] = f[i-1] * 2 % mod
	}
	left, right, ans := 0, size-1, 0
	for left <= right {
		if nums[left]+nums[right] > target {
			right--
		} else {
			ans = (ans + f[right-left]) % mod
			left++
		}
	}
	return ans
}

// 1292. 元素和小于等于阈值的正方形的最大边长 https://leetcode.cn/problems/maximum-side-length-of-a-square-with-sum-less-than-or-equal-to-threshold/
// 给你一个大小为 m x n 的矩阵 mat 和一个整数阈值 threshold。请你返回元素总和小于或等于阈值的正方形区域的最大边长；如果没有这样的正方形区域，则返回 0 。
func maxSideLength(mat [][]int, threshold int) int {
	m, n := len(mat), len(mat[0])
	pre_sum := make([][]int, m+1)
	pre_sum[0] = make([]int, n+1)
	for i := 0; i <= n; i++ {
		pre_sum[0][i] = 0
	}
	for i := 1; i <= m; i++ {
		pre_sum[i] = make([]int, n+1)
		pre_sum[i][0] = 0
		for j := 1; j <= n; j++ {
			pre_sum[i][j] = pre_sum[i-1][j] + pre_sum[i][j-1] - pre_sum[i-1][j-1] + mat[i-1][j-1]
		}
	}
	ans := 0
	for i := 1; i <= m; i++ {
		for j := 1; j <= n; j++ {
			max_l := min(m-i, n-j) + 1
			for l := ans + 1; l <= max_l; l++ {
				if pre_sum[i+l-1][j+l-1]-pre_sum[i-1][j+l-1]-pre_sum[i+l-1][j-1]+pre_sum[i-1][j-1] <= threshold {
					ans++
				} else {
					break
				}
			}
		}
	}
	return ans
}

// 1300. 转变数组后最接近目标值的数组和 https://leetcode.cn/problems/sum-of-mutated-array-closest-to-target/
// 给你一个整数数组 arr 和一个目标值 target ，请你返回一个整数 value ，使得将数组中所有大于 value 的值变成 value 后，数组的和最接近  target （最接近表示两者之差的绝对值最小）。如果有多种使得和最接近 target 的方案，请你返回这些整数中的最小值。请注意，答案不一定是 arr 中的数字。
func findBestValue(arr []int, target int) int {
	sum, right, size := 0, 0, len(arr)
	for i := 0; i < size; i++ {
		if arr[i] > right {
			right = arr[i]
		}
		sum += arr[i]
	}
	if sum <= target {
		return right
	}
	left := target / size
	getSum := func(mid int) int {
		tem_sum := 0
		for i := 0; i < size; i++ {
			if arr[i] < mid {
				tem_sum += arr[i]
			} else {
				tem_sum += mid
			}
		}
		return tem_sum
	}
	pos := sort.Search(right+1-left, func(i int) bool {
		return getSum(i+left) >= target
	}) + left
	if getSum(pos)-target >= target-getSum(pos-1) {
		return pos - 1
	} else {
		return pos
	}
}

// 1802. 有界数组中指定下标处的最大值 https://leetcode.cn/problems/maximum-value-at-a-given-index-in-a-bounded-array/
// 给你三个正整数 n、index 和 maxSum 。你需要构造一个同时满足下述所有条件的数组 nums（下标 从 0 开始 计数）：nums.length == n。nums[i] 是 正整数 ，其中 0 <= i < n。abs(nums[i] - nums[i+1]) <= 1 ，其中 0 <= i < n-1。nums 中所有元素之和不超过 maxSum。nums[index] 的值被 最大化。返回你所构造的数组中的 nums[index] 。
func maxValue(n int, index int, maxSum int) int {
	maxSum -= n
	var min_side, max_side int
	if index <= n-index-1 {
		min_side, max_side = index, n-index-1
	} else {
		min_side, max_side = n-index-1, index
	}
	if min_side*min_side >= maxSum {
		return 1 + int(math.Sqrt(float64(maxSum)))
	}
	maxSum -= min_side * min_side
	if (min_side*2+n)*(max_side-min_side)/2 >= maxSum {
		return int(cal(1, 0, 0))
	}
	maxSum -= (1 + min_side*2 + n) * (max_side - min_side) / 2
	return 1 + max_side + maxSum/n
}

func cal(a float64, b float64, c float64) float64 {
	return (-b + math.Sqrt(b*b-4*a*c)) / (2 * a)
}

// 1901. 寻找峰值 II https://leetcode.cn/problems/find-a-peak-element-ii/
// 一个 2D 网格中的 峰值 是指那些 严格大于 其相邻格子(上、下、左、右)的元素。给你一个 从 0 开始编号 的 m x n 矩阵 mat ，其中任意两个相邻格子的值都 不相同 。找出 任意一个 峰值 mat[i][j] 并 返回其位置 [i,j] 。你可以假设整个矩阵周边环绕着一圈值为 -1 的格子。
func findPeakGrid(mat [][]int) []int {
	m := len(mat)
	pos := sort.Search(m, func(i int) bool {
		if i == m-1 {
			return true
		}
		idx := getMaxIdx(mat[i])
		return mat[i][idx] > mat[i+1][idx]
	})
	return []int{pos, getMaxIdx(mat[pos])}
}

func getMaxIdx(arr []int) int {
	size := len(arr)
	m, ans := arr[0], 0
	for i := 1; i < size; i++ {
		if arr[i] > m {
			m, ans = arr[i], i
		}
	}
	return ans
}

// 1648. 销售价值减少的颜色球 https://leetcode.cn/problems/sell-diminishing-valued-colored-balls/
// 你有一些球的库存 inventory ，里面包含着不同颜色的球。一个顾客想要 任意颜色 总数为 orders 的球。这位顾客有一种特殊的方式衡量球的价值：每个球的价值是目前剩下的 同色球 的数目。比方说还剩下 6 个黄球，那么顾客买第一个黄球的时候该黄球的价值为 6 。这笔交易以后，只剩下 5 个黄球了，所以下一个黄球的价值为 5 （也就是球的价值随着顾客购买同色球是递减的）给你整数数组 inventory ，其中 inventory[i] 表示第 i 种颜色球一开始的数目。同时给你整数 orders ，表示顾客总共想买的球数目。你可以按照 任意顺序 卖球。请你返回卖了 orders 个球以后 最大 总价值之和。由于答案可能会很大，请你返回答案对 109 + 7 取余数 的结果。
func maxProfit(inventory []int, orders int) int {
	mod := int64(1e9 + 7)
	size := len(inventory)
	right := -1
	for i := 0; i < size; i++ {
		right = max(inventory[i], right)
	}
	price := sort.Search(right, func(mid int) bool {
		sum := 0
		for i := 0; i < size; i++ {
			sum += max(inventory[i]-mid, 0)
		}
		return sum < orders
	})
	ans := int64(0)
	for i := 0; i < size; i++ {
		if inventory[i] > price {
			ans += int64((inventory[i] - price) * (price + inventory[i] + 1) / 2)
			orders -= (inventory[i] - price)
		}
	}
	ans += int64(price * orders)
	return int(ans % mod)
}

// 1562. 查找大小为 M 的最新分组 https://leetcode.cn/problems/find-latest-group-of-size-m/
// 给你一个数组 arr ，该数组表示一个从 1 到 n 的数字排列。有一个长度为 n 的二进制字符串，该字符串上的所有位最初都设置为 0 。在从 1 到 n 的每个步骤 i 中（假设二进制字符串和 arr 都是从 1 开始索引的情况下），二进制字符串上位于位置 arr[i] 的位将会设为 1 。给你一个整数 m ，请你找出二进制字符串上存在长度为 m 的一组 1 的最后步骤。一组 1 是一个连续的、由 1 组成的子串，且左右两边不再有可以延伸的 1 。返回存在长度 恰好 为 m 的 一组 1  的最后步骤。如果不存在这样的步骤，请返回 -1 。
func findLatestStep(arr []int, m int) int {
	type pair struct{ left, right int }
	size, cnt, ans := len(arr), 0, -1
	ranges := make([]pair, size+1)
	for i := 0; i <= size; i++ {
		ranges[i] = pair{-1, -1}
	}
	for i := 0; i < size; i++ {
		left, right := arr[i], arr[i]
		if arr[i] > 1 && ranges[arr[i]-1].left != -1 {
			left = ranges[arr[i]-1].left
			if ranges[arr[i]-1].right-ranges[arr[i]-1].left+1 == m {
				cnt--
			}
		}
		if arr[i] < size && ranges[arr[i]+1].right != -1 {
			right = ranges[arr[i]+1].right
			if ranges[arr[i]+1].right-ranges[arr[i]+1].left+1 == m {
				cnt--
			}
		}
		ranges[left], ranges[right] = pair{left, right}, pair{left, right}
		if right-left+1 == m {
			cnt++
		}
		if cnt > 0 {
			ans = i + 1
		}
	}
	return ans
}

// 1201. 丑数 III https://leetcode.cn/problems/ugly-number-iii/
// 给你四个整数：n 、a 、b 、c ，请你设计一个算法来找出第 n 个丑数。丑数是可以被 a 或 b 或 c 整除的 正整数 。
func nthUglyNumber(n int, a int, b int, c int) int {
	L1 := a / gcd(a, b) * b
	L2 := a / gcd(a, c) * c
	L3 := c / gcd(c, b) * b
	LL := L1 / gcd(L1, c) * c
	return sort.Search(int(2*1e9), func(i int) bool {
		return (i/a + i/b + i/c - i/L1 - i/L2 - i/L3 + i/LL) >= n
	})
}

func gcd(a, b int) int {
	for a != 0 {
		a, b = b%a, a
	}
	return b
}

// 911. 在线选举 https://leetcode.cn/problems/online-election/
// 给你两个整数数组 persons 和 times 。在选举中，第 i 张票是在时刻为 times[i] 时投给候选人 persons[i] 的。对于发生在时刻 t 的每个查询，需要找出在 t 时刻在选举中领先的候选人的编号。在 t 时刻投出的选票也将被计入我们的查询之中。在平局的情况下，最近获得投票的候选人将会获胜。实现 TopVotedCandidate 类：TopVotedCandidate(int[] persons, int[] times) 使用 persons 和 times 数组初始化对象。int q(int t) 根据前面描述的规则，返回在时刻 t 在选举中领先的候选人的编号。
type pair struct{ time, person int }
type TopVotedCandidate struct {
	result []pair
}

func TopVotedCandidateConstructor(persons []int, times []int) TopVotedCandidate {
	m, max_person, size := 0, 0, len(persons)
	cnt := make([]int, size)
	result := make([]pair, size+1)
	result[0] = pair{-1, 0}
	for i := 0; i < size; i++ {
		cnt[persons[i]]++
		if cnt[persons[i]] >= m {
			m = cnt[persons[i]]
			max_person = persons[i]
		}
		result[i+1] = pair{times[i], max_person}
	}
	return TopVotedCandidate{result}
}
func (this *TopVotedCandidate) Q(t int) int {
	return this.result[sort.Search(len(this.result), func(i int) bool {
		return this.result[i].time > t
	})-1].person
}

// 1146. 快照数组 https://leetcode.cn/problems/snapshot-array/
/*
 * 实现支持下列接口的「快照数组」- SnapshotArray：
 * SnapshotArray(int length) - 初始化一个与指定长度相等的 类数组 的数据结构。初始时，每个元素都等于 0。
 * void set(index, val) - 会将指定索引 index 处的元素设置为 val。
 * int snap() - 获取该数组的快照，并返回快照的编号 snap_id（快照号是调用 snap() 的总次数减去 1）。
 * int get(index, snap_id) - 根据指定的 snap_id 选择快照，并返回该快照指定索引 index 的值。
 */
type SnapshotArray struct {
	snap_id int
	record  [][]ChangeRecord
}

type ChangeRecord struct {
	snap_id int
	val     int
}

func SnapshotArrayConstructor(length int) SnapshotArray {
	return SnapshotArray{0, make([][]ChangeRecord, length)}
}

func (this *SnapshotArray) Set(index int, val int) {
	size := len(this.record[index])
	if size != 0 && this.record[index][size-1].snap_id == this.snap_id {
		this.record[index][size-1].val = val
	} else {
		this.record[index] = append(this.record[index], ChangeRecord{this.snap_id, val})
	}
}

func (this *SnapshotArray) Snap() int {
	this.snap_id++
	return this.snap_id - 1
}

func (this *SnapshotArray) Get(index int, snap_id int) int {
	size := len(this.record[index])
	pos := sort.Search(size, func(i int) bool {
		return this.record[index][i].snap_id > snap_id
	})
	if pos > 0 {
		return this.record[index][pos-1].val
	} else {
		return 0
	}
}

// 981. 基于时间的键值存储 https://leetcode.cn/problems/time-based-key-value-store/
/*
 * 设计一个基于时间的键值数据结构，该结构可以在不同时间戳存储对应同一个键的多个值，并针对特定时间戳检索键对应的值。实现 TimeMap
 * TimeMap() 初始化数据结构对象
 * void set(String key, String value, int timestamp) 存储键 key、值 value，以及给定的时间戳 timestamp。
 * String get(String key, int timestamp)。返回先前调用 set(key, value, timestamp_prev) 所存储的值，其中 timestamp_prev <= timestamp 。如果有多个这样的值，则返回对应最大的  timestamp_prev 的那个值。如果没有值，则返回空字符串（""）。
 */
type TimeMap struct {
	mp map[string][]Stamp
}
type Stamp struct {
	timestamp int
	value     string
}

func TimeMapConstructor() TimeMap {
	return TimeMap{make(map[string][]Stamp)}
}

func (this *TimeMap) Set(key string, value string, timestamp int) {
	this.mp[key] = append(this.mp[key], Stamp{timestamp, value})
}

func (this *TimeMap) Get(key string, timestamp int) string {
	pairs := this.mp[key]
	i := sort.Search(len(pairs), func(i int) bool { return pairs[i].timestamp > timestamp })
	if i > 0 {
		return pairs[i-1].value
	}
	return ""
}

// 662. 二叉树最大宽度 https://leetcode.cn/problems/maximum-width-of-binary-tree/
// 给你一棵二叉树的根节点 root ，返回树的 最大宽度 。树的 最大宽度 是所有层中最大的 宽度 。每一层的 宽度 被定义为该层最左和最右的非空节点（即，两个端点）之间的长度。将这个二叉树视作与满二叉树结构相同，两端点间会出现一些延伸到这一层的 null 节点，这些 null 节点也计入长度。题目数据保证答案将会在  32 位 带符号整数范围内。
func widthOfBinaryTree(root *TreeNode) int {
	type pair struct {
		node  *TreeNode
		index uint64
	}
	q := make([]pair, 3000)
	q[0] = pair{root, 1}
	ans, l, r := 0, 0, 1
	for l < r {
		ans = max(ans, int(q[r-1].index-q[l].index))
		right := r
		for l < right {
			node, idx := q[l].node, q[l].index
			if node.Left != nil {
				q[r] = pair{node.Left, idx * 2}
				r++
			}
			if node.Right != nil {
				q[r] = pair{node.Right, idx*2 + 1}
				r++
			}
			l++
		}
	}
	return ans
}

// 九坤-01. 可以读通讯稿的组数 https://leetcode.cn/contest/ubiquant2022/problems/xdxykd/
// 校运动会上，所有参赛同学身上都贴有他的参赛号码。某班参赛同学的号码记于数组 nums 中。假定反转后的号码称为原数字的「镜像号码」。如果 两位同学 满足条件：镜像号码 A + 原号码 B = 镜像号码 B + 原号码 A，则这两位同学可以到广播站兑换一次读通讯稿的机会，为同班同学加油助威。请返回所有参赛同学可以组成的可以读通讯稿的组数，并将结果对10^9+7取余。
func numberOfPairs(nums []int) int {
	mod := int(1e9 + 7)
	mp := make(map[int]int)
	for _, num := range nums {
		origin, reverse := num, 0
		for num != 0 {
			reverse = reverse*10 + num%10
			num /= 10
		}
		mp[reverse-origin]++
	}
	ans := 0
	for _, val := range mp {
		ans = (ans + val*(val-1)/2) % mod
	}
	return ans
}

// 九坤-03. 数字默契考验 https://leetcode.cn/contest/ubiquant2022/problems/uGuf0v/
// 某数学兴趣小组有 N 位同学，编号为 0 ~ N-1，老师提议举行一个数字默契小测试：首先每位同学想出一个数字，按同学编号存于数组 numbers。每位同学可以选择将自己的数字进行放大操作，每次在以下操作中任选一种（放大操作不限次数，可以不操作）：将自己的数字乘以 2,将自己的数字乘以 3,若最终所有同学可以通过操作得到相等数字，则返回所有同学的最少操作次数总数；否则请返回 -1。
func minOperations(numbers []int) int {
	lcf := 0
	size := len(numbers)
	type pair struct {
		x, y int
	}
	cnt := make([]pair, size)
	max_x, max_y := 0, 0
	for i := 0; i < size; i++ {
		x, y, num := 0, 0, numbers[i]
		for num%2 == 0 {
			num /= 2
			x++
		}
		for num%3 == 0 {
			num /= 3
			y++
		}
		if lcf == 0 {
			lcf = num
		} else if lcf != num {
			return -1
		}
		if x > max_x {
			max_x = x
		}
		if y > max_y {
			max_y = y
		}
		cnt[i] = pair{x, y}
	}
	ans := 0
	for i := 0; i < size; i++ {
		ans += (max_x - cnt[i].x + max_y - cnt[i].y)
	}
	return ans
}

// 6171. 和相等的子数组 https://leetcode.cn/problems/find-subarrays-with-equal-sum/
// 给你一个下标从 0 开始的整数数组 nums ，判断是否存在 两个 长度为 2 的子数组且它们的 和 相等。注意，这两个子数组起始位置的下标必须 不相同 。如果这样的子数组存在，请返回 true，否则返回 false 。子数组 是一个数组中一段连续非空的元素组成的序列。
func findSubarrays(nums []int) bool {
	size := len(nums)
	mp := make(map[int]struct{})
	for i := 1; i < size; i++ {
		tmp := nums[i] + nums[i-1]
		_, ok := mp[tmp]
		if ok {
			return true
		} else {
			mp[tmp] = struct{}{}
		}
	}
	return false
}

// 6172. 严格回文的数字 https://leetcode.cn/problems/strictly-palindromic-number/
// 如果一个整数 n 在 b 进制下（b 为 2 到 n - 2 之间的所有整数）对应的字符串 全部 都是 回文的 ，那么我们称这个数 n 是 严格回文 的。给你一个整数 n ，如果 n 是 严格回文 的，请返回 true ，否则返回 false 。如果一个字符串从前往后读和从后往前读完全相同，那么这个字符串是 回文的 。
func isStrictlyPalindromic(n int) bool {
	return false
}

// 6173. 被列覆盖的最多行数 https://leetcode.cn/problems/maximum-rows-covered-by-columns/
// 给你一个下标从 0 开始的 m x n 二进制矩阵 mat 和一个整数 cols ，表示你需要选出的列数。如果一行中，所有的 1 都被你选中的列所覆盖，那么我们称这一行 被覆盖 了。请你返回在选择 cols 列的情况下，被覆盖 的行数 最大 为多少。
func maximumRows(mat [][]int, cols int) int {
	m, n := len(mat), len(mat[0])
	calculate := func(state int) int {
		cnt := 0
		for i := 0; i < n; i++ {
			if state&(1<<i) != 0 {
				cnt++
			}
		}
		if cnt != cols {
			return 0
		}
		ret := 0
		for i := 0; i < m; i++ {
			var j int
			for j = 0; j < n; j++ {
				if mat[i][j] == 1 && (state&(1<<j)) != 0 {
					break
				}
			}
			if j == n {
				ret++
			}
		}
		return ret
	}
	ans := 0
	end := (1 << cols)
	for i := 1; i < end; i++ {
		tmp := calculate(i)
		ans = max(tmp, ans)
	}
	return ans
}

// 6143. 预算内的最多机器人数目 https://leetcode.cn/problems/maximum-number-of-robots-within-budget/
// 你有 n 个机器人，给你两个下标从 0 开始的整数数组 chargeTimes 和 runningCosts ，两者长度都为 n 。第 i 个机器人充电时间为 chargeTimes[i] 单位时间，花费 runningCosts[i] 单位时间运行。再给你一个整数 budget 。运行 k 个机器人 总开销 是 max(chargeTimes) + k * sum(runningCosts) ，其中 max(chargeTimes) 是这 k 个机器人中最大充电时间，sum(runningCosts) 是这 k 个机器人的运行时间之和。请你返回在 不超过 budget 的前提下，你 最多 可以 连续 运行的机器人数目为多少。
func maximumRobots(chargeTimes []int, runningCosts []int, budget int64) int {
	n := len(chargeTimes)
	dq := make([]int, n)
	sum := int64(0)
	begin, end, l, ans := 0, -1, 0, 0
	for r := 0; r < n; r++ {
		for begin >= end && chargeTimes[r] >= chargeTimes[dq[end]] {
			end--
		}
		end++
		dq[end] = r
		sum += int64(runningCosts[r])
		for begin >= end && int64(r-l+1)*sum+int64(dq[begin]) > budget {
			if dq[begin] == l {
				begin++
			}
			sum -= int64(runningCosts[l])
			l++
		}
		ans = max(ans, r-l+1)
	}
	return ans
}

// 6167. 检查相同字母间的距离 https://leetcode.cn/problems/check-distances-between-same-letters/
// 给你一个下标从 0 开始的字符串 s ，该字符串仅由小写英文字母组成，s 中的每个字母都 恰好 出现 两次 。另给你一个下标从 0 开始、长度为 26 的的整数数组 distance 。字母表中的每个字母按从 0 到 25 依次编号（即，'a' -> 0, 'b' -> 1, 'c' -> 2, ... , 'z' -> 25）。在一个 匀整 字符串中，第 i 个字母的两次出现之间的字母数量是 distance[i] 。如果第 i 个字母没有在 s 中出现，那么 distance[i] 可以 忽略 。如果 s 是一个 匀整 字符串，返回 true ；否则，返回 false 。
func checkDistances(s string, distance []int) bool {
	size := len(s)
	pos := make([]int, 32)
	for i := 0; i < 32; i++ {
		pos[i] = -1
	}
	for i := 0; i < size; i++ {
		tmp := int(s[i] - 'a')
		if pos[tmp] != -1 {
			if distance[tmp] != i-pos[tmp]-1 {
				return false
			}
		} else {
			pos[tmp] = i
		}
	}
	return true
}

// 6168. 恰好移动 k 步到达某一位置的方法数目 https://leetcode.cn/problems/number-of-ways-to-reach-a-position-after-exactly-k-steps/
// 给你两个 正 整数 startPos 和 endPos 。最初，你站在 无限 数轴上位置 startPos 处。在一步移动中，你可以向左或者向右移动一个位置。给你一个正整数 k ，返回从 startPos 出发、恰好 移动 k 步并到达 endPos 的 不同 方法数目。由于答案可能会很大，返回对 109 + 7 取余 的结果。如果所执行移动的顺序不完全相同，则认为两种方法不同。注意：数轴包含负整数。
func numberOfWays(startPos int, endPos int, k int) int {
	mod := int64(1e9 + 7)
	inv := make([]int64, k+1)
	inv[1] = 1
	for i := 2; i <= k; i++ {
		inv[i] = (mod - mod/int64(i)) * inv[mod%int64(i)] % mod
	}
	C := func(n, k int) int64 {
		ret := int64(1)
		for i := 1; i <= k; i++ {
			ret = ret * int64(n-i+1) % mod * inv[i] % mod
		}
		return ret
	}
	var diff int
	if startPos < endPos {
		diff = endPos - startPos
	} else {
		diff = startPos - endPos
	}
	if diff > k || (k-diff)%2 != 0 {
		return 0
	}
	return int(C(k, (k-diff)/2))
}

// 6169. 最长优雅子数组 https://leetcode.cn/problems/longest-nice-subarray/
// 给你一个由 正 整数组成的数组 nums 。如果 nums 的子数组中位于 不同 位置的每对元素按位 与（AND）运算的结果等于 0 ，则称该子数组为 优雅 子数组。返回 最长 的优雅子数组的长度。子数组 是数组中的一个 连续 部分。注意：长度为 1 的子数组始终视作优雅子数组。
func longestNiceSubarray(nums []int) int {
	pos := make([]int, 32)
	for i := 0; i < 32; i++ {
		pos[i] = -1
	}
	end_mp := make(map[int]int)
	prev_mp := make(map[int]int)
	size := len(nums)
	for i := 0; i < size; i++ {
		mask := 1
		for j := 0; j < 31; j++ {
			if nums[i]&(mask<<j) != 0 {
				if pos[j] != -1 {
					end_mp[pos[j]] = i
					prev_mp[i] = pos[j]
				}
				pos[j] = i
			}
		}
	}
	left, ans := 0, 0
	for left < size {
		end := end_mp[left]
		right := left + 1
		for right < end {
			tmp, ok := end_mp[right]
			if ok {
				end = min(tmp, end)
			}
			right++
		}
		ans = max(ans, right-left+1)
		left = prev_mp[right] + 1
	}
	return ans
}

// 6170. 会议室 III https://leetcode.cn/problems/meeting-rooms-iii/
// 给你一个整数 n ，共有编号从 0 到 n - 1 的 n 个会议室。给你一个二维整数数组 meetings ，其中 meetings[i] = [starti, endi] 表示一场会议将会在 半闭 时间区间 [starti, endi) 举办。所有 starti 的值 互不相同 。会议将会按以下方式分配给会议室：每场会议都会在未占用且编号 最小 的会议室举办。如果没有可用的会议室，会议将会延期，直到存在空闲的会议室。延期会议的持续时间和原会议持续时间 相同 。当会议室处于未占用状态时，将会优先提供给原 开始 时间更早的会议。返回举办最多次会议的房间 编号 。如果存在多个房间满足此条件，则返回编号 最小 的房间。半闭区间 [a, b) 是 a 和 b 之间的区间，包括 a 但 不包括 b 。
func mostBooked(n int, meetings [][]int) int {
	sort.Slice(meetings, func(i, j int) bool {
		return meetings[i][0] < meetings[j][0]
	})
	cnt := make([]int, n)
	t := make([]int, n)
	for i := 0; i < n; i++ {
		t[i] = 0
		cnt[i] = 0
	}
	size := len(meetings)
	for i := 0; i < size; i++ {
		best, find := 0, false
		for j := 0; !find && j < n; j++ {
			if t[j] <= meetings[i][0] {
				t[j] = meetings[i][1]
				cnt[j]++
				find = true
			}
			if t[j] < t[best] {
				best = j
			}
		}
		if !find {
			cnt[best]++
			t[best] += meetings[i][1] - meetings[i][0]
		}
	}
	ans := 0
	for i := 1; i < n; i++ {
		if cnt[i] > cnt[ans] {
			ans = i
		}
	}
	return ans
}
