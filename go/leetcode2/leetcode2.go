package leetcode2

import (
	"math/bits"
	"math/rand"
	"sort"
	"strings"
	"time"
)

// 九坤-02. 池塘计数 https://leetcode.cn/contest/ubiquant2022/problems/3PHTGp/
// 最近的降雨，使田地中的一些地方出现了积水，field[i][j] 表示田地第 i 行 j 列的位置有：若为 W, 表示该位置为积水；若为 ., 表示该位置为旱地。已知一些相邻的积水形成了若干个池塘，若以 W 为中心的八个方向相邻积水视为同一片池塘。请返回田地中池塘的数量。
func lakeCount(field []string) int {
	m, n := len(field), len(field[0])
	grid := make([][]byte, m)
	for i := 0; i < m; i++ {
		grid[i] = []byte(field[i])
	}
	dirs := [][]int{{0, 1}, {0, -1}, {1, 0}, {-1, 0}, {1, 1}, {1, -1}, {-1, 1}, {-1, -1}}
	var dfs func(int, int)
	dfs = func(x, y int) {
		if x <= 0 || x >= m || y <= 0 || y >= n || grid[x][y] == '.' {
			return
		}
		grid[x][y] = '.'
		for i := 0; i < 8; i++ {
			dfs(x+dirs[i][0], y+dirs[i][1])
		}
	}
	ans := 0
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			if grid[i][j] == 'W' {
				dfs(i, j)
				ans++
			}
		}
	}
	return ans
}

// 6176. 出现最频繁的偶数元素 https://leetcode.cn/problems/most-frequent-even-element/
// 给你一个整数数组 nums ，返回出现最频繁的偶数元素。如果存在多个满足条件的元素，只需要返回 最小 的一个。如果不存在这样的元素，返回 -1
func mostFrequentEven(nums []int) int {
	cnt := make([]int, 50001)
	for i := 0; i <= 50000; i++ {
		cnt[i] = 0
	}
	for num := range nums {
		if num%2 == 0 {
			cnt[num/2]++
		}
	}
	ans := 0
	for i := 1; i <= 50000; i++ {
		if cnt[i] > cnt[ans] {
			ans = i
		}
	}
	if cnt[ans] == 0 {
		return -1
	}
	return ans * 2
}

// 6177. 子字符串的最优划分 https://leetcode.cn/problems/optimal-partition-of-string/
// 给你一个字符串 s ，请你将该字符串划分成一个或多个 子字符串 ，并满足每个子字符串中的字符都是 唯一 的。也就是说，在单个子字符串中，字母的出现次数都不超过 一次 。满足题目要求的情况下，返回 最少 需要划分多少个子字符串。注意，划分后，原字符串中的每个字符都应该恰好属于一个子字符串。
func partitionString(s string) int {
	state, ans := 0, 0
	for _, ch := range s {
		tem := int(ch - 'a')
		if state&(1<<tem) != 0 {
			ans++
			state = 0
		}
		state |= (1 << tem)
	}
	ans++
	return ans
}

// 6178. 将区间分为最少组数 https://leetcode.cn/problems/divide-intervals-into-minimum-number-of-groups/
// 给你一个二维整数数组 intervals ，其中 intervals[i] = [lefti, righti] 表示 闭 区间 [lefti, righti] 。你需要将 intervals 划分为一个或者多个区间 组 ，每个区间 只 属于一个组，且同一个组中任意两个区间 不相交 。请你返回 最少 需要划分成多少个组。如果两个区间覆盖的范围有重叠（即至少有一个公共数字），那么我们称这两个区间是 相交 的。比方说区间 [1, 5] 和 [5, 8] 相交。
func minGroups(intervals [][]int) int {
	size := len(intervals)
	m := int(1e6 + 5)
	diff := make([]int, m)
	for i := 0; i < m; i++ {
		diff[i] = 0
	}
	for i := 0; i < size; i++ {
		diff[intervals[i][0]]++
		diff[intervals[i][1]+1]--
	}
	ans, sum := 0, 0
	for i := 0; i < m; i++ {
		sum += diff[i]
		if sum > ans {
			ans = sum
		}
	}
	return ans
}

func max(a, b int) int {
	if a < b {
		return b
	}
	return a
}

type SegmentTree struct {
	nums []int
	size int
	sum  func(int, int) int
}

func SegmentTreeConstructor(size int, sum func(int, int) int) *SegmentTree {
	t := SegmentTree{nums: make([]int, 4*size), size: size, sum: sum}
	for i := 0; i < size*4; i++ {
		t.nums[i] = 0
	}
	return &t
}

func (tree *SegmentTree) update(cur int, left int, right int, pos int, val int) {
	if left == right && left == pos {
		tree.nums[cur] = tree.sum(tree.nums[cur], val)
		return
	}
	mid := left + (right-left)/2
	if pos <= mid {
		tree.update(cur*2, left, mid, pos, val)
	} else {
		tree.update(cur*2+1, mid+1, right, pos, val)
	}
	tree.nums[cur] = tree.sum(tree.nums[cur*2], tree.nums[cur*2+1])
}

func (tree *SegmentTree) Update(pos int, val int) {
	tree.update(1, 0, tree.size-1, pos, val)
}

func (tree *SegmentTree) getSum(cur int, s int, t int, left int, right int) int {
	if left <= s && t <= right {
		return tree.nums[cur]
	}
	mid := s + (t-s)/2
	sum := 0
	if left <= mid {
		sum = tree.sum(tree.getSum(cur*2, s, mid, left, right), sum)
	}
	if mid < right {
		sum = tree.sum(sum, tree.getSum(cur*2+1, mid+1, t, left, right))
	}
	return sum
}

func (tree *SegmentTree) GetSum(left int, right int) int {
	return tree.getSum(1, 0, tree.size-1, left, right)
}

// 6206. 最长递增子序列 II https://leetcode.cn/problems/longest-increasing-subsequence-ii/
// 给你一个整数数组 nums 和一个整数 k 。找到 nums 中满足以下要求的最长子序列：子序列 严格递增子序列中相邻元素的差值 不超过 k 。请你返回满足上述要求的 最长子序列 的长度。子序列 是从一个数组中删除部分元素后，剩余元素不改变顺序得到的数组。
func lengthOfLIS(nums []int, k int) int {
	size := len(nums)
	st := SegmentTreeConstructor(100001, max)
	dp := make([]int, size)
	ans := 1
	for i := 0; i < size; i++ {
		dp[i] = st.GetSum(max(0, nums[i]-k), nums[i]-1) + 1
		st.Update(nums[i], dp[i])
		if dp[i] > ans {
			ans = dp[i]
		}
	}
	return ans
}

// 2404. 出现最频繁的偶数元素 https://leetcode.cn/problems/most-frequent-even-element/
// 给你一个整数数组 nums ，返回出现最频繁的偶数元素。如果存在多个满足条件的元素，只需要返回 最小 的一个。如果不存在这样的元素，返回 -1 。
func countDaysTogether(arriveAlice string, leaveAlice string, arriveBob string, leaveBob string) int {
	arrive_a, _ := time.Parse("2006-01-02", "2022-"+arriveAlice)
	arrive_b, _ := time.Parse("2006-01-02", "2022-"+arriveBob)
	leave_a, _ := time.Parse("2006-01-02", "2022-"+leaveAlice)
	leave_b, _ := time.Parse("2006-01-02", "2022-"+leaveBob)
	var max_arrive, min_leave time.Time
	if arrive_a.After(arrive_b) {
		max_arrive = arrive_a
	} else {
		max_arrive = arrive_b
	}
	if leave_a.After(leave_b) {
		min_leave = leave_b
	} else {
		min_leave = leave_a
	}
	if max_arrive.After(min_leave) {
		return int(min_leave.Sub(max_arrive).Hours() / 24)
	} else {
		return 0
	}
}

// 2405. 子字符串的最优划分 https://leetcode.cn/problems/optimal-partition-of-string/
// 给你一个字符串 s ，请你将该字符串划分成一个或多个 子字符串 ，并满足每个子字符串中的字符都是 唯一 的。也就是说，在单个子字符串中，字母的出现次数都不超过 一次 。满足题目要求的情况下，返回 最少 需要划分多少个子字符串。注意，划分后，原字符串中的每个字符都应该恰好属于一个子字符串。
func matchPlayersAndTrainers(players []int, trainers []int) int {
	sort.Ints(players)
	sort.Ints(trainers)
	m, n, i, j := len(players), len(trainers), 0, 0
	ans := 0
	for i < m && j < n {
		for j < n && players[i] > trainers[j] {
			j++
		}
		if j < n {
			ans++
			j++
			i++
		}
	}
	return ans
}

// 2406. 将区间分为最少组数 https://leetcode.cn/problems/divide-intervals-into-minimum-number-of-groups/
// 给你一个二维整数数组 intervals ，其中 intervals[i] = [lefti, righti] 表示 闭 区间 [lefti, righti] 。你需要将 intervals 划分为一个或者多个区间 组 ，每个区间 只 属于一个组，且同一个组中任意两个区间 不相交 。请你返回 最少 需要划分成多少个组。如果两个区间覆盖的范围有重叠（即至少有一个公共数字），那么我们称这两个区间是 相交 的。比方说区间 [1, 5] 和 [5, 8] 相交。
func smallestSubarrays(nums []int) []int {
	size := len(nums)
	st := SegmentTreeConstructor(size+1, func(i1, i2 int) int {
		return i1 | i2
	})
	for i := 0; i < size; i++ {
		st.Update(i+1, nums[i])
	}
	ans := make([]int, size)
	right := 1
	for left := 1; left <= size; left++ {
		m := st.GetSum(left, size)
		right = max(left, right)
		for st.GetSum(left, right) < m {
			right++
		}
		ans[left-1] = (right - left + 1)
	}
	return ans
}

// 2407. 最长递增子序列 II https://leetcode.cn/problems/longest-increasing-subsequence-ii/
// 给你一个整数数组 nums 和一个整数 k 。找到 nums 中满足以下要求的最长子序列：子序列 严格递增，子序列中相邻元素的差值 不超过 k 。请你返回满足上述要求的 最长子序列 的长度。子序列 是从一个数组中删除部分元素后，剩余元素不改变顺序得到的数组。
func minimumMoney(transactions [][]int) int64 {
	sum, size := int64(0), len(transactions)
	for i := 0; i < size; i++ {
		tem := int64(transactions[i][1] - transactions[i][0])
		if tem < 0 {
			sum += tem
		}
	}
	ans := int64(0)
	for i := 0; i < size; i++ {
		tem := int64(transactions[i][1] - transactions[i][0])
		if tem < 0 {
			ans = max64(ans, int64(transactions[i][0])-(sum-tem))
		} else {
			ans = max64(ans, int64(transactions[i][0])-sum)
		}
	}
	return ans
}

func max64(a, b int64) int64 {
	if a < b {
		return b
	}
	return a
}

// 1640. 能否连接形成数组 https://leetcode.cn/problems/check-array-formation-through-concatenation/
// 给你一个整数数组 arr ，数组中的每个整数 互不相同 。另有一个由整数数组构成的数组 pieces，其中的整数也 互不相同 。请你以 任意顺序 连接 pieces 中的数组以形成 arr 。但是，不允许 对每个数组 pieces[i] 中的整数重新排序。如果可以连接 pieces 中的数组形成 arr ，返回 true ；否则，返回 false 。
func canFormArray(arr []int, pieces [][]int) bool {
	mp := make(map[int]int)
	size1 := len(pieces)
	for i := 0; i < size1; i++ {
		mp[pieces[i][0]] = i
	}
	size := len(arr)
	for i := 0; i < size; {
		pos, exist := mp[arr[i]]
		if exist {
			for _, num := range pieces[pos] {
				if num == arr[i] {
					i++
				} else {
					return false
				}
			}
		} else {
			return false
		}
	}
	return true
}

type ListNode struct {
	Val  int
	Next *ListNode
}

// 707. 设计链表 数据结构设计 https://leetcode.cn/problems/design-linked-list/
/*
 * 设计链表的实现。您可以选择使用单链表或双链表。单链表中的节点应该具有两个属性：val 和 next。val 是当前节点的值，next 是指向下一个节点的指针/引用。如果要使用双向链表，则还需要一个属性 prev 以指示链表中的上一个节点。假设链表中的所有节点都是 0-index 的。在链表类中实现这些功
 * get(index)：获取链表中第 index 个节点的值。如果索引无效，则返回-1。
 * addAtHead(val)：在链表的第一个元素之前添加一个值为 val 的节点。插入后，新节点将成为链表的第一个节点。
 * addAtTail(val)：将值为 val 的节点追加到链表的最后一个元素。
 * addAtIndex(index,val)：在链表中的第 index 个节点之前添加值为 val  的节点。如果 index 等于链表的长度，则该节点将附加到链表的末尾。如果 index 大于链表长度，则不会插入节点。如果index小于0，则在头部插入节点。
 * deleteAtIndex(index)：如果索引 index 有效，则删除链表中的第 index 个节点。
 */
type MyLinkedList struct {
	Head *ListNode
	Tail *ListNode
}

func Constructor() MyLinkedList {
	return MyLinkedList{Head: nil, Tail: nil}
}

func (this *MyLinkedList) Get(index int) int {
	p := this.Head
	for index != 0 {
		if p == nil {
			return -1
		}
		p = p.Next
		index--
	}
	if p == nil {
		return -1
	}
	return p.Val
}

func (this *MyLinkedList) AddAtHead(val int) {
	node := &ListNode{Val: val, Next: this.Head}
	this.Head = node
	if this.Tail == nil {
		this.Tail = node
	}
}

func (this *MyLinkedList) AddAtTail(val int) {
	node := &ListNode{Val: val, Next: nil}
	if this.Tail == nil {
		this.Head, this.Tail = node, node
	} else {
		this.Tail.Next = node
		this.Tail = node
	}
}

func (this *MyLinkedList) AddAtIndex(index int, val int) {
	if index <= 0 {
		this.AddAtHead(val)
		return
	}
	prev := this.Head
	index--
	for index > 0 {
		if prev == nil {
			return
		}
		prev = prev.Next
		index--
	}
	if prev != nil {
		prev.Next = &ListNode{Val: val, Next: prev.Next}
		if this.Tail == prev {
			this.Tail = prev.Next
		}
	}
}

func (this *MyLinkedList) DeleteAtIndex(index int) {
	if index == 0 {
		tem := this.Head.Next
		if tem == nil {
			this.Head, this.Tail = nil, nil
		} else {
			this.Head = tem
		}
		return
	}
	prev := this.Head
	index--
	for index != 0 {
		if prev == nil {
			return
		}
		prev = prev.Next
		index--
	}
	if prev != nil && prev.Next != nil {
		tem := prev.Next
		prev.Next = tem.Next
		if this.Tail == tem {
			this.Tail = prev
		}
	}
}

// 135. 分发糖果 贪心 https://leetcode.cn/problems/candy/
// n 个孩子站成一排。给你一个整数数组 ratings 表示每个孩子的评分。你需要按照以下要求，给这些孩子分发糖果：每个孩子至少分配到 1 个糖果。相邻两个孩子评分更高的孩子会获得更多的糖果。请你给每个孩子分发糖果，计算并返回需要准备的 最少糖果数目 。
func candy(ratings []int) int {
	size := len(ratings)
	left, right := make([]int, size), make([]int, size)
	left[0] = 1
	for i := 1; i < size; i++ {
		if ratings[i] > ratings[i-1] {
			left[i] = left[i-1] + 1
		} else {
			left[i] = 1
		}
	}
	right[size-1] = 1
	for i := size - 2; i >= 0; i-- {
		if ratings[i] > ratings[i+1] {
			right[i] = right[i+1] + 1
		} else {
			right[i] = 1
		}
	}
	ans := 0
	for i := 0; i < size; i++ {
		ans += max(left[i], right[i])
	}
	return ans
}

// 382. 链表随机节点 随机选择 https://leetcode.cn/problems/linked-list-random-node/
// 给你一个单链表，随机选择链表的一个节点，并返回相应的节点值。每个节点 被选中的概率一样 。实现 Solution 类：Solution(ListNode head) 使用整数数组初始化对象。int getRandom() 从链表中随机选择一个节点并返回该节点的值。链表中所有节点被选中的概率相等。
type ListRandom struct {
	arr []int
}

func listRandomConstructor(head *ListNode) ListRandom {
	arr := make([]int, 0, 10000)
	for head != nil {
		arr = append(arr, head.Val)
		head = head.Next
	}
	return ListRandom{arr}
}

func (this *ListRandom) GetRandom() int {
	return this.arr[rand.Intn(len(this.arr))]
}

// 1545. 找出第 N 个二进制字符串中的第 K 位 递归分治 https://leetcode.cn/problems/find-kth-bit-in-nth-binary-string/
// 给你两个正整数 n 和 k，二进制字符串  Sn 的形成规则如下：S1 = "0"。当 i > 1 时，Si = Si-1 + "1" + reverse(invert(Si-1))。其中 + 表示串联操作，reverse(x) 返回反转 x 后得到的字符串，而 invert(x) 则会翻转 x 中的每一位（0 变为 1，而 1 变为 0）。例如，符合上述描述的序列的前 4 个字符串依次是：S1 = "0"，S2 = "011"，S3 = "0111001"，S4 = "011100110110001"。请你返回  Sn 的 第 k 位字符 ，题目数据保证 k 一定在 Sn 长度范围以内。
func findKthBit(n int, k int) byte {
	if assist(n, k) {
		return '1'
	}
	return '0'
}

func assist(n, k int) bool {
	if n == 1 || k == 1 {
		return false
	}
	t := (1 << (n - 1)) - 1
	if k <= t {
		return assist(n-1, k)
	} else if k == t+1 {
		return true
	} else {
		return !assist(n-1, 2*t+2-k)
	}
}

// 6188. 按身高排序 https://leetcode.cn/problems/sort-the-people/
// 给你一个字符串数组 names ，和一个由 互不相同 的正整数组成的数组 heights 。两个数组的长度均为 n 。对于每个下标 i，names[i] 和 heights[i] 表示第 i 个人的名字和身高。请按身高 降序 顺序返回对应的名字数组 names 。
func sortPeople(names []string, heights []int) []string {
	size := len(heights)
	index := make([]int, size)
	for i := 0; i < size; i++ {
		index[i] = i
	}
	sort.Slice(index, func(i, j int) bool {
		return heights[index[i]] > heights[index[j]]
	})
	ans := make([]string, size)
	for i := 0; i < size; i++ {
		ans[i] = names[index[i]]
	}
	return ans
}

// 6190. 找到所有好下标 https://leetcode.cn/problems/find-all-good-indices/
// 给你一个大小为 n 下标从 0 开始的整数数组 nums 和一个正整数 k 。对于 k <= i < n - k 之间的一个下标 i ，如果它满足以下条件，我们就称它为一个 好 下标：下标 i 之前 的 k 个元素是 非递增的 。下标 i 之后 的 k 个元素是 非递减的 。按 升序 返回所有好下标。
func goodIndices(nums []int, k int) []int {
	size := len(nums)
	no_increase, no_decrease := make([]int, size), make([]int, size)
	no_increase[0] = 1
	no_decrease[0] = 1
	for i := 1; i < size; i++ {
		if nums[i] > nums[i-1] {
			no_increase[i] = 1
			no_decrease[i] = no_decrease[i-1] + 1
		} else if nums[i] < nums[i-1] {
			no_decrease[i] = 1
			no_increase[i] = no_increase[i-1] + 1
		} else {
			no_decrease[i] = no_decrease[i-1] + 1
			no_increase[i] = no_increase[i-1] + 1
		}
	}
	ans := make([]int, 0, size-2*k)
	for i := k; i < size-k; i++ {
		if no_increase[i-1] >= k && no_decrease[i+k] >= k {
			ans = append(ans, i)
		}
	}
	return ans
}

// 2116. 判断一个括号字符串是否有效 https://leetcode.cn/problems/check-if-a-parentheses-string-can-be-valid/
// 一个括号字符串是只由 '(' 和 ')' 组成的 非空 字符串。如果一个字符串满足下面 任意 一个条件，那么它就是有效的：字符串为 ().它可以表示为 AB（A 与 B 连接），其中A 和 B 都是有效括号字符串。它可以表示为 (A) ，其中 A 是一个有效括号字符串。给你一个括号字符串 s 和一个字符串 locked ，两者长度都为 n 。locked 是一个二进制字符串，只包含 '0' 和 '1' 。对于 locked 中 每一个 下标 i ：如果 locked[i] 是 '1' ，你 不能 改变 s[i] 。如果 locked[i] 是 '0' ，你 可以 将 s[i] 变为 '(' 或者 ')' 。如果你可以将 s 变为有效括号字符串，请你返回 true ，否则返回 false 。
func canBeValid(s string, locked string) bool {
	size := len(s)
	mn, mx := 0, 0
	for i := 0; i < size; i++ {
		if locked[i] == '1' {
			var diff int
			if s[i] == '(' {
				diff = 1
			} else {
				diff = -1
			}
			mx += diff
			mn = max(mn+diff, (i+1)%2)
		} else {
			mx += 1
			mn = max(mn-1, (i+1)%2)
		}
		if mx < mn {
			return false
		}
	}
	return mn == 0
}

type UnionSet struct {
	Rank []int
	Fa   []int
}

func UnionSetConstructor(n int) UnionSet {
	us := UnionSet{Rank: make([]int, n), Fa: make([]int, n)}
	for i := 0; i < n; i++ {
		us.Rank[i] = 1
		us.Fa[i] = i
	}
	return us
}

func (us *UnionSet) Find(x int) int {
	if us.Fa[x] != x {
		us.Fa[x] = us.Find(us.Fa[x])
	}
	return us.Fa[x]
}

func (us *UnionSet) Join(x, y int) bool {
	fx, fy := us.Find(x), us.Find(y)
	if fx == fy {
		return false
	}
	if us.Rank[fx] < us.Rank[fy] {
		fx, fy = fy, fx
	}
	us.Rank[fx] += us.Rank[fy]
	us.Fa[fy] = fx
	return true
}

// 6191. 好路径的数目 https://leetcode.cn/problems/number-of-good-paths/
// 给你一棵 n 个节点的树（连通无向无环的图），节点编号从 0 到 n - 1 且恰好有 n - 1 条边。给你一个长度为 n 下标从 0 开始的整数数组 vals ，分别表示每个节点的值。同时给你一个二维整数数组 edges ，其中 edges[i] = [ai, bi] 表示节点 ai 和 bi 之间有一条 无向 边。一条 好路径 需要满足以下条件：开始节点和结束节点的值 相同 。开始节点和结束节点中间的所有节点值都 小于等于 开始节点的值（也就是说开始节点的值应该是路径上所有节点的最大值）。请你返回不同好路径的数目。注意，一条路径和它反向的路径算作 同一 路径。比方说， 0 -> 1 与 1 -> 0 视为同一条路径。单个节点也视为一条合法路径。
func numberOfGoodPaths(vals []int, edges [][]int) int {
	n := len(vals)
	graph := make([][]int, n)
	us := UnionSetConstructor(n)
	for i := 0; i < n-1; i++ {
		graph[edges[i][0]] = append(graph[edges[i][0]], edges[i][1])
		graph[edges[i][1]] = append(graph[edges[i][1]], edges[i][0])
	}
	idx := make([]int, n)
	for i := 0; i < n; i++ {
		idx[i] = i
	}
	sort.Slice(idx, func(i, j int) bool {
		return vals[idx[i]] < vals[idx[j]]
	})
	ans := n
	for _, x := range idx {
		fx := us.Find(x)
		for _, y := range graph[x] {
			y = us.Find(y)
			if y == fx || vals[y] > vals[x] {
				continue
			}
			if vals[y] == vals[x] {
				ans += us.Rank[fx] * us.Rank[y]
				us.Rank[fx] += us.Rank[y]
			}
			us.Fa[y] = fx
		}
	}
	return ans
}

// LCP 65. 舒适的湿度 https://leetcode.cn/problems/3aqs1c/
// 力扣嘉年华为了确保更舒适的游览环境条件，在会场的各处设置了湿度调节装置，这些调节装置受控于总控室中的一台控制器。控制器中已经预设了一些调节指令，整数数组operate[i] 表示第 i 条指令增加空气湿度的大小。现在你可以将任意数量的指令修改为降低湿度（变化的数值不变），以确保湿度尽可能的适宜：控制器会选择 一段连续的指令 ，从而进行湿度调节的操作；这段指令最终对湿度影响的绝对值，即为当前操作的「不适宜度」在控制器所有可能的操作中，最大 的「不适宜度」即为「整体不适宜度」。请返回在所有修改指令的方案中，可以得到的 最小 「整体不适宜度」。
func unSuitability(operate []int) int {
	m, inf := 0, int(1e9)
	for _, op := range operate {
		m = max(m, op)
	}
	m *= 2
	dp := make([][]int, 2)
	dp[0], dp[1] = make([]int, m+1), make([]int, m+1)
	dp[1][0] = 0
	for i, op := range operate {
		cur := i & 1
		prev := cur ^ 1
		for j := 0; j <= m*2; j++ {
			dp[cur][j] = inf
		}
		for j := 0; j <= m*2; j++ {
			if dp[prev][j] == inf {
				continue
			}
			// var p1, p2 int
			if i < len(dp)-op {

			} else {

			}
			if i >= op {

			} else {

			}
			// dp[i&1][j] = min(p1, p2)
		}
	}
	return 0
}

func getKthMagicNumber(k int) int {
	dp := make([]int, k)
	dp[0] = 1
	p3, p5, p7 := 1, 1, 1
	for i := 1; i < k; i++ {
		x2, x3, x5 := dp[p3]*3, dp[p5]*5, dp[p7]*7
		dp[i] = min(min(x2, x3), x5)
		if dp[i] == x2 {
			p3++
		}
		if dp[i] == x3 {
			p5++
		}
		if dp[i] == x5 {
			p7++
		}
	}
	return dp[k-1]
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// 6192. 公因子的数目 枚举 https://leetcode.cn/problems/number-of-common-factors/
// 给你两个正整数 a 和 b ，返回 a 和 b 的 公 因子的数目。如果 x 可以同时整除 a 和 b ，则认为 x 是 a 和 b 的一个 公因子 。
func commonFactors(a int, b int) int {
	ans := 0
	for i := 1; i <= a; i++ {
		if a%i == 0 && b%i == 0 {
			ans++
		}
	}
	return ans
}

// 6193. 沙漏的最大总和 暴力模拟 https://leetcode.cn/problems/maximum-sum-of-an-hourglass/
// 给你一个大小为 m x n 的整数矩阵 grid 。按以下形式将矩阵的一部分定义为一个 沙漏 ：返回沙漏中元素的 最大 总和。注意：沙漏无法旋转且必须整个包含在矩阵中。
func maxSum(grid [][]int) int {
	m, n := len(grid), len(grid[0])
	ans := 0
	for i := 0; i < m-2; i++ {
		for j := 0; j < n-2; j++ {
			tem := grid[i][j] + grid[i][j+1] + grid[i][j+2] + grid[i+1][j+1] + grid[i+2][j] + grid[i+2][j+1] + grid[i+2][j+2]
			if tem > ans {
				ans = tem
			}
		}
	}
	return ans
}

// 6194. 最小 XOR 暴力模拟 https://leetcode.cn/problems/minimize-xor
// 给你两个正整数 num1 和 num2 ，找出满足下述条件的整数 x ：x 的置位数和 num2 相同，且x XOR num1 的值 最小.注意 XOR 是按位异或运算。返回整数 x 。题目保证，对于生成的测试用例， x 是 唯一确定 的。整数的 置位数 是其二进制表示中 1 的数目。
func minimizeXor(num1 int, num2 int) int {
	c2 := bits.OnesCount(uint(num2))
	if c2 >= bits.Len(uint(num1)) {
		return 1<<c2 - 1
	}
	c1 := bits.OnesCount(uint(num1))
	for ; c2 < c1; c2++ {
		num1 &= num1 - 1
	}
	for x := ^num1; c2 > c1; c2-- {
		num1 |= x & -x
		x &= x - 1
	}
	return num1
}

// 6195. 对字母串可执行的最大删除数 最长公共前缀，动态规划 https://leetcode.cn/problems/maximum-deletions-on-a-string/
// 给你一个仅由小写英文字母组成的字符串 s 。在一步操作中，你可以：删除 整个字符串 s ，或者对于满足 1 <= i <= s.length / 2 的任意 i ，如果 s 中的 前 i 个字母和接下来的 i 个字母 相等 ，删除 前 i 个字母。例如，如果 s = "ababc" ，那么在一步操作中，你可以删除 s 的前两个字母得到 "abc" ，因为 s 的前两个字母和接下来的两个字母都等于 "ab" 。返回删除 s 所需的最大操作数。
func deleteString(s string) int {
	size := len(s)
	lcp := make([][]int, size+1)
	lcp[size] = make([]int, size+1)
	for i := 0; i <= size; i++ {
		lcp[size][i] = 0
	}
	for i := size - 1; i >= 0; i-- {
		lcp[i] = make([]int, size+1)
		lcp[i][size] = 0
		for j := size - 1; j >= 0; j-- {
			if s[i] == s[j] {
				lcp[i][j] = lcp[i+1][j+1] + 1
			}
		}
	}
	dp := make([]int, size)
	dp[size-1] = 1
	for i := size - 1; i >= 0; i-- {
		dp[i] = 0
		for j := 1; i+j*2 <= size; j++ {
			if lcp[i][i+j] >= j {
				dp[i] = max(dp[i], dp[i+j])
			}
		}
		dp[i]++
	}
	return dp[0]
}

// 777. 在LR字符串中交换相邻字符 双指针 https://leetcode.cn/problems/swap-adjacent-in-lr-string/
// 在一个由 'L' , 'R' 和 'X' 三个字符组成的字符串（例如"RXXLRXRXL"）中进行移动操作。一次移动操作指用一个"LX"替换一个"XL"，或者用一个"XR"替换一个"RX"。现给定起始字符串start和结束字符串end，请编写代码，当且仅当存在一系列移动操作使得start可以转换成end时， 返回True。
func canTransform(start string, end string) bool {
	return false
}

// 2105. 给植物浇水 II https://leetcode.cn/problems/watering-plants-ii/
// Alice 和 Bob 打算给花园里的 n 株植物浇水。植物排成一行，从左到右进行标记，编号从 0 到 n - 1 。其中，第 i 株植物的位置是 x = i 。每一株植物都需要浇特定量的水。Alice 和 Bob 每人有一个水罐，最初是满的 。他们按下面描述的方式完成浇水： Alice 按 从左到右 的顺序给植物浇水，从植物 0 开始。Bob 按 从右到左 的顺序给植物浇水，从植物 n - 1 开始。他们 同时 给植物浇水。如果没有足够的水 完全 浇灌下一株植物，他 / 她会立即重新灌满浇水罐。不管植物需要多少水，浇水所耗费的时间都是一样的。不能 提前重新灌满水罐。每株植物都可以由 Alice 或者 Bob 来浇水。如果 Alice 和 Bob 到达同一株植物，那么当前水罐中水更多的人会给这株植物浇水。如果他俩水量相同，那么 Alice 会给这株植物浇水。给你一个下标从 0 开始的整数数组 plants ，数组由 n 个整数组成。其中，plants[i] 为第 i 株植物需要的水量。另有两个整数 capacityA 和 capacityB 分别表示 Alice 和 Bob 水罐的容量。返回两人浇灌所有植物过程中重新灌满水罐的 次数 。
func minimumRefill(plants []int, capacityA int, capacityB int) int {
	left, right := 0, len(plants)-1
	a, b, ans := capacityA, capacityB, 0
	for left < right {
		if a < plants[left] {
			a = capacityA
			ans++
		}
		a -= plants[left]
		left++
		if b < plants[right] {
			b = capacityB
			ans++
		}
		b -= plants[right]
		right--
	}
	if left == right {
		if a >= b && a < plants[left] {
			ans++
		}
		if a < b && b < plants[right] {
			ans++
		}
	}
	return ans
}

// 6200. 处理用时最长的那个任务的员工 https://leetcode.cn/problems/the-employee-that-worked-on-the-longest-task/
// 共有 n 位员工，每位员工都有一个从 0 到 n - 1 的唯一 id 。给你一个二维整数数组 logs ，其中 logs[i] = [idi, leaveTimei] ：idi 是处理第 i 个任务的员工的 id ，且leaveTimei 是员工完成第 i 个任务的时刻。所有 leaveTimei 的值都是 唯一 的。注意，第 i 个任务在第 (i - 1) 个任务结束后立即开始，且第 0 个任务从时刻 0 开始。返回处理用时最长的那个任务的员工的 id 。如果存在两个或多个员工同时满足，则返回几人中 最小 的 id
func hardestWorker(n int, logs [][]int) int {
	size1 := len(logs)
	max_id, max_time, prev := -1, -1, 0
	for i := 0; i < size1; i++ {
		if logs[i][1]-prev > max_time {
			max_time = logs[i][1] - prev
			max_id = logs[i][0]
		} else if logs[i][1]-prev == max_time && logs[i][0] < max_id {
			max_id = logs[i][0]
		}
		prev = logs[i][1]
	}
	return max_id
}

// 88. 合并两个有序数组 https://leetcode.cn/problems/merge-sorted-array/description/
// 给你两个按 非递减顺序 排列的整数数组 nums1 和 nums2，另有两个整数 m 和 n ，分别表示 nums1 和 nums2 中的元素数目。请你 合并 nums2 到 nums1 中，使合并后的数组同样按 非递减顺序 排列。注意：最终，合并后数组不应由函数返回，而是存储在数组 nums1 中。为了应对这种情况，nums1 的初始长度为 m + n，其中前 m 个元素表示应合并的元素，后 n 个元素为 0 ，应忽略。nums2 的长度为 n 。
func merge(nums1 []int, m int, nums2 []int, n int) {
	p1, p2 := m-1, n-1
	tail := m + n - 1
	for tail >= 0 {
		flag := p1 == -1 || (p2 != -1 && nums1[p1] < nums2[p2])
		if flag {
			nums1[tail] = nums2[p2]
			p2--
		} else {
			nums1[tail] = nums1[p1]
			p1--
		}
		tail--
	}
}

// 6202. 使用机器人打印字典序最小的字符串 https://leetcode.cn/problems/using-a-robot-to-print-the-lexicographically-smallest-string/
// 给你一个字符串 s 和一个机器人，机器人当前有一个空字符串 t 。执行以下操作之一，直到 s 和 t 都变成空字符串：删除字符串 s 的 第一个 字符，并将该字符给机器人。机器人把这个字符添加到 t 的尾部。删除字符串 t 的 最后一个 字符，并将该字符给机器人。机器人将该字符写到纸上。请你返回纸上能写出的字典序最小的字符串。
func robotWithString(s string) string {
	size := len(s)
	min_ch := make([]byte, size+1)
	min_ch[size] = 'z' + 1
	for i := size - 1; i >= 0; i-- {
		min_ch[i] = min_byte(min_ch[i+1], s[i])
	}

	stk := make([]byte, size)
	top := -1
	var ans strings.Builder
	for i := 0; i < size; i++ {
		top++
		stk[top] = s[i]
		for top >= 0 && stk[top] <= min_ch[i+1] {
			ans.WriteByte(stk[top])
			top--
		}
	}
	return ans.String()
}

func min_byte(a, b byte) byte {
	if a < b {
		return a
	}
	return b
}

// 6203. 矩阵中和能被 K 整除的路径 https://leetcode.cn/problems/paths-in-matrix-whose-sum-is-divisible-by-k/
// 给你一个下标从 0 开始的 m x n 整数矩阵 grid 和一个整数 k 。你从起点 (0, 0) 出发，每一步只能往 下 或者往 右 ，你想要到达终点 (m - 1, n - 1) 。请你返回路径和能被 k 整除的路径数目，由于答案可能很大，返回答案对 109 + 7 取余 的结果。
func numberOfPaths(grid [][]int, k int) int {
	mod := int64(1e9 + 7)
	m, n := len(grid), len(grid[0])
	dp := make([][][]int64, 2)
	for i := 0; i < 2; i++ {
		dp[i] = make([][]int64, n+1)
		for j := 0; j <= n; j++ {
			dp[i][j] = make([]int64, k)
		}
	}
	for i := 0; i < m; i++ {
		cur := i & 1
		prev := cur ^ 1
		for j := 0; j < k; j++ {
			dp[cur][0][j] = 0
		}
		if i == 0 {
			dp[cur][0][0] = 1
		}
		for j := 1; j <= n; j++ {
			for l := 0; l < k; l++ {
				p := (l - grid[i][j-1]%k + k) % k
				dp[cur][j][l] = (dp[cur][j-1][p] + dp[prev][j][p]) % mod
			}
		}
	}
	return int(dp[(m-1)&1][n][0])
}

// 856. 括号的分数 https://leetcode.cn/problems/score-of-parentheses/
// 给定一个平衡括号字符串 S，按下述规则计算该字符串的分数：() 得 1 分。AB 得 A + B 分，其中 A 和 B 是平衡括号字符串。(A) 得 2 * A 分，其中 A 是平衡括号字符串。
func scoreOfParentheses(s string) int {
	size, ch_top, val_top := len(s), -1, -1
	type pair struct{ pos, val int }
	ch_stk := make([]int, size)
	val_stk := make([]pair, size)
	ans := 0
	for idx, ch := range s {
		if ch == ')' {
			val := 0
			for val_top >= 0 && val_stk[val_top].pos >= ch_stk[ch_top] {
				val += val_stk[val_top].val
				val_top--
			}
			if val == 0 {
				val = 1
			} else {
				val *= 2
			}
			if val_top < 0 {
				ans += val
			} else {
				val_stk[val_top].val += val
			}
			ch_top--
		} else {
			ch_top++
			ch_stk[ch_top] = idx
			val_top++
			val_stk[val_top] = pair{val: 0, pos: idx}
		}
	}
	return ans
}

// 6208. 有效时间的数目 https://leetcode.cn/problems/number-of-valid-clock-times/
// 给你一个长度为 5 的字符串 time ，表示一个电子时钟当前的时间，格式为 "hh:mm" 。最早 可能的时间是 "00:00" ，最晚 可能的时间是 "23:59" 。在字符串 time 中，被字符 ? 替换掉的数位是 未知的 ，被替换的数字可能是 0 到 9 中的任何一个。请你返回一个整数 answer ，将每一个 ? 都用 0 到 9 中一个数字替换后，可以得到的有效时间的数目。
func countTime(time string) int {
	ans := 1
	if time[0] == '?' {
		if time[1] == '?' {
			ans = 24
		} else if time[1] <= '3' {
			ans = 3
		} else {
			ans = 2
		}
	} else if time[1] == '?' {
		if time[0] == '2' {
			ans = 4
		} else {
			ans = 10
		}
	}
	if time[3] == '?' {
		ans *= 6
	}
	if time[4] == '?' {
		ans *= 10
	}
	return ans
}

// 6209. 二的幂数组中查询范围内的乘积 https://leetcode.cn/problems/range-product-queries-of-powers/
// 给你一个正整数 n ，你需要找到一个下标从 0 开始的数组 powers ，它包含 最少 数目的 2 的幂，且它们的和为 n 。powers 数组是 非递减 顺序的。根据前面描述，构造 powers 数组的方法是唯一的。同时给你一个下标从 0 开始的二维整数数组 queries ，其中 queries[i] = [lefti, righti] ，其中 queries[i] 表示请你求出满足 lefti <= j <= righti 的所有 powers[j] 的乘积。请你返回一个数组 answers ，长度与 queries 的长度相同，其中 answers[i]是第 i 个查询的答案。由于查询的结果可能非常大，请你将每个 answers[i] 都对 109 + 7 取余 。
func productQueries(n int, queries [][]int) []int {
	var pre_sum [33]int
	pre_sum[0] = 0
	for i := 0; i < 32; i++ {
		if n&(1<<i) != 0 {
			pre_sum[i+1] = i
		}
	}
	for i := 1; i <= 32; i++ {
		pre_sum[i] += pre_sum[i-1]
	}
	size := len(queries)
	ans := make([]int, size)
	for i := 0; i < size; i++ {
		ans[i] = fastPow(pre_sum[queries[i][1]+1] - pre_sum[queries[i][0]])
	}
	return ans
}

func fastPow(n int) int {
	ans, x_pow, mod := int64(1), int64(2), int64(1e9+7)
	for n > 0 {
		if n&1 != 0 {
			ans = x_pow * ans % mod
		}
		x_pow = x_pow * x_pow % mod
		n /= 2
	}
	return int(ans)
}

// 6210. 最小化数组中的最大值 https://leetcode.cn/problems/minimize-maximum-of-array/
// 给你一个下标从 0 开始的数组 nums ，它含有 n 个非负整数。每一步操作中，你需要：选择一个满足 1 <= i < n 的整数 i ，且 nums[i] > 0 。将 nums[i] 减 1 。将 nums[i - 1] 加 1 。你可以对数组执行 任意 次上述操作，请你返回可以得到的 nums 数组中 最大值 最小 为多少。
func minimizeArrayValue(nums []int) int {
	mx := 0
	for _, num := range nums {
		mx = max(mx, num)
	}
	return sort.Search(mx+1, func(mid int) bool {
		size := len(nums)
		arr := make([]int64, size)
		for idx, num := range nums {
			arr[idx] = int64(num)
		}
		for i := size - 1; i > 0; i-- {
			diff := arr[i] - int64(mid)
			if diff > 0 {
				arr[i] = int64(mid)
				arr[i-1] += diff
			}
		}
		return arr[0] <= int64(mid)
	})
}
