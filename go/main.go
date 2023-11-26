package main

func constructDistancedSequence(n int) []int {
	size := 2*n - 1
	ret := make([]int, size)
	queue := make([]int, n-1)
	for i := 0; i < size; i++ {
		ret[i] = 0
	}
	for i := 0; i < n-1; i++ {
		queue[i] = i + 2
	}
	pos := n - 1
	tmp := make([]int, 0, n)
	for i := 0; i < size; i++ {
		if ret[i] != 0 {
			continue
		}
		pos--
		for pos >= 0 {
			top := queue[pos]
			if i+top < size && ret[i+top] == 0 {
				ret[i] = top
				ret[i+top] = top
				break
			} else {
				pos--
				tmp = append(tmp, top)
			}
		}
		if pos < 0 {
			ret[i] = 1
			pos++
		}
		for j := len(tmp) - 1; j >= 0; j-- {
			queue[pos] = tmp[j]
			pos++
		}
	}
	return ret
}

func main() {
	constructDistancedSequence(3)
}
