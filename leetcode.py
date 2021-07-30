'''
Author: zanilia
Date: 2021-07-30 13:01:40
LastEditTime: 2021-07-30 16:23:44
Descripttion: 
'''
from typing import List
class Solution:
    def findJudge(self, n: int, trust: List[List[int]]) -> int:
        if n==1:
            return 1
        cnt = [0 for i in range(n+1)]
        for a,b in trust:
            cnt[a] -= 1
            cnt[b] += 1
        for i in range(n+1):
            if cnt[i]==n-1:
                return i
        return -1
    def findCircleNum(self, isConnected: List[List[int]]) -> int:
        size = len(isConnected)
        def bfs(pos):
            if(pos<0 or pos>=size):
                return False
            have_found = isConnected[pos][pos]
            isConnected[pos][pos] =0 
            for i in range(size):
                if isConnected[pos][i]:
                    isConnected[pos][i] = 0
                    bfs(i)
                    have_found = True
            return have_found
        ans = 0
        for i in range(size):
            if(bfs(i)):
                ans += 1
        return ans
    def canVisitAllRooms(self, rooms: List[List[int]]) -> bool:
        have_visited = set()
        def bfs(pos):
            if pos in have_visited:
                return
            have_visited.add(pos) 
            for  i in rooms[pos]:
                bfs(i)
        bfs(0)
        return len(have_visited) == len(rooms)
    def findSmallestSetOfVertices(self, n: int, edges: List[List[int]]) -> List[int]:
        cnt = [0 for i in range(n)]
        ans = []
        for a,b in edges:
            cnt[b] +=1
        for i in range(n):
            if(cnt[i]!=0):
                ans.append(i)
        return ans
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def minCameraCover(self, root: TreeNode) -> int:
        def observe(node:TreeNode):
            a = dfs(node.right,False,False,True)+dfs(node.left,True,True,True)+1
            b = dfs(node.left,False,False,True)+dfs(node.right,True,True,True)+1
            c = dfs(node,True,True,True)
            return min(a,b,c)
        def dfs(node:TreeNode,mintor:bool,this_observed:bool,parent_observed:bool)->int:
            if not node:
                return 0
            if mintor:
                return dfs(node.left,False,True,True)+dfs(node.right,False,True,True)
            if this_observed:
                return dfs(node.left,False,False,True)+dfs(node.right,False,False,True)
            if parent_observed:
                return observe(node)
            else:
                return dfs(node.right,False,True,True)+dfs(node.left,False,True,True)+1
        return dfs(root,False,False,True)

        # if not node:
        #         return 0
        #     if is_left:
        #         return dfs(node.right,False,False,True)+dfs(node.left,True,True,True)+1
        #     return dfs(node.left,False,False,True)+dfs(node.right,True,True,True)+1