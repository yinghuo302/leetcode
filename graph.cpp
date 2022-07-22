#include <bits/stdc++.h>
using namespace std;
// 207课程表 https://leetcode-cn.com/problems/course-schedule/  210课程表 II https://leetcode-cn.com/problems/course-schedule-ii/
// 你这个学期必须选修 numCourses 门课程，记为 0 到 numCourses - 1 。在选修某些课程之前需要一些先修课程。 先修课程按数组 prerequisites 给出，其中 prerequisites[i] = [ai, bi] ，表示如果要学习课程 ai 则 必须 先学习课程  bi 。例如，先修课程对 [0, 1] 表示：想要学习课程 0 ，你需要先完成课程 1 。请你判断是否可能完成所有课程的学习？如果可以，返回 true ；否则，返回 false 。
class Solution {
    int pos;
    bool valid;
public:
    vector<int> findOrder(int numCourses, vector<vector<int>>& prerequisites) {
        vector<vector<int>> edge(numCourses);
        char visited[numCourses];
        for(int i=0;i<numCourses;++i)
            visited[i] = 0;
        vector<int> res(numCourses);
        pos = numCourses;
        valid = true;
        for(auto &prerequire:prerequisites)
            edge[prerequire[1]].push_back(prerequire[0]);
        for(int i=0;(i<numCourses)&&valid;++i)
            if(!visited[i])
                dfs(edge,res,i,visited);
        if(valid)
            return res;
        else
            return {};
    }
    void dfs(const vector<vector<int>>& edge,vector<int>& res,int x,char* visited){
        visited[x] = 1;
        for(auto &y: edge[x]){
            if(!visited[y]){
                dfs(edge,res,y,visited);
                if(!valid)
                    return ;
            }
            else if(visited[y]==1){
                valid =  false;
                return ;
            }
        }
        res[--pos] = x;
        visited[x] = 2;
    }
};
// 310. 最小高度树 https://leetcode-cn.com/problems/minimum-height-trees/
// 给你一棵包含 n 个节点的树，标记为 0 到 n - 1 。给定数字 n 和一个有 n - 1 条无向边的 edges 列表（每一个边都是一对标签），其中edges[i] = [ai, bi] 表示树中节点 ai 和 bi 之间存在一条无向边。可选择树中任何一个节点作为根。当选择节点 x 作为根节点时，设结果树的高度为 h 。在所有可能的树中，具有最小高度的树（即，min(h)）被称为 最小高度树 。 请你找到所有的 最小高度树 并按 任意顺序 返回它们的根节点标签列表。 拓扑排序
vector<int> findMinHeightTrees(int n, vector<vector<int>>& edges) {
    if (n == 1)
        return {0};
    if (n == 2)
        return {0, 1};
    int degree[n];
    memset(degree, 0, n<<2);
    vector<vector<int>> graph(n, vector<int>());
    for (vector<int>& edge : edges){
        ++degree[edge[0]];
        ++degree[edge[1]];
        graph[edge[0]].push_back(edge[1]);
        graph[edge[1]].push_back(edge[0]);
    }
    queue<int> q;
    for(int i=0;i<n;++i)
        if(degree[i]==1)
            q.emplace(i);
    while(n>2){
        int size = q.size();
        n -= size;
        for(int i=0;i<size;++i){
            int curr = q.front();
            q.pop();
            degree[curr] = 0; // 无向图，记录是否访问过该节点
            for (int next : graph[curr]){
                if (degree[next] != 0){
                    --degree[next];
                    if (degree[next] == 1)
                        q.push(next);
                }
            }
        }
    }
    vector<int> ans;
    while (!q.empty()){
        ans.push_back(q.front());
        q.pop();
    }
    return ans;
}
// 329矩阵中的最长递增路径 https://leetcode-cn.com/problems/longest-increasing-path-in-a-matrix/
// 给定一个 m x n 整数矩阵 matrix ，找出其中 最长递增路径 的长度。对于每个单元格，你可以往上，下，左，右四个方向移动。你不能 在 对角线 方向上移动或移动到 边界外（即不允许环绕）。 拓扑排序，记忆深度优先搜索
int longestIncreasingPath(vector<vector<int>>& matrix) {
    auto getDegree = [](vector<vector<int>>& matrix,int i,int j){
        int res = 0,m = matrix.size(), n = matrix[0].size();
        if(i>0&&matrix[i][j]>matrix[i-1][j])
            ++res;
        if(j>0&&matrix[i][j]>matrix[i][j-1])
            ++res;
        if(i+1<m&&matrix[i][j]>matrix[i+1][j])
            ++res;
        if(j+1<n&&matrix[i][j]>matrix[i][j+1])
            ++res;
        return res;
    };
    int m= matrix.size(), n = matrix[0].size();
    int* in_degree[m];
    queue<int> q;
    for(int i=0;i<m;++i){
        in_degree[i] = new int [n];
        for(int j=0;j<n;++j){
            in_degree[i][j] = getDegree(matrix,i,j);
            if(in_degree[i][j]==0)
                q.emplace((i<<8)+j);
        }
    }
    int count = 0;
    while(!q.empty()){
        int size = q.size();
        ++count;
        for(int i=0;i<size;++i){
            int x = q.front();
            q.pop();
            int y = x&0xFF;
            x >>= 8;
            if(x>0&&matrix[x][y]<matrix[x-1][y]){
                --in_degree[x-1][y];
                if(in_degree[x-1][y]==0)
                    q.emplace(((x-1)<<8)+y);
            }
            if(y>0&&matrix[x][y]<matrix[x][y-1]){
                --in_degree[x][y-1];
                if(in_degree[x][y-1]==0)
                    q.emplace((x<<8)+y-1);
            }
            if(x+1<m&&matrix[x][y]<matrix[x+1][y]){
                --in_degree[x+1][y];
                if(in_degree[x+1][y]==0)
                    q.emplace(((x+1)<<8)+y);
            }
            if(y+1<n&&matrix[x][y]<matrix[x][y+1]){
                --in_degree[x][y+1];
                if(in_degree[x][y+1]==0)
                    q.emplace((x<<8)+y+1);
            }
        }
    }
    return count;
}