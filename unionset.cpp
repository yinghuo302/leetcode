/*
 * @Author: zanilia
 * @Date: 2022-02-17 13:14:18
 * @LastEditTime: 2022-02-17 13:14:45
 * @Descripttion: 
 */
#include <bits/stdc++.h>
using namespace std;
// 684. 冗余连接 https://leetcode-cn.com/problems/redundant-connection/
// 给定往一棵 n 个节点 (节点值 1～n) 的树中添加一条边后的图。删除一条边使其变为树
vector<int> findRedundantConnection(vector<vector<int>>& edges) {
    UnionSet us(edges.size()+1);
    for(auto &edge:edges)
        if(!us.join(edge[0],edge[1]))
            return edge;
    return {};
}
class UnionSet{
    int* rank;
    int* fa;
public:
    UnionSet():rank(new int[20]),fa(new int [20]){
        for(int i=0;i<20;++i){
            rank[i] = 1;
            fa[i] = i;
        }
    }
    UnionSet(int n):rank(new int[n]),fa(new int [n]){
        for(int i=0;i<n;++i){
            rank[i] = 1;
            fa[i] = i;
        }
    }
    ~UnionSet(){
        delete[] rank;
        delete[] fa;
    }
    int find(int x){
        if(fa[x]!=x)
            fa[x] = find(fa[x]);
        return fa[x];
    }
    bool join(int x,int y){
        int fx = find(x),fy= find(y);
        if(fx==fy)
            return false;
        if(rank[fx]<rank[fy]){
            int tem = fy;
            fy = fx;
            fx = tem;
        }
        rank[fx] += rank[fy];
        fa[fy] = fx;
        return true;
    }
};
// 685. 冗余连接 II https://leetcode-cn.com/problems/redundant-connection-ii/ 
// 有根树指满足以下条件的有向图。有根树指满足以下条件的有向图。该树只有一个根节点，所有其他节点都是该根节点的后继。该树除了根节点之外的每一个节点都有且只有一个父节点，而根节点没有父节点。返回一条能删除的边，使得剩下的图是有 n 个节点的有根树多的一条边可能形成双父节点，或者有向环，如果两种情况都有则答案唯一(判断哪一个父亲节点造成了环)，否则返回后出现的那一个
vector<int> findRedundantDirectedConnection(vector<vector<int>>& edges) {
    int n = edges.size();
    int parent[n+1];
    memset(parent,0,n*4+4);
    int double_p = -1;
    int circle = -1;
    UnionSet us(n+1);
    for(int i=0;i<n;++i){
        int x = edges[i][0],y = edges[i][1];
        if(parent[y])
            double_p = i;
        else{
            parent[y] = x;
            if(!us.join(x,y))
                circle = i;
        }
    }
    if(double_p==-1)
        return edges[circle];
    else if(circle!=-1)
        return {parent[edges[double_p][1]],edges[double_p][1]};
    else
        return edges[double_p];
}
// 399 除法求值https://leetcode-cn.com/problems/evaluate-division/
// 给你一个变量对数组 equations 和一个实数值数组 values 作为已知条件，其中 equations[i] = [Ai, Bi] 和 values[i] 共同表示等式 Ai / Bi = values[i] 。每个 Ai 或 Bi 是一个表示单个变量的字符串。另有一些以数组 queries 表示的问题，其中 queries[j] = [Cj, Dj] 表示第 j 个问题，请你根据已知条件找出 Cj / Dj = ? 的结果作为答案。返回 所有问题的答案 。如果存在某个无法确定的答案，则用 -1.0 替代这个答案。如果问题中出现了给定的已知条件中没有出现的字符串，也需要用 -1.0 替代这个答案。注意：输入总是有效的。你可以假设除法运算中不会出现除数为 0 的情况，且不存在任何矛盾的结果。
class UnionFind{
    int * parent;
    double* weight;
public:
    UnionFind():parent(new int[20]),weight(new double[20]){
        for(int i=0;i<20;++i){
            parent[i] = i;
            weight[i] = 1;
        }
    }
    UnionFind(int n):parent(new int[n]),weight(new double[n]){
        for(int i=0;i<n;++i){
            parent[i] = i;
            weight[i] = 1;
        }
    }
    int find(int u){
        if(u==parent[u])
            return u;
        int tem = parent[u];
        parent[u] = find(tem);
        weight[u] *= weight[tem];
        return parent[u];
    }
    bool join(int u,int v,double w){
        int u_root = find(u),v_root = find(v);
        if(u_root==v_root)
            return false;
        parent[u_root] = v_root;
        weight[u_root] = weight[v]*w/weight[u];
        return true;
    }
    double getWeight(int u,int v){
        int u_root = find(u),v_root = find(v);
        if(u_root==v_root)
            return weight[u]/weight[v];
        else
            return -1;
    }
};
vector<double> calcEquation(vector<vector<string>>& equations, vector<double>& values, vector<vector<string>>& queries) {
    int id = 0;
    unordered_map<string,int> mp;
    int size = equations.size();
    UnionFind uf(size*2);
    for(int i=0;i<size;++i){
        if(!mp.count(equations[i][0])){
            mp.emplace(equations[i][0],id);
            ++id;
        }
        if(!mp.count(equations[i][1])){
            mp.emplace(equations[i][1],id);
            ++id;
        }
        uf.join(mp.find(equations[i][0])->second,mp.find(equations[i][1])->second,values[i]);
    }
    int size1 = queries.size();
    vector<double> res(size1);
    for(int i=0;i<size1;++i){
        auto i1 = mp.find(queries[i][0]),i2 = mp.find(queries[i][1]);
        if(i1==mp.end()||i2==mp.end())
            res[i] = -1;
        else
            res[i] = uf.getWeight(i1->second,i2->second);
    }
    return res;
}