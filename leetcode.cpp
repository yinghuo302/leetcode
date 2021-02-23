#include <bits/stdc++.h>
using namespace std;
int minPatches(vector<int> &nums, int n)
{
}
int findMaximizedCapital(int k, int W, vector<int> &Profits, vector<int> &Capital)
{
}
int strStr(string haystack, string needle)
{
    int h_size = haystack.size(), n_size = needle.size(), i = 0, j = 0;
    while (j < n_size)
    {
        if (haystack[i] == needle[j])
        {
            ++i;
            ++j;
        }
        else
        {
            i = i - j + 1;
            j = 0;
            if (i >= h_size)
                return -1;
        }
    }
    return i - n_size;
}
//股票价格跨度，股票价格小于或等于今天价格的最大连续日数，包括今天
class StockSpanner
{
private:
    stack<int> s;
    vector<int> prices;

public:
    StockSpanner()
    {
    }
    int next(int price)
    {
        while (!s.empty() && price >= prices[s.top()])
            s.pop();
        int num = 0;
        if (s.empty())
            num = prices.size() + 1;
        else
            num = prices.size() - s.top();
        prices.push_back(price);
        s.push(prices.size() - 1);
        return num;
    }
};
//设计推特
typedef int User_Id;
typedef int Tweet_time;
typedef int Tweet_Id;
struct TweetNode
{
    Tweet_Id id;
    Tweet_time time;
    TweetNode *next;
    TweetNode(int id, int time) : id(id), time(time), next(NULL) {}
    TweetNode() : id(-1), time(-1), next(NULL) {}
};
class User
{
private:
    TweetNode *tweets;

public:
    unordered_set<User_Id> followees;
    User() : tweets(NULL) {}
    User(int id) : tweets(NULL) { followees.emplace(id); }
    void like(int like_id)
    {
        followees.emplace(like_id);
    }
    void dislike(int like_id)
    {
        int size = followees.size(), i = 0;
        unordered_set<int>::iterator followee = followees.begin();
        while (followee != followees.end())
            if (like_id == *followee)
                followees.erase(followee);
    }
    TweetNode *get_tweet()
    {
        return tweets;
    }
    void post_tweet(int tweet_id, int time)
    {
        if (!tweets)
            tweets = new TweetNode(tweet_id, time);
        else
            tweets->next = new TweetNode(tweet_id, time);
    }
};
class Twitter
{
private:
    unordered_map<User_Id, User *> users;
    Tweet_time time;
    TweetNode *mergeTwoLists(TweetNode *l1, TweetNode *l2)
    {
        if (l1 == NULL)
            return l2;
        if (l2 == NULL)
            return l1;
        if (l1->time > l2->time)
        {
            l1->next = mergeTwoLists(l1->next, l2);
            return l1;
        }
        else
        {
            l2->next = mergeTwoLists(l1, l2->next);
            return l2;
        }
    }
    TweetNode *deep_copy(TweetNode *head)
    {
        if (!NULL)
            return NULL;
        TweetNode *root = new TweetNode(head->id, head->time), *p = root;
        while (head)
        {
            p->next = new TweetNode(head->id, head->time);
            p = p->next;
        }
        p = root->next;
        delete (root);
        return p;
    }
    void deep_del(TweetNode *head)
    {
        if (!head)
            return;
        TweetNode *p = head->next;
        while (p)
        {
            delete (head);
            head = p;
            p = p->next;
        }
        delete (head);
    }

public:
    Twitter()
    {
        time = 0;
    }
    void postTweet(int userId, int tweetId)
    {
        if (users.empty())
        {
            users.emplace(userId, new User(userId));
            users[userId]->post_tweet(tweetId, time);
            return;
        }
        unordered_map<User_Id, User *>::iterator usr_iter = users.find(userId);
        if (users.end() == usr_iter)
        {
            users.emplace(userId, new User(userId));
            usr_iter->second->post_tweet(tweetId, time);
        }
        else
        {
            usr_iter->second->post_tweet(tweetId, time);
        }
        ++time;
    }
    vector<int> getNewsFeed(int userId)
    {
        TweetNode *r = NULL;
        for (auto &followee : users[userId]->followees)
        {
            r = mergeTwoLists(r, deep_copy(users[followee]->get_tweet()));
        }
        vector<int> res;
        TweetNode *h = r;
        while (res.size() <= 10 && r)
        {
            res.push_back(r->id);
            r = r->next;
        }
        deep_del(h);
        return res;
    }
    void follow(int followerId, int followeeId)
    {
        if (users.empty())
            return;
        unordered_map<User_Id, User *>::iterator usr_iter = users.find(followerId);
        if (users.end() != usr_iter)
            usr_iter->second->like(followeeId);
    }
    void unfollow(int followerId, int followeeId)
    {
        if (users.empty())
            return;
        unordered_map<User_Id, User *>::iterator usr_iter = users.find(followerId);
        if (users.end() != usr_iter)
            usr_iter->second->dislike(followeeId);
    }
};
//在数组中找两个数和为target
vector<int> twoSum(vector<int> &nums, int target)
{
    unordered_map<int, int> m;
    int size = nums.size();
    for (int i = 0; i < size; ++i)
    {
        unordered_map<int, int>::iterator a = m.find(target - nums[i]);
        if (a != m.end())
            return {i, a->second};
        m.emplace(nums[i], i);
    }
    return {};
}
//连续1的个数
int findMaxConsecutiveOnes(vector<int> &nums)
{
    int size = nums.size();
    int res = 0, count = 0;
    for (int i = 0; i < size; ++i)
    {
        if (nums[i])
            ++count;
        else if (count > res)
            res = count;
    }
    return (count > res) ? count : res;
}
//如果矩阵上每一条由左上到右下的对角线上的元素都相同，那么这个矩阵是托普利茨矩阵。
bool isToeplitzMatrix(vector<vector<int>> &matrix){
    int n1 = matrix.size(), n2 = matrix[0].size();
    for (int i = 1; i < n1;i++) 
        for (int j = 1; j < n2; j++) 
            if (matrix[i][j] != matrix[i - 1][j - 1]) 
                    return false; 
    return true;
}
//
int longestSubarray(vector<int>& nums, int limit) {
    int size=nums.size(),left=0,right=0,res=0;
    multiset<int> s;
    while(right<size){
        s.insert(nums[right]);
        while(*s.begin()-*s.rbegin()>limit){
            s.erase(s.find(nums[left]));
            ++left;
        }
        if(right-left+1>res)
            res=right-left+1;
        ++right;
    }
    return res;
}
int maxSatisfied(vector<int>& customers, vector<int>& grumpy, int X) {
    int size=customers.size(),no_use=0,use=0;
    for(int i=0;i<size;++i)
        if(!grumpy[i])
            no_use+=customers[i];
    int left=0,right=X,tem=0;
    for(int i=0;i<right&&i<size;++i)
        if(grumpy[i])
            tem+=customers[i];
    while(right<size){
        tem=tem+grumpy[right]-grumpy[left];
        if(tem>use)
            use=tem;
        ++right;++right;
    }
    return no_use+use;
}