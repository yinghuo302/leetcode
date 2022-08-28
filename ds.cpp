#include <bits/stdc++.h>
using namespace std;
/* 
 * 请你设计并实现一个满足  LRU (最近最少使用) 缓存 约束的数据结构。实现 LRUCache 类：
 * LRUCache(int capacity) 以 正整数 作为容量 capacity 初始化 LRU 缓存
 * int get(int key) 如果关键字 key 存在于缓存中，则返回关键字的值，否则返回 -1 。
 * void put(int key, int value) 如果关键字 key 已经存在，则变更其数据值 value ；如果不存在，则向缓存中插入该组 key-value 。如果插入操作导致关键字数量超过 capacity ，则应该 逐出 最久未使用的关键字。
 * 函数 get 和 put 必须以 O(1) 的平均时间复杂度运行。
 */
class LRUCache {
    struct LinkList{
        int key;
        int val;
        LinkList* next;
        LinkList* prev;
        LinkList():key(-1),val(-1),next(NULL),prev(NULL){}
        LinkList(int key,int value):key(key),val(value),next(NULL),prev(NULL){}
        LinkList(int key,int value,LinkList* next):key(key),val(value),next(next){}
    };
private:
    unordered_map<int,LinkList*> data;
    LinkList* head;
    LinkList* tail;
    int capacity;
    void moveToHead(LinkList* node){
        if(node==head)
            return ;
        node->prev->next = node->next;
        if(node==tail)
			tail = node->prev;
		else
            node->next->prev = node->prev;
        node->next = head;
        head->prev = node;
        node->prev = NULL;
        head = node;
    }
    void addToHead(LinkList* node){
        if(!head)
            head = tail = node;
        else{
            node->next = head;
            head->prev = node;
            head = node;
        }
    }
    void removeTail(){
        tail->prev->next = NULL;
        LinkList *tem = tail;
        tail = tail->prev;
        data.erase(tem->key);
        delete tem;
    }
public:
    LRUCache(int capacity):capacity(capacity+1),head(NULL),tail(NULL){}
    int get(int key) {
        auto i1=data.find(key);
        if(i1!=data.end()){
            moveToHead(i1->second);
            return i1->second->val;
        }
        return -1;
    }
    int put(int key, int value){
        unordered_map<int,LinkList*>::iterator i1=data.find(key);
        if(i1!=data.end()){
            i1->second->val=value;
            moveToHead(i1->second);
        }
        else{
            data.emplace(key,new LinkList(key,value));
            addToHead(data[key]);
            if(data.size()==capacity)
                removeTail();
        }
		return 1;
    }
};
// 460. LFU 缓存 https://leetcode.cn/problems/lfu-cache/
/* 
 * 请你为 最不经常使用（LFU）缓存算法设计并实现数据结构。实现 LFUCache 类：
 * LFUCache(int capacity) - 用数据结构的容量 capacity 初始化对象
 * int get(int key) - 如果键 key 存在于缓存中，则获取键的值，否则返回 -1 。
 * void put(int key, int value) - 如果键 key 已存在，则变更其值；如果键不存在，请插入键值对。当缓存达到其容量 capacity 时，则应该在插入新项之前，移除最不经常使用的项。在此问题中，当存在平局（即两个或更多个键具有相同使用频率）时，应该去除 最近最久未使用 的键。
 * 为了确定最不常使用的键，可以为缓存中的每个键维护一个 使用计数器 。使用计数最小的键是最久未使用的键。当一个键首次插入到缓存中时，它的使用计数器被设置为 1 (由于 put 操作)。对缓存中的键执行 get 或 put 操作，使用计数器的值将会递增。
 * 函数 get 和 put 必须以 O(1) 的平均时间复杂度运行。
 */
class LFUCache {
    int capacity;
    struct Node{
        int val;
        int freq;
        list<int>::iterator node;
        Node() = default;
        Node(int val,int freq,list<int>::iterator node):val(val),freq(freq),node(node){}
    };
    unordered_map<int,Node> data_mp;
    unordered_map<int,list<int>> freq_mp;
    int min_fq;
public:
    LFUCache(int capacity):capacity(capacity),min_fq(0){
        data_mp.reserve(capacity);
        freq_mp.reserve(capacity);
    }
    int get(int key) {
        auto i = data_mp.find(key);
        if(i==data_mp.end())
            return -1;
        int freq = i->second.freq;
        i->second.freq++;
        freq_mp[freq].erase(i->second.node);
        if(freq_mp[freq].empty()){
            if(freq==min_fq)
                min_fq += 1;
            freq_mp.erase(freq);
        }
    }
    void put(int key, int value) {
        auto i = data_mp.find(key);
        if(i==data_mp.end()){
            if(data_mp.size()==capacity){
                auto iter = freq_mp[min_fq].back();
                data_mp.erase(iter);
                freq_mp[min_fq].pop_back();
                if(freq_mp[min_fq].empty())
                    freq_mp.erase(min_fq);
            }
            freq_mp[1].emplace_front(key);
            data_mp[key] = Node(value,1,freq_mp[1].begin());
            return ;
        }
        i->second.val = value;
        int freq = i->second.freq;
        i->second.freq++;
        freq_mp[freq].erase(i->second.node);
        if(freq_mp[freq].empty()){
            if(freq==min_fq)
                min_fq += 1;
            freq_mp.erase(freq);
        }
    }
};
// 355. 设计推特 https://leetcode.cn/problems/design-twitter/
/* 
 * 设计一个简化版的推特(Twitter)，可以让用户实现发送推文，关注/取消关注其他用户，能够看见关注人（包括自己）的最近 10 条推文。实现 Twitter 类：
 * Twitter() 初始化简易版推特对象 
 * void postTweet(int userId, int tweetId) 根据给定的 tweetId 和 userId 创建一条新推文。每次调用此函数都会使用一个不同的 tweetId. 
 * List<Integer> getNewsFeed(int userId) 检索当前用户新闻推送中最近  10 条推文的 ID 。新闻推送中的每一项都必须是由用户关注的人或者是用户自己发布的推文。推文必须 按照时间顺序由最近到最远排序 。
 * void follow(int followerId, int followeeId) ID 为 followerId 的用户开始关注 ID 为 followeeId 的用户。
 * void unfollow(int followerId, int followeeId) ID 为 followerId 的用户不再关注 ID 为 followeeId 的用户。
 */
typedef int User_Id;
typedef int Tweet_time;
typedef int Tweet_Id;
struct TweetNode{
    Tweet_Id id;
    Tweet_time time;
    TweetNode *next;
    TweetNode(int id, int time) : id(id), time(time), next(NULL) {}
    TweetNode() : id(-1), time(-1), next(NULL) {}
};
class User{
private:
    TweetNode *tweets;
public:
    unordered_set<User_Id> followees;
    User() : tweets(NULL) {}
    User(int id) : tweets(NULL) { followees.emplace(id); }
    void like(int like_id){
        followees.emplace(like_id);
    }
    void dislike(int like_id){
        int size = followees.size(), i = 0;
        unordered_set<int>::iterator followee = followees.begin();
        while (followee != followees.end())
            if (like_id == *followee)
                followees.erase(followee);
    }
    TweetNode *get_tweet(){
        return tweets;
    }
    void post_tweet(int tweet_id, int time){
        if (!tweets)
            tweets = new TweetNode(tweet_id, time);
        else
            tweets->next = new TweetNode(tweet_id, time);
    }
};
class Twitter{
private:
    unordered_map<User_Id, User *> users;
    Tweet_time time;
    TweetNode *mergeTwoLists(TweetNode *l1, TweetNode *l2){
        if (l1 == NULL)
            return l2;
        if (l2 == NULL)
            return l1;
        if (l1->time > l2->time){
            l1->next = mergeTwoLists(l1->next, l2);
            return l1;
        }
        else{
            l2->next = mergeTwoLists(l1, l2->next);
            return l2;
        }
    }
    TweetNode *deep_copy(TweetNode *head){
        if (!NULL)
            return NULL;
        TweetNode *root = new TweetNode(head->id, head->time), *p = root;
        while (head){
            p->next = new TweetNode(head->id, head->time);
            p = p->next;
        }
        p = root->next;
        delete (root);
        return p;
    }
    void deep_del(TweetNode *head){
        if (!head)
            return;
        TweetNode *p = head->next;
        while (p){
            delete (head);
            head = p;
            p = p->next;
        }
        delete (head);
    }
public:
    Twitter(){
        time = 0;
    }
    void postTweet(int userId, int tweetId){
        if (users.empty()){
            users.emplace(userId, new User(userId));
            users[userId]->post_tweet(tweetId, time);
            return;
        }
        unordered_map<User_Id, User *>::iterator usr_iter = users.find(userId);
        if (users.end() == usr_iter){
            users.emplace(userId, new User(userId));
            usr_iter->second->post_tweet(tweetId, time);
        }
        else
            usr_iter->second->post_tweet(tweetId, time);
        ++time;
    }
    vector<int> getNewsFeed(int userId){
        TweetNode *r = NULL;
        for (auto &followee : users[userId]->followees){
            r = mergeTwoLists(r, deep_copy(users[followee]->get_tweet()));
        }
        vector<int> res;
        TweetNode *h = r;
        while (res.size() <= 10 && r){
            res.push_back(r->id);
            r = r->next;
        }
        deep_del(h);
        return res;
    }
    void follow(int followerId, int followeeId){
        if (users.empty())
            return;
        unordered_map<User_Id, User *>::iterator usr_iter = users.find(followerId);
        if (users.end() != usr_iter)
            usr_iter->second->like(followeeId);
    }
    void unfollow(int followerId, int followeeId){
        if (users.empty())
            return;
        unordered_map<User_Id, User *>::iterator usr_iter = users.find(followerId);
        if (users.end() != usr_iter)
            usr_iter->second->dislike(followeeId);
    }
};
// 1206. 设计跳表 https://leetcode.cn/problems/design-skiplist/
// 不使用任何库函数，设计一个 跳表 。跳表 是在 O(log(n)) 时间内完成增加、删除、搜索操作的数据结构。跳表相比于树堆与红黑树，其功能与性能相当，并且跳表的代码长度相较下更短，其设计思想与链表相似。跳表中有很多层，每一层是一个短的链表。在第一层的作用下，增加、删除和搜索操作的时间复杂度不超过 O(n)。跳表的每一个操作的平均时间复杂度是 O(log(n))，空间复杂度是 O(n)。在本题中，你的设计应该要包含这些函数：bool search(int target) : 返回target是否存在于跳表中。void add(int num): 插入一个元素到跳表。bool erase(int num): 在跳表中删除一个值，如果 num 不存在，直接返回false. 如果存在多个 num ，删除其中任意一个即可。注意，跳表中可能存在多个相同的值，你的代码需要处理这种情况。
class Skiplist {
    #define MAX_SKIPLIST_LEVEL 32
    static const int S = 0xFFFF;
    static const int SP = 0x3FFF;
    struct Node{
        int val;
        Node* next[MAX_SKIPLIST_LEVEL];
        Node(){memset(next,0,sizeof(next));}
        Node(int val):val(val){memset(next,0,sizeof(next));}
    };
    Node* head;
    int level;
    static int randomLevel(){
        for(int i=1;i<MAX_SKIPLIST_LEVEL;++i){
            if((rand()&S)>SP)
                return i;
        }
        return MAX_SKIPLIST_LEVEL;
    }
public:
    Skiplist() {
        head = new Node();
        level = 0;
        srand(time(NULL));
    }
    bool search(int target) {
        Node* p = head;
        for(int i=level-1;i>=0;--i)
            while(p->next[i]&&p->next[i]->val<target)
                p = p->next[i];
        p = p->next[0];
        if(p&&p->val==target)
            return true;
        return false;
    }
    void add(int num) {
        vector<Node*> update(MAX_SKIPLIST_LEVEL, head);
        Node *curr = head;
        int lv = randomLevel();
        level = max(level, lv);
        Node *node = new Node(num);
        for (int i =level-1;i>=0;i--) {
            while (curr->next[i] && curr->next[i]->val < num)
                curr = curr->next[i];
            node->next[i] = curr->next[i];
            curr->next[i] = node;
        }
    }
    bool erase(int num) {
        vector<Node*> update(MAX_SKIPLIST_LEVEL,NULL);
        Node *cur = head,*node = NULL;
        for (int i=level-1; i>=0;i--) {
            while (cur->next[i] && cur->next[i]->val < num) 
                cur = cur->next[i];
            if(cur->next[i]&&cur->next[i]->val==num){
                node = cur->next[i];
                cur->next[i] = node->next[i];
            }
        }
        if(!node)
            return false;
        delete node;
        while (level>1&&!(head->next[level-1]))
            level--;
        return true;
    }
};