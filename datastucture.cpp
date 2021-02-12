#include<bits/stdc++.h>
using namespace std;
class MyStack {
private:
    queue<int> s;
public:
    MyStack() {
    }
    void push(int x) {
    s.push(x);
    }
    int pop(){
        int size=s.size();
        for(int i=0;i<size-1;++i){
            s.push(s.front());s.pop();
        }
        size=s.front();s.pop();
        return size;
    }
    int top() {
        int size=s.size();
        for(int i=0;i<size-1;++i){
            s.push(s.front());s.pop();
        }
        s.push(s.front());size=s.front();s.pop();
        return size;
    }
    bool empty() {
        return s.empty();
    }
};
class MyQueue {
public:
    MyQueue() {
    }
    void push(int x) {
    }
    int pop() {
    }
    int peek() {
    }
    bool empty() {
    }
};