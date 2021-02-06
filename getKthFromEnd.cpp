#include<iostream>
using namespace std;
struct ListNode {
    int val;
    ListNode *next;
    ListNode():next(NULL){}
    ListNode(int x) : val(x), next(NULL) {}
};
ListNode* getKthFromhead(ListNode* head, int k) {//头节点序号为0
    for(int i=1;i<=k;++i){
        head=head->next;
    }
    return head;
}
ListNode* getKthFromEnd(ListNode* head, int k) {
    ListNode *p1=head, *p2=getKthFromhead(head,k);
    for(;p2;p2=p2->next,p1=p1->next);
    return p1;
}
ListNode* removeNthFromEnd(ListNode* head, int n) {
    ListNode *p=head;
    for(int i=1;i<=n;++i)
        p=p->next;
    if(!p){
        p=head->next;delete(head);return p;
    }
    ListNode *ptr=head;
    for(;ptr;ptr=ptr->next,p=p->next);
    p=ptr->next;
    ptr->next=p->next;delete(p);
    return head;
}