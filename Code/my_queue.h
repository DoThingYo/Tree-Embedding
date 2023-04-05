#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <assert.h>

typedef struct my_queue {
	//stores the elements
	int* arr;

	//stores its maximum size
	int capacity;

	//stores the front position
	int front;

	//stores the rear position
	int rear;
}Queue;

bool isEmpty(Queue* Q)
{
	return Q->front == Q->rear;
}

bool isFull(Queue* Q)
{
	return Q->front == (Q->rear + 1) % Q->capacity;
}


int NNZ(Queue* Q)
{
	if(isEmpty(Q)){
		return 0;
	}
	if(isFull(Q)){
		return Q->capacity;
	}

	int front = Q->front;

	int rear = Q->rear;

	if(front < rear){
		return rear - front;
	}
	else{
		return Q->capacity - (front - rear);
	}
}


void enqueue(Queue *Q, int e)
{
	/*if(isFull(Q))
	{
		printf("error Queue full");
		throw "The queue is full, no room to enqueue!";
	}*/
	assert(!isFull(Q));
	Q->arr[Q->rear] = e;
	Q->rear = (Q->rear + 1) % Q->capacity;
}



int dequeue(Queue* Q)
{
	/*if(isEmpty(Q))
	{
		printf("error Queue empty");
		throw "The Queue is empty, no element to deque!";
	}*/
	assert(!isEmpty(Q));
	int e = Q->arr[Q->front];
	Q->front = (Q->front + 1) % Q->capacity;
	return e;
}


int get_front(Queue* Q)
{
	return Q->arr[Q->front];
}

//Display each element in the Queue
void display(Queue* Q) {
	int start = Q->front;
	int end = Q->rear;
	printf("Q->front = %d, Q->rear = %d\n", start, end);
	for (int i = start; i != end; i++) 
	{
		printf("%d->", Q->arr[i]);
	}
	printf("\nEnd!");
	printf("\n");
}


