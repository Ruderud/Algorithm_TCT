#2021-04-12

'''
자료구조
오버플로:데이터가 꽉찬 상태에서 삽입(push)할때 발생
언더플로:데이터가 하나도없는 상태에서 삭제(pop)할때 발생

-

스택 = 박스쌓기 -> 선입후출, 후입선출 구조
append()는 리스트 최후단에 추가,
pop()은 리스트 최후단에서 삭제

stack = [5,2,3,1]일때
print(stack) -> 최하단 원소부터 출력 : [5,2,3,1]
print(stack[::-1]) -> 최상단 원소부터 출력 : [1,3,2,5]

큐(Queue) = 놀이공원줄 = 공정한 자료구조 = 선입선출
append는 같음
pop대신 popleft를 사용 (list의 왼쪽에서 뺀다->선입선출)
파이선에서 큐 구현시, collections 모듈의 deque 자료구조를 활용함.

from collections import deque
queue = deque()

이는 일반 리스트자료형에비해 데이터 넣고빼는 속도가 빠르고 효율적임. 또한 queue라이브러리 보다 간단하기도함.

-

재귀함수
함수내에서 자기자신을 다시금 호출하는 함수.

def recursive_f(i):
    if i == 100:
        return
    print(i,'번째 재귀함수에서',i+1,'번째 재귀함수를 호출합니다.')
    recursive_f(i+1)
    print(i,'번째 재귀함수를 종료합니다.')
recursive_f(1)

이러한 재귀함수는 펙토리얼을 계산할때 유용함.

def factorial(n):               #일반 정의계산
    result = 1
    for i in range(1,n+1):
        result*=i
    return print(result)

def factorial_re(n):            #재귀 정의계산
    if n<=1:
        return 1
    return n*factorial_re(n-1)

print(factorial_re(5))

이는 점화식꼴로 이해하자.

-

DFS : Depth-First Search -> 깊은부분을 우선적으로 탐색하는 알고리즘 (탐색순서는 137p확인)

그래프는 노드와 간선으로 이루어지고, 노드를 정점이라고한다.
그래프탐색 : 하나의 노드를 시작으로 다수의 노드를 방문하는것.
두노드가 간선으로 이어져있을때 이를 '두 노드는 인접하다' 라고함 (O-O)

그래프 표현방법은 크게 2가지 - 인접행렬(adjacency Matrix;2차원배열로 그래프의 연결관계표현), 인접리스트(adj. list;리스트로 연결관계 표현) - 로 나뉨
직접 간선으로 이어져있지 않은 노드는 무한의 비용이라고 작성함 #책 135,136참조

INF = 99999999999

#인접행렬로 표기 -> 노드갯수가 많을수록 메모리 사용 낭비
graph = [
    [0,7,5],
    [7,0,INF],
    [5,INF,0]
]
print(graph)

#인접리스트로 표기 -> 메모리는 적게사용. 그러나 느리다->왜냐? 연결데이터를 하나씩 확인해야하기 때문
graph = [[] for _ in range(3)]

graph[0].append((1,7))  #(노드,거리), 0에서 1,2를 갈때
graph[0].append((2,5))

graph[1].append((0,7))  #1에서 0갈때

graph[2].append((0,5))  #2에서 0갈떄

print(graph)



BFS : Breadth First Search 너비 우선탐색 -> 가까운노드부터 탐색한다. 선입선출방식인 큐 자료구조를 이용한다.
큐 기준이기에 구현이 간단하고, deque라이브러리 사용이 추천된다.
또한 시간도 O(N)시간이 수행되며 일반적으로 DFS보다 빠르다.


'''

#DFS 예제

def dfs(graph,v,visited):
    visited[v] = True
    print(v,end=' ')
    for i in graph[v]:
        if not visited[i]:
            dfs(graph,i,visited)

graph=[
    [],                 # 그림의 0번노드는 없으며, 이어진것도 없으므로 빈것으로 처리.
    [2,3,8],            # 그림의 1번노드와 인접한 노드리스트
    [1,7],              # 그림의 2번노드와 인접한 노드리스트...
    [1,4,5],
    [3,5],
    [3,4],
    [7],
    [2,6,8],
    [1,7]
]

visited = [False] * 9

dfs(graph, 1 ,visited)



#BFS

from collections import deque

def bfs(graph,start,visited):
    queue = deque([start]) #큐 구현을 위해 deque라이브러리 사용

    visited[start]=True #현재노드 방문처리

    while queue:
        v= queue.popleft() #큐에서 하나의 원소를 뽑아서 출력한다.
        print(v,end=' ')

        for i in graph[v]:
            if not visited[i]:
                queue.append(i)
                visited[i]=True