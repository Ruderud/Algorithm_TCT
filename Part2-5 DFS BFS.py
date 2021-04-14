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

'''
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

visited = [False] * 9 #첫 공백과 8개의 노드를 리스트 자료형으로 만들었다.

dfs(graph, 1 ,visited)

print()
'''

#BFS

'''
from collections import deque

def bfs(graph,start,visited):
    queue = deque([start]) #큐 구현을 위해 deque라이브러리 사용

    visited[start]=True #현재노드 방문처리

    while queue:
        v= queue.popleft() #큐에서 '앞에서부터' 하나의 원소를 뽑아서 출력한다. -> 선입선출!
        print(v,end=' ')

        for i in graph[v]:      #각 노드에 연결된 첫번째 항목부터(낮은수부터) 검사를 시작한다
            if not visited[i]:
                queue.append(i)     #방문하지 않은 노드일경우 큐의 '맨뒤'에 추가하고 해당 노드를 방문처리한다.
                visited[i]=True

graph=[
    [],
    [2,3,8],
    [1,7],
    [1,4,5],
    [3,5],
    [3,4],
    [7],
    [2,6,8],
    [1,7]
]

visited = [False] * 9

dfs(graph,1,visited)
'''


#실전문제3 음료수 얼려 먹기
'''
메모
1,1, 1,2...등으로 검색할 좌표를 지정하자.

해당노드를 bfs를 이용하여 방문순서를 만들고, 만들어진 방문순서 리스트 한덩어리 = 아이스크림 1개이다.

총 방문순서 리스트 갯수를 count해서 출력한다.

'''

'''입력 구현끝
n,m = map(int,input().split())

d = [[0] * m for _ in range(n)]

tray = []
for _ in range(n):
    tray.append(list(map(str,input())))
    if n == len(tray):
        break

print(n,m)
print(d)
print(tray)

n,m = 4,5
d=[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
tray = [['0', '0', '1', '1', '0'], ['0', '0', '0', '1', '1'], ['1', '1', '1', '1', '1'], ['0', '0', '0', '0', '0']]

graph = []


for x in range(m):
    for y in range(n):
        if x+1>m or x-1<0 or y+1>n or y-1<n:
            continue
        if tray[y][x+1] == '0':
            graph.append(tray[y][x+1])
        if tray[y][x-1] == '0':
            graph.append(tray[y][x-1])
        if tray[y+1][x] == '0':
            graph.append(tray[y+1][x])
        if tray[y-1][x] == '0':
            graph.append(tray[y-1][x])

print(graph)
'''

#해설
'''
n,m = map(int,input().split())

graph = []
for i in range(n):
    graph.append(list(map(int,input())))

print(graph)

def dfs(x,y):
    if x<=-1 or x>=n or y<=-1 or y>=m:      #범위벗어나면 false출력
        return False
    if graph[x][y] == 0:
        graph[x][y] = 1                     #방문안했으면 1로 바꿈(i,j를 통해서 dfs검사시 이중으로 0인 덩어리를 세는것을 막기위함)
        dfs(x-1,y)                          #사방의 반칸들에 대한 처리는 재귀화로 처리함. 
        dfs(x+1,y)
        dfs(x,y-1)
        dfs(x,y+1)
        return True
    return False                            #이외의 표내의 1들은 false값 반환

result = 0
for i in range(n):
    for j in range(m):
        if dfs(i,j) == True:                #사방에 이어진 0덩어리를 다세면 +1을 한다.
            result += 1

print(result)
'''


#실전문제4 미로 탈출
'''
맵에서 첫시작에서 인접한 1을 가져오고 거기로이동.

1,1에서 2,1을 가져오고, 1,1에서 갈곳이 없으면 빠져나감.

다음으로 2,1을 가져오고 갈곳이 2,2가 있으므로 가져옴
(반복)
2,2에서 2,3을 가져옴

2,3에서 갈수잇는 1,3과 2,4를 가져옴 

1,3을 먼저 수색->갈수있는곳이 없음 ->false

2.4를 수색-> 갈곳이있음->2,5->2,6 ->3,6->4,6

4,6에서 문제임. 갈수있는 곳이 많은데 최단경로가 5,6을 거쳐서 6,6을 가면된다는것을 어떻게 정의?




'''

'''입력구현끝
n,m = map(int,input().split())

graph = []
for i in range(n):
    graph.append(list(map(int,input())))

print(n,m)
print(graph)
'''
print('-'*50)


'''
n,m=5,6
graph = [
    [1, 0, 1, 0, 1, 0], 
    [1, 1, 1, 1, 1, 1], 
    [0, 0, 0, 0, 0, 1], 
    [1, 1, 1, 1, 1, 1], 
    [1, 1, 1, 1, 1, 1]
    ]

from collections import deque

def bfs(graph,start,visited):
    queue = deque([start])
    print(queue)
    visited[start] = True

    while queue:
        for x in range(n):
            for y in range(m):
                
for x in range(m):
    for y in range(n):
        if x+1>m or x-1<0 or y+1>n or y-1<n:
            continue
        if tray[y][x+1] == '0':
            graph.append(tray[y][x+1])
        if tray[y][x-1] == '0':
            graph.append(tray[y][x-1])
        if tray[y+1][x] == '0':
            graph.append(tray[y+1][x])
        if tray[y-1][x] == '0':
            graph.append(tray[y-1][x])



        v = queue.popleft()
        print(v,end=' ')




        for i in graph[v]:
            if not visited[i]:
                queue.append(i)
                visited[i]=True

visited= [False] * (n*m)  #나중에 true수 = 이동한 횟수

bfs(graph,graph[0][0],visited)

print(visited)

'''


#해설

from collections import deque

n,m = map(int,input().split())

graph = []
for i in range(n):
    graph.append(list(map(int,input())))

#상하좌우
dx=[-1,1,0,0]
dy=[0,0,-1,1]

def bfs(x,y):
    queue = deque()
    queue.append((x,y))

    while queue:  #queue가 빌때까지 반복함.
        x,y = queue.popleft()   #선입선출
        for i in range(4):      #선출된 좌표의 4방향검색
            nx = x + dx[i]
            ny = y + dy[i]

            if nx<0 or ny<0 or nx>=n or ny>=n:      #선출좌표의 4방향이 그레프의 바깥부분일때 예외처리
                continue

            if graph[nx][ny] == 0:                  #선출좌표의 4방향이 0일때 = 벽일때 예외처리
                continue

            if graph[nx][ny] == 1:                  #선출좌표 4방향중 갈수있는 길(1)이 있을때
                graph[nx][ny] = graph[x][y] + 1     #해당 갈수있는 길의 숫자에 +1을 한다.( 1->2->3...꼴)
                queue.append((nx,ny))
    return graph[n-1][m-1]                  #목적지에 도착했을때의 좌표위치값(=이동하는데 걸린횟수)을 출력한다

print(bfs(0,0))








print('-'*50)
