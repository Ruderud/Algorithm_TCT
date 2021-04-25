# 2021-04-15

#최단거리 알고리즘은 1)다익스트라(dijkstra) 2)플로이드 워셜 3)벨만 포드 3가지로 되어있는데, 1,2만 다룬다.(빈출유형)
#최단거리 알고리즘은 그리디+다이나믹 알고리즘이 섞인 유형이다.


#다익스트라
#다익스트라 최단경로 알고리즘은 '특정노드'에서 출발-> '다른노드'로 가는 각각의 최단경로를 구해주는 알고리즘
#음의간선이 없을때 정상동작하므로, 음의 간선이 없는 현실의 최단경로(GPS)계산에 사용된다.

'''원리

1. 출발노드 설정
2. 최단거리 테이블 초기화
3. 미방문 노드중 최단거리가 가장 짧은 노드 선택
4. 해당노드를 거쳐 다른노드로 가는 비용을 계산하여 최단거리 테이블을 갱신  -> 이 방식때문에 그리디 알고리즘으로 분류
5. 3~4과정을 반복.

'''

#시간복잡도는 O(V^2), V=노드의 갯수
# +대부분의 코테는 최단 거리를 묻지, 최단경로를 출력해달라고는 하지않기에 그냥 거리값을 도출하는것 위주로만 정리한다.
# ++입력 데이터가 많기에, sys.std.readline()을 이용한다.

'''
import sys
input = sys.stdin.readline
INF = int(1e9) #10억을 의미한다. 1,000,000,000

n,m = map(int,input().split())       #노드갯수, 간선갯수 입력
start = int(input())                 #시작노드번호 입력
graph = [ [] for i in range(n+1) ]   #각 노드에 연결된 노드에 대한 정보를 입력할 리스트 생성
visited = [False] * (n+1)            #방문 여부 체크리스트
distance = [INF] * (n+1)             #초기거리 무한값으로 초기화 (start노드 -> i번 노드까지의 최단거리 데이터 저장 [i])

for _ in range(m):
    a,b,c = map(int,input().split())    #a번 노드에서 b번 노드로 가는 비용이 c이다.
    graph[a].append((b,c))

#print(graph)
#[[], [(2, 2), (3, 5), (4, 1)], [(3, 3), (4, 2)], [(2, 3), (6, 5)], [(3, 3), (5, 1)], [(3, 1), (6, 2)], []]
#       ㄴ> (2,2)는 1번노드에서 2번노드까지의 거리가 2 라는 의미의 배열. graph[1]을 호출시 [(2, 2), (3, 5), (4, 1)]로 (목표노드,묙표노드까지의 거리)리스트가 만들어짐.



def get_smallest_node():
    min_value = INF         #최소값을 처음에 무한으로 초기화한다.
    index = 0               #가장 최단거리가 짧은 노드(인덱스를 0으로 초기화한다.)
    for i in range(1,n+1):
        if distance[i] < min_value and not visited[i]:
            min_value = distance[i]
            index = i
    return index

def dijkstra(start):
    distance[start] = 0             #시작노드에 대해 초기화
    visited[start] = True
    for j in graph[start]:
        distance[j[0]] = j[1]       #검색할 노드의 start기준으로 입력된 graph값을 가져와서 distance list에 추가한다.
    for _ in range(n-1):            #시작노드를 제외한 전체 n-1개의 노드에 대해 반복 (n-1번 수행)
        now = get_smallest_node()   #처음검색할 노드값의 graph값을 distance에 입력했으므로, 최단거리가 짧은 노드를 now로 반환한다.
        visited[now] = True         #현재 최단거리가 가장짫은 노드를 가져오고, 방문처리함
        for j in graph[now]:
            cost = distance[now] + j[1]         #현재검색중인 노드와 연결된 다른 노드까지의 거리를 계산함. (start노드 <-> 검색하는 대상노드간 거리 <-> 대상노드에서 다른 노드까지의 거리)
            if cost < distance[j[0]]:               #현재노드를 거쳐서 다른노드로 이동하는 거리 < 현재노드에서 목표노드로 이동하는 거리 일경우
                distance[j[0]] = cost               #ex) 1->4->3 의 거리값은 4, 1->3은 5이므로 위에 해당. 이럴때는 1->4->3과 같이 현재 검색중인 노드를 거친 값(=cost)을 거리값에 입력한다.

dijkstra(start)

for i in range(1,n+1):
    if distance[i] == INF:      #도달할수없는 경우, 무한이라고 출력
        print("INFINITY")
    else:
        print(distance[i])
'''

#입력값과 결과값은 p232의 표와 그 결과에 따른다.





#개선된 다익스트라 알고리즘 -> 이를 이용하면 O(V^2)을 O(ElogV)로 줄일수있음. V는 노드갯수, E는 간선갯수다.
#시간을 줄인방법은 노드를 하나씩 선형적으로 검색하지않고, 최단거리가 가장 짧은 노드를 더욱 빠르게 찾는방법을 도입했기때문. (by Heap)

#Heap은 우선순위 Queue를 구현할때 사용한다. (스텍은 선입후출, 큐는 선입선출임을 상기) 따라서 우선순위 큐는 우선순위가 가장 높은데이터를 먼저 삭제한다.
#파이썬에서 호출시에는 heapq를 불러서 사용한다.(or PriorityQueue)

#최소힙은 값이 낮은데이터를 먼저 삭제하고, 최대힙은 값이 높은데이터를 먼저삭제한다. 다익스트라에서는 전자가 유리하겠지?
#-(음)의값을 입력해놓고 최소힙을 이용해서 꺼낸다음 다시 -부호를 붙여서 최대값을 가져오는방법도 있다.
#우선순위큐에서는 튜플(거리,노드)로 입력한다. 순서를 바꾼이유는 거리가 낮은순으로 가져와야하기때문.
#이를 이용했을때의 장점은 위처럼 get_smallest_node함수를 정의해서 쓸필요가 없다는 점이다. 자동으로 우선순위큐에서 가져올순서를 정해놨기 때문이다.

#개선 다익스트라
'''
import heapq                        #힙큐를 불러온다.
import sys

input = sys.stdin.readline
INF = int(1e9)

n,m = map(int,input().split())
start = int(input())
graph = [[] for _ in range(n+1)]
distance = [INF] * (n+1)            #여기까지는 동일

for _ in range(m):
    a,b,c = map(int,input().split())
    graph[a].append((b,c))

def dijkstra_rv(start):     #여기서부터 달라짐
    q=[]

    heapq.heappush(q,(0,start))         #시작노드에 대한 큐는 q=(거리=0,시작노드)로 입력한다.
    distance[start] = 0
    while q:                            #큐가 True일때 = 값이 있을떄
        dist, now = heapq.heappop(q)    #힙큐의 (거리,노드)값을 가져온다. 이는 낮은값순.
        
        if distance[now] < dist:        #현재검색중인 노드의 거리가 힙큐에 저장된 최소거리보다 클경우 패스시킨다.
            continue

        for i in graph[now]:
            cost = dist + i[1]          #현재 검색중인 노드(a)와 인접한 노드(b)의 까지의 거리값(cost)를 계산한다.
            if cost < distance[i[0]]:           #계산결과 그 거리값이 기존의 start에서 인접한노드(b)까지 가는 값보다 작을시 거리값을 갱신하고, 이를 힙큐에 추가한다
                distance[i[0]] = cost
                heapq.heappush(q,(cost,i[0]))

dijkstra_rv(start)              #이하는 동일

for i in range(1,n+1):
    if distance[i] == INF:      
        print("INFINITY")
    else:
        print(distance[i])
'''

'''
기존 다익스트라 계산과 개선 다익스트라 계산의 직관적인차이.
개선 다익스트라계산의 힙큐에 짧은것 순서로 가져오는것이 아닌, 모든 노드를 가져와서 패스하지않고 다 검사한다면 그것이 바로 기존 다익스트라 계산(전수검사)법이다.
이때문에 개선방법의 처리횟수의 최대는 O(ElogE) (E;간선갯수) 가 되는데, E는 항상 V^2 이하이기때문에(모든노드를 서로 연결시의 최대값) 그렇다.

'''




#폴로이드 워셜 알고리즘
#다익스트라는 "한지점"에서 다른노드들까지의 최소거리를 계산했다면, 폴로이드는 모든노드에서 다른 모든노드까지의 최소거리를 계산한다. 
#N개의 노드에 대해 각각O(N^2)의 연산(2차원리스트사용)을 하므로, 총 시간복잡도는 O(N^3)
#다이나믹 프로그래밍을 이용함. -> 점화식에 맞게 2차원 리스트를 갱신한다.

'''
k를 현재 탐색대상인 노드라고하고 A,B는 k를 제외한 나머지 N-1개중 2개를 순서를 고려한방법으로 고른 조합이다.(n-1P2)
min(A->B, A->k->B)을 판단해서 갱신해 나가는것이 기본적인 방법이다. 

K 1 2 3 4
1 x x x x
2 x x o o 
3 x o x o
4 x o o x -> k가 1일때 이런느낌으로 (k,k)의 종/횡/대각선을 제외한 부분(O)을 위의 min점화식을 이용, k를 전 노드에 대해서 점화식적 검사를 수행.
(O구간의 갯수는 n-1P2 = 3P2 = 6임을 알수있다.)
'''


#폴로이드 워셜 알고리즘
'''
INF = int(1e9)
n = int(input())        #노드갯수
m = int(input())        #간선갯수
graph = [[INF] * (n+1) for _ in range(n+1)]

for a in range(1,n+1):
    for b in range(1,n+1):
        if a==b:                #자기자신으로 가는 노드(k->k)는 0의값으로 배정.
            graph[a][b] = 0

for _ in range(m):
    a,b,c = map(int,input().split())
    graph[a][b] = c             #a->b의 거리비용은 c

for k in range(1,n+1):
    for a in range(1,n+1):
        for b in range(1,n+1):
            graph[a][b] = min(graph[a][b],graph[a][k]+graph[k][b])      #a->b vs a->k->b

for a in range(1,n+1):
    for b in range(1,n+1):
        if graph[a][b] == INF:          #도달불가는 무한으로 출력
            print('INFINITY',end=" ")
        else:
            print(graph[a][b],end=' ')
    print()     #1 -> 1,2,3,4 // 2->1,2,3,4 ... 꼴로 보기좋게 출력하기위함
'''

'''결과값
0 4 8 6 
3 0 7 9 
5 9 0 4 
7 11 2 0 
(a,b)에 해당하는값이 a에서 b까지 가는데 걸리는 최단거리값.
'''


#------------------------------

#실전문제2 미래 도시
'''
start지정 -> 다익스트라방법 (x)

특징은 쌍방향이동이 가능하며, 무조건 k를 먼저 가야지, x를 들렀다가 k를 갔다가 다시 x를 가는것은 -1을 반환하게끔함.

'''

"""#해설 -> 이런유형은 플로이드 워셜을 사용함.

INF = int(1e9)

n,m = map(int,input().split())
graph = [[INF] * (n+1) for _ in range(n+1)]
'''
[
    [INF INF INF INF INF INF]
    [INF INF INF INF INF INF]
    [INF INF INF INF INF INF]
    [INF INF INF INF INF INF]
    [INF INF INF INF INF INF]
    [INF INF INF INF INF INF]
]
꼴
'''

for a in range(1,n+1):
    for b in range(1,n+1):
        if a == b:
            graph[a][b] = 0  #위 그래프에서 좌상->우하에 해당하는 대각선을 0으로 만든다 = 자기자신으로 가는 경로길이는 0으로 할당

print(graph)

for _ in range(m):
    a,b = map(int,input().split())
    graph[a][b] = 1
    graph[b][a] = 1                 #a<->b간 경로길이는 1로 만든다.

print(graph)
'''
[
    [INF, INF, INF, INF, INF, INF],
    [INF, 0,   1,   1,   1,   INF],
    [INF, 1,   0,   INF, 1,   INF],
    [INF, 1,   INF, 0,   1,   1  ],
    [INF, 1,   1,   1,   0,   1  ],
    [INF, INF, INF, 1,   1,   0  ]
]
'''

x,k = map(int,input().split())

for k in range(1,n+1):
    for a in range(1,n+1):
        for b in range(1,n+1):
            graph[a][b] = min(graph[a][b], graph[a][k]+graph[k][b])  #플로이드 워셜 알고리즘 수행. a->b까지의 최단경로 vs a->k->b

print(graph)
'''
[
    [INF, INF, INF, INF, INF, INF], 
    [INF, 0, 1, 1, 1, 2],
    [INF, 1, 0, 2, 1, 2],
    [INF, 1, 2, 0, 1, 1],
    [INF, 1, 1, 1, 0, 1],
    [INF, 2, 2, 1, 1, 0]
]
'''
distance = graph[1][k] + graph[k][x]   #1->k까지의 최단경로 + k->x까지의 최단경로 = 1이 k를 들렀다가 x로 가는 경로의 최소값.

if distance >= INF:
    print("-1")
else : 
    print(distance)
"""


#실전문제3 전보 (64:06)
#해석의 차이임. 나는 정보교환이 이루어지기 위해서는 서로 다른 두 노드 사이에 단방향 경로가 서로를 향하게끔 2개가 있어야한다고 생각했음(X->Y and Y->X)
#하지만 문제는 단방향 경로존재시 정보전달이 단방향으로 가능하다는 조건으로 문제를 풀었음.
'''
다익스트라 -개선써야할듯. 도시갯수3만, 통로갯수 20만이라 많음.

start도시에서 각도시에 도착하는데 걸리는 최단경로 리스트를 구하고, 그중 INF값이 아닌값 갯수 -1(자신은 제외해야하므로) = 수신하는도시
도달하는데 걸리는 시간중 최대값 = 총걸리는시간

이때 중요한거는 start에서 직접연결이 아니라, 다른 도시를 거쳐서 갈경우에는 서로 왕복통로가 존재해야하며 이걸 검사하는게 필요함 (핵심)

'''

'''
import heapq
import sys

input = sys.stdin.readline
INF = int(1e9)

n,m,c = map(int,input().split())        #c=start
graph = [ [] for _ in range(n+1)]
distance = [INF] * (n+1)
for _ in range(m):
    x,y,z = map(int,input().split())
    graph[x].append((y,z))

def dijkstra_sol(start):
    q=[]
    heapq.heappush(q,(0,start))
    distance[start] = 0
    while q:
        dist, now = heapq.heappop(q)
        if distance[now] < dist:        #현재검색중인게 비효율적이면 검색안한다. +여기다가 graph에서 서로 path가 이어져있는지 걸러내는거 필요
            continue

        for i in graph[now]:
            cost = dist + i[1]
            if now != start :
                                    #i[0] 검사대상경로가 향하는 목적지 노드값
                path_check=[]
                for p in range(len(graph[i[0]])):
                    path_check.append(graph[i[0]][p][0])
                if now not in path_check:
                    continue

            if cost < distance[i[0]]:
                distance[i[0]] = cost
                heapq.heappush(q,(cost,i[0]))

dijkstra_sol(c)

unable = distance.count(INF)

print(len(distance)-unable-1,end=' ')

result = []
for i in range(1,n+1):
    if distance[i] == INF:      
        continue
    else:
        result.append(distance[i])
print(max(result))
'''

'''#해설->이거 뭔가좀 이상함. 양방향 path가 존재해야 전달이 가능하다는 조건이있었는데 이걸 고려하지않았음.
import heapq
import sys

input = sys.stdin.readline
INF = int(1e9)

n,m,start = map(int,input().split())        #c=start
graph = [ [] for _ in range(n+1)]
distance = [INF] * (n+1)
for _ in range(m):
    x,y,z = map(int,input().split())
    graph[x].append((y,z))

def dijkstra_solution(start):
    q=[]
    heapq.heappush(q,(0,start))
    distance[start] = 0
    while q:
        dist, now = heapq.heappop(q)
        if distance[now] < dist:
            continue
        for i in graph[now]:
            cost = dist + i[1]
        if cost < distance[i[0]]:
            distance[i[0]] = cost
            heapq.heappush(q,(cost,i[0]))   #여기까지 기존 개선 다익스트라와 동일

dijkstra_solution(start)

count = 0
max_distance = 0
for d in distance:
    if d != INF :       #INF값을 걸러낸다
        count += 1      #도달가능한 노드의 갯수 (시작노드가 포함되어있다는것을 인지!)...A
        max_distance = max(max_distance,d)  #도달경로거리가 가장긴것을 구한다.
    
print(count-1,max_distance)         #여기서 시작노드가 포함되어있는 count에서 1을 빼주어서 보정함.
'''