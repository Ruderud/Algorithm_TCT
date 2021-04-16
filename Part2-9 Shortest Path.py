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