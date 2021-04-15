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


import sys
input = sys.stdin.readline
INF = int(1e9) #10억을 의미한다. 1,000,000,000

n,m = map(int,input().split())       #노드갯수, 간선갯수 입력
start = int(input())                 #시작노드번호 입력
graph = [ [] for i in range(n+1) ]   #각 노드에 연결된 노드에 대한 정보를 입력할 리스트 생성
visited = [False] * (n+1)            #방문 여부 체크리스트
distance = [INF] * (n+1)             #초기거리 무한값으로 초기화

for _ in range(m):
    a,b,c = map(int,input().split())    #a번 노드에서 b번 노드로 가는 비용이 c이다.
    graph[a].append((b,c))

def get_smallest_node():
    min_value = INF
    index = 0               #가장 최단거리가 짧은 노드(인덱스)
    for i in range(1,n+1):
        if distance[i] < min_value and not visited[i]:
            min_value = distance[i]
            index = i
    return index

def dijkstra(start):
    distance[start] = 0             #시작노드에 대해 초기화
    visited[start] = True
    for j in graph[start]:
        distance[j[0]] = j[1]
    for i in range(n-1):            #시작노드를 제외한 전체 n-1개의 노드에 대해 반복
        now = get_smallest_node()
        visited[now] = True         #현재 최단거리가 가장짫은 노드를 가져오고, 방문처리함
        for j in graph[now]:
            cost = distance[now] + j[1]         #현재노드와 연결된 다른 노드를 확인
            if cost < distance[j[0]]:               #현재노드를 거쳐서 다른노드로 이동하는 거리 < 현재노드에서 목표노드로 이동하는 거리 일경우
                distance[j[0]] = cost

dijkstra(start)

for i in range(1,n+1):
    if distance[i] == INF:      #도달할수없는 경우, 무한이라고 출력
        print("INFINITY")
    else:
        print(distance[i])