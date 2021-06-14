#37 플로이드

'''#해설

INF = 1e9

n = int(input())
m = int(input())

graph = [[INF] * (n+1) for _ in range(n+1)]

for a in range(1,n+1):
    for b in range(1,n+1):
        if a == b:
            graph[a][b] = 0

for _ in range(m):
    a,b,c = map(int,input().split())
    if c < graph[a][b]:
        graph[a][b] = c

for k in range(1,n+1):          #플로이드 알고리즘-> 중간에 k노드를 거쳐서 가는값이 더 작을경우를 반영함. 여기서 k는 중간에 k번노드를 지나갈떄의 경로길이를 계산하기위한 기준
    for a in range(1,n+1):      # +노드를 1~n번까지 한번씩 돌리면 최단경로가 무조건 나타난다. 1->2->1->3처럼 한번 들렀던 노드에 대한 검색을 할 필요가 없기 때문이다.
        for b in range(1,n+1):
            graph[a][b] = min(graph[a][b], graph[a][k]+graph[k][b])

for a in range(1,n+1):
    for b in range(1,n+1):
        if graph[a][b] == INF:
            print(0, end=' ')
        else:
            print(graph[a][b], end=' ')
    print()
'''


#38 정확한 순위
#대상 노드 i에 대해, 나머지 n-1개의 노드가 i노드까지 가는데 걸리는 거리값이 모두 INF가 아니라면, 해당 노드는 확정순위를 가지는 사람이다.

'''#내답: 시간 = 1만 + 1.25억 이하 + 2500만 + 500 = 1.5억(시간초과)? 
n, m = map(int,input().split())

array = [[0] * (n+1) for _ in range(n+1)]

for _ in range(m):
    a,b = map(int,input().split())
    array[a][b] = 1

for a in range(1,n+1):
    for b in range(1,n+1):
        if a == b:                  #자기자신으로 향하는것은 제외
            continue
        if array[a][b] != 0:
            for i in range(1,n+1):  #다른노드로 넘어갈수있는것들만 for문을 돌리는거기때문에 1.25억보다 적긴할텐데 얼마나 적어질지는 모르겠음
                if array[b][i] != 0:
                    array[a][i] = array[a][b] + array[b][i]

for x in range(1,n+1):              #교차점은 x표시
    array[x][x] = 'x'
    for y in range(1,n+1):          #대칭형으로 만들어주기
        array[x][y] = max(array[x][y], array[y][x])

count=0
for x in range(1,n+1):
    if 0 not in array[x][1:]:
        count+=1

print(count)
'''

'''#해설
INF = int(1e9)

n,m = map(int,input().split())
graph = [[INF] * (n+1) for _ in range(n+1)]

for a in range(1,n+1):
    for b in range(1,n+1):
        if a==b:
            graph[a][b] = 0

for _ in range(m):
    a,b = map(int, input().split())
    graph[a][b] = 1                         #여기까진 똑같음

for k in range(1,n+1):
    for a in range(1,n+1):
        for b in range(1,n+1):
            graph[a][b] = min(graph[a][b], graph[a][k]+graph[k][b])

result = 0

for i in range(1,n+1):
    count = 0
    for j in range(1,n+1):
        if graph[i][j] != INF or graph[j][i] != INF:        #그래프에서 (i,i)를 교차중심으로 하는 십자 항목들에 대한 검사 실행 -> [i][j] [j][i]둘중 하나라도 도달가능한 값이 존재시 정확한 순서가 정해짐
            count += 1
    if count == n:
        result+=1

print(result)
'''


#39 화성탐사 







#40 숨바꼭질
'''#플로이드 워셜 문제 (x) 이렇게하면 n = 20000이라서 시간제한에 걸림... -> 다익스트라를 써야함

import heapq
from sys import maxsize

INF = int(1e9)
n,m = map(int,input().split())
start = 1
graph = [[] for _ in range(n+1)]
distance = [INF] * (n+1)

for _ in range(m):
    a,b = map(int,input().split())
    graph[a].append((b,1))
    graph[b].append((a,1))

def dijkstra_rv(start):
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
                heapq.heappush(q,(cost,i[0]))

dijkstra_rv(start)

# del distance[0]
# distance_max = max(distance)
# recommend_index = distance.index(distance_max)
# others = distance.count(distance_max)
# print(recommend_index+1, distance_max, others)


#해답: 다익스트라를 푸는 과정은 똑같음. 마지막 결과 출력이 쪼금더 효율적

#다익스트라 로직은 동일
max_dist = 0
max_node = 0
result = []

for i in range(1,n+1):
    if max_dist < distance[i]:
        max_node = i
        max_dist = distance[i]
        result = [max_node]
    elif max_dist == distance[i]:
        result.append(i)

print(max_node, max_dist, len(result))
'''