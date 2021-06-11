#37 플로이드

#해설

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


