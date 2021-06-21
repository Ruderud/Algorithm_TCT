#Q41 여행 계획

'''#내답: 플로이드워셜 알고리즘을 이용하여 서로 도달가능한 최단거리 그래프를 만들고, 여행계획대로 움직이면서 INF를 만나면 NO, 전부 무사히 통과하면 YES를 출력
#시간복잡도:  500^3 = 125,000,000 ->?? 1억넘어서 안되네...
n,m = map(int,input().split())
graph = []
INF = 1e9
for i in range(n):
    line = list(map(int, input().split()))
    for j in range(len(line)):
        if i != j and line[j] == 0:
            line[j] = INF
    graph.append(line)
plan = list(map(int, input().split()))

for k in range(n):
    for a in range(n):
        for b in range(n):
            if a==b:
                continue
            graph[a][b] = min(graph[a][b], graph[a][k]+graph[k][b])

start = plan[0]
available = []
for target in plan[1:]:
    if graph[start][target] == INF:
        available.append('NO')
        break
    else:
        start = target
available.append('YES')

print(available[0])
'''

#해설:plan에 해당하는 노드 요소들이, 서로 이동가능하도록 그룹되어있는 노드 그룹내에 전부 속하는지를 확인하는 방법을 사용
#parent = [0, 1, 2, 3, 4, 5...n]까지 만들어놓고, 맵을 한줄씩 입력할때마다, 연결되어있는 노드중 가장 작은번호의 노드(여기서는 1이되겠다)를 그룹대표번호로 선정,
#해당 노드와 연결된 노드들을 전부 대표 노드번호(1)화 한다. 그래서 1~n번까지 각노드가 어떤 그룹에 속해있는지 확인할 수 있게끔한 후, 
#plan의 노드가 한가지 그룹내에 다 들어있는지 확인해서 yes,no를 가린다.

#여기서 parent가 최종적으로 대표노드가 될때까지 재귀적으로 거슬러 올라가게함
def find_parent(parent, x):
    if parent[x] != x:
        parent[x] = find_parent(parent, parent[x])
    return parent[x]

#두 노드가 연결되어있을때, 더작은 노드를 대표번호화 하는과정
def union_parent(parent, a, b):
    a = find_parent(parent,a)
    b = find_parent(parent,b)
    if a<b:
        parent[b] = a
    else:
        parent[a] = b

n,m = map(int,input().split())
parent = [0] * (n+1)

#처음에는 각 노드번호를 대표로하는 n개의 그룹꼴로 만든다.
for i in range(1,n+1):
    parent[i] = i

#한줄씩 순차적으로 입력하면서, i번째줄의 j번째 노드를 하나씩 검사하여 그값이 1일때(연결되어있을때), i번 노드와 j번노드중 작은번호의 노드값으로 병합한다
for i in range(n):
    data = list(map(int, input().split()))
    for j in range(n):
        if data[j] == 1:
            union_parent(parent, i+1, j+1)
        
plan=list(map(int,input().split()))

result = True

#여행계획에서 하나씩 번호를 목적지노드를 가져와서, 이것의 최종 대표노드가 방문 노드 전체에 대해 전부 같은지 확인한다.
for i in range(m-1):
    if find_parent(parent, plan[i]) != find_parent(parent, plan[i+1]):
        result=False

if result:
    print('YES')
else:
    print('NO')

