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

'''#해설:plan에 해당하는 노드 요소들이, 서로 이동가능하도록 그룹되어있는 노드 그룹내에 전부 속하는지를 확인하는 방법을 사용
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
'''

#Q42 탑승구

'''#내답. 시간초과난다 -> 최대 10억이라 bisect를 써보려했는데 잘안댐
g = int(input())
p = int(input())
gi = []
for _ in range(p):
    gi.append(int(input()))

gi_landing = [True] * g

count=0

for land in gi:
    
    if True in gi_landing[:land]:
        for i in range(land-1,-1,-1):
            if gi_landing[i]:
                gi_landing[i] = False
                count+=1
                break
    else:
        break

print(count)
'''

#해설: 서로소집합알고리즘을 사용했음.
#도킹/비도킹 게이트로 구분해서, n번 게이트 도킹시 바로 왼쪽(n-1) 게이트의 그룹으로 합집합연산을 수행해서 묶는다. (n번노드는 항상 갈수있는 가장 큰 번호의 게이트)
#그렇기에 최종적으로 가상의 0번게이트와 연합되었다 = 더이상 도킹불가능하다 의 의미를 가짐. -> 도킹하려는 비행기의 최종 도달게이트가 0번일시 중단하고 도킹된 비행기 갯수 출력

'''
def find_parent(parent,x):
    if parent[x] != x:
        parent[x] = find_parent(parent,parent[x])
    return parent[x]

def union_parent(parent, a, b):
    a = find_parent(parent, a)
    b = find_parent(parent, b)
    if a < b:
        parent[b] = a
    else:
        parent[a] = b

g = int(input())
p = int(input())

parent = [0] * (g+1)

for i in range(1, g+1):                     #최종도달게이트를 자기자신으로 초기화
    parent[i] = i

result = 0

gi = []
for _ in range(p):
    gi.append(int(input()))

for land in gi:
    data = find_parent(parent, land)        #현재 도킹하려는 비행기의 최종도달게이트를 확인
    if data == 0:                           #만일 최종도달게이트가 0이면 더이상 도킹불가능하다는 의미 -> 중단
        break
    union_parent(parent, data, data-1)      #도킹 가능시, 최종도달게이트와, 최종도달게이트 바로 옆 게이트를 묶어서 최종도달게이트를 갱신시킨다.
    result+=1

print(result)
'''


'''#Q43
#0번 노드부터 순차적으로 노드와 노드를 연결해서 하나의 노드로 만든다.
#이때 새로 편입할 노드가 그룹에 속하기 위해 사용되는 최소한의 값만 선택해서 추가하고, 이를 min_cost에 합산한다
#모든 노드가 연결되었을때, 전체 max_cost - min_cost를 결과로 출력 
# => 크루스칼 알고리즘(산장트리 구현; 최소비용의 사이클 생성확인); O(NlogN)이므로 100만의 시간복잡도

def find_parent(parent,x):
    if parent[x] != x:
        parent[x] = find_parent(parent, parent[x])
    return parent[x]

def union_parent(parent,a,b):
    a = find_parent(parent,a)
    b = find_parent(parent,b)
    if a<b:
        parent[b] = a
    else:
        parent[a] = b

n, m = map(int, input().split())
roads = []
parent = [0] * n
max_cost = 0
min_cost = 0

for _ in range(m):
    a = list(map(int,input().split()))
    roads.append([a[2],a[0],a[1]])         #[비용,노드a,노드b]순으로 전환 저장

for i in range(n):                         #부모노드 자기자신으로 초기화
    parent[i] = i

roads.sort()                               #비용 오름차순으로 정렬

for road in roads:                         #낮은비용의 길부터 하나씩 순차적으로 가져옴.
    cost,a,b = road                        #전체비용을 계산하기위해 처음가져온값은 max에넣고, 해당도로의 양쪽 노드의 부모노드가 같은 부모노드를 공유하는지 확인
    max_cost+=cost
    if find_parent(parent,a) != find_parent(parent,b):  #만일 부모노드가 서로 달라서 연결해야한다면(전체노드를 순환할수있는 사이클을 생성하기위해서) 두노드를 합하고, 그때의 비용을 최소값에 할당
        union_parent(parent,a,b)
        min_cost+=cost

print(max_cost-min_cost)

#해설; 상동
'''

#Q44 행성 터널

#해설
#최소신장트리를 만드는 방법을 응용해야함 (크루스칼 알고리즘)
#x,y,z 각각에 대한 축별 거리를 고려하고, 이를 정렬하여 3 * (n-1)개의 간선을 검사하게한다.

def find_parent(parent,x):  #부모노드 찾기
    if parent[x] != x:
        parent[x] = find_parent(parent, parent[x])
    return parent[x]

def union_parent(parent,a,b):   #두 원소가 속한 집합 합치기
    a = find_parent(parent,a)
    b = find_parent(parent,b)
    if a<b:
        parent[b] = a
    else:
        parent[a] = b

n = int(input())
parent = [0] * (n+1)

edges = []  #간선담을 리스트
result = 0

for i in range(1,n+1):  #부모노드 자기자신으로 초기화
    parent[i] = i

x,y,z = [], [], []

for i in range(1,n+1):
    data = list(map(int,input().split()))
    x.append((data[0], i))
    y.append((data[1], i))
    z.append((data[2], i))

x.sort()    #x,y,z축 각각에 대해 위치값을 오름차순으로 정리. (위치, index)로 값을 저장하여 값을 구분할 수 있게끔함
y.sort()
z.sort()

for i in range(n-1):     #x,y,z축값 각각에 대해, 오름차순 정리한 노드간 거리를 계산해서 edges에 저장함. -> edges에서 이제 x,y,z축값 각각의 구분x
    edges.append((x[i+1][0] - x[i][0], x[i][1], x[i+1][1]))
    edges.append((y[i+1][0] - y[i][0], y[i][1], y[i+1][1]))
    edges.append((z[i+1][0] - z[i][0], z[i][1], z[i+1][1]))

print(x,'\n',y,'\n',z,'\n')
print(edges)
edges.sort()            #노드간 축거리 계산한결과를 다시 오름차순으로 정리 
print(edges)

for edge in edges:      #노드간 간선을 하나씩 가져와서 두 노드가 이미 이어져있는지 확인하고, 이어져있으면 패스, 이어져있지않으면 union으로 연결하고 cost를 더함
    cost,a,b = edge
    if find_parent(parent,a) != find_parent(parent,b):
        union_parent(parent,a,b)
        result+=cost

print(result)