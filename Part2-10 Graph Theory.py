#2021-04-19

#DFS/aFS문제와 최단경로 문제들은 그래프알고리즘의 일부이다.
#서로 다른 객체가 연결되어있다 -> 이런유형은 그래프알고리즘이라는 것을 떠올려야한다.

#최단경로-다익스트라에서 우선순위 큐가 사용되었는데, 이때 사용되는 최소/최대힙은 트리자료구조에 속한다!
#p268에 그래프와 트리의 방향성/순환성등의 특징이 있다. 참고

#그래프의 구현방법은 1)인접행렬(2차원배열), 2)인접리스트(리스트사용) 두가지로 나뉜다. 


#서로소 집합(Union-find자료구조; 공통원소가 없는 집합)
#이 집합구조는 Union(합집합연산;2개의 원소를 가지는 집합을 하나로 합치는것), Find(특정 원소가 속한 집합이 어떤 집합인지 찾는것)로 조작가능하다.


'''서로소집합 계산 알고리즘
1. Union 연산을 확인-> 서로 연결된 두 노드 A,B를 확인
    1-1. A와 B의 루트노드 A',B'를 각각 찾는다        -> 이를 이해하는게 중요. 1-1을 재수행할때 루트노드 = 부모노드라는것을 항상 염두에 두어야함
    1-2. A'를 B'의 부모 노드로 설정 (B'가 A'를 가르키도록한다. (B'->A'))
2. 모든 Union 연산을 수행할때까지 위과정을 반복
대개 이과정에서 부모노드값은 자식보다 작게한다 (A':1, B':3)

해당연산의 특징은 루트를 찾기 위해서(최종부모노드까지의 루트)는 '재귀적으로' 하위에서부터 반복적으로 거슬러 올라가야한다는 점이다.
'''

'''#예시
def find_parent1(parent,x):
    if parent[x] != x:              #해당 노드의 부모노드가 자기자신이 아니라면 = 자신이 루트노드가 아니라면 -> 루트노드를 찾을떄까지 재귀호출
        return find_parent(parent, parent[x])
    return x

def union_parent(parent,a,b):       #각각의 두노드의 루트노드의 크기를 비교해서, 각 노드(a,b)의 루트노드를 더 낮은 루트노드를 가지는 값으로 한다
    a = find_parent(parent,a)       #이 과정을 통해서 두 원소가 속한 집합을 합친다.
    b = find_parent(parent,b)
    if a<b:
        parent[b] = a
    else:
        parent[a] = b

v,e = map(int,input().split())
parent = [0] * (v+1)    #초기 부모테이블을 0으로 초기화

for i in range(1,v+1):
    parent[i] = i       #초기 각노드의 부모테이블을 자기자신으로 초기화

for i in range(e):
    a,b = map(int,input().split())     #입력받은 값으로 union 작업을 수행함
    union_parent(parent,a,b)

print('각 원소가 속한 집합:',end='')
for i in range(1,v+1):
    print(find_parent(parent,i),end=' ')        #union 작업 수행후, 각 원소가 속한 집합을 출력함.->각 원소의 루트노드(부모노드)값 출력
                                                #3번째 값이 1이면 노드 3은 1이라는 루트노드 하에 있다는것을 알려줌

print()

print('부모 테이블:',end='')
for i in range(1,v+1):
    print(parent[i],end=' ')                    #부모 테이블의 값을 출력
                                                #3번째 값이 2이면, 노드3의 부모노드는 2라는것을 알려줌.
'''

#위의 find_parent함수는 비효율적으로 작동하는데, 예를들어 1<-2<-3<-4<-5 형태의 그래프가 만들어진다면, (각각 부모노드 출력값은 1 1 2 3 4가 된다.)
#find함수는 모든 노드(V)와 그 노드의경로(E)를 탐색해야하므로, 시간복잡도는 O(VE)가 되어 비효율적이다 -> 경로압축을 통해서 해결가능!!

'''#기존의 find_parent함수 경로압축을 이용한 개선! 
def find_parent(parent,x):
    if parent[x] != x:
        parent[x] = find_parent(parent,parent[x])
    return parent[x]
'''
#위처럼 자기자신이 경로함수가 아닐경우, 해당 노드의 루트노드를 바로 부모노드로 잡게하기때문에 시간복잡도를 개선할 수 있다.
#1<-2<-3<-4<-5 각각의 노드에 대해 부모노드 출력값은 1 1 1 1 1 이 된다.


#서로소 집합을 이용한 사이클 판별 (간선에 방향성이없는 무향 그래프에서만 사용가능하다!)
#서로소집합은 무방향 그래프에서 사이클 판별시 사용할 수 있다.

'''사이클 판별 알고리즘
1. 각 간선을 확인하며 두 노드의 루트 노드를 확인한다.
    1-1. 루트노드가 다를때 -> 두 노드에 대한 Union연산 수행
    1-2. 루트노드가 같을때 -> 사이클이 발생한것!
2. 그래프에 포함되어있는 모든 간선에 대해 1.의 과정 수행.
'''
#이러한 방법은 그래프에 포함되어있는 간선 E개를 모두 하나씩 확인하며, 각각의 연산에 대해 Union&Find를 호출한다.

'''#예시
def find_parent(parent,x):  #경로압축 응용
    if parent[x] != x:
        parent[x] = find_parent(parent,parent[x])
    return parent[x]

def union_parent(parent,a,b):
    a = find_parent(parent,a)
    b = find_parent(parent,b)
    if a<b:
        parent[b] = a
    else:
        parent[a] = b

v,e = map(int,input().split())
parent = [0] * (v+1)
for i in range(1,v+1):
    parent[i] = i

cycle = False #사이클 발생여부 판단인자

for i in range(e):
    a,b = map(int,input().split())
    if find_parent(parent,a) == find_parent(parent,b):
        cycle = True
        break
    else:
        union_parent(parent,a,b)

if cycle:
    print('사이클이 발생했다.')
else:
    print('사이클이 발생하지 않았다.')
''' # 1-2-3-1 의 삼각구조 (1,2/1,3/2,3)를 입력했을때, 사이클이 존재하므로 발생했다는 결과를 반환한다.




#신장트리(Spanning Tree)
#모든노드가 서로 연결되어있지만, 사이클이 존재하지않는 트리를 말함. (예시는 p280~281)
#간선갯수 = 노드갯수 -1 이라는 특징을 가짐(일종의 트리이기 때문)

#이러한 신장트리를 최소한의 비용으로 찾아야할 때(ex-최소비용으로 모든 도시를 연결)가 있는데, 이때 '크루스칼 알고리즘'을 사용함 

#크루스칼 알고리즘은 그리디 알고리즘에 속한다.
'''알고리즘
1. 간선데이터를 비용에 따라 오름차순으로 정렬
2. 간선을 하나씩 확인하면서, 현재 간선이 사이클을 발생하는지 확인
    2-1. 사이클이 발생하지않는경우 -> 최소 신장트리에 포함
    2-2. 사이클이 발생하는경우 -> 제외시킴
3. 모든 간선에 대해 2를 반복
'''

'''#에시
def find_parent(parent,x):
    if parent[x] != x:
        parent[x] = find_parent(parent,parent[x])
    return parent[x]

def union_parent(parent,a,b):
    a = find_parent(parent,a)
    b = find_parent(parent,b)
    if a<b:
        parent[b] = a
    else:
        parent[a] = b

v,e = map(int,input().split())
parent = [0] * (v+1)

edges = []          #모든 간선을 담을 리스트
result = 0          #신장트리의 총 간선값을 담을 변수

for i in range(1,v+1):
    parent[i]=i
for _ in range(e):
    a,b,cost = map(int,input().split())
    edges.append((cost,a,b))            #입력받은 2개의 노드와 그 노드를 잇는 간선의 비용을 정렬하기 용이하게 하기위해 cost를 맨앞으로 지정하여 edges에 입력

edges.sort()

for edge in edges:
    cost,a,b = edge
    if find_parent(parent,a) != find_parent(parent,b):      #각각의 부모노드가 서로 속해있지않는경우, union작업후 그 간선값을 결과에 더함
        union_parent(parent,a,b)
        result += cost

print(result)
'''
#크루스칼 알고리즘의 시간복잡도는 O(ElogE)이다. -> 이 알고리즘에서 가장 시간을 많이 소모하는구간이 간선정렬이고 그 정렬에 대한 복잡도가 O(ElogE)이기때문
#서로소 집합 알고리즘은 간선정렬보다 매우 작기떄문에 무시된다.


#위상정렬
#순서가 정해져있는 작업을 수행해야할때 사용하는 알고리즘 (ex- 일반화학이수->유기화학이수->고분자공학 순서로 수업을 듣는것처럼)
#진입차수; 특정 노드로 들어오는 간선의 갯수
#시간복잡도 O(V+E) ; 순서대로 모든 노드를 확인하면서(v) 해당노드에서 출발하는 간선을 제거(e)해야하기 때문.
'''알고리즘
1. 진입차수가 0인 노드를 큐에 넣는다.
2. 큐가 빌 때까지 다음의 과정을 반복한다.
    2-1. 큐에서 원소를 꺼내 해당 노드에서 출발하는 간선을 그래프에서 제거한다.
    2-2. 새롭게 진입차수가 0이 된 노드를 큐에 넣는다.
'''
#이때, 모든 노드를 방문하기전에 큐가 비게되면 Cycle이 존재한다는 것을 반증한다. (대부분의 위상정렬에서는 사이클이 발생하지않는경우가 많긴함)
#또한 위상정렬은 답이 여러가지인것이 특징이다.(한 노드에 진입하는 노드가여러개일 경우)

'''#예시
from collections import deque

v,e = map(int,input().split())
indegree = [0] * (v+1)              #모든 노드에 대한 진입차수 0으로 초기화.
graph = [[] for i in range(v+1)]    #모든 간선정보를 담기위한 그래프 초기화.

for _ in range(e):
    a,b = map(int,input().split())
    graph[a].append(b)              #노드 a에서 b로 이동
    indegree[b]+=1                  #b의 진입차수 1증가

def topology_sort():
    result = []
    q = deque()                     #queue기능 수행을위해서 라이브러리 사용

    for i in range(1,v+1):          #해당 노드의 진입차수가 0이될경우, 이를 큐에 삽입
        if indegree[i] == 0:
            q.append(i)
    
    while q:
        now = q.popleft()           #큐에서 선입선출 -> 수행할 노드를 꺼냄
        result.append(now)          #작업한 노드를 result리스트에 추가함(순서화)

        for i in graph[now]:        #해당 노드가 다른노드와의 간선이 있다면, '해당노드'의 진입차수에서 -1씩 제거 
            indegree[i] -= 1

            if indegree[i] == 0:    #그렇게 진입차수를 제거하다가 0이되면, 새롭게 진입차수가 0이되는 노드를 큐에 삽입
                q.append(i)
    
    for i in result:
        print(i,end=' ')
topology_sort()
'''
