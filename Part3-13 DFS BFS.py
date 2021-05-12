#Q15 특정 거리의 도시 찾기
'''
bfs 적으로 생각. 
도시갯수 30만개, 길 갯수는 100만개이므로 완전탐색하면 타임오버뜰듯.
거리를 계산하다가 지금까지의 최단거리이상이 될경우 다음위치로 탐색하게끔하는것이 필요

생각이 바로안난다

'''


'''
from collections import deque

n, m, k, x = map(int,input().split())
array=[]
for _ in range(x):
    i, j = map(int,input().split())
    array.append((i,j))

visited = [False] * (n+1)
INF = 1e9

min_line= [INF] * (n+1)

def Bfs(array,start,visited,min_line):
    queue = deque([start])
    visited[start] = True
    queue.append(start)
    

    while queue:
        v = queue.popleft()
        count=0
        for x,y in array:
            if x == v and visited[y] == false:  #검사노드를 스타트로하는 도로를 가져왔고, 목적지가 방문한적 없다면
                count+=1
                min_line[y] = min(min_line[y], count) #y까지 가는데 걸리는 최소거리를 지금까지 카운트한것과 저장되어있는 최소값하고 비교해보고 갱신한다
                Bfs(array, y, visited, min_line)        #다시 y에서 다른 노드로 갈수있는것을 검색한다

            else :  #순차적으로 노드를 가져올때, 검사대상인 v에 대한 도로값이 아니라면 다음 도로를 가져옴 // 이미 도착
                continue
'''

'''#풀이
from collections import deque

n, m, k, x = map(int,input().split())
graph = [[] for _ in range(n+1)]

for _ in range(m):
    a,b = map(int,input().split())
    graph[a].append(b)          #그래프 리스트에 노드 a에서 갈수있는 노드 b로 입력한다
    
#(1,2),(1,3),(2,3),(2,4)일때 -> 단방향이라는것을 염두!
#[ [], [2,3], [3,4] ] 

distance = [-1] * (n+1)
distance[x] = 0 #출발도시 자신까지의 거리는 0으로 설정

q = deque([x]) 

while q:                #스타트노드에서 각 노드까지의 최단거리를 계산
    now = q.popleft()

    for next_node in graph[now]:
        if distance[next_node] == -1 :  #만일 다음에 갈 노드가 -1 = 방문하지않은 노드라면
            distance[next_node] = distance[now] + 1         #현재노드까지의 거리에 + 1한값을 할당
            q.append(next_node)         #그리고 다음에 갈 노드를 큐에 할당

check = False
for i in range(1,n+1):
    if distance[i] == k:        #저장된 거리값을 하나씩 확인하면서 각 노드까지의 거리값중, k값과잁치한 노드는 출력하고 check에 하나라도값이 있음을 확인
        print(i)
        check = True

if check == False:
    print(-1)                   #저장된 모든 거리값을 검사했을때, k값과 일치한값이없어서 check = false일때 -1을 출력함
'''

#Q16 연구소
'''
일단 맵의 0인자리에 1 3개를 배정하는 조합을 모두 만들어서 각각의 경우를 모두 체크하게끔해야함 (콤비네이션 이용)

체크방법
1. 1(벽) 3개를 조합에따라 배정
2. 2(바이러스)를 인접한 0에 최대한 퍼트림
3. 퍼진후의 맵 전체에서 0의 갯수를 센다

체크하면서 0의 값이 최대일때 갱신하는 방법을 이용함.
가로세로가 해봤자 8x8이라 전수검사해도 ㄱㅊ을듯?

dfs적 재귀화 이용해서 바이러스 감염 ㄱㄱ
'''

'''#내답
from itertools import combinations
import copy

n, m = map(int,input().split())
array = []
for _ in range(n):
    array.append(list(map(int,input().split())))

zero_site = []
virus_site = []
for i in range(n):
    for j in range(m):
        if array[i][j] == 0:
            zero_site.append([i,j])     #0의 위치를 저장
        elif array[i][j] == 2:
            virus_site.append([i,j])    #2의 위치를 저장

def built_wall(array,plan):     #원래의 맵과 짓는방법을 집어넣었을때 지어진 결과맵을 반환하는 함수
    for i in plan:
        array[i[0]][i[1]] = 1
    return array

def count_zero(array):    #0의 갯수를 세는 함수
    count=0
    for line in array:
        for element in line:
            if element == 0:
                count+=1
    return count

def infested_array(x,y):     #감염된후의 맵을 반환하는 함수 -> 각 방향별로 재귀화

    dx = [-1,0,1,0]
    dy = [0,1,0,-1]

    for i in range(4):
        nx = x + dx[i]
        ny = y + dy[i]

        if nx >= 0 and nx < n and ny >= 0 and ny < m:
            if complete_array[nx][ny] == 0:
                complete_array[nx][ny] = 2
                infested_array(nx,ny)


how_to_build = list(combinations(zero_site, 3))   #빈땅중 3개를 골라서 벽을 세우는 조합을 만든다

count = []      

for plan in how_to_build:   #조합을 하나씩 가져옴
    c = copy.deepcopy(array)        #원래의 맵을 c에 카피함
    complete_array = built_wall(c, plan)        #카피한 c맵에 벽짓는 조합대로 벽을 지음

    for virus in virus_site:
        infested_array(virus[0],virus[1])        #바이러스에 따른 감염후 맵상태를 구하고

    after_count = count_zero(complete_array)          #완전히 퍼졌을때의 맵 내의 0갯수를 센다
    #print(complete_array)
    
    count.append(after_count)     #퍼진상태의 맵내의 0갯수와 기존의 count갯수중, 더 큰값을 갱신한다
    #print(count)

print(max(count))            #모든검사를 끝낸후, 감염 후 잔존 0갯수의 최대값을 출력
'''

'''#해설
n, m = map(int, input().split())
data = []                          #벽설치전 원래 맵리스트
temp = [[0] * m for _ in range(n)] #벽설치후 맵 리스트

for _ in range(n):
    data.append(list(map(int,input().split())))

dx = [-1,0,1,0]
dy = [0,-1,0,1]

result =0

def virus(x,y):         #감염시키는함수
    for i in range(4):
        nx = x + dx[i]
        ny = y + dy[i]
        if nx >= 0 and nx < n and ny >= 0 and  ny < m:      #맵범위내에 대해 감염가능하다면, 감염한 결과를 temp에 저장하고 재귀
            if temp[nx][ny] == 0:
                temp[nx][ny] = 2
                virus(nx, ny)

def get_score():    #감염되지 않은 맨땅(0)의 갯수를 세는 함수
    score = 0
    for i in range(n):
        for j in range(m):
            if temp[i][j] == 0:
                score += 1
    return score

def dfs(count):
    global result
    if count == 3:                          #벽(1)이 3개설치됐을때
        for i in range(n):
            for j in range(m):
                temp[i][j] = data [i][j]    #그 결과를 data에서 temp로 복사한다
    
        for i in range(n):
            for j in range(m):
                if temp[i][j] == 2:         #복사한 temp에 대해서 하나씩 순차적으로 검사할때, 바이러스(2)가 발견되면 전염함수를 실행한다
                    virus(x, y)
        
        result = max (result, get_score())  #전염과정이 끝나고 남은 0의 갯수가 큰수를 계속해서 갱신한다
        return
    
    for i in range(n):                      #벽 설치(아직 전부 설치하지 않았을때)
        for j in range(m):
            if data[i][j] == 0:             #순차적으로 빈땅을 찾아서, 빈땅에 벽을 세우고 카운트+1한다
                data[i][j] = 1
                count += 1
                dfs(count)                  #재귀화해서 count가 3이될때까지 벽을 계속 세운다 -> 3개가 됐을때 result를 한번 계산하고...(1)
                data[i][j] = 0
                count -= 1                  #(1)에서 이어서 설치한 마지막 3번째 벽을 지우고 스텍을 1개 깎는다 -> 이후 3번째스택의 다음 0에대해 벽을 세우고 반복한다
                                            #ex) 0이 1,2,3,4,5 로 되어있다면 [1,2,3]->[1,2]->[1,2,4]->[1,2]->[1,2,5]->[1,2]->[1]->[1,3]->[1,3,4]...이런방식으로 흘러감

dfs(0)
print(result)
'''

#Q17 경쟁적 전염

'''#내답
n, k = map(int,input().split())
array = []
virus_site = [[] for _ in range(k+1)]

for i in range(n):
    line=list(map(int,input().split()))
    array.append(line)
    for j in range(len(line)):  #1 0 2 i=0
        if line[j] == 0:
            continue
        else :
            virus_site[line[j]].append([i,j])

s, x, y = map(int,input().split())

#북동남서
dx = [-1,0,1,0]
dy = [0,1,0,-1]

def infest(array):
    new_array = array
    for num in range(1,k+1):
        for site in virus_site[num]:
            x,y = site[0],site[1]

            for i in range(4):
                nx=x+dx[i]
                ny=y+dy[i]
                if nx<0 or nx>=n or ny<0 or ny>=n:   #바이러스가 움직일칸이 범위밖이면 패스
                    continue

                if array[nx][ny] == 0:  #바이러스가 갈곳이 비어서 이동가능하다면 이동
                    new_array[nx][ny] = num    #맵에 표시
                else:                   #그밖의 상황은 패스
                    continue
    return new_array
    
def renew_virus_site(array,virus_site):
    new_virus_site = [[] for _ in range(k+1)]       #이전의 바이러스 위치를 초기화
    for i in range(n):
        for j in range(n):
            if array[i][j] == 0:    #여전히 빈칸이면 패스
                continue
            else :
                new_virus_site[array[i][j]].append([i,j])   #바이러스 번호에 맞는 순서의 site리스트에 위치값을 저장
    return new_virus_site


for i in range(s):      #s초 수행 = s번 수행
    print(i+1,'초쨰')
    array = infest(array)            #바이러스 순차적 감염
    virus_site = renew_virus_site(array,virus_site)  #감염후 맵의 상태에 따라 virus_site를 갱신함 
    print('감염후 맵',array)
    print(virus_site)

print(array[x-1][y-1])  #x,y위치의 상태를 출력
'''

#해설: 기본적으로 bfs적으로 해결가능함. 낮은번호부터 바이러스를 감염시키게 큐에다가 바이러스를 할당하면된다.return

from collections import deque

n,k = map(int,input().split())

graph = []      #맵데이터를 담는 리스트
data = []       #바이러스 정보를 담는 리스트

for i in range(n):
    graph.append(list(map(int,input().split())))
    for j in range(n):
        if graph[i][j] != 0:
            data.append((graph[i][j],0,i,j))        #초기 맵 입력시 바이러스가 있다면 (바이러스숫자,입력시간,x,y)데이터 순으로 입력한다

data.sort() #바이러스 정보를 바이러스 숫자 순으로 정렬
q = deque(data) #큐에 바이러스 정보를 할당함

target_s, target_x, target_y = map(int,input().split())

#북동남서
dx = [-1,0,1,0]
dy = [0,1,0,-1]

while q:
    virus, s, x, y = q.popleft()
    if s == target_s:
        break               #s시간이 다되거나, 바이러스가 움직이는 큐가 다 비게되면 중지

    for i in range(4):
        nx=x+dx[i]
        ny=y+dy[i]

        if nx>=0 and ny>=0 and nx<n and ny<n:       #움직일수 있는 범위내이고 움직일곳이 0이라면
            if graph[nx][ny] == 0:
                graph[nx][ny] = virus               #움직일 곳을 virus번호로 뒤집어씌우고 큐에 해당 바이러스정보를 추가한다
                q.append((virus,s+1,nx,ny))         #위에서 이미 바이러스 순서대로 리스트를 정리했기때문에 이렇게 순차적으로 추가해도 순서가 꼬일일은 없음

print(graph[target_x-1][target_y-1])








