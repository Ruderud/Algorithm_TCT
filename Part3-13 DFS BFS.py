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

'''#해설: 기본적으로 bfs적으로 해결가능함. 낮은번호부터 바이러스를 감염시키게 큐에다가 바이러스를 할당하면된다
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
'''

#18  괄호 변환
'''
균형잡힌 문자열 = [ [올바른 문자열], [올바르지 않은 문자열]]

조건
1. 빈문자열 -> 빈문자열로 반환
2. 입력문자열 -> u(균형잡인 문자열) + v(균형잡힌문자열, 공백가능) //이두개는 각각이 더이상 쪼개지지 않을때까지 반복

문자열을 위 규칙대로 잘랐을때 맨앞이 (이면 올바름, )이면 올바르지 않음임
'''


'''#내답 -> 정확도 32/100
w = str(input())

data=[]

def divide_str(w):
    i=0
    a=w[i]

    while a.count('(') != a.count(')'):
        i+=1
        a += w[i]
    
    data.append(a)
    
    if i+1 < len(w):
        divide_str(w[i+1:])
    else:
        data.append('')

def sol(w):
    answer = ''

    divide_str(w)
    i=0

    stack = []

    for element in data:
        if element == '' or element[0] == '(':    #올바른 배열 또는 공백배열을 만나게되었을때 -> 올바르지않은배열이 스텍에 있을때 /없을때로 구분
            if not stack:                         #스택에 뭐가 없을때
                answer += element                 #그냥 출력

            else:                                    #스텍에 뭔가가 있을때
                answer += '(' + element + ')'        #일단 현재 올바른 배열이나 공백배열을 괄호에 싸서 먼저 출력

                while stack:                         #스택을 모두뺄때까지 반복
                    a = stack.pop()                  #스텍의 '뒤에서부터' 하나씩 순차적으로 빼옴
                    answer += a[-2:0:-1]             #빼온 스택의 앞뒤문자열을 제외하고 역순으로 answer에 추가함
                
        else:                                     #올바르지 않은 배열을 만났을때
            stack.append(element)                 #스텍에 추가함
    
    return answer

print(sol(w))
'''

'''#해답 근데 이것도 정확도점수는 52점밖에안나오는데 ㅡㅡ
def balanced_index(p):          #문자열 p에 대해서, 균형잡힌 문자열의 위치 인덱스를 알려줌; 괄호왼쪽과 오른쪽의 갯수차이를 이용, 0이될때의 위치를 반환
    count = 0
    for i in range(len(p)):
        if p[i] == '(':
            count += 1
        else:
            count -= 1
        
        if count == 0:
            return i
    
def check_proper(p):            #문자열 p에 대해서, 올바른 괄호문자열인지 확인
    count = 0
    for i in p:
        if i == '(':
            count += 1
        else:
            if count == 0:
                return False
            count -= 1          #왼쪽괄호의 갯수를 더하고 오른쪽 괄호를 빼다가, 0인 상태에서 오른쪽괄호를 만나게되면(짝이 안맞게되면) false반환
    return True                 #무사히 문자열 모든 위치를 확인한다면 True반환

def solution(p):
    answer=''
    if p == '':
        return answer
    index = balanced_index(p)
    u = p[:index+1]
    v = p[index+1:]

    if check_proper(u):
        answer = u + solution(v)

    else:
        answer = '(' + solution(v) + ')' + u[-2:0:-1]
    return answer

print(solution('()))((()'))
'''

#19 연산자 끼워 넣기
'''

'''

'''#내답
from itertools import permutations

n = int(input())
number = list(map(int,input().split()))
calculator = list(map(int,input().split()))

calculator_rawlist=[]

for i in range(4):                      #사칙연산 (0,1,2,3)에 매치해서 리스트에 나열함
    for num in range(calculator[i]):
        calculator_rawlist.append(i)

cal_match = list(permutations(calculator_rawlist,len(calculator_rawlist)))      #같은 부호일때 위치를 구별한 조합을 생성

def Remove(lst):        #동일 결과를 나타내는 리스트를 제거
     res = []
     check = set({})        #비어있는상태의 set인 {} 를 만든다
  
     for x in lst:          #조합을 하나씩 가져와서 튜플화 시킨다. (기존의 리스트타입은 헤시가능하지않기 때문 -> 변동가능성이 없는값만이 set에서 중복을제거할 때 사용가능)
         hsh = tuple(x)
         if hsh not in check:       #가져온 조합이 체크리스트에있는거면 패스, 없는거면 결과리스트에 추가하고 체크리스트에 추가함.
             res.append(x)
             check.add(hsh)
               
     return res

def calculate(number,comb):     #숫자 순서 1 2 3 4 5 6 과 여러 연산자 배열중 하나인 + + - * / 를 수행
    result=number[0]

    for i in range(len(comb)):
        if comb[i] == 0:                        # +부호
            result = result + number[i+1]
        elif comb[i] == 1:
            result = result - number[i+1]
        elif comb[i] == 2:
            result = result * number[i+1]
        else :                                  # /부호일때, 이전까지의 결과값이 음수일때와 양수로일때로 나눔
            if result <0:
                result = ( abs(result) // number[i+1] ) * -1
            else :
                result = result // number[i+1]
    return result

Remove(cal_match)

INF = 1e9
max_min = [(INF*-1),INF]

for comb in cal_match:
    result = calculate(number,comb)
    max_min[0] = max(max_min[0],result)
    max_min[1] = min(max_min[1],result)

print(max_min[0])
print(max_min[1])
'''

'''#해설 여기답은 dfs적으로 풀었으나, from itertools import product 이런함수를 쓸수도있음 -> n=4일때 사칙연산중 중복을 허용하여 3개를 뽑아서 나열함 
n=int(input())
data = list(map(int,input().split()))
add, sub, mul, div = map(int,input().split())           #잔여 부호 갯수를이용하여 dfs를 돌림

min_value = 1e9
max_value = -1e9

def dfs(i,now):
    global min_value, max_value, add, sub, mul, div
    if i == n:      #연산자를 모두 사용했을때, 최소/최대값을 갱신
        min_value = min(min_value,now)
        max_value = max(max_value,now)

    else:
        if add>0:                   # +부호가 남아있을때
            add -= 1                #하나를 가져와서
            dfs(i+1, now+data[i])   #현재까지 더한 숫자에 지금가져온 숫자를 더한 연산을 해서 dfs연산을 재귀화
            add += 1                #연산이 끝나면 사용한 +부호를 다시 되돌려놓음 (가지가 뻗어나가야하므로)

        if sub>0:
            sub -= 1
            dfs(i+1, now-data[i])
            sub +=1
        
        if mul>0:
            mul -= 1
            dfs(i+1, now*data[i])
            mul += 1
        
        if div>0:
            div -= 1
            dfs(i+1, int(now/data[i]))      #나눗셈할때는 소수부분을 절삭
            div += 1

dfs(1,data[0]) #연산시작
print(max_value)
print(min_value)
'''


#Q20 감시피하기 (45분)


'''#내답. 좀 비효율적인부분이 많이보이지만 시간내(50분)내에 풀었음
from itertools import combinations

n = int(input())
array = []
for _ in range(n):
    array.append(list(map(str,input().split())))

teacher = []
student = []
empty = []

for i in range(n):                  #각각의 선생/학생/빈공간정보를 따로 돌리지않고, 위에 data input시 같이 수행했다면 더 효율적
    for j in range(n):
        if array[i][j] == 'T':
            teacher.append((i,j))
        elif array[i][j] == 'S':
            student.append((i,j))
        else :
            empty.append((i,j))

object_comb = list(combinations(empty,3))

def build(array,comb):
    built_array = array
    for x,y in comb:
        built_array[x][y] = 'O'
    return built_array

def search(built_array,teacher,n,comb):     #여기서 맵범위 바깥에 해당하는 시야를 제거했다면 메모리 다이어트가능 + for문을 하나 더 썼다면 굳이 x,y에 일일히 +-1씩 할필요없었을듯
    for x,y in teacher:

        for i in range(1,n):
            if (x,y-i) in comb:             #좌측시야 추가. 이때 장애물이 설치되어있다면 더이상 해당방향으로 시야를 늘리지않고 끝냄
                break
            search_site.append((x,y-i))

        for i in range(1,n):
            if (x,y+i) in comb:             #우측 추가
                break
            search_site.append((x,y+i))

        for i in range(1,n):
            if (x-i,y) in comb:             #위쪽 추가
                break
            search_site.append((x-i,y))

        for i in range(1,n):
            if (x+i,y) in comb:             #아래쪽쪽 추가
                break
            search_site.append((x+i,y))

def confirm(student,search_site):
    for i in student:
        if i in search_site:        #선생시야안에 학생이 있다면 true반환
            return True
    return False

result=0
for comb in object_comb:
    built_array = build(array,comb)             #조합에 따라서 맵에다가 조형물을 설치한값을 출력
    search_site = []
    search(built_array,teacher,n,comb)          #장애물 설치한 맵에 대해서 교사가 감시하는 지역을 search_site에 저장
    if not confirm(student,search_site):        #감시하는 지역내에 학생이 있는지 확인해서, 학생이 선생시야내에 없으면 결과값을 1로하여 검사를 끝냄
        result = 1
        break

if result == 1:
    print("YES")
else:
    print("NO")
'''

'''#해답. 해답도 combination 라이브러리를 사용했음 대신, 선생별로 watch()메서드를 따로만들어서 검사하게끔 하는 방식을 이용

from itertools import combinations

n = int(input())
board = []
teachers = []
spaces = []

for i in range(n):
    board.append(list(input().split()))
    for j in range(n):
        if board[i][j] == "T":
            teachers.append((i,j))
        if board[i][j] == "X":
            spaces.append((i,j))

def watch(x,y,direction):       #방향0,1,2,3에 따라 각각 좌/우/상/하 를 직선으로 검사하고, 중간에 학생이 발견되면 true, 장애물이면 false, 맨땅이면 다음칸 검색함
    if direction == 0:
        while y >= 0:
            if board[x][y] == 'S':
                return True
            if board[x][y] == 'O':
                return False
            y-=1
    if direction == 1:
        while y < n:
            if board[x][y] == 'S':
                return True
            if board[x][y] == 'O':
                return False
            y+=1
    if direction == 2:
        while x >= 0:
            if board[x][y] == 'S':
                return True
            if board[x][y] == 'O':
                return False
            x-=1
    if direction == 3:
        while x < n:
            if board[x][y] == 'S':
                return True
            if board[x][y] == 'O':
                return False
            x+=1
    return False

def process():                  #선생님위치를 하나씩 각자가져오면서 시야를 확인함
    for x,y, in teachers:
        for i in range(4):
            if watch(x,y,i):
                return True
    return False

find = False                    #일단은 발견된다는것을 디폴트값으로 설정

for data in combinations(spaces,3):
    for x,y in data:
        board[x][y] = 'O'       #조합에 따라 장애물을 설치하고
    if not process():
        find = True             #탐색과정을 수행했을때 발각되지 않는다면 true로 전환하고 거기서 탐색종료
        break
    for x,y in data:            #설치한 장애물을 다시 제거하는 과정을 통해 데이터사용량 절약
        board[x][y] = 'X'

if find:
    print('YES')
else:
    print('NO')
'''

#Q21

#음료수얼리기 응용! dfs를 이용해서 덩어리를 형성하고, 덩어리내의 각 좌표에 해당하는 값을 전부 더한뒤 연합한 좌표 갯수로 나눠서 다시 배분한다.

'''#timeover -> 여러개의 유니온일때와, 인구 분배후 인구이동이 다시가능한것을 구현하지못함
n,L,R = map(int,input().split())
array = []
for _ in range(n):
    array.append(list(map(int,input().split())))

#상하좌우 순서로 검사
dx = [-1,1,0,0]
dy = [0,0,-1,1]

visited = []

def union_nation(x,y):
    if x<0 or x>=n or y<0 or y>=n or (x,y) in visited :
        return False
    
    visited.append((x,y))
    for i in range(4):
        nx = x + dx[i]
        ny = y + dy[i]
        if 0<= nx <n and 0<=ny<n and L<=abs(array[x][y] - array[nx][ny])<=R and (nx,ny) not in visited:        
                    #다음으로 이동할 위치가 범위내에 있으면서, 방문하지 않았으며, 현재위치의 인구수와 그 차이값이 LR범주내라면
            print((nx,ny))
            visited.append((nx,ny))                 #다음이동할 위치를 방문처리하고
            union_nation(nx,ny)                     #다시 재귀화 한다
    return visited


count=0
while union_nation(0,0):                       #연합이 있다면 아래의 내용을 수행한다.
    union_elements = []
    union_elements.append(union_nation(0,0))
                                            #연합을 형성한다 ex [[(0,1),(0,2)], [(1,0),(1,1),(1,2) ] 꼴
    while union_elements.count(False):
        union_elements.remove(False)

    for union in union_elements:            #여러연합중 한 연합을 가져옴
        total = 0
        for x,y in union:                   #가져온 연합내의 나라의 인구수를 다 더함
            total += array[x][y]

        for x,y in union:                   #더한 총 인구수를 연합내 나라 갯수로 나누고, 그것을 연합내 각 나라에 분배함
            array[x][y] = total // len(union)
    print(array)
    count+=1

print(count)
'''

'''#해답. bfs사용해서 풀었음. union이라는 nxn행렬을 하나더만들고 전부 -1로 초기화 시킨후, 0부터시작해서 n^2 -1 번까지 돌고, 마지막으로 n^2+1번째를 진입하면 while문탈출
#그리고 중간에 조건에 따라 인구이동이 가능해지면, 그 연합국은 동일 union좌표에 같은 index번호를 가지게해서 구별하게끔했음.
from collections import deque

n,l,r = map(int,input().split())
graph = []
for _ in range(n):
    graph.append(list(map(int,input().split())))

dx = [-1,0,1,0]
dy = [0,-1,0,1]

result = 0

def process(x,y,index):         #나라(x,y)위치와 현재 연합상황을 입력해서 수행
    united = []
    united.append((x,y))

    q = deque()
    q.append((x,y))             #큐에 현재탐색중인 나라의 위치를 입력
    union[x][y] = index         #연합을 나타내는 행렬(index)에 index수로 무슨연합인지 남김
    summary = graph[x][y]       #현재 검사중인 국가의 인구수 = 전체인구수화 함
    count = 1                   #현재연합 국가의수(시작이므로 1)

    while q:
        x, y = q.popleft()
        for i in range(4):
            nx = x + dx[i]
            ny = y + dy[i]

            if 0<=nx<n and 0<=ny<n and union[nx][ny] == -1 :    #맵내의 인접한 나라가 아직 연합에 속하지 않았을때
                if l <= abs(graph[nx][ny] - graph[x][y]) <= r:  #인접나라와의 인구차이가 인구이동이 가능할정도라면
                    q.append((nx,ny))

                    union[nx][ny] = index                       #연합을 나타내는 행렬에 인접한나라를 현재의 연합에 추가하고
                    summary += graph[nx][ny]                    #전체인구수에 추가되는나라의 인구를 더함
                    count += 1                                  #연합나라수를 추가
                    united.append((nx,ny))                      #연합된 나라리스트에 추가

    for i,j in united:
        graph[i][j] = summary//count        #연합작업이 끝나면 연합국가간에 인구를 분배
    return count

total_count = 0     #총 수행횟수

while True :
    union = [[-1] * n for _ in range(n)]            #연합index를 나타내는 nxn행렬을 하나더 생성하고 전부 -1로 초기화함
    index = 0
    for i in range(n):
        for j in range(n):
            if union[i][j] == -1:                   #연합indx행렬을 순차적으로 -1인것을 하나씩 가져와서 연합처리를 수행하고, 
                print(index)
                process(i,j,index)                  #연합처리가 끝나면 다음 연합번호로 넘어감.
                index += 1                          #이때 인덱스는 같은 연합이라면 +되지않고 process함수에 의해 모두 돌아서 union에 같은 index 숫자를먹여버리므로 index가 변화하지않고 처음 연합을 시작한 index 숫자로 뒤집어 씌워짐
    print(graph)
    print(union)
    if index == n*n:        #각각이 연합될일이없어서 index가 0부터 n-1까지 싹다돌고 n이 되었을때 while문을 탈출한다.
        break
    total_count+=1          #인구이동 한사이클을 전부 수행하면 횟수를 +1한다

print(total_count)
'''

#Q22 블록 이동하기
'''
무조건 (1,1)(1,2)에 있기때문에, 위치숫자를 +하는 방법으로 나아가야함

로봇의 가로세로 구분은 x값의 차이가 0이면 가로, 1이면 세로로 판별가능

8가지 로봇이 움직이는것을 각각 구현하고, 움직임을 bfs로 돌려보자

움직임 구현끝

'''


'''
def robot_shape():
    global robot
    return abs(robot[0][0] - robot[1][0])     #0이면 수평, 1이면 수직이다.

def move_r():
    global robot
    if robot[1][1]+1 < n:       #오른쪽 로봇칸의 오른쪽에 맵이 남아있으며
        if robot_shape() == 0 and board[robot[1][0]][robot[1][1]+1] == 0: #로봇이 수평이고 로봇의 오른쪽칸이 빈칸이면
            robot[0][1]+=1
            robot[1][1]+=1
    return False

def move_l():
    global robot
    if robot[0][1]-1 >= 0:       #왼쪽 로봇칸의 왼쪽에 맵이 남아있으며
        if robot_shape() == 0 and board[robot[0][0]][robot[0][1]-1] == 0: #로봇이 수평이고 로봇의 왼쪽칸이 빈칸이면
            robot[0][1]-=1
            robot[1][1]-=1
    return False

def turn_r_r90(): #오른쪽 중심으로 시계반대방향 90도 전환
    global robot
    if robot[0][0]+1 < n and robot[1][0]+1 < n and robot_shape() == 0 : #로봇의 아래 두칸이 존재하며 수평
        if board[robot[0][0]+1][robot[0][1]] == 0 and board[robot[1][0]+1][robot[1][1]] == 0: #아래두칸이 전부 빈칸일경우
            robot[0][0] = robot[1][0]
            robot[0][1] = robot[1][1]       #로봇의 첫번째 칸을 윗칸으로 하고
            robot[1][0] = robot[1][0]+1
            robot[1][1] = robot[1][1]       #두번째 칸을 아랫칸으로 한다.
    return False

def turn_r_90(): #오른쪽 중심으로 시계방향 90도 전환
    global robot
    if robot[0][0]-1 >= 0 and robot[1][0]-1 >= 0 and robot_shape() == 0 : #로봇의 위 두칸이 존재하며
        if board[robot[0][0]-1][robot[0][1]] == 0 and board[robot[1][0]-1][robot[1][1]] == 0: #위두칸이 전부 빈칸일경우
            robot[0][0] = robot[1][0]-1
            robot[0][1] = robot[1][1]       #로봇의 첫번째 칸을 윗칸으로 하고
            robot[1][0] = robot[1][0]
            robot[1][1] = robot[1][1]       #두번째 칸을 아랫칸으로 한다.
    return False

def turn_l_90(): #왼쪽 중심으로 시계방향 90도 전환
    global robot
    if robot[0][0]+1 < n and robot[1][0]+1 < n and robot_shape() == 0 : #로봇의 아래 두칸이 존재하며
        if board[robot[0][0]+1][robot[0][1]] == 0 and board[robot[1][0]+1][robot[1][1]] == 0: #아래두칸이 전부 빈칸일경우
            robot[0][0] = robot[0][0]
            robot[0][1] = robot[0][1]       #로봇의 첫번째 칸을 윗칸으로 하고
            robot[1][0] = robot[0][0]+1
            robot[1][1] = robot[0][1]       #두번째 칸을 아랫칸으로 한다.
    return False

def turn_l_r90(): #왼쪽 중심으로 시계반대방향 90도 전환
    global robot
    if robot[0][0]-1 >= 0 and robot[1][0]-1 >= 0 and robot_shape() == 0 : #로봇의 위 두칸이 존재하며
        if board[robot[0][0]-1][robot[0][1]] == 0 and board[robot[1][0]-1][robot[1][1]] == 0: #위두칸이 전부 빈칸일경우
            robot[0][0] = robot[0][0]-1
            robot[0][1] = robot[0][1]       #로봇의 첫번째 칸을 윗칸으로 하고
            robot[1][0] = robot[0][0]+1
            robot[1][1] = robot[0][1]       #두번째 칸을 아랫칸으로 한다.
    return False

def move_u():
    global robot
    if robot[0][0]-1 >= 0:       #위쪽 로봇칸의 아래에 맵이 남아있으며
        if robot_shape() == 1 and board[robot[0][0]-1][robot[0][1]] == 0: #로봇이 수직이고 로봇의 위쪽칸이 빈칸이면
            robot[0][0]-=1
            robot[1][0]-=1
    return False

def move_d():
    global robot
    if robot[1][0]+1 < n:       #아래쪽 로봇칸의 아래에 맵이 남아있으며
        if robot_shape() == 1 and board[robot[1][0]-1][robot[1][1]] == 0: #로봇이 수직이고 로봇의 아랫칸이 빈칸이면
            robot[0][0]+=1
            robot[1][0]+=1
    return False

def turn_u_90(): #위쪽 중심으로 시계방향 90도 전환
    global robot
    if robot[0][1]-1 >= 0 and robot[1][1]-1 >= 0 and robot_shape() == 1: #로봇의 왼쪽 두칸이 존재하며 수직
        if board[robot[0][0]][robot[0][1]-1] == 0 and board[robot[1][0]][robot[1][1]-1] == 0: #완쪽두칸이 전부 빈칸일경우
            robot[1][0] = robot[0][0]
            robot[1][1] = robot[0][1]     #위블록을 뒤로
            robot[0][0] = robot[1][0]
            robot[0][1] = robot[1][1]-1   #밑블록을 앞으로
    return False

def turn_u_r90(): #위쪽 중심으로 시계반대방향 90도 전환
    global robot
    if robot[0][1]+1 < n and robot[1][1]+1 < n and robot_shape() == 1: #로봇의 오른쪽 두칸이 존재하며
        if board[robot[0][0]][robot[0][1]+1] == 0 and board[robot[1][0]][robot[1][1]+1] == 0: #오른쪽두칸이 전부 빈칸일경우
            robot[0][0] = robot[0][0]
            robot[0][1] = robot[0][1]
            robot[1][0] = robot[0][0]
            robot[1][1] = robot[0][1]+1
    return False

def turn_d_90(): #아래 중심으로 시계방향 90도 전환
    global robot
    if robot[0][0]+1 < n and robot[1][0]+1 < n and robot_shape() == 1: #로봇의 오른쪽 두칸이 존재하며
        if board[robot[0][0]][robot[0][1]+1] == 0 and board[robot[1][0]][robot[1][1]+1] == 0: #오른쪽두칸이 전부 빈칸일경우
            robot[0][0] = robot[1][0]
            robot[0][1] = robot[1][1]
            robot[1][0] = robot[0][0]
            robot[1][1] = robot[0][1]+1
    return False

def turn_d_r90(): #아래 중심으로 시계반대방향 90도 전환
    global robot
    if robot[0][1]-1 >= 0 and robot[1][1]-1 >= 0 and robot_shape() == 1: #로봇의 왼쪽 두칸이 존재하며 수직
        if board[robot[0][0]][robot[0][1]-1] == 0 and board[robot[1][0]][robot[1][1]-1] == 0: #완쪽두칸이 전부 빈칸일경우
            robot[1][0] = robot[1][0]
            robot[1][1] = robot[1][1]     #밑블록이 뒤가되고
            robot[0][0] = robot[1][0]
            robot[0][1] = robot[1][1]-1   #위블록이 앞이됨
    return False



board = [
    [0,0,0,0,0],
    [0,0,0,1,0],
    [0,0,0,1,1],
    [1,1,0,0,1],
    [0,0,0,0,0]
    ]

n = len(board)
robot = [[0,0],[0,1]]

from collections import deque

count=0

while [n,n] not in robot:
    move_r()
    count+=1
    if not move_r():
        turn()
        if not turn():


'''

from collections import deque

def get_next_pos(pos, board):
    next_pos = []
    pos = list(pos)
    pos1_x, pos1_y, pos2_x, pos2_y = pos[0][0], pos[0][1], pos[1][0], pos[1][1] #보기좋게 값을 배정

    #상하좌우
    dx = [-1,1,0,0]
    dy = [0,0,-1,1]
    for i in range(4):
        pos1_next_x, pos1_next_y, pos2_next_x, pos2_next_y = pos1_x + dx[i], pos1_y + dy[i], pos2_x + dx[i], pos2_y + dy[i]

        if board[pos1_next_x][pos1_next_y] == 0 and board[pos2_next_x][pos2_next_y] == 0:
                next_pos.append({(pos1_next_x,pos1_next_y), (pos2_next_x,pos2_next_y)})
    
    if pos1_x == pos2_x:        #수평일경우
        for i in [-1,1]:            # 위쪽으로/아래쪽으로
            if board[pos1_x+i][pos1_y] == 0 and board[pos2_x+i][pos2_y] == 0:   #위쪽/아래쪽 두칸이 각각 비어있을경우
                next_pos.append({(pos1_x,pos1_y), (pos1_x+i,pos1_y)})
                next_pos.append({(pos2_x,pos2_y), (pos2_x+i,pos2_y)})
    elif pos1_y == pos2_y:      #수직일경우
        for i in [-1,1]:            #왼쪽회전/오른쪽회전
            if board[pos1_x][pos1_y+i] == 0 and board[pos2_x][pos2_y+i] == 0:   #왼쪽/오른쪽 두칸이 각각 비어있을경우
                next_pos.append({(pos1_x,pos1_y), (pos1_x,pos1_y+i)})
                next_pos.append({(pos2_x,pos2_y), (pos2_x,pos2_y+i)})
    return next_pos

def solution(board):
    n = len(board)
    new_board = [[1] * (n+2) for _ in range(n+2)]           #가로세로가 한칸씩 늘어난 맵을 만들고(모두 벽으로 초기화)
    for i in range(n):
        for j in range(n):
            new_board[i+1][j+1] = board[i][j]               #원래의 맵을 다시 입력

    
    q = deque()
    visited = []
    pos = {(1,1), (1,2)}            #set 형을 사용해서 로봇의 앞뒤 순서를 구별할 필요가 없게끔 했음(이거 사용못하면 로봇앞/뒤 or 위/아래 구분해서 구현해야함 -> 개길어짐 -> 개 비효율적)
    q.append((pos,0))
    visited.append(pos)

    while q:
        pos, cost = q.popleft()
        
        if (n,n) in pos:
            return cost
        
        for next_pos in get_next_pos(pos, new_board):
            if next_pos not in visited:
                q.append((next_pos, cost+1))
                visited.append(next_pos)
    return 0


board = [
    [0,0,0,1,0],
    [0,0,0,1,0],
    [0,1,0,1,1],
    [1,1,0,0,1],
    [0,0,0,0,0]
    ]


print(solution(board))