# 2021-04-11

print('='*50)

#예제 4-1 상하좌우
'''
memo
공간의 크기 NxN이 주어진다.
그리고 다음으로 움직임이 입력된다. 

움직임의 입력 총합수는 최종위치가 (x,y)라고할때, x+y-2=이동횟수 이다.(이게 왜필요함?)

입력한것을 u몇개 d몇개 각각 세고, r->+1, l->-1, u->-1 d->+1로 하여 계산한다. dic이용?
'''

'''내 풀이
n = int(input())
move = list(map(str,input().split()))
mv_type = {'U':-1,'D':+1,'L':-1,'R':+1}

x = 1
y = 1 

for i in move:
    if i == "U" or i == "D":
        if 0 == x + mv_type.get(str(i)) or n < x + mv_type.get(str(i)) :
            pass
        else : x+=mv_type.get(str(i))
    if i == "L" or i == "R":
        if 0 == y + mv_type.get(str(i)) or n < y + mv_type.get(str(i)):
            pass
        else : y+=mv_type.get(str(i))

print(x,y)
'''

'''해설
n = int(input())
x,y=1,1                  #굳이 x,y,를 따로 줄바꿈해서 쓰지않아도된다
plans = input().split()

#상 하 좌 우에 따른 이동방향 지정
dx = [0,0,-1,1]
dy = [-1,1,0,0]
move_types = ['L','R','U','D']

for plan in plans:

    for i in range(len(move_types)):
        if plan == move_types[i]:           # 첫입력 R에 대해 -> i에 1이 할당 -> 같아짐 -> dx[1] = 0, dy[1] = 1이므로 1,1 -> 1,2가된다
            nx = x + dx[i]
            ny = y + dy[i]
        
        if nx<1 or ny<1 or nx>n or ny>n:            #주어진 NxX바깥을 벗어날경우 무시. (U 입력시)
            continue

        x,y = nx, ny                        #이동
'''


#예제 4-2 시각 
'''memo
N을 넣었을때 00시 00분 00초 ~ N시 59분 59초까지의 range를 가지는 list를 가지게 한다. (최대24*60*60=86400 ㄱㅊ)

1시간(00시00분00초~00시59분59초)사이에 3이들어간 시간이 몇번오는지를 먼저 세는 방법을 구현(A)
이를 N에 3이 있을때로 (3,13,이외)으로 나누어 곱하여 그결과를 출력하게끔하자.

생각은 ㄱㅊ은데 구현할때 00시00분00초 -> 000000 으로 만들어서 3의 여부를 find를 이용했으면 금방했다
'''

'''
num=int(input())

count=0
for h in range(num+1):      #N시 59분 59초까지이므로 num으로만 하면 n-1시 59분 59초까지로밖에 안나와서 오답이 나온다!
    for m in range(60):
        for s in range(60):
            t = str(h)+str(m)+str(s)        #t를 따로 정의하지않 78행에 바로 써넣었으면 더 간단했을듯.
            if '3' in t:
                count+=1

print(count)
'''

#실전문제2 왕실의 나이트 (20분 over)
'''memo
이동방법은 좌+2상-1, 좌+2하+1 등 총 8가지가 있다. 각각의 좌표움직임을 [2,-1,0,0]이런꼴로 정의한다.
움직인 좌표위치가 1~8범위를 넘어설경우 count하지 않게끔하자.

그 결과값을 dic을 이용하여 행은 매칭을 시키던가, 아니면 ascii값을 이용해서 변환하던가 하자.
'''

'''
p = input()

x_mt= {"a":1,"b":2,"c":3,"d":4,"e":5,"f":6,"g":7,"h":8}

result=[]
x = int(x_mt.get(p[0]))
y = int(p[1])

#상하/좌우 상,좌 는 - 하,우는 +
move = [[2,1],[2,-1],[-2,1],[-2,-1],[1,2],[-1,2],[1,-2],[-1,-2]]

count=0
for i in range(8):
    dx=move[i][0]
    dy=move[i][1]
    print(x+dx,y+dy)
    if x+dx<1 or y+dy<1 or x+dx>8 or y+dy>8:
        continue
    count+=1

print(count)
'''

'''#해설
input_data = input()
row = int(input_data[1])
column = int(ord(input_data[0])) - int(ord('a')) + 1 #아스키코드값 'a'을 빼서 좌표를 숫자화했음.(ord)이용

steps = [(2,1),(2,-1),(-2,1),(-2,-1),(1,2),(-1,2),(1,-2),(-1,-2)] #여기서 움직임은 tuple로 했음.

result = 0
for step in steps:
    next_row = row + step[0]
    next_column = column + step[1]
    if next_row>=1 and next_row<=8 and next_column>=1 and next_column<=8:
        result+=1

print(result)
'''

#실전문제3 게임 개발
'''
외곽은 전부 물(1)로 되어있음을 염두.

입력을 일단 다 받고 맵을 이중리스트로 만든다

첫위치좌표 = 맵[x][y] -> 이동한내역리스트에 추가

보고있는 방향의 다음 땅 좌표 -> 
서 3일경우 맵[x][y-1]
남 2일경우 맵[x+1][y]
동 1일경우 맵[x][y+1]
북 0일결우 맵[x-1][y] 

이때 해당 좌표값이 0(땅)일경우 위치좌표는 보고있는 방향의 다음땅이 된다.

    0이면 일단 보고있는 좌표가 이동한 내역에 있는경우, 이는 pass
    그렇지않으면 count+=1
    주위의 좌표가 이동한 내역과 1(물)밖에없으면 종료하고 count를 출력

'''

'''입력
volume = list(map(int,input().split()))
start = list(map(int,input().split()))

map_data=[]


for i in range(volume[0]):
    map_data.append(list(map(int,input().split())))
    if len(map_data)==volume[1]:
        break

print(volume)
print(start)
print(map_data)


volume = [4, 4]
start = [1, 1, 0]
map_data = [[1, 1, 1, 1], [1, 0, 0, 1], [1, 1, 0, 1], [1, 1, 1, 1]]

#북동남서 순서
way = [0,1,2,3]

#방향전환
see = way.index(start[2]) -1
search_way = way[see]

#위치
position = [map_data[start[0]],map_data[start[1]]]

#방문한 땅 이력
save=[]
save.append(position)

next_p=
while True:
    see = way.index(start[2]) -1
    if see == 3:
        next_p = [position[0],position[1]-1]
    elif see == 2:
        next_p = [position[0]+1,position[1]]
    elif see == 1:
        next_p = [position[0],position[1]+1]
    elif see == 0:
        next_p = [position[0]-1,position[1]]
    print(next_p)

'''

#해설
#n은 세로, m은 가로
n,m = map(int,input().split())

#리스트 컴프리핸션을 이용하여 NxM크기의 2차원 리스트를 만들었음(0으로 초기화 해놓았음) -> 입력된 맵에 쓰는게아님!
d = [[0] * m for _ in range(n)]

#현재캐릭터의 위치x,y와 방향(dir)을 입력
x,y,dir = map(int,input().split())

d[x][y] = 1 #현재의 위치를 방문처리 = 1은 물이므로 갈수없는곳으로 처리 ->이래야 나중에 방향전환시 있었던 땅을 1로 인식해서 안간다.

#전체맵정보 입력
array=[]
for i in range(n):
    array.append(list(map(int,input().split())))

print(d)
print(array)

#북-동-남-서 방향정리
dx = [-1,0,1,0] 
dy = [0,1,0,-1]

#왼쪽으로 회전
def turn_left():
    global dir 
    dir -= 1  #방향값에서 -1씩해서 서 남 동 북 순서로 방향을 전환시킴
    if dir == -1 :
        dir = 3         #북(0)에서 -1하여 서쪽으로 바꿨을떄 3의값을 다시 할당해줌

#시뮬레이션

count = 1 #첫위치를 포함해야함
turn_time = 0

while True:
    #왼쪽회전
    turn_left()
    nx = x + dx[dir]  #방향 전환할때마다 정의한 방향정리에 따라 바라보는곳의 좌표값을 가져온다
    ny = y + dy[dir]
#idea! -> 방향값이 0,1,2,3인것은 대놓고 좌표로 쓰라고 주는 힌트였음.

    if d[nx][ny] == 0 and array[nx][ny] ==0:   #바라보는곳이 방문한 땅이 아니면서, 좌표값상에서 땅(0)이면 이동
        d[nx][ny] = 1                          #방문한 위치값을 저장
        x = nx 
        y = ny
        count += 1                             #새로운땅으로 갔으므로 count+1
        turn_time = 0                          #마찬가지로 새로운 땅으로 
        continue

    else:
        turn_time += 1                         #방향전환했을때 본곳이 갔던땅이거나 바다일때 회전수 +1
    
    if turn_time == 4 :                        #4방향 다돌아봤을때 갈곳이 없을떄
        nx = x - dx[dir]
        ny = y - dy[dir]                       #현재 바라보고있는 방향의 반대편으로 이동방향을 잡고

        if array[nx][ny] == 0:                 #뒤에 땅이있는경우 이동한다.
            x = nx
            y = ny

        else:                                   #이마저도 못하면 시뮬레이션을 종료한다.
            break
        turn_time=0                            #뒤로갔으면 다시 회전수 0으로 돌린다.(시뮬레이션이 끝나지않았을때를 고려)

print(count)        #결과값 출력








print("="*50)