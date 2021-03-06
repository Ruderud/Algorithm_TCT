#Q31 금광
'''
완전탐색하기엔 애매한데
근데 극한의 상황에선 완탐말고 로직을 만들어낼만한 구석이 없음
일단 움직이는 방법을 고려하지않고 중복을 제외한 세로칸에서 하나씩 선택해서 m개의 금광칸을 캐는 조합을 싹다 만들어놓고
거기서 움직일수 없는 상황인 조합을 제거하는정도면 완전탐색 가능할거같은데?


'''



'''
t = int(input())

test = [[] for _ in range(t)]     #t개 케이스리스트를만듬

for i in range(t):
    n, m = map(int,input().split())
    test[i].append([n,m])

    data = list(map(int,input().split()))
    array = []
    for x in range(n):
        array.append(data[:m])
        del data[:m]
        
    test[i].append(array)
'''
'''
test = [
    [[3, 4],
    [[1, 3, 3, 2], 
     [2, 1, 3, 1], 
     [0, 6, 4, 7]]], 
     
    [[4, 4], 
    [[1, 3, 1, 5], 
     [2, 2, 4, 1], 
     [5, 0, 2, 3], 
     [0, 6, 1, 2]]]
     ]

from abc import abstractmethod
from itertools import combinations

def next(num,n):
    next = [num-1,num,num+1]
    if -1 in next:
        del next[0]
    if n in next:
        del next[-1]
    return next



def simulation():
    for i in range(n):
        simulation()
    #여기서 재귀화 하면서 진행할 수 있는 모든 루트에 대해서 채굴값을 result에 저장하게하면 될거같은데


for i in test:
    n, m = i[0][0], i[0][1]
    test_array = i[1]
    result = []
    simulation()
    print(max(result))
'''

'''#해답 좌상,좌,좌하단에 대해서 가장 많은 금을 가지고있는 경우를 저장하면서 진행하는것을 구현
"""
1332    x532    xx8   2    xxx(14)
2141 -> x341 -> xx(12)1 -> xxx(13) -> 이중 19가 가장 큰값!
0647    x847    xx(12)7    xxx(19)
"""

for tc in range(int(input())):
    n,m = map(int,input().split())
    array = list(map(int,input().split()))

    dp = []
    index = 0
    for i in range(n):                      #입력받은 한줄짜리 array값을 nxm행렬화
        dp.append(array[index:index+m])
        index += m


    for j in range(1,m):    #좌측에서 올수있게, 좌측에서 한칸떨어져서부터 연산시작
        for i in range(n):

            if i == 0:          #좌상단에서 올때/맨윗단이면 없음처리
                left_up = 0
            else:
                left_up = dp[i-1][j-1]
            
            if i == n-1:        #좌하단에서 올때/맨아랫단이면 없음처리
                left_down = 0
            else:
                left_down = dp[i+1][j-1]
            
            left = dp[i][j-1]   #좌측에서 올때
            dp[i][j] = dp[i][j] + max(left_up,left_down,left)       #3방향에서 올때의 값중 가장 큰값을 현재좌표에 저장함
    
    result = 0
    for i in range(n):
        result = max(result, dp[i][m-1])

    print(result)
'''

#Q32 정수 삼각형

'''#내답
n = int(input())
array = []

for i in range(n):
    array.append(list(map(int,input().split())))

for x in range(1,n):
    for y in range(len(array[x])):
        
        if y == 0:
            left = 0
        else:
            left = array[x-1][y-1]

        if y == len(array[x])-1 :
            top = 0
        else:
            top = array[x-1][y]

        array[x][y] = array[x][y] + max(left, top)

print(max(array[n-1]))
'''

#해설-Q31과 문제유형이 같으며, 따라서 답변도 위와 그 맥락이 동일하므로 작성생략

#Q33 퇴사
'''
데이터는 
[[], [3,5], [5,20], [1,10], [1,20], [2,15], [4,40], [2,200] ]꼴로 받음 (n+1개 리스트)

처음에 index + Ti > N인것들은 전부 데이터를 1,0으로 바꿔버림(그날 하루는 수입없는것)
이떄 수입이 0인것은 무조건 수락으로 함

그리고 순차적으로 1부터 불러와서 1일차업무를 할때와 안할때를 나눔
1일차 Y -> 4일차 Y -> 5일차 Y -> 7일차 Y -> 8일차 끝 result = 45 
          4일차 N -> ...
1일차 N -> 2일차 Y -> 7일차(1,0) Y -> 8일차 끝
          2일차 N -> 

이런식으로 순차적으로 전수조사시킴.
n이 해봤자 15라서 ㄱㅊ
'''

'''#내답 -> 왜 work에서 날짜가 n초과라 멈춰야하는데도 계속 돌아가지?
n = int(input())
schedule = [[]]
for _ in range(n):
    schedule.append(list(map(int,input().split())))

print(schedule)

#n = 7
#schedule = [[], [3, 10], [5, 20], [1, 10000], [1, 20], [2, 15], [4, 40], [2, 200]]

for i in range(1,n+1):          #퇴사일넘는계획은 걸리는시간을 1, 보수를 0으로 만든다
    if i + schedule[i][0] > n:
        schedule[i][0] = 1
        schedule[i][1] = 0

print(schedule)

date = 1
total=0
yes_or_no = ['Y', 'N']

result = []


def work(date,total):
    if date > n:                     #n+1일차일시 퇴사하고 지금까지번돈(total)을 income에 넣음
        return result.append(total)

    else :
        for select in yes_or_no:
            
            if select == 'Y':
                total += schedule[date][1]
                date += schedule[date][0]
                work(date, total)

            else:
                total += 0
                date += 1
                work(date,total)
    
work(date,total)

print(max(result), result)
'''

"""#해답
'''
dp[i] = max(p[i]+dp[t[i]+i], max_value) = i번째 날부터 마지막날까지 낼 수 있는 최대 이익

max_value = 현재까지의 최대상담금액
p[i] = i번째 날의업무 수행시 얻는 이익
t[i] + i = i번째날의 업무수행시 끝나는 날의 날짜

따라서 p[i]+dp[t[i]+i]의 의미는 i번쨰 날의 업무수행보수를 추가한, i번째일까지의 최대이익이 된다.

dp[i]와 max_value를 0으로 초기화해서 뒤에서부터 역순으로 계산해나가며, 
i번째 날의 업무 수행완료일이 마지막날을 초과하는 경우는 수행하지않고, 지금까지의 최대 이익을 유지시켜버리게 하므로

이러한 계산방법은 중간에 (N,10000)과 같은 n이 아슬아슬하게 마지막날까지 지속되는 대신 엄청 큰 튀는 보수가 생기는 경우가 생기면
해당 일자(iN)에서 i+N일까지 사이에는 어떠한 일도 없었던것으로 치고, dp[N+i]로써 그 종료일 이후에 얻을 수 있는 최대의 보수만 고려하기 때문에
튀는 케이스를 놓치지 않을 수 있다.


ex) i = 6(7일)은 time이 n을 넘어버리므로 패스
    i = 5(6일) 상동
    i = 4(5일) -> dp[4] = max(15+dp[6](=아직 0) -> 15, 0  )
    i = 3 -> (dp[4] =) 15 + 20 = 35 ->dp[3] =35
    i = 2 -> dp[4](=35) + p[2](=10) = 45로써 최대값이 갱신된다.
    i = 1 -> dp[6](=0) + 20은 지금까지의 최대값 45보다 작으므로 무시
    i = 0 -> dp[4](=35) + p[0](=10) = 45로써 최대값이 유지된다.

    결과값은 45 (단, 일하는 방식은 1일 4일 5일 // 3일 4일 5일 두가지 방법이 있다. 최대값만 구하면되는것이기에 큰의미없음)
'''
n = int(input())
t = []              #상담을 완료하는데 걸리는시간
p = []              #상담을 완수했을때 받는 보수

dp = [0] * (n+1)    #i일까지의 최대수익 초기값을 0으로 초기화
max_value = 0       #현재 검사하는 최대 상담금액 (i번째일)을 계속 갱신하면서 hold하는 역할. 이것또한 0으로 초기화

for _ in range(n) :
    x, y = map(int,input().split())
    t.append(x)
    p.append(y)

for i in range(n-1,-1,-1):      #뒤에서부터 거꾸로확인, n=7일때 6,5,4,3,2,1,0순으로 확인
    time = t[i] + i             #time은 i번째 날짜의 업무가 끝나는 날을 알려줌

    if time <= n:               #상담이 퇴사전에 끝날때
        dp[i] = max(p[i] + dp[time], max_value) #해당날짜까지의 보수 최대값을 리스트에 저장
        max_value = dp[i]                       #역순으로 계산하면서 hold할 값을 최대값으로 갱신함
    
    else:
        dp[i] = max_value                       #퇴사일 이후에 끝나는일은 하면 안되기에, 지금까지의 최대값이 해당날짜까지의 최대값이 된다.


print(max_value)
"""

#Q34 병사배치하기
'''
하향구간을 박스화해서 다음박스의 최대값을 기준으로하여, 현재박스에서 해당값 이하의 갯수(a)와 다음박스의 갯수(b)를 비교해서 현재박스에 추가하는 방식으로 반복

'''

'''#내답
n = int(input())
array = list(map(int,input().split()))
boxes = []
start=0

def compare_count(box, compare_value):
    count = 0
    for i in box:
        if i < compare_value:
            count+=1
    return count

for i in range(len(array)):

    if i+1 < n and array[i] < array[i+1] :
        boxes.append(array[start:i+1])
        start += i+1

    if i == n-1:
        if min(boxes[-1]) < array[i]:
            boxes.append([array[i]])
        else:
            boxes[-1].append(array[i])

main_box = boxes[0]

for box in boxes[1:]:
    compare_value = box[0]
    a = compare_count(main_box, compare_value)
    b = len(box)

    if a < b :                                  #뒷박스길이가 더길어서 가져다 붙이는게 좋을떄
        del main_box[-a:]
        main_box = main_box + box
    elif a > b:                                 #뒷박스길이가 더 짧아서 현재박스보다 작은값들만 붙이는게 좋을때
        c = compare_count(box, min(main_box))
        del box[-c:]
    else:                                       #뒷박스길이와 현재박스 비교박스길이가 같을떄
        if min(box) < min(main_box):                #현재박스의 비교박스최소값이 더 클때는 지금을 유지
            continue
        elif min(box) > min(main_box):              #뒷박스의 최소값이 더 클때는 해당 비교구간을 뒷박스로 바꿈
            del main_box[-a:]
            main_box = main_box + box
        else:                                       #양쪽의 비교한 박스의 최소값이 같을때는 뒷박스의 최소값을 뒤에 붙여서 길이를 늘림
            main_box.append(min(box))

print(boxes)
print(main_box)
print(n - len(main_box))
'''

'''#해답
"""
가장긴 증가하는 부분 수열유형문제

사용한 점화식은 D[i] = max(D[i], D[j]+1) if array[j] < array[i] 이다
        (D는 주어진 배열i번째값을 마지막으로하는배열의 길이 최대값)
ex array = [10,20,10,30,20,50]일때
    [1,1,1,1,1,1]
    [1,2,1,1,1,1]
    [1,2,1,1,1,1]
    [1,2,1,3,1,1] -> i=3에대해 처리한다. 이때 array[3]=30을 끝값으로하는 최대값은 10 20 (10->제거) 30 이라 10 20 30인 3개가 된다.
    [1,2,1,3,2,1]
    [1,2,1,3,2,4] ->
즉, 주어진배열에서 순차적으로 값을 집어넣었을때 이전최대값보다 큰값이라, 더 길이가 늘어날수있는지 주어진 배열의 index에 대해서 하나씩 확인하면된다

=> 이러한 개념을 문제에서 반대로 적용하면된다.(가장)
+ 슬라이싱해서 비교할때, 검사중인 값과 동일한 level이라면 해당 검사중인 값도 배열에 추가해서 길게해야 최소한의 배출병사를 계산가능
"""
n = int(input())
array = list(map(int,input().split()))

array.reverse()                         #주어진 배열을 뒤집어서, 오름차순으로 가장 긴 배열을 가지게끔 만들어보면된다
dp = [1] * n                            #해당 인덱스위치값을 배열의 끝으로하는 최대길이를 저장할 배열

for i in range(1,n):                    #비교할대상은 index기준 1번부터 n번까지이다.
    for j in range(0,i):                #각 최대 비교값을 i위치로하는 array앞에서부터의 배열을 슬라이싱
        if array[j] <= array[i]:        #슬라이싱한것을 앞부터 순차적으로 검사할때 검사값이 끝값보다 작거나 같으면 
            dp[i] = max(dp[i], dp[j]+1)         # 검사를 위해 슬라이싱한 맨 끝값의 dp vs 현재 검사중인 위치에 대한dp +1 해서 큰값을 해당 dp[i]에 저장

print(n-max(dp))                        #n에서 가장 긴배열의 길이만큼 뺴면, 이것이 최소한으로 뺴야할 병사의 수가 된다.
'''


#Q35 못생긴 수
# f(n) = f(n-1) +1, +2, +3... 씩하면서 2,3,5로 나눠지는지 확인


'''#내답 n번째까지 순차적으로 2,3,5로 나눠서 만족하는경우를 저장하고 출력하게끔했음 -> 시간초과!
n = int(input())

numbers = [1]

def div(num):
    if num == 1:
        return True
    elif num % 2 == 0:
        return div(num/2)
    elif num % 3 == 0:
        return div(num/3)
    elif num % 5 == 0:
        return div(num/5)
    else:
        return False

counter = 1

if n == 1:
    print(1)
else:
    i=1
    while counter < n:
        target = numbers[-1] + i

        if div(target):
            numbers.append(target)
            counter+=1
            i=1
        elif not div(target):
            i+=1
    print(numbers[-1])
'''

'''#해설-> 기존의 수에 2,3,5를 곱하면 만족하는 수가 나온다는것을 이용해서 계산
"""
min(next2, next3, next5) 조합
2 3 5   ->2 쓴다
4 3 5   ->3 쓴다
4 6 5   ->4 쓴다
6 6 5   ->5 쓴다
6 6 10  ->6 쓴다 이때 2의배수 6과 3의배수 6둘다 갱신하는데, ugly배열의 4번째(i2=4)*2인 8과 ugly배열의 3번째(i3=3)*3인 9로 갱신
8 9 10
10 9 10
...
"""
n = int(input())

ugly = [0] * n 
ugly[0] = 1

i2 = i3 = i5 = 0

next2, next3, next5 = 2,3,5

for l in range(1,n):
    ugly[l] = min(next2, next3, next5)

    if ugly[l] == next2:
        i2 += 1
        next2 = ugly[i2] * 2
    if ugly[l] == next3:
        i3 += 1
        next3 = ugly[i3] * 3
    if ugly[l] == next5:
        i5 += 1
        next5 = ugly[i5] * 5

print(ugly[n-1])
'''

#Q36 편집거리


'''#내답
a = str(input())
b = str(input())

len_a = len(a)
len_b = len(b)

frame_a, frame_b = [0] * len_b, [0] * len_b

for i in range(len_b):
    frame_b[i] = b[i]

index = 0
for i in range(len_a):
    for j in range(index,len_b):
        #만일 a의 프레임이 비어있고, 해당위치의 알파벳이 a의 i번째 알파뱃과 같으며, 이때 집어넣는다.
        if frame_a[j] == 0 and a[i] == frame_b[j]:
            frame_a[j] = a[i]
            index = j+1     #문자의 집어넣는 순서가 뒤바뀌지않게 처음 검사하는 위치를 조정
            break

print(frame_a.count(0))
'''

'''#해답
#2차원 dp 테이블을 만들고, 좌상->우하 방향으로 "결과값"의 의미를 담은 계산을 수행
# 1. 교차점에서 서로 문자가 같을때 = 좌상단값을 그대로 가져옴 (a문자의 해당 위치의 알파벳을 그대로 살림)
# 2. 교차점에서 서로 문자가 다를때
# 2-1. 좌상단->교차점 = 위치알파벳을 그대로 가져오되, +1하여 b의 해당위치 알파벳으로의 변환
# 2-2. 좌측->교차점 = +1 새로운 알파벳을 추가로 삽입
# 2-3. 상단->교차점 = +1 해당위치 알파벳을 지우는 작업
# 위의 dp[i][j] = if 같을때는 dp[i-1][j-1], else는 min(dp[i][j-1](=삽입), dp[i-1][j](=삭제), dp[i-1][j-1](=변환)) +1(작업) 으로 수행

def edit_dist(str1, str2):
    n = len(str1)
    m = len(str2)

    dp = [[0] * (m+1) for _ in range(n+1)] #세로축이 a, 가로축이 b의 문자열길이 각각의 +1한 2차원 dp테이블 생성

    for i in range(1,n+1):
        dp[i][0] = i
    for j in range(1,m+1):
        dp[0][j] = j

    for i in range(1, n+1):
        for j in range(1, m+1):

            if str1[i-1] == str2[j-1]: #교차점 문자가 같을때
                dp[i][j] = dp[i-1][j-1]
            
            else:
                dp[i][j] = 1 + min(dp[i][j-1], dp[i-1][j], dp[i-1][j-1])
    
    return dp[n][m]

str1 = input()
str2 = input()

print(edit_dist(str1, str2))
'''



