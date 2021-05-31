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

#해답-이게뭔소리지? 다시봐야겠닼
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