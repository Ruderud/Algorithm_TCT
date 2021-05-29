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