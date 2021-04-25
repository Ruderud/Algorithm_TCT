#2021-04-10

print('='*50)

#3-1 거스름돈 문제.

n = 1260

f = n // 500
n_f = n - (500*f)

o = n_f // 100
n_f_o = n_f - (100*o)

ff = n_f_o // 50
n_f_o_ff = n_f_o - (50*ff)

t = n_f_o_ff // 10

print("500원 %s개, 100원 %s개, 50원 %s개, 10원 %s개." %(f,o,ff,t))


n = 1260 #풀이

count = 0

coin_types = [500,100,50,10]

for coin in coin_types:
    count += n // coin
    n %= coin

print(count)

print('-'*50)

#실전문제2 - 큰수의법칙
''' 내개 생각한거
 두번쨰 입력된 인자들 ex 2 4 5 4 6 에서 pop을 통해 가장 큰 숫자 순으로 index와 그 값을 map으로 뽑아내고,
 (가장큰거 * k) + 두번째 큰거 1번 + (가장큰거 * k) ...순으로 계산하는데, 이때 count에 k번 + 1 + k본 +1...순으로 수행해서 m이상시 더하는것을 중지
 이후 더한 result를 출력
 for i in m을 해서 m회 반복계산하는데, 그속에서 for j in k를 사용해서 최대값을 k번더하고 두번째값을 한번더하는것을 while문으로 돌린다.
'''

#풀이
'''
n,m,k=map(int,input().split())         #공백기준 n,m,k각각에 숫자 할당하기
data = list(map(int,input().split()))  #공백기준으로 n개의 수입력받기

data.sort()
first = data[-1]                       #제일큰숫자
second = data[-2]                      #두번째로 큰 숫자

result=0
"""
while 1:                #first를 k번 더하고 second를 한번더하는것을 무한히 반복한다.
    for i in range(k):      #k번 제일큰 수를 더한다
        if m == 0:              #이때 m번 다 더했다면(m=0) while문을 종료한다
            break
        result+=first       
        m-=1                #1번더할때마다 m에서 1씩뺀다
    if m == 0 :                 
        break
    result+=second          #두번째로 큰수를 한번만 더한다
    m-=1                    #덧셈을 한번했으므로 m에서 1를 차감한다.

print(result)
"""

#여기서 가장 큰수와 두번째로 큰수가 더해지는것은 일정한 수열이 있는것을 파악한다면 더 효율을 높일수있다.
# 6 6 6 5 // 6 6 6 5 //  6 6 6 5 .... 반복이다.
# 그렇기에 총 더하는 횟수 = ( m / ( k(6을 더하는 횟수) + 1(5를 더하는 횟수) ) ) * k  +  M % (k+1)(m을 k+1로 나누고 남은 나머지) 이다.

result+= (((m/(k+1))*k) + (m%(k+1)))*first
result+= (m//(k+1))*second

print(result)
'''

#실전문제3 - 숫자카드게임 (12:33 걸림)
'''
memo
nxm 행렬을 먼저 입력했을때 이를 [[1,2,3] [4,5,6] [7,8,9] [10,11,12]] 꼴로 정리하자.
    1 2 3\n
    4 5 6\n
    7 8 9\n   -> 먼저 \n을 기준으로 행을 나누고, 그 다음 각각의 행을 공백을 기준으로 정리하자
    아니다 이거 잘안된다 그냥 n번 입력받고 이걸 list화해서 각각의 행에서 최소값을 바로 얻어내는게 낫다
속 리스트는 한개의 행에있는 숫자들을 의미하도록 하자.

그리고 각 행마다의 최소값을구하고, 이를 새로운 리스트로 만들자.

새롭게 만들어낸 리스트에서의 최대값을 출력하자.

'''


'''내 풀이
n,m = map(int,input().split())

card =[]
for i in range(n):
    data=list(map(int,input().split()))
    card.append(min(data))

print(max(card))
'''



#해설
''' min 이용
n,m = map(int,input().split())

result=0
for i range(n):
    data=list(map(int,input().split()))
    min_value=min(data)
    result = max(result, min_value)

print(result)

2중 반복문 이용
n,m = map(int,input().split())

result=0
for i range(n):
    data=list(map(int,input().split()))
    min_value= 10001
    for a in data:
        min_value = min(min_value,a)
    result = max(result, min_value)

print(result)
'''

#실전문제4 - 1이될때까지 (9:21 걸림)
'''
memo

값 n이 주어지고 이를 1. -1씩하거나, 2 k로 나누거나(나머지가 0일때만가능) 해서 n=1을 가장 빠르게 만드는것이 목표이다.

둘다 정수라는 전제하에, 항상 k로 나누는것이 가장 빠르다.

k=2일때서야 겨우 2에서 -1을 하나, 2로 나누나 그 효과가 똑같기 때문이다. 그러니 최대한 k로 자주나눠야함.

일단 n,k값을 각각 받고, while문을 사용해서 계속 반복하게한다. (두시행중하나를 실시할때마다 result+=1)

if문 2개를 사용하여 %k==0일때 수행, 그이외는 -1씩!

'''

n,k=map(int,input().split())

result=0
while n!=1:
    if n%k==0:
        n/=k
        result+=1
    else:
        n-=1
        result+=1

print(result)

#풀이 +@ -> 100억이상의 큰수일때, 이를 빠르게 계산하기 위해서는 n이 k의 배수가 되도록 효율적으로 빼게해야함.

n,k=map(int,input().split())

result=0
while 1:
    target = (n//k)*k           #n에 가장 가까운 k의 배수를 구하기
    result += (n-target)        #-1씩 빼는 횟수 = n에서 k의 배수가 될때까지 빼는 횟수 -> 이를 result에 추가.
    n = target
    if n<k:                     #더이상 k로 나눌수 없을때 while문 탈출
        break
    result+=1
    n//=k

result+=(n-1)                   #마지막에 남은 수n에 대해 -1씩 빼는 횟수를 더함
print(result)













print('='*50)