#23 국영수
'''
sort()는 NlogN이므로, n=100000이면 10만*5 = 50만이니까 sort는 이중으로는 쓰면안댐
그러면 데이터를 받으면서 바로 결과값에 집어넣는 방식 이용

데이터를 처음받았을때는 일단 result에 바로 넣어버리고

두번쨰부터 숫자를 비교시작함.

입력받으면서 바로 자리를 배정하는 방식


'''

'''
n = int(input())
data = []
result = []

for a in range(n):
        raw = list(map(str,input().split()))
        for i in range(3):
            raw[i+1] = int(raw[i+1])
        data.append(raw[0])
 
data = [
['Junkyu', 50, 60, 100],
['Sangkeun', 80, 60, 50],
['Sunyoung', 80, 70, 100],
['Soong', 50, 60, 90],
['Haebin', 50, 60, 100],
['Kangsoo', 60, 80, 100],
['Donghyuk', 80, 60, 100],
['Sei', 70, 70, 70],
['Wonseob', 70, 70, 90],
['Sanghyun', 70, 70, 80],
['nsj', 80, 80, 80],
['Taewhan', 50, 60, 90]
]

kor = sorted(data, key=lambda name : name[1])

a = sorted(kor, key=lambda name : if name[1] ==)


for i in a:
    print(i)
'''

'''#해답
#sort() 함수의 key값을 지정해서 맟춤정렬이 가능하다. (모르면 개고생)
"""
n = int(input())
data = []

for _ in range(n):
    data.append(input().split())  #[['a', '1', '2', '3'], ... ['b', '1', '5', '7']] 꼴로 입력된다. 
"""

data = [ #이름,국,영,수 순서
['Junkyu', 50, 60, 100],
['Sangkeun', 80, 60, 50],
['Sunyoung', 80, 70, 100],
['Soong', 50, 60, 90],
['Haebin', 50, 60, 100],
['Kangsoo', 60, 80, 100],
['Donghyuk', 80, 60, 100],
['Sei', 70, 70, 70],
['Wonseob', 70, 70, 90],
['Sanghyun', 70, 70, 80],
['nsj', 80, 80, 80],
['Taewhan', 50, 60, 90]
]

data.sort(key=lambda x : (-int(x[1]),              int(x[2]),                    -int(x[3]),                      x[0]                   ))
#                        국어성적을 기준으로 내림차순 정렬/국어성적이 같으면 수학점수 오름차순정렬/수학성적까지 같으면 영어성적 내림차순 정렬/모든성적이 같으면 이름알파벳순 정렬
#되도록 element를 튜플로 받아서 정렬을 사용하자
for name in data:
    print(name[0])
'''

#24 안테나
'''
평균을 쓰면 편차가 큰값이 평균값에 큰 영향을 미치기때문에 옳지못한 결과가 도출된다


'''


'''#내답 괜히 어렵고 비효율적임
n = int(input())
house = list(map(int,input().split()))

average = sum(house) // len(house)

cost = 0
for i in house:
    cost += abs(average-i)

while 1: #비교할 코스트가 현재 코스트보다 작다면 계속 진행함

    for move in [1,-1]:
        if move == 1:   #오른쪽한칸 이동후 코스트 총합계산
            compare_cost_r = 0
            now_r = average + move
            for i in house:
                compare_cost_r += abs(now_r-i)
            
        else :
            compare_cost_l = 0
            now_l = average + move
            for i in house:
                compare_cost_l += abs(now_l-i)
        
    if compare_cost_r > compare_cost_l:
        compare_cost = compare_cost_l
        now = now_l
    elif compare_cost_r < compare_cost_l:
        compare_cost = compare_cost_r
        now = now_r
    else :
        compare_cost_l = compare_cost_r
        compare_cost = compare_cost_l
        now = now_l
    
    if cost < compare_cost:
        print(average)
        break
    elif cost == compare_cost:
        print( min(average,now))
        break
    else:
        cost = compare_cost
        average = now
'''

'''#해답 정확히 중간값에 해당하는 원소의 집에 지으면된다는것을 알아야함
#1 2 3 5 8 9 -> 평균값인 4.66..과 가까운값인 3 또는 5에서 항상 최소값을 얻을 수 있음

n = int(input())
house = list(map(int,input().split()))
house.sort()

print(house[(n-1)//2])
'''

#Q25 실패율


'''#내답 정확도점수 70.4 나머지는 런타임에러
N= 5
stages = [2,1,2,6,2,4,3,3]

def solution(N, stages):
    clear = [[] for _ in range(N+2)]        #편의상 n개의 스테이지, 올클, 첫 빈칸용을 추가

    for i in range(len(stages)):
        clear[stages[i]].append(stages[i])

    clear_rate = [[] for _ in range(N+2)]

    for i in range(1,N+1):
        count = 0
        for j in range(i,N+2):
            count += len(clear[j])
        rate = len(clear[i]) / count
        clear_rate[i].append(rate)

    result = []
    for i in range(1,N+1):
        result.append((clear_rate[i][0],i))

    result.sort(key = lambda x : -x[0])

    result = [i[1] for i in result]
    return result

solution(N,stages)
'''

'''#해설 런타임오류 없이 패스
def solution(N,stages):
    answer = []
    length = len(stages)

    for i in range(1,N+1):
        count = stages.count(i)     #1스테이지부터 해당스테이지에 머물러있는 사람수를 더함

        if length == 0:             #실패율계산. 머무른사람이 없다면 0으로 출력
            fail = 0
        else:
            fail = count / length

        answer.append((i,fail))
        length -= count             #계산이 끝난 스테이지의 사람들 수를 제거해서 다음층에 도달한 사람들의 수를 만든다

    answer = sorted(answer, key = lambda x : x[1], reverse=True)

    answer = [i[0] for i in answer]
    return answer
'''

#Q26 카드 정렬하기
'''
10,20,30,40을 정렬한다고하면

    10+20
    +
    10+20+30
    +
    10+20+30+40
->이런식의 총합이 최소값이 되므로, 오름차순정렬후, 앞부터 순차적으로 더하게하는것이 최소값

다만 dp적 사고를 접목해서 재귀화 구현할때, 계산한 데이터를 저장시키고, 이것을 다음계산시에 쓰게해서 시간및 메모리복잡도를 낮춰야함
'''

#내답 -> 틀렸음 앞에서 순차적으로 더하는것만 생각하면 순차적으로 더한값이 그 뒤에있는 원소간 합보다 큰경우를 피할 수 없음.
#Ex) 10 20 21 22 일때, 내답은 154지만 이상적인 조합은 30+43+73으로 146이어야함
#차라리 데이터리스트에서 최소값을 빼내고 뺴낸 위치에 inf를 대체, 다음 최소값을 빼내고 inf를 대체하고 빼낸 두값을 더한다음에 데이터 리스트에 추가하고
#이를 재귀적으로 반복하는것이 더 나았을듯

'''
n = int(input())
data = []
for _ in range(n):
    data.append(int(input()))

data.sort()                             #n = 10만일때 -> 50만 사용

sum_data = [0] * (n-1) 

def cal(n):
    if n == 1:
        return data[0]
    
    if n == 2:
        return data[0] + data[1]
    
    sum_data[0] = data[0] + data[1] 
    for i in range(3,n+1):                              #n = 10만일때 -> 10만 사용 , total 60만사용
        sum_data[i-2] = sum_data[i-3] + data[i-1]

    return sum(sum_data)

print(cal(n))
'''

''' 이방법을 써도 min / index 과정에서 O(N)을 두번이나 사용하므로 오히려 pop보다 효율이 안좋아지므로 그만둔다
n = int(input())
data = []
for _ in range(n):
    data.append(int(input()))

n = 4
data = [10,20,40,50]

INF = 1e9

def pop_min_value(data):        #data 리스트내에서 최소값을 가져오고, 그 자리에 INF값을 대체한다. 1회연산시 최대 20만
    a = min(data)
    a_index = data.index(a)
    data[a_index] = INF
    return a

print(data)

record = []

#최대 10만번 for문연산
for _ in range(n-1) :         #data내에 INF 갯수가 n+1개가 될때까지 수행. ex)원소 3개일때 모든연산후 data = [inf,inf,inf,inf,val]상태
    a = pop_min_value(data) + pop_min_value(data)  #20만 + 20만 = 40만
    data.append(a)
    record.append(a)
    print(data)

print(sum(record))
'''


'''#해설 뒤에있는 원소간 합이 순차적 합보다 작은 경우를 비교해서 항상 최소값끼리 선택해서 더하도록 함 -> heapq로 구현함(최소값을 찾는데 걸리는 시간은 logN 보장)

# + 최대/최소값을 항상 빼와야할때는 힙큐 방식이 제일좋고, 그외의 n번째 값같은 극값이 아닌 다른 정렬상태의 값을 필요로한다면 tree구조가 좋다.
# 힙큐방식은 구조 트리를 모두 정렬해놓지않는다! 최대or최소값만 관심이 있기 때문에, 그 값만 도출시켜놓고 나머지 값은 흐트러져있는 상태임.
# 그렇기때문에 최소/최대 값을 빼내고, 새로운 최대/최소값을 올려야할때마다 logN회의 부모-자식노드간 교환작업을 통해 최대/최소값을 얻는다

import heapq

n = int(input())

heap = []
for i in range(n):
    data = int(input())
    heapq.heappush(heap,data)   #힙 리스트에 힙큐자료구조로써 모든데이터를 입력한다

result = 0

while len(heap) != 1:                   #카드뭉치가 모두 합쳐질때까지 연산수행
    first = heapq.heappop(heap)         #최소값을 first에,
    second = heapq.heappop(heap)        #그다음 최소값을 second에 heapq 구조를 이용해서 할당하고

    sum_value = first + second
    result += sum_value
    heapq.heappush(heap, sum_value)     #힙큐 자료구조에 최소값을 더한 값을 다시 집어넣는다

print(result)
'''


