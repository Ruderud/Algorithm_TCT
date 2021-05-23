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