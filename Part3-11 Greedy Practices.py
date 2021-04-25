#2021-04-25

import time

#Q1 모험가길드 (time over)
'''memo
최대 형성그룹갯수

모두를 넣을필요는없음.

sort정렬

23122->1 2 2 2 3
1
22
->23은 남겨두기
1,1,1,1,2,5

'''

'''
#n=int(input())
#traveler=list(map(int,input().split()))
n=6
traveler=[5,1,1,1,2,1]
traveler.sort()
print(traveler)
team=0

for i in traveler:          #traveler의 앞에서 하나씩 가져옴
    print(i)
    if traveler[i-1]:       #만일 가져온숫자에 해당하는만큼 모험자가 남아있다면
        del traveler[:i]    #앞에서부터 가져온 숫자만큼의 모험자를 뺀다.
        team+=1             #팀갯수 +1

    else:                   #가져온 숫자에 해당하는 만큼의 모험자가 없다면 중지         -> 이런방법금지!!
                                                                               for문 순회중에 list를 건드리면 문제가생기기때문!
                                                                               for문은 리터레이터에 의해 index 기준으로 하나씩 가져오기때문에
                                                                               리스트내용이 변하게되면 혼돈이 생기게된다.
        break
    print(traveler)
print(team)
'''

'''#해설. 일단 오름차순으로 정해서 공포도기준으로 순서대로 가져와서 그룹화 하는거는 맞음.
st=time.time()
n=int(input())
data=list(map(int,input().split()))
#n=5
#data=[2,3,1,2,2]
data.sort()

team=0
member=0

for i in data:
    member+=1
    if member >=i:      #공포도순서로 정렬했기때문에, 맴버의 수가 가져온 공포도 이상이라면 해당 팀을 묶고 팀 갯수에 +1시킨다.
                        #팀에 마지막으로 집어넣는 모험자의 공포도가 제일 높기때문에, 해당 모험자의 공포도만큼 팀원수가 된다면 그 시점에서 팀을 완성할 수 있기때문
        team+=1
        member=0
print(team)

end=time.time()

print("걸린시간 해설",end-st)
'''

'''+@Counter 함수
from collections import Counter

c=[1,1,2,2,3,3]

d=Counter(c)        #d = Counter({1: 2, 2: 2, 3: 2}) 
print(d.get(0))     #none출력
'''

'''#+@ 포인터(=플레그;c에서 사용하는 그 포인터 개념과는 다르다!)를 이용한처리방법 -> 확실히 더 빠르다!

st=time.time()

#n=int(input())
#data=list(map(int,input().split()))

n=5
data=[2,3,1,2,2]
sorted_arr = sorted(data,reverse=True)

n=len(data)
plag = 0            #첫플레그는 첫번째 캐릭에 꽂음
answer = 0

while plag < n :
    horror_p = sorted_arr[plag]   #플래그 꽃인 캐릭의 공포도를 가져옴
    if (plag+horror_p-1) < n :      #index계산 (플래그index+공포도-1)
        answer+=1
        plag+=horror_p              #첫 파티원의 공포도만큼 파티원을 무조건 넣어야하므로(내림차순정렬이므로 더 큰 공포도를 가진사람은 들어오지않음)
    else:                           #플레그를 해당 공포도만큼 이동시킨다.
        answer=-1                   #파티원수 이상으로 플레그가 옮겨져서 계산범위를 넘어간다면 해당 파티는 구성할수없으므로 만들던것을 제외시킴(-1)
        break                       #그리고 종료

print(answer)

end=time.time()

print("걸린시간 춘",end-st)
'''


#Q2 곱하기 혹은 더하기 (19:29)
'''로직
무조건 ->순서로만 연산이 이루어지게한다
이는 for문으로 앞에서 하나씩 가져온다고 생각하기

최대한 큰수를 만들려면 어떻게 해야할까? 
간단하다!
대상숫자와 그 뒤에숫자를 서로 +,*한 결과값을 비교해서 더 큰값으로 연산케하면된다.

02984
'''

'''
num=input()

result=int(num[0])          #개선! 83,86번줄처럼 str위치값을 int화하면 굳이 리스트화해서 가져오지않아도된다!

for i in range(1,len(num)):
    int_num=int(num[i])
    if result*int_num >= result+int_num:
        result=result*int_num
    else:
        result=result+int_num

print(result)
'''

'''#해설. 여기서는 결과값 비교가 아니라, 가져온수가 1이하일경우는 더하고 2이상일경우에는 곱하는 연산으로 구분했다.
data=input()
result=int(data[0])
for i in range(1,len(data)):
    num=int(data[i])
    if num <=1 or result <=1:
        result+=num
    else:
        result+=num
print(result)
'''