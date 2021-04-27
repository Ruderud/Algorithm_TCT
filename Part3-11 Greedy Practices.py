#2021-04-25

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


#Q3 문자열 뒤집기 (time over)
'''
문자열을 연속적인 0의갯수와 1의 갯수를 세고, 
더많은것을 flip한다.
-> 그냥 연속적인 덩어리갯수가 더 적은 수 = 수행해야하는 횟수이다.

오타 안생기게 조심해야한다!
'''

'''#해설 : 0001110010 -> 전부 1로 바꾸는데 걸리는 횟수 or 전부 0으로 바꾸는데 걸리는횟수를 비교해서 최소값을 반환함
data = input()
count0=0
count1=0

if data[0] == "1":      #처음숫자가 1이면 0으로 바꾸는데 걸리는 횟수(count0)에 +1 // 첫 숫자가 0이면 1로 바꾸는데 걸리는 횟수에 +1
    count0+=1
else:
    count1+=1

for i in range(len(data)-1):
    if data[i] != data[i+1]:    #문자열 자리하나씩 이동하다가, 달라지게 되면 이때 다음숫자가 1이라면 count0에 1을 함
        if data[i+1]=="1":      #전부 0으로 바꾸기 : '0'0011100 -> 000'1'1100 (1이 처음으로 나오는 순간만 count0에 +1) -> 000'1'11'0'0 
            count0+=1
        else:
            count1+=1

print(min(count0,count1))
'''


#Q4 만들 수 없는 금액
'''
1 1 2 3 9
처음에 1을 검사한다 처음 가져온수는 1이다. 가져온수가 검사하는 수보다 크지않으므로 만들수 있다. 검사할 수에 가져온수를 더한다(2)
2를 검사한다. 다음가져온수는 1이다. 가져온수가 검사하는 수보다 크지않으므로 만들 수 있다. 검사할 수에 가져온 수를 더한다(3)
3을 검사한다. 다음가져온수는 2이다. 가져온수가 검사하는 수보다 크지않으므로 만들 수 있다. 검사할 수에 가져온 수를 더한다(5)
5를 검사한다. 다음 가져온수는 3이다. 5는 3보다 크므로 만들 수 있다. 둘을 더한다(8)
8을 검사한다. 다음 가져온수는 9이다. 8은 9보다 작다. 만들수 없다. 정지한다.

처음에 가져오는 수가 1이 아니면 무조건 1을 반환할 수 밖에없다.

처음에 가져오는 수가 1이면 2를 검사대상으로 삼게된다.
다음가져오는 수가 2이면 
'''
'''
n=int(input())
coin=list(map(int,input().split()))
coin.sort()

check=1

for i in coin:
    if check < i:           #많은것이 담겨있는 if문임. dp의 bottom-up적 사고가 필요.
                            #처음에 check에 1을 할당함으로써, check는 최소 코인들의 합으로 만들 수 있는 종류중, 최대값보다 항상 1이 크게 된다.
                            #또한 check가 가져온 코인값보다 작다는것은, 가져온 코인을 기존조합으로 만들수 있는 종류에 섞어넣었을때 연속적이지 않은
                            #끊어지는 링크가 생긴다는 의미와 동일하다 (1,2,3,"4, 6,"7,8,9...)->5가 누락과 같은 상황.
                            #이때 check는 이전조합의 가장 큰수보다 1이 높다 = 만들수 없는 종류중 가장 낮은 수 가 되므로 해당 check값이 목표값이 된다.
        print(check)
        break
    check+=i

'''


#Q5 볼링공 고르기 (9:20)
'''
우선 입력된 무게 1 3 2 3 2에 각각 index기준으로 dict화 한다? 굳이?

그냥 오름차순 1 2 2 3 3으로 하고 i번째와 j번째(구간은 n-i)에서 하나씩 매칭하면서 i의 무게값과 j의 무게값이 같으면 패스, 다르면 카운트 +1
-> 사실 오름차순 할필요도 없음. 
'''

'''#내풀이
n,m= map(int,input().split())
data=list(map(int,input().split()))

count=0
for i in range(n):
    for j in range(i+1,n):
        if data[i]!=data[j]:
            count+=1
        else:
            pass

print(count)
'''

'''#해설 -> 순차적으로 올라가면서 검사하는것은 똑같지만, 여기서는 리스트변수에 이를 저장해가면서 계산했음

n,m= map(int,input().split())
data=list(map(int,input().split()))

array=[0]*11

for x in data:
    array[x] += 1  #무게를 array의 index로하여, 무게별 공의 갯수를 array에 저장한다.

result = 0
for i in range(1,m+1):  #1부터 최대무게까지 공의 무게를 순차적으로 가져와서 검사
    n-=array[i]         #전체갯수 - 검사하는 무게의 공의 갯수 = 검사하는 무게 외의 다른 모든 무게의 공의 갯수
    result += array[i] * n      # 검사대상 * 검사대상 외의 무게 공 총 갯수 -> 단, 여기서 for문을 통해 검사대상 공의 무게는 순차적으로 올라가고
                                #이때문에 이전 검사 무게보다 낮은 공은 검사하지않음 -> 중복제외!
print(result)
'''


#Q6 무지의 먹방 라이브 (29:34)

'''#내답 -> 답은 정확히 나오지만, 시간에서 다 걸려버림 
def solution(food_times, k):
    
    dish_len = len(food_times) #음식종류 갯수
    round_count=0              #다음에 먹을 음식번호
    complete = [0] * dish_len  #다먹었는지 확인. 다먹으면 1로 표기
    
    for x in range(k+1):
        print(round_count,food_times)
        while complete[round_count]!=0:
            round_count+=1
            if round_count==dish_len:
                round_count=0
        if x==k:
            answer=round_count+1

        food_times[round_count]-=1

        if food_times[round_count]==0:
            complete[round_count]=1
        round_count+=1
        if round_count==dish_len:
                round_count=0

    return answer
'''

#해설 시간이 지나고난 후 먹어야할 음식을 구하기만 하면 되는것! -> 순차적으로 수행할 필요없이 시간을 바로 food times에서 빼버린다.
#0이 되버린 음식은 heapq를 이용해서 순차적으로 먹는데 시간이 적게 걸리는 순서대로 지워나가면서 현재 남은시간과 남은 음식시간을 비교하며 결과 도출

import heapq

def solution(food_times,k):

    if sum(food_times) <=k:         #우선, 모든 음식을 먹는데 걸리는 시간보다 k가 크면 음식을 다먹어버렸기때문에 -1을 반환시킴
        return -1

    q=[]
    for i in range(len(food_times)):
        heapq.heappush(q, (food_times[i],i+1))      #(음식을 다먹는데 걸리는 시간,음식번호)로 힙큐에 저장한다.
        
    sum_value = 0               #먹기위해 사용한시간
    privious = 0                #직전에 다먹은 음식의 시간
    length = len(food_times)    #남은 음식의 갯수

    #일단 먹기위해 사용한시간 + (현재검사중인 음식을 다먹는데 걸리는 시간 * 남은음식종류)이 k이상이 될때까지 계속 반복한다.
    #음식하나를 제거할때마다 heapq에서 다음으로 다먹을수있는 음식을 가져온다.
    while sum_value + ( (q[0][0] - privious) * length ) <= k:       #[5,7,4],14 기준, 가장먼저 다먹게 되는 음식은 3번이다.
        print(sum_value,q)                                          #3번음식을 다먹는다면 1번 2번음식도 4(=privious)씩 먹었다는 의미이다
        now = heapq.heappop(q)[0]                                   #now는 검사대상음식을 먹는데 걸리는 총 시간을 의미한다.
        print(sum_value,now, q)
        sum_value += (now - privious) * length                      #지금까지 먹기위해 사용한 총 시간에 현재 음식을 다먹는데 걸리는 시간을 더한다.
        print(sum_value)
        length -= 1                                 #다먹은 음식은 종류 갯수에서 제거한다.
        privious = now                              #이제 다먹은 음식의 총시간을 갱신 (3번음식을 다먹을때는 4(+4), 1번음식을 다먹을때는 5(+1))
    '''
    0 [(4, 3), (7, 2), (5, 1)] -> 첫번쨰 힙큐리스트 -> (4,3)을 가져온다
    0 4 [(5, 1), (7, 2)]       -> 제일 먼저다먹을 수 있는 3번음식의 걸리는시간 4에 대해 검사 시작 -> 완료후에도 12 <= k(=14)이므로 계속 수행
    12       ->3번음식을 다먹는데 걸리는 시간은 12
    12 [(5, 1), (7, 2)]         -> 두번째 힙큐리스트 -> (5,1)을 가져온다.
    12 5 [(7, 2)]               -> 두번째로 다먹을 수 있는 음식 1번에 대해서 가져온다. -> 완료후 
    14       -> 1번음식은 3번음식을 먹으면서 이미 4의 시간이 빠져서 1이 남았다. 그렇기에 1번음식을 다먹는데 걸리는시간은 1(1번음식시간) * 2(남은음식종류 1,2번) = 2가 더해진 14이다
    ->이제 while문의 거짓조건이 되므로 빠져나온다.
    '''
    result = sorted(q, key= lambda x : x[1])            #계산이 끝나면 남은 q를 번호 순서대로 다시금 정렬한다.
    return result[ (k - sum_value) % length][1]         #초기입력 k시간 - 지금까지 먹는데 사용한 시간 = k시간중 남은 시간 
                                                        # -> 이것을 남은 음식 종류로 나누어서 그 나머지번호의 음식 번호값을 반환
                                                        #(이미 위에서 더이상 k시간중 남은 시간으로 인해 다먹어서 0이되지 않게끔 계산했기 때문에 무조건 이상없이 결과가 나온다.)

print(solution([5,7,4],14))

    
    