'''

from bisect import bisect_left, bisect_right        

a = [1,2,3,4,5,6,6,8]

print(bisect_left(a,5)) #왼쪽에서부터 정렬되어있는 배열a에 5를 집어넣을 위치 (왼쪽기준이니까 현재 a내에있는 5의 왼쪽 인덱스인 4를 반환)

print(bisect_right(a,5))    #이번에는 오른쪽을 기준으로 넣을 위치를 알려줌 (a의 5위치의 오른쪽이므로 인덱스값 5를 반환함)

def count_value_range(a,left_v,right_v):            #O(logN)의 시간으로 계산한다!
    right_index = bisect_right(a, right_v)
    left_index = bisect_left(a, left_v)
    return right_index - left_index

print(count_value_range(a,6,6))     #6 갯수를 알려줌 (2)
print(count_value_range(a,3,6))     #3<= x <=6인 값 갯수를 알려줌 (5)

'''

#Q27 정렬된 배열에서특정 수의 개수 구하기 (시간복잡도는 무조건 logN이하를 만족해야함)

'''#내답. bisect 라이브러리와 응용함수 위에서 익힌거 바로 써먹어서 개꿀
from bisect import bisect_left, bisect_right

def count_value(a,l_v,r_v):
    l_id = bisect_left(a,l_v)
    r_id = bisect_right(a,r_v)
    return r_id - l_id

n,x = map(int,input().split())
data = list(map(int,input().split()))

num = count_value(data,x,x)

if num != 0 :
    print(num)
else:
    print(-1)
'''

'''#해설 -> 이진탐색을 구현해서 만든것과 위의 bisect라이브러리 응용방법 두가지를 모두 보여줬음. 후자는 내가 위에서 했으니 전자만 작성해봄

def count_by_value(array,x):
    n = len(array)

    a = first(array,x,0,n-1)        #x가 처음등장한 인덱스계산
    if a == None:
        return 0                    #위에서 계산을 수행했을때, 그 값이 없다면 0을 반환한다

    b = last(array,x,0,n-1)         #x가 마지막으로 등장한 인덱스 계산
    
    return b-a +1

def first(array,target,start,end):           #첫 x가 등장하는 위치를 계산하는 함수정의 -> 2진검색법이용
    if start > end:
        return None                     #시작점이 끝점보다 크다면 None반환          None의 타입은 'Nonetype'이다. 하지만 불린값의 false처럼 작동

    mid = (start + end) // 2

    #중앙값이 목표값이되면 중앙값의 위치를 출력 (단, 이때 해당값을 가지는 원소중 가장 왼쪽에 있는 값의 인덱스만 출력한다! ;target이 중앙값의 왼쪽값보다 클때(ex: 2,'3',3,3,4))
    if (mid == 0 or target > array[mid-1]) and array[mid] == target:  
        return mid
    
    #중앙값이 목표값 이상이면, 끝값을 중앙값으로 하는범위로 다시금 재검색 (중앙기준 좌측검색)
    elif array[mid] >= target:
        return first(array,target, start, mid-1)
    
    #중앙값이 목표값 미만이면, 처음값을 중앙값으로 하는 범위로 다시금 재검색 (중앙기준 우측검색)
    else :
        return first(array,target,mid+1,end)

def last(array,target,start,end):
    if start > end:
        return None

    mid = (start + end) // 2

    if (mid == n-1 or target < array[mid+1]) and array[mid] == target:      # ex: 3이목표값일때 2,3,3,'3',4 에서  '3'의 위치를 출력
        return mid

    #first와는 달리 목표값보다 '초과'일때 좌측을 검색하게함. 왜냐하면 동일한게 여러개있을때 마지막 목표값의 index를 구하기 위함임
    elif array[mid] > target:
        return last(array,target,start,mid-1)
    
    else :
        return last(array,target,mid+1,end)

n, x = map(int,input().split())
array = list(map(int,input().split()))

count = count_by_value(array,x)

if count == 0:
    print(-1)
else:
    print(count)
'''


#Q28 고정점 찾기

#고정점은 index값과 그index의 value가 같은 값을 말한다

'''#내답 binary search를 이용해서 구했음
n = int(input())
data = list(map(int,input().split()))

def check_fix_value(data,n):

    a = bi_check(data,0,n-1)        
    if a == None:
        return -1
    return a

def bi_check(data,start,end): 
    if start > end:
        return None

    mid = (start + end) // 2
c
    if data[mid] == mid:  
        return mid
    
    elif data[mid] > mid:              #index값보다 해당위치 value값이 크다면 그 뒤의 index중에는 고정값이 없다!
        return bi_check(data, start, mid-1)
    
    else :                              #index값보다 해당위치 value값이 작다면 mid index이전에는 고정값이 없다!
        return bi_check(data,mid+1,end)

print(check_fix_value(data,n))
'''

#해설 - 내가 작성한 답과 동일한 구성, 양상이므로 생략했음


#Q29 공유기 설치
'''

중간값을 갱신하면서 찾아나가다가 발견했을때, 해당 중간값 첫 범위의 양끝값에서 각각 얼마나 떨어져있는지 비교하여, 더 작은값을 선택한다.
(양끝값의 중앙값으로부터 떨어진 거리가 작은곳에 설치할수록, 새로설치한 공유기와 양끝간 거리가 멀어지기 때문!)

-> 예시에서 12489일때, 처음 5의 중간실제값을 계산하고 양쪽을 이진검색하여 가장 5와 인접한 값을 찾아내는것을 못했음
'''


'''
n, c = map(int,input().split())
data = []
for _ in range(n):
    data.append(int(input()))

data.sort()
print('-')
def binary_search(data,start,end):
    global c
    print('000')

    if start>end:
        return None
    print(111)
    mid_index = (start + end) // 2                  #중간 위치값
    mid_value = (data[start] + data[end]) // 2      #실제 중간값

    # if start+1 == end:      #만일 구간요소가 처음과 끝밖에없다면 처음과 끝값차이를 반환하면서 공유기한대 설치
    #     c-=1
    #     return data[end] - data[start]
    # print(222)

    if data[mid_index] == mid_value:     #주어진 구간의 중간위치값이 실제 해당구간의 중간값이 되었을때 해당실제 중간값과 양끝값간 거리중 짧은거리를 반환하면서 공유기 한대설치
        c-=1
        return router.append(min((data[mid_index]-data[start]), (data[end]-data[mid_index])))

    elif data[mid_index] < mid_value:       #중간 위치값이 실제 중간값보다 클경우 왼쪽을 검색
        print(444)
        binary_search(data,start,mid_index)
        

    else:
        print(555)
        binary_search(data,mid_index,end)
        

router = []

c -= 2
if c == 0:
    print(abs(data[0] - data[-1]))      #설치할게 2개밖에없으면 양끝에 설치하고 그거리를 반환

else:                                       #설치할게 남아있으면 이제 연산시작
    while c:    #설치할 공유기가 0이되면 탈출
        binary_search(data,0,n-1)

print(router,router[-1])
'''

'''#해답-로직이 좀 어려움
"""
n,c = 5,3
1 2 4 8 9 
최대거리는 1 - 9인 8이다.
이렇게 하면 2개만 설치하니까 1개가 남는다.
그래서 최대거리를 반으로 줄인 4로 다시 검색해본다
1-2는 1이라 안된다. 1-4는 3이라 안된다. 1-8은 7이라 4보다 크니까 만족한다. / 8-9는 1이라 안된다 -> 총2개 설치되는데, 설치할개수 3개보다 적으니 거리를 반으로 또 줄인다 [?,4-1]
최대거리를 2로 줄여서 검색해본다
1-2는 1이라 안된다. 1-4는 3이라 2보다 크니까 만족한다. 4-8은 4라 2보다 크니 만족한다. -> 3개를 모두 사용했으니 만족한다. [2,4-1]
이때 [2,4-1] = [2,3]상태이므로 더 구간을 조일수 왼쪽값을 더 늘려볼여지가 있다 -> [3,3]으로 구간을 조여서 3으로 다시 시행해본다
....1-4는 3이라 된다. 4-8은 4라서 된다. 3개를 모두 설치했다. 구간도 [3,3]으로 더이상 줄일 수 있는 여지가 없다. -> 종료하고 3을 출력한다
"""

n, c = map(int,input().split())
array = []
for _ in range(n):
    array.append(int(input()))
array.sort()

start = 1      #최소거리는 한칸으로 설정                                                            ->책에서는 array[1]-array[0]이라 되어있지만 오류같음
end = abs(array[0] - array[-1]) #최대거리는 양끝값의 차이값으로 설정 -> [start,end] = [1,end]꼴
result = 0

while (start<=end) :        #[start,end]를 조일수있는 여지가 있으면 계속해서 반복수행한다
    mid = (start+end) // 2  #end값으로 설정할 중간값
    value = array[0]        #첫값을 기준으로 더해나가본다
    count = 1               #첫집에 공유기 설치
    for i in range(1,n):
        if array[i] >= value + mid:     #첫값에 중간값을 더한것이 i번째값보다 작거나 같은경우
            value = array[i]            #다음에 더할 첫값은 i번쨰 값으로하고
            count+=1                    #설치한 횟수를 1번 추가
        
    if count >= c:          #c개이상 공유기를 설치할 수 있는 경우(간격이 좁을때)
        start = mid+1       #앞값을 중간값보다 1큰 값을 배정해본다
        result = mid        #출력할 값에 현재의 중간값을 배정해놓는다 (1큰값으로 돌려봐서 안되는경우 출력하기 위해)

    else :                  #c개미만으로 공유기를 설치하는경우
        end = mid -1        #끝값을 중간값보다 1작게해서 검사해본다

print(result)
'''