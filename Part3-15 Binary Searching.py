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

    if data[mid] == mid:  
        return mid
    
    elif data[mid] > mid:              #index값보다 해당위치 value값이 크다면 그 뒤의 index중에는 고정값이 없다!
        return bi_check(data, start, mid-1)
    
    else :                              #index값보다 해당위치 value값이 작다면 mid index이전에는 고정값이 없다!
        return bi_check(data,mid+1,end)

print(check_fix_value(data,n))
'''

#해설 - 내가 작성한 답과 동일한 구성, 양상이므로 생략했음