#2021-04-14

#순차탐색이란 리스트 안에 있는 특정 데이터를 찾기 위해 앞에서부터 데이터를 하나씩 순서대로 확인하는 방법.

#순차탐색 소스

'''
def sequential_search(n,target,array):
    for i in range(n):                  #n개의 범위에서 검색
        if array[i] == target:          #검색대상과 같은 것을 찾으면 현재의 위치를 반홤(0~n-1인것을 1~n화)
            return i+1

print("생성할 원소 개수를 입력 후, 한칸띄우고 찾을 문자열을 입력")
input_data = input().split()
n = int(input_data[0])  #원소 갯수
target = input_data[1]  #찾고싶은 문자열

print('앞서 입력한 원소 갯수만큼 문자열을 입력. 구분은 띄어쓰기 기준.')
array = input().split()

print(sequential_search(n,target,array))
'''

#이러한 메커니즘이므로 시간복잡도는 최악기준 O(N)이다.


#이진탐색(binary search) -> 정렬되어있어야만 사용할 수 있지만, 매우빠르다! (시간복잡도 O(logN))
#탐색범위를 절반씩 나눠가면서 수행함. -> 이상적인수행시 절반씩 나눠가면서 하기에 logN이 되는것.


#이진탐색 - 재귀이용
'''
def binary_search1(array,target,start,end):
    if start>end:
        return None
    mid = (start+end)//2 #중간점의 위치는 처음과 끝의 인덱스값의 중간(소숫점버림) -> 0~9일때, 중간점은 4.5->'4'가된다.
    if array[mid] == target: #중간점의 값이 목표값이되면(찾으면) 중간값의 위치를 반환
        return mid  
    elif array[mid] > target:
        return binary_search1(array,target,start,mid-1) #만일 중간값이 목표값보다 크다면, 중간값위치-1을 끝값으로 하여 재귀화(왼쪽탐색)
    else:                                              #중간값이 목표값보다 작으면 중간값위치+1을 처음값으로 하여 재귀(오른쪽탐색)
        return binary_search1(array,target,mid+1,end)

n, target = list(map(int,input().split()))

array = list(map(int,input().split()))


result = binary_search1(array,target,0,n-1)
if result == None:
    print("없다")
else:
    print(result+1)
'''

#이진탐색-반복문이용
'''
def binary_search2(array,target,start,end):
    if start>end:
        return None
    mid = (start+end)//2 
    if array[mid] == target: 
        return mid  
    elif array[mid] > target:
        end = mid - 1 
    else:
        start = mid + 1
    return None

n, target = list(map(int,input().split()))

array = list(map(int,input().split()))

result = binary_search2(array,target,0,n-1)
if result == None:
    print("없다")
else:
    print(result+1)
'''

#트리자료구조란? 노드(정보를가진 개체)와 노드의 연결로 표현된 그래프 자료구조의 일종. 계층적이고 정렬된 데이터 다루기에 편리
'''
트리는 부모-자식 노드관계로 표현된다.
최상단 노드는 루트노드, 최하단은 단말노드라고한다.
트리에서 일부를 때어내도 트리구조이며, 이는 서브트리라고 한다.

이진탐색트리 -> 크기 조건이 왼쪽자식노드<부모노드<오른쪽자식노드 의 관계를 가진다.
'''


#빠르게 입력하기
#input()사용시 시간초과될수도 있음. 이럴떄는 sys라이브러리의 readline()을 이용하면 빠르게 입력가능.

'''
import sys
input_data = sys.stdin.readline().rstrip()  #readline()이용시, 입력후 엔터자체를 줄바꿈(공백)문자로 인식한다.
                                            #rstrip()을 이용하면 오른쪽 끝의 공백을 지우게되므로 이 공백문자를 지울 수 있게된다.
print(input_data)
'''


#실전문제2 부품 찾기  (17:47)  탐색범위가 100만이하라서 그냥 순차탐색했음. 이진탐색도 만들어보자.
'''
n = int(input())
nn = list(map(int,input().split()))

m = int(input())
mm = list(map(int,input().split()))

def search(n,target,array):
    for i in range(n):
        if array[i] == target:
            return 'yes'
        elif array[i] != target:
            continue
    return 'no'

for j in range(m):
    target = mm[j]
    print(search(n,target,nn),end=' ')
'''

'''해설-이진탐색-재귀
def binary_search(array,target,start,end):
    while start<=end:
        mid = (start+end)//2
        if array[mid] == target:
            return mid
        elif array[mid] > target:
            return binary_search(array,target,start,mid-1)
        else :
            return binary_search(array,target,mid+1,end)
    return None

n = int(input())
array = list(map(int,input().split()))
array.sort()
m = int(input())
x = list(map(int,input().split()))

for i in x:
    result = binary_search(array,i,0,n-1)
    if result != None:
        print('yes', end=" ")
    else:
        print('no', end=" ")
'''

'''계수정렬
n=int(input())
array=[0]*1000001 #최대 100만개의 데이터를 다루므로 +1까지 해주어야 한다.'

for i in input().split():
    array[int(i)] = 1   #가지고있는 제품들을 입력했을때 해당 제품에 1을 표기해서 가지고있음을 나타냄

m=int(input())
x = list(map(int,input().split()))

for i in x:
    if array[i] == 1:
        print('yes',end=' ')
    else :
        print('no',end=' ')
'''


'''set을 이용한 집합자료형 -> 가장 효율적임. 가지고있는 재고를 중복을 제외하여 기록후, 원하는 물건이 그중에 하나라도 있는지를 검사.
n=int(input())
array = set(map(int,input().split()))

m = int(input())
x = list(map(int,input().split()))

for i in x:
    if i in array:
        print('yes',end=' ')
    else:
        print('no',end=' ')
'''


#실전문제3 떡볶이 떡 만들기
'''
n=떡갯수, m=손님이 가져갈 떡의 총길이 = 원래떡에서 잘린양의 총합

손님이 m만큼 달라고한다. 

떡이 a,b,c,d..길이로 있다고할때, 높이 k만큼 자른다.

즉 m= (a-k)+(b-k)+(c-k)+(d-k)....이 된다.
이때 ( )내의 값이 음수일경우, 0이되도록한다.

이렇게하려면 일단 가지고있는 떡을 오름차순으로 정렬한다.

k가 최대값을 가지려면 잘라낸떡이 딱 m만큼이 되어야한다.

첫값과 끝값의 평균 = 중간값으로 잘라보고 그 값이 m일때 m값을 출력
총합이 m보다 크면 중간값을 +1씩하고
총합이 m보다 작으면 중간값을 -1씩한다.
될때까지 계속 돌린다.

4, 6.3
19 15 10 17 -> k=14.9 이런건 생각하지말까

a,b,c,d중 k보다 작거나 같은것(b,c)은 잘라봤자 의미없다.
더 큰것에서 k를 뺀것을 총합해보고, 이게 더 작
'''

'''#작성한답. 이진탐색을 사용할때, 처음에 중간값을 사용해서 잘라보고 총합량에 부족하거나 남을때마다 +1,-1씩 조정한다.
n,m = map(int,input().split())
array = list(map(int,input().split()))

array.sort()

def binary_cut(array,target,start,end):
    while 1:
        if start>end:
            return None
        mid = (start+end)//2
        result=0
        for i in array:
            if i-array[mid] <=0:
                result+=0
            else:
                result+=(i-array[mid])
        if result == target:
            return array[mid]
        elif result > target:
            array[mid]+=1
            continue
        else:
            array[mid]-=1
            continue

print(binary_cut(array,m,0,n-1))
'''


'''해설-이진탐색개념을 응용해서 자르고 덜자르고 그값을 직접적으로 조정하면서 값을 구했다.
n,m = list(map(int,input().split()))
array = list(map(int,input().split()))

start = 0
end = max(array) #가지고있는 떡의 가장 큰값.

result = 0
while start<=end:       #떡크기가0보다 크면 계속 반복한다.
    total = 0
    mid = (start+end)//2
    for x in array:
        if x> mid:
            total +=x - mid   #중간값이 떡보다 작아서 잘리는것이 생길때만 결과값에 추가하는 방식으로 간략화
    if total < m:
        end = mid -1      #최대값을 중간값-1으로 바꾼다.(왼쪽부분탐색)
    else:
        result = mid        #자른떡양이 충분할경우, 덜잘라본다(오른쪽부분 탐색) 그리고 최대한 덜잘랐을때가 답이므로, 여기서 결과값을 result에 기록
        start = mid +1

print(result)
'''


