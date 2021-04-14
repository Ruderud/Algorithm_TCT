# 2021-04-13

'''
정렬은 여러가지가있지만,
선택정렬/삽입정렬/퀵정렬/계수정렬이 주로 사용된다.

내림차순 정렬 = 오름차순 + reverse(O(N))이다.
'''

#선택정렬(가장 원시적). 무작위로 배열된 숫자 0~9를 오름차순 정리. O(N^2)의 시간복잡도를 가진다. ((N*N+1)/2). 직관적으로는 2중 for반복문이니까!

array = [7,5,9,0,3,1,6,2,4,8]

for i in range(len(array)):
    min_index = i
    for j in range(i+1, len(array)):
        if array[min_index] > array [j]:    #만일 현재 작업중인 위치 i의 값보다 작은 값이 그 뒤에 있다면, 그중 가장 작은값을가져올때까지 계속 비교하고,
            min_index = j                   #비교했을때 작은값을 가진위치를 최소값의 위치로써 지정하면서 i+1번부터 끝까지 비교해나간다.
    array[i], array[min_index] = array[min_index], array[i]  #현재작업중인 위치의 값과 그 뒤에있는 값중 가장 작은값의 위치를 스왑한다.
    print(array)

print('-'*50)

#스왑 과정은 리스트위치값 기준, a, b = b,a로하면 [a,b] 가 [b,a]로 바뀐다.

#효율상으로는 정말 구리지만, 코테에서 특정 리스트내의 가장 작은 값을 찾는일이 있기때문에 선택정렬에 익숙해질 필요는 있다.





#삽입정렬. 정렬대상을 앞에 정렬되어있는 것들 중에 적절한 위치 사이에 끼워넣는방식의 정렬. [0]위치의값은 정렬되어있다고 간주하고, [1]위치부터 정렬시작한다.
#거의 정렬되어있는 상태에서 사용시 매우빠르다. 이중 for문이기에 O(N^2)의 시간복잡도이며, 대체로 선택정렬보다는 빠르다.

array = [7,5,9,0,3,1,6,2,4,8]

for i in range(1,len(array)):
    for j in range(i,0,-1):         #인덱스 i, i-1, i-2, ... 0의 순서로 j를 가져온다.
        if array[j] < array[j-1]:       #i=1일때 -> 5 < 7이므로, 둘의 위치를 바꾼다.
            array[j], array[j-1] = array[j-1], array[j]
        else :                          #i=2일때 -> 9는 5,7보다 크므로 그냥 놔두고 넘어간다.
            break
    print(array)

print(array)

print('-'*50)





#퀵정렬. 병합정렬과 더불어 가장빠른 정렬이다. (시간복잡도 : O(NlogN))
#정렬작업을 수행하기 위해 기준으로 잡는값을 피벗이라고 한다. 피벗보다 작은값을 왼쪽, 큰값을 오른쪽에 두게끔 하는 행위를 분할 또는 파티션이라고 한다.
#리스트의 가장 왼쪽데이터를 피벗으로 삼을때, 그리고 이미 데이터가 정렬되어있을때는 매우 느려서 최악의 경우 O(N^2)의 시간복잡도를 가진다.
#그렇기에 이러한 약점을 보완하기 위해서 기본 정렬 라이브러리를 이용하는데, 그러면 O(NlogN)의 시간복잡도를 보장하게된다. 아무튼 그렇다.

array = [7,5,9,0,3,1,6,2,4,8]

def quick_sort(array, start, end):  #start와 end는 array에서의 index값을 의미하게된다.
    if start >= end:  #원소가 1개 (len(array)-1 = 0)일때 종료한다. 
        return
    pivot = start       #첫 start숫자를 피벗으로 둔다.
    left = start +1     #피벗보다 1큰수를 좌, len(array)-1을 우로 둔다. [피벗, left,.........., right ]꼴을 생각하자.
    right = end
    while left <= right:        #피벗보다 큰수(left)를 찾을때까지 반복. (right와 left가 교차되어야 false)
        while left <=end and array[left] <=array[pivot]: #left위치의 값이 피벗보다 커야한다. 그렇지않으면 left를 오른쪽으로 한칸 이동.
            left +=1
        while right > start and array[right] >= array[pivot]: #마찬가지로 right위치의 값이 피벗보다 작아야하며, 그렇지않으면 왼쪽으로 한칸 이동.
            right -=1
        if left > right:                          #만일 위의 과정을통해서 조작하다가 left와 right가 교차할시, 피벗과 left위치를 스왑한다.
            array[right], array[pivot] = array[pivot], array[right]
        else:
            array[left], array[right] = array[right], array[left]       #엇갈리지 않는다면 left와 right를 위치를 스왑
    quick_sort(array,start,right-1)         #위의 과정을 수행하면서 분할된 좌측과 우측을 각각 정렬수행한다. 
    quick_sort(array,right+1,end)

quick_sort(array,0,len(array)-1)
print(array)

#파이썬의 장점을 살린 정렬소스코드......이게 더 이해하기편하다...

def quick_sort_short(array):
    if len(array) <=1:
        return array        #array리스트가 한개 이하의 원소를 가지게되면 종료시킨다.
    
    pivot = array[0]        #array의 첫 원소를 피벗으로 삼기
    tail = array[1:]        #피벗이외의 리스트를 정의

    left_side = [x for x in tail if x <= pivot]     #분할된 왼쪽 부분
    right_side = [x for x in tail if x > pivot]     #분할된 오른쪽 부분

    return quick_sort_short(left_side) + [pivot] + quick_sort_short(right_side)

print(quick_sort_short(array))

print('-'*50)




#계수정렬. 특정조건일때만 매우 빠른방법. (데이터 크기범위가 제한되어, 정수형태로 표현가능할때만!)
#일반적으로 Max-Min 차이가 백만 이하일떄 효과적이다. 너무 범위가 크면 안됨
#별도의 리스트를 선언하고, 그안에 정렬정보를 담기때문에 리스트의 크기가 클수록 힘들다.
#성적 정리같은거 할떄 유용하다.
#계수정렬의 시간복잡도는 O(N+K)이다. N:데이터 개수, K:데이터중 최대값의 크기 (0부터 정수로 갯수를 세기 때문) -> 기수정렬과 더불어 가장빠르다!
##기수정렬은 계수정렬보다 약간느리지만, 처리가능한 정수크기가 더큼.

array = [7,5,9,0,3,1,6,2,9,1,4,8,0,5,2] #0~9 사이의 중복되는 숫자의 무작위 배열

count = [0] * (max(array)+1) #array의 최대값보다 하나 더 큰것까지(0~10)담을 수 있는 리스트를 만든다.왜냐면 0~9는 총 10가지니까

for i in range(len(array)):
    count[array[i]] += 1        #[0,0,0....,0] -> 0이 10개 배정된 count에 각 0~9에 해당하는 수의 자리에 +1씩 배정
'''
0 1 2 3 4 5 6 7 8 9

0 0 0 0 0 0 0 1 0 0 꼴이 된다고 생각하자
'''

for i in range(len(count)):     #count 리스트의 각 자리별 카운트된 횟수만큼 해당자리 값 = 정렬한 array의 값을 반복해서 출력한다.
    for j in range(count[i]):   #0 0 1 1 2 2 3 4 5 5 6 7 8 9 9 로 출력된다.
        print(i,end=' ')

#단점! 공간복잡도가 극심. 0과 999999 이 2개뿐인데도 100만개의 원소리스트를 정의해야하기때문.



#파이썬의 sorted()함수는 퀵정렬보다 좀느리지만, O(NlogN)의 복잡도를 보장한다.
#sorted()는 별도의 정의된 리스트를 반환하지만, sort()는 기존의 리스트를 그냥 바로 정렬해준다.

#dict의 key를 이용가능하다. 예시는 튜플을 이용해서 묶은 리스트이다.

array = [('바나나',2),('사과',5),('당근',3)]

def setting(data):
    return data[1]

result = sorted(array,key=setting)
print(result)

#정렬문제는 다음의 3가지 유형을 가진다 
'''
1.정렬라이브러리로 풀이가능한 문제.
2.정렬 알고리즘 원리를 묻는 문제. -> 선택/삽입/퀵정렬에 대한 원리를 알고있어야한다.
3.더빠른 정렬이 필요한 문제. -> 퀵으로는 풀수없고 계수정렬등의 다른 정렬을 이용하거나, 기존알고리즘의 구조적 개선을 거쳐야함.
'''


#실전문제2 위에서 아래로. (5분컷)
'''
n = int(input())

data=[]

for _ in range(n):
    data.append(int(input()))

#data.sort()
#data.reverse()

data = sorted(data, reverse=True)  #한줄로 끝내기

for i in data:      #바로 앞에서부터 순서대로 데이터를 가져오면 굳이 data[i]이런걸 쓸필요없다.
    print(i,end=' ')
'''

#실전문제3 성적이 낮은 순서로 학생 출력하기

"""
n = int(input())

data=[]
for _ in range(n):
    input_data = input().split()
    data.append((input_data[0], int(input_data[1])))

print(n)
print(data)

data = sorted(data, key=lambda student: student[1]) 
'''
리스트내에 리스트가 있거나, 튜플로 묶인게있다면 각각의 리스트나 튜블내의 특정값을 기준으로 정렬가능하다.
data = [ (김,50), (이,60), (박,30) ]
data = sorted(data, key = lambda student : student[1])
       오름차순정렬(정렬대상 리스트, key= lambda student -> 정렬기준이 될 data내부의 리스트/튜플의 이름정의(student) : student[1]인자를 기준으로 정렬 )

'''
for stu in data:    #정렬된 data내의 리스트/튜플의 인자를 출력.
    print(stu[0], end=' ')
"""



#실전문제4 두 배열의 원소 교체

#a값의 최소값과 b값의 최대값을 비교하여 b쪽값이 더크면 비교한 값끼리 k번 교체한다. 단 이떄, a의값이 b보다 크면 교체를 중단한다.

n,k = map(int,input().split())

a = list(map(int,input().split()))
b = list(map(int,input().split()))

for i in range(k):
    if min(a) <= max(b):
        aa=a.index(min(a))
        bb=b.index(max(b))
        a[aa],b[bb] = b[bb],a[aa]
    else: break


#result=0
#for num in a:
#    result+=num

print(sum(a))

'''해설
여기서는 a는 오름차순, b는 내림차순으로 정렬해서 a[0] <-> b[0] 교체수행을 k번시킨다. (단 여기서도 a쪽이 b쪽보다 작을때 교체한다는 조건필수)

n,k = map(int,input().split())
a = list(map(int,input().split()))
b = list(map(int,input().split()))

a.sort()
b.sort(reverse=True)

for i range(k):
    if a[i]<b[i]:
        a[i],b[i] = b[i],a[i]
    else:
        Break
print(sum(a))
'''

