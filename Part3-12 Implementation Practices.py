#2021-04-29

#Q7 럭키 스트레이트 (07:13)
'''
자릿수가 항상 2의 배수로 들어오므로 이를 반으로 나눠서 각각 더해서 결과를 비교하면된다.
6자리수면 0,1,2,3,4,5
'''

'''내답
n = input()

left=0
right=0
for i in range(len(n)):
    if i < (len(n)/2):
        left+=int(n[i])
    else :
        right+=int(n[i])

if left == right:
    print("LUCKY")
else :
    print(("READY"))
'''

'''#해설-> 여기서는 절반의 좌측에서 계속 더하고, 나머지 절반의 우측에서 계속 빼서 그결과가 0이면 좌우 양쪽의 숫자의 합이 같다는 것을 이용했음.
n=input()
length=len(n)
summery=0

for i in range(length//2):
    summery+=int(n[i])

for i in range(length//2,length):
    summery-=int(n[i])

if summery==0:
    print("LUCKY")
else :
    print("READY")
'''

#Q8 문자열 재정렬 (12:35)
'''
알파벳 대문자의 아스키코드값 기준으로 리스트를 분류해서 정수들은 합산시키고 문자열들은 sort해서 순차적으로 출력시킨다!
'''

'''#내답 -> 숫자가 없을때 0이 출력되어버린다! 이를 간과했군...
s=input()
str_type=[]
int_type=[]
 #대문자 알파벳 아스키코드 65~90, 숫자0부터 9는 48~57

for i in range(len(s)):
    if ord(s[i]) >=65 and ord(s[i]) <=90:
        str_type.append(s[i])
    else :
        int_type.append(int(s[i]))      #여기서 리스트에 따로저장하지않고 바로 total=0을 선언하고 저장했으면 더 간결해진다.

str_type.sort()
total=sum(int_type)

for i in str_type:
    print(i,end='')         #그리고 마지막글자에대해 end=''를 안나오게해야지
if total != 0:
    print(total)
'''

'''#해설
data=input()
result=[]
value=0

for x in data:
    if x.isalpha():             # .isalpha()메소드는 알파벳이면 여부를 T/F로 반환한다.
        result.append(x)
    else:
        value+=int(x)

result.sort()
if value !=0:
    result.append(str(value))       #숫자가 배열내에 있을때만 결과값을 출력하게 한다. 이게없으면 숫자가없는경우 맨뒤에 0이 붙기때문이다.

print(''.join(result))              # ''.join(list)를쓰면 list[ "A","K","B","12"] 에서 리스트의 각요소 사이에 ''을 끼워넣어서 문자열로 출력한다. 
print(result)                       # 이는 결과물을 문자열로 나열해서 출력할때 자주쓰므로 매우중요!!
'''


#Q9 문자열압축
'''
aabbaccc -> 2a2ba3

abcabcdede -> abcabc2de(x) -> 2abcdede(best!)

ababcdcdababcdcd -> 2ab2cd2ab2cd(12) -> 2ababcdcd(9, best!)

일단 1개단위로 압축하는것을 만들어본다...
그결과의 문자열길이를 계산해본다
2개단위로 압축해본다... 문자열길이계산....3개단위...4개단위... 각각의결과값을 결과값 리스트에 저장하고 그결과값중 최소값을 반환하면되지않을까?

그러면 일단 1개단위로 압축하는것을 만드는것이 급선무

앞에서부터 글자를 하나씩 가져와서 그게 이전값하고 같다면 카운트+1, 
다르다면 이전비교용글자에 가져온 글자를 할당하고/카운트를 0으로 하고/문자열만드는용 리스트에 count->이전비교용글자순으로 할당한다

'''


'''일단 겨우 1글자짜리 압축은 성공함. 근데 2개씩 가져오는건 어케하지..
data=input()
present_word=''
count=0
zip_word_list=[]

for x in range(len(data)):
    if present_word == data[x]:
        count+=1
        continue

    else :
        if count>1:
            zip_word_list.append(str(count))
        if present_word != '':
            zip_word_list.append(present_word)
        present_word=data[x]
        count=1
    
zip_word_list.append(str(count))
zip_word_list.append(present_word)

print(zip_word_list,present_word,count)
print(''.join(zip_word_list))
'''

'''#해답. 압축된 문자열을 출력할 필요없다! 그냥 그결과값만 내면된다! 쓸데없는 중간구현에 목매지말것! 문자열을 1개단위로 맟춰본것과 2개단위로 맟춰본것과 그 값만 비교하자

s = input()
answer = len(s)    #처음에는 오리지널 문자열길이를 결과값에 대입

for step in range(1,len(s)//2+1):   #1,2,3... 부터 문자열 절반까지 압축단위를 점차 늘림
    compressed = ''
    prev = s[0:step]                #앞에서 step번 문자열까지 슬라이싱함
    count=1                         #첫문자를 포함해야하니 1부터 시작
    for j in range(step,len(s),step):       #range(3,10,3)이라고 생각하면 3,6,9를 가져온다고 생각하면된다. range맨뒷글자는 다음가져올 숫자까지의 거리
        if prev == s[j:j+step]:
            count+=1                #만일 이전에 가져온 문자열과 동일할시 +1
        else:
            compressed += str(count) + prev if count >= 2 else prev     #이전문자열과 다를시,    1) count가 2 이상이면 compressed문자열에 count수를 문자열로서 쓰고 그뒤에 prev문자를 추가한다
            prev = s[j:j+step]                                          #                   2) count가 2 미만이면 count수 없아 prev문자만 compressed문자열에 추가한다.
            count = 1
    compressed += str(count) + prev if count >= 2 else prev             #마지막에 모든연산이 끝난뒤, 맨뒷글자에 대해 count+뒷문자열 or 뒷문자열 추가 작업을 수행해준다
    answer = min(answer, len(compressed))                               #압축단위를 늘리면서 더 작은값을 answer에 기록한다.
'''

#10 자물쇠와 열쇠
'''
key를 90도씩 돌리기 + key를 lock의 (1,1)....(1,m),(2,1)...(2,m)....(m,m)까지 완전탐색!
3x3의 경우에는 90도씩돌리기 (4) * lock에 순차적으로 대보기 (m*m번) = 최대 4*m*m = 36번까지 수행해야 답이 나올수 있음.
가져다댔을때 arraylock의 1과 key의 1이 겹쳐진다면 break시키고 다음 시행으로 넘어가게함.
매번 copy를 쓰면 메모리가 너무많이들텐데...heapq를 만들어서 할당하는식으로 순차적으로 키배열을 집어넣고 빼고를 반복해야함 
-> 완전탐색 자체는 최대 20x20일때 4*20*20번 = 640,000번 수행하므로 가능한 수준임.
초기입력된 모양 -> 순차적대보기 -> 90도 회전 ->순차적대보기... 90도 회전 할때마다 count+=1을하고 4가되는순간 false처리하고 break
순차적 대보기는 1부터 m*m회까지 count+=을하면서 m*m회에 다다르면 0으로 초기화하고 90도 회전시킴
'''


key = [[0,0,0], [1,0,0], [0,1,1]]     #예시의 키
lock = [[1,1,1], [1,1,0], [1,0,1]]     #에시의 자물쇠

'''내가 풀어낸 내용
n=len(array_key) #3
m=len(array_lock)

count_c = 0 #회전횟수
count_t = 0 #대본횟수

def circulation(array_key,n): #90도 회전시키는 함수
    old_array=array_key
    new_array=[[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            new_array[i].append(old_array[n-1-j][i])
    return new_array
    
print(circulation(array_key,n))


def try_lock_off(array_key,array_lock,n,m):
    global count_c, count_t
    for _ in range(4):
        for _ in range(len(array_lock)**2):
            #맟춰보는연산 True/False로 답내기...
            
            if '연산결과' == true:  #연산결과 열쇠가 맟춰진다면 true를 반환하고 연산을 멈춤
                print(true)
                break
            else:                 #안맟춰져서 false를 반환되면 회전시키고 다음연산으로 넘어감
                circulation(array_key,n)
                continue
'''

#해답
'''
lock이 
001
110
111 배열일때

new lock array
000 000 000
090 000 000
000 000 000

000 001 000
000 110 000
000 111 000

000 000 000
000 000 000
000 000 000 이렇게 원래 lock 배열의 제곱크기로 늘려서 하나씩 확인시키는것이다. 편의상 제곱크기로 늘리는게 편하기도하고 숫자 9위치가 실제 (1,1)위치이라 이런방식을 선택

(1,1)부터시작, (1,2),(1,3)... 순으로 key를 집어넣어서 해당위치의 값이 2가되면 돌기가 겹치는것이므로 패스, 2가 없을때 중앙의 lock이 전부 1로 가득차있지 않으면 패스
'''



def rotate_90degree (key):         #회전함수. 이거는 시계방향으로 90도 회전시킨다.
    n = len(key)          #세로길이
    m = len(key[0])       #가로길이
    result = [ [0] * n for _ in range(m)] #회전결과값을 할당할 테이블. 이전array의 가로-> 세로, 세로->가로가 된다.
    for i in range(n):
        for j in range(m):
            result[j][n-i-1] = key[i][j]
    return result

def check(new_lock):                 #중앙 lock이 풀렸는지 풀리지 않았는지 체크하는 함수
    lock_length = len(new_lock) // 3 #중앙의 진짜 lock의 길이를 계산한다. 위 아래를 쳐내니까 1/3에 해당
    for i in range(lock_length, lock_length*2):
        for j in range(lock_length, lock_length*2):     #중앙의 lock에 해당하는 부분만 검사를 시작
            if new_lock[i][j] != 1:     #1외의 숫자 (0or2)가 있다면 거짓값을 반환!
                return False
            return True                 #모두 1로 가득차있다 = 자물쇠와 일치한다면 true반환
    
def solution(key,lock):
    n = len(lock)
    m = len(key)

    new_lock = [ [0] * (3*n) for _ in range(n*3)] #new_lock으로 자물쇠를 검증할 행렬을 확장
    for i in range(n):
        for j in range(n):
            new_lock[i+n][j+n] = lock[i][j]        #그리고 new lock의 정중앙에 lock배열을 할당한다
    
    for rotation in range(4): #90도회전은 4번수행하면 끝!
        key = rotate_90degree(key)  #1번회전키 -> 2번회전 -> 3번회전 -> 4번회전 = 원래의 키모습. 즉 1번회전한키부터 시작한다 어차피 완탐이라 뭘먼저하든 상관은 없다
        for x in range(n*2):
            for y in range(n*2):        #키길이자체가 n의 길이를 가지고있으므로 움직이는범위는 n*2만큼움직여야 new_lock범위내에서 다 움직인다.
                
                for i in range(m):
                    for j in range(m):
                        new_lock[x+i][y+j]+=key[i][j]   #new_lock배열에 key의 값을 순서대로 더해본다...(A)
                
                if check(new_lock) == True:             #unlock됐다면 true값을 반환한다
                    return True

                for i in range(m):
                    for j in range(m):
                        new_lock[x+i][y+j]-=key[i][j]   #(A)에서 new_lock에 더했던 key값을 다시 빼서 원래의 new_lock배열로 되돌린다
    return False    #위 과정을 전부 수행했는데 답이안나왔다면 키가 배열에 맞지 않는것이므로 false를 반환한다
                
                    

    
    
