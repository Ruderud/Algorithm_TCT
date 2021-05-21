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