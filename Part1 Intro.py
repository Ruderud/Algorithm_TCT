#대충 책앞부분 참조하라는내용

https://www.google.com/

#구동에 걸린시간 계산
import time

start_time=time.time()

a=1
b=2
print(a+b)

end_time=time.time()
print(end_time-start_time)

#박오 (BigO)표기법
'''
빠르다 |  O(1) : 상수시간
        O(logN) : 로그시간
        O(N) : 선형시간
        O(NlogN) : 로그선형시간
        O(N^2) : 이차시간
        O(N^3) : 삼차시간
느리다    O(2^N) : 지수시간


'''
