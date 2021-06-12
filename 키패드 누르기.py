
numbers = [1, 3, 4, 5, 8, 2, 1, 4, 5, 9, 5]  #"LRLLLRLLRRL"
hand = "right"

def solution(numbers, hand):
    answer = []

    left = (4,1)
    right = (4,3)

    numpad = { 1:(1,1), 2:(1,2), 3:(1,3), 4:(2,1), 5:(2,2), 6:(2,3), 7:(3,1), 8:(3,2), 9:(3,3), 0:(4,2) }

    for num in numbers:
        if num == 1 or num == 4 or num == 7:
            answer.append('L')
            left = numpad[num]
        elif num == 3 or num == 6 or num == 9:
            answer.append('R')
            right = numpad[num]
        else:                       #2,5,8,0일때
            target = numpad[num]

            len_l = abs(left[0]-target[0]) + abs(left[1]-target[1])
            len_r = abs(right[0]-target[0]) + abs(right[1]-target[1])

            if len_l < len_r:
                answer.append('L')
                left = numpad[num]
            elif len_l > len_r:
                answer.append('R')
                right = numpad[num]
            else:                   #거리가같을떄
                if hand == "right":
                    answer.append('R')
                    right = numpad[num]
                else:
                    answer.append('L')
                    left = numpad[num]

    return ''.join(answer)

print(solution(numbers, hand))