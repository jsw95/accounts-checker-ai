
l = ['marian', 'jane', '11', '11', 'loeki']


def check_for_name(l):

    def check(pos):
        while l[pos] == '11':
            pos -= 1
            check(pos)

        return l[pos]

    return [check(i) for i in range(len(l))]


print(check_for_name(l))

