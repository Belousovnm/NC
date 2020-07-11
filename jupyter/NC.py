from numba import njit
import numpy as np
from sympy import *
from tqdm import tqdm, tnrange, tqdm_gui, notebook
import operator
import re
from itertools import permutations, combinations
from sympy.physics.quantum import TensorProduct
init_printing(False)


class NC:
    def __init__(self, N, M):
        self.N = N
        self.M = M
        X = [[0 for j in range(N)] for i in range(M)]
        for i in range(M):
            for j in range(N):
                X[i][j] = Symbol('x' + str(i + 1) + str(j + 1),
                                 commutative=False)
        XX = X
        X = Matrix(X)

        z = [0 for i in range(N)]
        for i in range(N):
            z[i] = Symbol('z' + str(i + 1), commutative=False)
        Z = diag(*z)

        u = Symbol('u', commutative=False)
        U = Matrix(u * ones(N))

        P = [[0 for j in range(N)] for i in range(M)]
        for i in range(M):
            for j in range(N):
                P[i][j] = Symbol('p' + str(i + 1) + str(j + 1),
                                 commutative=False)
        PP = P
        P = Matrix(P)
        P_u = Symbol('p_u', commutative=False)
        PP_u = Matrix(P_u * ones(N))
        P_v = Symbol('p_v', commutative=False)
        v = Symbol('v', commutative=False)

        if N == M:
            Y = [[0 for j in range(N)] for i in range(N)]
            for i in range(N):
                for j in range(N):
                    for a in range(M):
                        Y[i][j] += XX[j][a] * PP[i][a]
                    if i == j:
                        Y[i][j] += (M - i - 1)
            YY = Matrix(Y)
        else:
            YY = 'N/A'

        l = [0 for i in range(M)]
        for i in range(M):
            l[i] = Symbol('l' + str(i + 1), commutative=False)
        L = diag(*l)

        uz = [0 for j in range(N)]
        for i in range(N):
            uz[i] = Symbol('uz' + str(i + 1), commutative=False)

        F = [[[[0 for j in range(N)] for i in range(N)] for a in range(M)]
             for flag in range(2)]
        G = [[0 for j in range(M)] for i in range(M)]
        EEE = [[[0 for i in range(N)] for j in range(M)] for a in range(M)]
        for i in range(M):
            for j in range(M):
                for a in range(N):
                    EEE[i][j][a] = -uz[a] * XX[j][a] * PP[i][a]
                    G[i][j] += EEE[i][j][a]
                    # if i == j:
                    # F[1][i][j][a] = EEE[i][j][a]
                    # F[0][j][j][0] = P_u - l[j]
                if i == j:
                    G[i][j] += P_u - 1
        G = Matrix(G)

        q = [0 for i in range(N)]
        for i in range(N):
            q[i] = Symbol('Q' + str(i + 1), commutative=False)
        Q = diag(*q)

        self.X = X
        self.P = P
        self.Z = Z
        self.U = U
        self.P_u = P_u
        self.YY = YY
        self.L = L
        self.G = G
        self.uz = uz
        self.Q = Q

    def sigma(self):
        return list(permutations([x for x in range(self.N)]))

    def c(self, i, j):
        sigma = self.sigma()
        if sigma[j][i] != i:
            return 1
        else:
            return 0

    # def a(i, j):
    #     if c(self,i, j) == 0:
    #         return 0
    #     else:
    #         return i

    @staticmethod
    def ss4iss(foo):
        inpt = str(foo).split('+')
        out = []
        for x in inpt:
            w = x.split('-')
            for y in w:
                out.append(y)
        if out[0] == '':
            out.pop(0)
        return out

    @staticmethod
    def pstor_legacy(foo):
        global pstorage, pnumber
        current = str(foo)
        pstorage = []
        pnumber = []
        pattern = r'p'
        i = -1
        while current.find(pattern) != -1:
            i += 1
            pstorage.append(0)
            pnumber.append(0)
            pstorage[i] = current.find(pattern)
            pnumber[i] = int(current[current.find(pattern) +
                                     1:current.find(pattern) + 3])
            current = current[current.find(pattern) + 4:]
            if i != 0:
                pstorage[i] += pstorage[i - 1] + 4
        return pstorage, pnumber

    @staticmethod
    def pstor(foo):
        #         global pstorage, pnumber
        current = str(foo)
        pstorage = []
        pnumber = []
        regex = re.compile('[p]\d{2}')
        pattern = regex.findall(current)
        for i in range(len(pattern)):
            pt = pattern[i]
            pstorage.append(0)
            pnumber.append(0)
            pstorage[i] = current.find(pt)
            pnumber[i] = int(current[current.find(pt) + 1:current.find(pt) +
                                     3])
            current = current[current.find(pt) + 4:]
            if i != 0:
                pstorage[i] += pstorage[i - 1] + 4
        return pstorage, pnumber

    @staticmethod
    def xstor(foo):
        #         global xstorage, xnumber
        current = str(foo)
        xstorage = []
        xnumber = []
        pattern = r'x'
        i = -1
        while current.find(pattern) != -1:
            i += 1
            xstorage.append(0)
            xnumber.append(0)
            xstorage[i] = current.find(pattern)
            xnumber[i] = int(current[current.find(pattern) +
                                     1:current.find(pattern) + 3])
            current = current[current.find(pattern) + 4:]
            if i != 0:
                xstorage[i] += xstorage[i - 1] + 4
        return xstorage, xnumber

    @staticmethod
    def is_normal(foo):
        flag = True
        xstorage = NC.xstor(foo)[0]
        pstorage = NC.pstor(foo)[0]
        for i in range(len(xstorage)):
            for j in range(len(pstorage)):
                if pstorage[j] < xstorage[i]:
                    flag = False
        return flag

    @staticmethod
    def is_normalorder(foo):
        flag = True
        xstorage = NC.xstor(foo)[0]
        pstorage = NC.pstor(foo)[0]
        xnumber = NC.xstor(foo)[1]
        pnumber = NC.pstor(foo)[1]

        for i in range(len(xstorage)):
            for j in range(len(pstorage)):
                if pstorage[j] < xstorage[i]:
                    flag = False
        if pnumber != sorted(pnumber):
            return False
        if xnumber != sorted(xnumber):
            return False
        return flag

    # @staticmethod
    def rdet(self, foo):
        result = 0
        sigma = self.sigma()
        c = self.c
        for j in range(len(sigma)):
            current = 1
            sm = 0
            for i in range(self.N):
                current = current * foo[i, sigma[j][i]]
                sm += c(i, j)
            if sm == 0:
                sgn = 1
            elif (sm % 2) == 0:
                sgn = -1
            else:
                sgn = 1
            if sgn == 1:
                result += current
            else:
                result += -current
        return result

    # @staticmethod
    def cdet(self, foo):
        result = 0
        sigma = self.sigma()
        c = self.c
        for j in range(len(sigma)):
            current = 1
            sm = 0
            for i in range(self.N):
                current = current * foo[sigma[j][i], i]
                sm += c(i, j)
            if sm == 0:
                sgn = 1
            elif (sm % 2) == 0:
                sgn = -1
            else:
                sgn = 1
            if sgn == 1:
                result += current
            else:
                result += -current
        return result

    #Capelli = expand(rdet(X)*rdet(P) - rdet(YY))
    #Capelli = Capelli.subs(h, 1)

    @staticmethod
    def subsets(S):
        sets = []
        len_S = len(S)
        for i in range(1 << len_S):
            subset = [S[bit] for bit in range(len_S) if i & (1 << bit)]
            sets.append(subset)
        return sets

    #S = [i for i in range(N)]
    #subsets(S)

    # @staticmethod
    def is_plus(self, foo):
        foo = str(foo)
        index = 0
        sign = []
        if str(foo)[0] == '-':
            for x in self.ss4iss(foo):
                sign.append(str(foo)[index])
                index += len(x) + 1
        else:
            foo = '-' + foo
            for x in self.ss4iss(foo):
                sign.append(str(foo)[index])
                index += len(x) + 1
            sign[0] = '+'
        return sign

    @staticmethod
    def myinstance(foo):
        if len(foo[0]) == 1:
            return 'int'
        else:
            return 'list'

    @staticmethod
    def FNFxp(foo, index=0):
        result = []
        if isinstance(foo, str):
            result.append(foo)
        elif isinstance(foo, list):
            for item in foo:
                result.append(item)
        current = result[index]
        while not is_normal(current):
            xxstor = xstor(current)
            ppstor = pstor(current)
            sspar = ''
            for i in range(len(xxstor[0])):
                for j in range(len(ppstor[0])):
                    if current == 'p' + str(ppstor[1][j]) + '*' + 'x' + str(
                            xxstor[1][i]):
                        #                     print('\n In ' + str(current)+' report0: '+'p'+str(ppstor[1][j]) + ' on position '+str(ppstor[0][j])+  ' does not commute with '+'x'+ str(xxstor[1][i]) + ' on position ' + str(xxstor[0][i]))
                        if ppstor[1][j] == xxstor[1][i]:
                            sspar = '1'
                        else:
                            sspar = ''
                        temp = current.replace(
                            'p' + str(ppstor[1][j]) + '*' + 'x' +
                            str(xxstor[1][i]), 'x' + str(xxstor[1][i]) + '*' +
                            'p' + str(ppstor[1][j]))
                        result.remove(current)
                        result.insert(index, temp)
                        current = temp
                        if ssapr != '':
                            result.append(sspar)
    #                     print(result)
                    else:
                        if j != len(ppstor[0]) - 1:
                            if ppstor[0][j] < xxstor[0][i] and ppstor[0][
                                    j + 1] > xxstor[0][i]:
                                if ppstor[1][j] != xxstor[1][i]:
                                    #                                  print( '\n In ' + str(current)+' report1: '+'p'+str(ppstor[1][j]) + ' on position '+str(ppstor[0][j])+  ' commutes with '+'x'+ str(xxstor[1][i]) + ' on position ' + str(xxstor[0][i]))
                                    temp = current.replace(
                                        'p' + str(ppstor[1][j]) + '*' + 'x' +
                                        str(xxstor[1][i]),
                                        'x' + str(xxstor[1][i]) + '*' + 'p' +
                                        str(ppstor[1][j]))
                                    result.remove(current)
                                    result.insert(index, temp)
                                    current = temp
    #                                 print(result)
                                else:
                                    #                                 print('\n In ' + str(current)+' report2: '+'p'+str(ppstor[1][j]) + ' on position '+str(ppstor[0][j])+  ' does not commute with '+'x'+ str(xxstor[1][i]) + ' on position ' + str(xxstor[0][i]))
                                    sspar = current[ppstor[0][j]:ppstor[0][j] +
                                                    7].replace(
                                                        'p' +
                                                        str(ppstor[1][j]) +
                                                        '*' + 'x' +
                                                        str(xxstor[1][i]), '',
                                                        1)
                                    sspar = current[:ppstor[0]
                                                    [j]] + sspar + current[
                                                        ppstor[0][j] + 7:]
                                    temp = current[ppstor[0][j]:ppstor[0][j] +
                                                   7].replace(
                                                       'p' +
                                                       str(ppstor[1][j]) +
                                                       '*' + 'x' +
                                                       str(xxstor[1][i]), 'x' +
                                                       str(xxstor[1][i]) +
                                                       '*' + 'p' +
                                                       str(ppstor[1][j]))
                                    temp = current[:ppstor[0][
                                        j]] + temp + current[ppstor[0][j] + 7:]
                                    result.remove(current)
                                    result.insert(index, temp)
                                    current = temp
                                    sspar = sspar.replace('**', '*')
                                    if sspar[0] == '*':
                                        sspar = sspar[1:]
                                    elif sspar[-1] == '*':
                                        sspar = sspar[:-1]
                                    if sspar != '':
                                        result.append(sspar)
    #                                 print(result)
                        else:
                            if ppstor[0][j] < xxstor[0][i]:
                                if ppstor[1][j] != xxstor[1][i]:
                                    #                                 print('\n In ' + str(current)+' report3: '+'p'+str(ppstor[1][j]) + ' on position '+str(ppstor[0][j])+  ' commutes with '+'x'+ str(xxstor[1][i]) + ' on position ' + str(xxstor[0][i]))
                                    temp = current.replace(
                                        'p' + str(ppstor[1][j]) + '*' + 'x' +
                                        str(xxstor[1][i]),
                                        'x' + str(xxstor[1][i]) + '*' + 'p' +
                                        str(ppstor[1][j]))
                                    result.remove(current)
                                    result.insert(index, temp)
                                    current = temp
    #                                 print(result)
                                else:
                                    #                                 print('\n In ' + str(current)+' report4: '+'p'+str(ppstor[1][j]) + ' on position '+str(ppstor[0][j])+  ' does not commute with '+'x'+ str(xxstor[1][i]) + ' on position ' + str(xxstor[0][i]))
                                    sspar = current.replace(
                                        'p' + str(ppstor[1][j]) + '*' + 'x' +
                                        str(xxstor[1][i]), '', 1)
                                    temp = current.replace(
                                        'p' + str(ppstor[1][j]) + '*' + 'x' +
                                        str(xxstor[1][i]),
                                        'x' + str(xxstor[1][i]) + '*' + 'p' +
                                        str(ppstor[1][j]))
                                    result.remove(current)
                                    result.insert(index, temp)
                                    current = temp
                                    sspar = sspar.replace('**', '*')
                                    if sspar[0] == '*':
                                        sspar = sspar[1:]
                                    elif sspar[-1] == '*':
                                        sspar = sspar[:-1]
                                    if sspar != '':
                                        result.append(sspar)

    #                                 print(result)
        index += 1
        if index == len(result):
            return result
        return FNFxp(result, index)

    @staticmethod
    def FNFc(foo):
        if foo[-1] == '-':
            minus = True
        else:
            minus = False
        ans = ''
        current = foo.rstrip('-')
        if current.isdigit():
            if minus:
                return current + '-'
            else:
                return current

        xxstor = NC.xstor(current)
        ppstor = NC.pstor(current)
        uzzstor = NC.uzstor(current)
        puustor = NC.pustor(current)
        xxcurrent = ''
        ppcurrent = ''
        uzzcurrent = ''
        puucurrent = ''
        if uzzstor[0] != []:
            uzsort = sorted(uzzstor[1])
            uzcurrent = current[uzzstor[0][0]:uzzstor[0][-1] + 3]
            for i in range(len(uzsort)):
                uzzcurrent += uzcurrent[:uzcurrent.find(r'uz') + 4].replace(
                    str(uzzstor[1][i]), str(uzsort[i]))
                uzcurrent = uzcurrent[uzcurrent.find(r'uz') + 4:]
            ans += uzzcurrent + '*'

        if xxstor[0] != []:
            xsort = sorted(xxstor[1])
            xcurrent = current[xxstor[0][0]:xxstor[0][-1] + 3]
            for i in range(len(xsort)):
                xxcurrent += xcurrent[:xcurrent.find(r'x') + 4].replace(
                    str(xxstor[1][i]), str(xsort[i]))
                xcurrent = xcurrent[xcurrent.find(r'x') + 4:]
            ans += xxcurrent + '*'

        if puustor != []:
            pucurrent = current[puustor[0]:puustor[-1] + 3]
            ans += pucurrent + '*'

        if ppstor[0] != []:
            psort = sorted(ppstor[1])
            pcurrent = current[ppstor[0][0]:ppstor[0][-1] + 3]
            for i in range(len(psort)):
                ppcurrent += pcurrent[:pcurrent.find('p') + 4].replace(
                    str(ppstor[1][i]), str(psort[i]))
                pcurrent = pcurrent[pcurrent.find('p') + 4:]
            ans += ppcurrent + '*'

        ans = ans.rstrip('*')
        current = ans
        if minus:
            ans += '-'
        return ans

    @staticmethod
    def max_int(foo, count=1, answer=''):
        number = foo[:1]
        if number.isdigit():
            new_foo = foo[1:]
            count += 1
            answer += number
            return NC.max_int(new_foo, count, answer)
        else:
            return answer, count

    # @staticmethod
    def symsplit(self, foo):
        inpt = str(foo).split('+')
        iss = self.is_plus(foo)
        isss = []
        out = []
        out2 = []
        i = -1
        for x in inpt:
            w = x.split('-')
            for y in w:
                out.append(y)
        if out[0] == '':
            out.pop(0)
        for row in out:
            i += 1
            row = row.replace(' ', '')
            maxint, count = self.max_int(row)[0], self.max_int(row)[1]
            if row.isdigit():
                if maxint == '':
                    maxint = int(maxint)
                    for j in range(maxint):
                        out2.append(row[count - 1:])
                        isss.append(iss[i])
                else:
                    out2.append(row)
                    isss.append(iss[i])
            else:
                if maxint != '':
                    maxint = int(maxint)
                    for j in range(maxint):
                        out2.append(row[count:])
                        isss.append(iss[i])
                else:
                    out2.append(row)
                    isss.append(iss[i])
        return out2, isss

    @staticmethod
    def is_equal(plus, minus):
        new_plus = sorted(plus)
        new_minus = sorted(minus)
        return np.array_equal(new_plus, new_minus)

    @staticmethod
    def pustor(foo):
        global pustorage, punumber
        current = str(foo)
        pustorage = []
        punumber = []
        pattern = r'p_u'
        i = -1
        while current.find(pattern) != -1:
            i += 1
            pustorage.append(0)
            punumber.append(0)
            pustorage[i] = current.find(pattern)
            #         punumber[i] = int(current[current.find(pattern)+1:current.find(pattern)+3])
            current = current[current.find(pattern) + 4:]
            if i != 0:
                pustorage[i] += pustorage[i - 1] + 4
        return pustorage

    @staticmethod
    def uzstor(foo):
        global uzstorage, uznumber
        current = str(foo)
        uzstorage = []
        uznumber = []
        pattern = r'uz'
        i = -1
        while current.find(pattern) != -1:
            i += 1
            uzstorage.append(0)
            uznumber.append(0)
            uzstorage[i] = current.find(pattern)
            uznumber[i] = int(current[current.find(pattern) +
                                      2:current.find(pattern) + 3])
            current = current[current.find(pattern) + 4:]
            if i != 0:
                uzstorage[i] += uzstorage[i - 1] + 4
        return uzstorage, uznumber

    @staticmethod
    def lstor(foo):
        global lstorage, lnumber
        current = str(foo)
        lstorage = []
        lnumber = []
        pattern = r'l'
        i = -1
        while current.find(pattern) != -1:
            i += 1
            lstorage.append(0)
            lnumber.append(0)
            lstorage[i] = current.find(pattern)
            lnumber[i] = int(current[current.find(pattern) +
                                     1:current.find(pattern) + 2])
            current = current[current.find(pattern) + 3:]
            if i != 0:
                lstorage[i] += lstorage[i - 1] + 4
        return lstorage, lnumber

    @staticmethod
    def is_normaluz(foo):
        flag = True
        xstorage = NC.xstor(foo)[0]
        pstorage = NC.pstor(foo)[0]
        xnumber = NC.xstor(foo)[1]
        pnumber = NC.pstor(foo)[1]
        pustorage = NC.pustor(foo)
        uzstorage = NC.uzstor(foo)[0]

        for i in range(len(xstorage)):
            for j in range(len(pstorage)):
                if pstorage[j] < xstorage[i]:
                    flag = False
        for i in range(len(uzstorage)):
            for j in range(len(pustorage)):
                if pustorage[j] < uzstorage[i]:
                    flag = False
        for i in range(len(uzstorage)):
            for j in range(len(xstorage)):
                if xstorage[j] < uzstorage[i]:
                    flag = False
        for i in range(len(uzstorage)):
            for j in range(len(pstorage)):
                if pstorage[j] < uzstorage[i]:
                    flag = False
        for i in range(len(pustorage)):
            for j in range(len(xstorage)):
                if xstorage[j] > pustorage[i]:
                    flag = False
        for i in range(len(pustorage)):
            for j in range(len(pstorage)):
                if pstorage[j] < pustorage[i]:
                    flag = False
        return flag

    @staticmethod
    def is_normalorderuz(foo):
        flag = True
        xstorage = NC.xstor(foo)[0]
        pstorage = NC.pstor(foo)[0]
        xnumber = Nc.xstor(foo)[1]
        pnumber = NC.pstor(foo)[1]
        pustorage = NC.pustor(foo)
        uzstorage = NC.uzstor(foo)[0]
        uznumber = NC.uzstor(foo)[1]
        if uznumber != sorted(uznumber):
            return False
        if pnumber != sorted(pnumber):
            return False
        if xnumber != sorted(xnumber):
            return False
        return is_normaluz(foo)

    @staticmethod
    def symsplit2(foo):
        ssym = []
        regex = re.compile(r'[-+]\s')
        sym = regex.split(foo)
        for i in range(len(sym)):
            if i == 0:
                sym[i] = list(sym[i])
                del sym[i][0]
                sym[i] = "".join(sym[i])
            sym[i] = sym[i].replace(' ', '')
            ssym.append(sym[i])
        return ssym

    @staticmethod
    def FNFuz(foo, index=0):
        result = []
        if isinstance(foo, str):
            result.append(foo)
        elif isinstance(foo, list):
            for item in foo:
                result.append(item)
        current = result[index]
        #     print(result)
        while not NC.is_normaluz(current):
            uzzstor = NC.uzstor(current)
            ppustor = NC.pustor(current)
            xxstor = NC.xstor(current)
            ppstor = NC.pstor(current)
            sspar = ''
            signstorage = ''
            for i in range(len(uzzstor[0])):
                for j in range(len(ppustor)):

                    if current.rstrip('-') == 'p_u' + '*' + 'uz' + str(
                            uzzstor[1][i]):
                        if current.find('-') != -1:
                            signstorage = current[current.find('-'):]
                            current = current.rstrip('-')
    #                     print('report0')
                        sspar = 'uz' + str(uzzstor[1][i]) + '*' + 'uz' + str(
                            uzzstor[1][i])
                        sspar += '-'
                        temp = current.replace(
                            'p_u' + '*' + 'uz' + str(uzzstor[1][i]),
                            'uz' + str(uzzstor[1][i]) + '*' + 'p_u')

                        current += signstorage
                        temp += signstorage
                        sspar += signstorage

                        result.remove(current)
                        result.insert(index, temp)
                        current = temp
                        result.append(sspar)

    #                     print(result)
                    else:
                        if j != len(ppustor) - 1:
                            # if current.find('-') != -1:
                            #     signstorage = current[current.find('-'):]
                            #     current = current.rstrip('-')
                            if ppustor[j] + 4 == uzzstor[0][i] and ppustor[
                                    j + 1] > uzzstor[0][i]:
                                sspar = current.replace(
                                    'p_u' + '*' + 'uz' + str(uzzstor[1][i]),
                                    'uz' + str(uzzstor[1][i]) + '*' + 'uz' +
                                    str(uzzstor[1][i]), 1)
                                temp = current.replace(
                                    'p_u' + '*' + 'uz' + str(uzzstor[1][i]),
                                    'uz' + str(uzzstor[1][i]) + '*' + 'p_u')
                                result.remove(current)
                                result.insert(index, temp)
                                sspar = sspar.replace('**', '*')
                                if sspar[0] == '*':
                                    sspar = sspar[1:]
                                elif sspar[-1] == '*':
                                    sspar = sspar[:-1]
                                sspar += '-'
                                if sspar != '':
                                    result.append(sspar)
                                current = temp
    #                                 print(result)
                        else:
                            if ppustor[j] + 4 == uzzstor[0][i]:
                                #                             print('report2' +' '+ str(ppustor[j]) +' '+ str(uzzstor[0][i]))
                                sspar = current.replace(
                                    'p_u' + '*' + 'uz' + str(uzzstor[1][i]),
                                    'uz' + str(uzzstor[1][i]) + '*' + 'uz' +
                                    str(uzzstor[1][i]), 1)
                                temp = current.replace(
                                    'p_u' + '*' + 'uz' + str(uzzstor[1][i]),
                                    'uz' + str(uzzstor[1][i]) + '*' + 'p_u')
                                result.remove(current)
                                result.insert(index, temp)
                                sspar = sspar.replace('**', '*')
                                if sspar[0] == '*':
                                    sspar = sspar[1:]
                                elif sspar[-1] == '*':
                                    sspar = sspar[:-1]
                                sspar += '-'
                                if sspar != '':
                                    result.append(sspar)
                                current = temp
    #                                 print(result)

            xxstor = NC.xstor(current)
            ppstor = NC.pstor(current)
            spar = ''
            for ii in range(len(xxstor[0])):
                for jj in range(len(ppstor[0])):

                    if current.rstrip('-') == 'p' + str(
                            ppstor[1][jj]) + '*' + 'x' + str(xxstor[1][ii]):
                        if current.find('-') != -1:
                            signstorage = current[current.find('-'):]
                            current = current.rstrip('-')
                        if ppstor[1][jj] == xxstor[1][ii]:
                            sspar = '1'
                        else:
                            sspar = ''
                        temp = current.replace(
                            'p' + str(ppstor[1][jj]) + '*' + 'x' +
                            str(xxstor[1][ii]), 'x' + str(xxstor[1][ii]) +
                            '*' + 'p' + str(ppstor[1][jj]))

                        current += signstorage
                        temp += signstorage
                        sspar += signstorage

                        result.remove(current)
                        result.insert(index, temp)
                        current = temp
                        if sspar != '':
                            result.append(sspar)
    #                     print(result)
                    else:
                        if jj != len(ppstor[0]) - 1:
                            if ppstor[0][jj] + 4 == xxstor[0][
                                    ii] and ppstor[0][jj + 1] > xxstor[0][ii]:
                                if current.find('-') != -1:
                                    signstorage = current[current.find('-'):]
                                    current = current.rstrip('-')
                                if ppstor[1][jj] != xxstor[1][ii]:

                                    temp = current.replace(
                                        'p' + str(ppstor[1][jj]) + '*' + 'x' +
                                        str(xxstor[1][ii]),
                                        'x' + str(xxstor[1][ii]) + '*' + 'p' +
                                        str(ppstor[1][jj]))

                                    current += signstorage
                                    temp += signstorage

                                    result.remove(current)
                                    result.insert(index, temp)
                                    current = temp
    #                                 print(result)
                                else:

                                    sspar = current[
                                        ppstor[0][jj]:ppstor[0][jj] +
                                        7].replace(
                                            'p' + str(ppstor[1][jj]) + '*' +
                                            'x' + str(xxstor[1][ii]), '', 1)
                                    sspar = current[:ppstor[0]
                                                    [jj]] + sspar + current[
                                                        ppstor[0][jj] + 7:]
                                    temp = current[
                                        ppstor[0][jj]:ppstor[0][jj] +
                                        7].replace(
                                            'p' + str(ppstor[1][jj]) + '*' +
                                            'x' + str(xxstor[1][ii]),
                                            'x' + str(xxstor[1][ii]) + '*' +
                                            'p' + str(ppstor[1][jj]))
                                    temp = current[:ppstor[0]
                                                   [jj]] + temp + current[
                                                       ppstor[0][jj] + 7:]

                                    current += signstorage
                                    temp += signstorage

                                    result.remove(current)
                                    result.insert(index, temp)
                                    current = temp
                                    sspar = sspar.replace('**', '*')
                                    if sspar[0] == '*':
                                        sspar = sspar[1:]
                                    elif sspar[-1] == '*':
                                        sspar = sspar[:-1]
                                    if sspar != '':
                                        sspar += signstorage
                                        result.append(sspar)
    #                                 print(result)
                        else:
                            if ppstor[0][jj] + 4 == xxstor[0][ii]:
                                if current.find('-') != -1:
                                    signstorage = current[current.find('-'):]
                                    current = current.rstrip('-')
                                if ppstor[1][jj] != xxstor[1][ii]:
                                    temp = current.replace(
                                        'p' + str(ppstor[1][jj]) + '*' + 'x' +
                                        str(xxstor[1][ii]),
                                        'x' + str(xxstor[1][ii]) + '*' + 'p' +
                                        str(ppstor[1][jj]))

                                    current += signstorage
                                    temp += signstorage

                                    result.remove(current)
                                    result.insert(index, temp)
                                    current = temp

    #                                         print(result)
                                else:

                                    sspar = current.replace(
                                        'p' + str(ppstor[1][jj]) + '*' + 'x' +
                                        str(xxstor[1][ii]), '', 1)
                                    temp = current.replace(
                                        'p' + str(ppstor[1][jj]) + '*' + 'x' +
                                        str(xxstor[1][ii]),
                                        'x' + str(xxstor[1][ii]) + '*' + 'p' +
                                        str(ppstor[1][jj]))

                                    current += signstorage
                                    temp += signstorage

                                    result.remove(current)
                                    result.insert(index, temp)
                                    current = temp
                                    sspar = sspar.replace('**', '*')
                                    if sspar[0] == '*':
                                        sspar = sspar[1:]
                                    elif sspar[-1] == '*':
                                        sspar = sspar[:-1]
                                    if sspar != '':
                                        sspar += signstorage
                                        result.append(sspar)

            for j in range(len(ppustor)):
                for jj in range(len(ppstor[0])):
                    if ppustor[j] == ppstor[0][jj] + 4:
                        temp = current.replace(
                            'p' + str(ppstor[1][jj]) + '*' + 'p_u',
                            'p_u' + '*' + 'p' + str(ppstor[1][jj]))
                        result.remove(current)
                        result.insert(index, temp)
                        current = temp

            for j in range(len(ppustor)):
                for ii in range(len(xxstor[0])):
                    if ppustor[j] == xxstor[0][ii] - 4:
                        temp = current.replace(
                            'p_u' + '*' + 'x' + str(xxstor[1][ii]),
                            'x' + str(xxstor[1][ii]) + '*' + 'p_u')
                        result.remove(current)
                        result.insert(index, temp)
                        current = temp

            for i in range(len(uzzstor[0])):
                for jj in range(len(ppstor[0])):
                    if uzzstor[0][i] == ppstor[0][jj] + 4:
                        temp = current.replace(
                            'p' + str(ppstor[1][jj]) + '*' + 'uz' +
                            str(uzzstor[1][i]), 'uz' + str(uzzstor[1][i]) +
                            '*' + 'p' + str(ppstor[1][jj]))
                        result.remove(current)
                        result.insert(index, temp)
                        current = temp

            for i in range(len(uzzstor[0])):
                for ii in range(len(xxstor[0])):
                    if uzzstor[0][i] == xxstor[0][ii] + 4:
                        temp = current.replace(
                            'x' + str(xxstor[1][ii]) + '*' + 'uz' +
                            str(uzzstor[1][i]), 'uz' + str(uzzstor[1][i]) +
                            '*' + 'x' + str(xxstor[1][ii]))
                        result.remove(current)
                        result.insert(index, temp)
                        current = temp

        index += 1
        if index == len(result):
            answer = []
            for row in result:
                count = 0
                while row.find('-') != -1:
                    count += 1
                    row = row[:-1]
                if count % 2 == 1:
                    row += '-'
                answer.append(row)
            return answer
        return NC.FNFuz(result, index)

    @staticmethod
    def decode_multiples(foo):
        decoded = ''
        regmult = re.compile('\w{3}[*][*]\d')
        right = regmult.findall(foo)
        new = foo
        for j in range(len(right)):
            rr = right[j]
            decoded = ''
            for i in range(int(rr[5:])):
                decoded += rr[:3]
                decoded += '*'
            decoded = decoded.strip('*')
            new = new[:new.index(rr)] + new[new.index(rr):new.index(rr) +
                                            6].replace(
                                                rr, decoded,
                                                1) + new[new.index(rr) + 6:]
        # print( right)
        return new

    # @staticmethod
    def answer(self, Capelli):
        global plus, minus
        sym = self.symsplit(Capelli)
        plus = []
        minus = []
        for i in tnrange(len(sym[0])):
            if sym[1][i] == '+':
                for item in self.FNF(sym[0][i]):
                    plus.append(item)
            else:
                for item in self.FNF(sym[0][i]):
                    minus.append(item)
        ans = self.is_equal(plus, minus)
        setp = set(plus)
        setm = set(minus)
        diff = []
        diff.append(setp - setm)
        diff.append(setm - setp)

        if ans:
            anss = "Успех!. Возникло членов со знаком плюс — {0}, членов со знаком минус — {1} ".format(
                len(plus), len(minus))
        else:
            anss = "Возникает '{0}' поправок со знаком плюс и '{1}' поправок со знаком минус.".format(
                len(plus), len(minus))
        return anss  #, len(plus),sorted(plus), len(minus),sorted(minus), sorted(diff)

    # @staticmethod
    def answeruz(self, foo):
        global plus, minus
        sym = self.symsplit(foo)[0]
        wr = []
        for row in sym:
            wr += self.FNF(self.decode_multiples(row))[1:]
        plus = []
        minus = []

        for i in tnrange(len(wr)):
            if wr[i][-1] == '-':
                minus.append(wr[i].rstrip('-'))
            else:
                plus.append(wr[i])
        ans = self.is_equal(plus, minus)
        return ans, len(plus), len(minus)

    def answeruz_presentation(self, foo):
        global plus, minus
        sym = self.symsplit(foo)[0]
        wr = []
        for row in sym:
            wr += self.FNF(self.decode_multiples(row))[1:]
        plus = []
        minus = []

        for i in tnrange(len(wr)):
            if wr[i][-1] == '-':
                minus.append(wr[i].rstrip('-'))
            else:
                plus.append(wr[i])
        # return ans, len(plus), len(minus)
        ans = self.is_equal(plus, minus)
        setp = set(plus)
        setm = set(minus)
        diff = []
        diff.append(setp - setm)
        diff.append(setm - setp)

        if ans:
            anss = "Успех!. Возникло поправок со знаком плюс — {0}, поправок со знаком минус — {1} ".format(
                len(plus), len(minus))
        else:
            anss = "Возникает '{0}' поправок со знаком плюс и '{1}' поправок со знаком минус.".format(
                len(plus), len(minus))
        return anss  #, len(plus),sorted(plus), len(minus),sorted(minus), sorted(diff)

    @staticmethod
    def FNF(foo):
        result = []
        for item in NC.FNFuz(foo):
            result.append(NC.FNFc(item))
        return result


class xcls():
    def __init__(self):
        self.name = 'x'

    @staticmethod
    # # @njit
    def stor(foo):
        global xstorage, xnumber
        current = str(foo)
        xstorage = []
        xnumber = []
        regex = re.compile('[x]\d{2}')
        pattern = regex.findall(current)
        for i in range(len(pattern)):
            pt = pattern[i]
            xstorage.append(0)
            xnumber.append(0)
            xstorage[i] = current.find(pt)
            xnumber[i] = int(current[current.find(pt) + 1:current.find(pt) +
                                     3])
            current = current[current.find(pt) + 4:]
            if i != 0:
                xstorage[i] += xstorage[i - 1] + 4
        return xstorage, xnumber


class xx0cls():
    def __init__(self):
        self.name = 'xx'

    # # @njit
    @staticmethod
    def stor(foo):
        global xx0storage, xx0number
        current = str(foo)
        xx0storage = []
        xx0number = []
        regex = re.compile('[x][x]\d{1}')
        pattern = regex.findall(current)
        for i in range(len(pattern)):
            pt = pattern[i]
            xx0storage.append(0)
            xx0number.append(0)
            xx0storage[i] = current.find(pt)
            xx0number[i] = int(current[current.find(pt) + 2:current.find(pt) +
                                       3])
            current = current[current.find(pt) + 4:]
            if i != 0:
                xx0storage[i] += xx0storage[i - 1] + 4
        return xx0storage, xx0number


class bcls():
    def __init__(self):
        self.name = 'b'

    # # @njit
    @staticmethod
    def stor(foo):
        global bstorage, bnumber
        current = str(foo)
        bstorage = []
        bnumber = []
        regex = re.compile('[b]\d{2}')
        pattern = regex.findall(current)
        for i in range(len(pattern)):
            pt = pattern[i]
            bstorage.append(0)
            bnumber.append(0)
            bstorage[i] = current.find(pt)
            bnumber[i] = int(current[current.find(pt) + 1:current.find(pt) +
                                     3])
            current = current[current.find(pt) + 4:]
            if i != 0:
                bstorage[i] += bstorage[i - 1] + 4
        return bstorage, bnumber


class acls():
    def __init__(self):
        self.name = 'a'

    # # @njit
    @staticmethod
    def stor(foo):
        global astorage, anumber
        current = str(foo)
        astorage = []
        anumber = []
        regex = re.compile('[a]\d{2}')
        pattern = regex.findall(current)
        for i in range(len(pattern)):
            pt = pattern[i]
            astorage.append(0)
            anumber.append(0)
            astorage[i] = current.find(pt)
            anumber[i] = int(current[current.find(pt) + 1:current.find(pt) +
                                     3])
            current = current[current.find(pt) + 4:]
            if i != 0:
                astorage[i] += astorage[i - 1] + 4
        return astorage, anumber


class edxcls():
    def __init__(self):
        self.name = 'edx'

    # # @njit
    @staticmethod
    def stor(foo):
        global edxstorage, edxnumber
        current = str(foo)
        edxstorage = []
        pattern = r'edx'
        i = -1
        while current.find(pattern) != -1:
            i += 1
            edxstorage.append(0)
            edxstorage[i] = current.find(pattern)
            current = current[current.find(pattern) + 4:]
            if i != 0:
                edxstorage[i] += edxstorage[i - 1] + 4
        return edxstorage


class eecls():
    def __init__(self):
        self.name = 'ee'

    # # @njit
    @staticmethod
    def stor(foo):
        global eestorage, eenumber
        current = str(foo)
        eestorage = []
        eenumber = []
        regex = re.compile('[e][e]\d{1}')
        pattern = regex.findall(current)
        for i in range(len(pattern)):
            pt = pattern[i]
            eestorage.append(0)
            eenumber.append(0)
            eestorage[i] = current.find(pt)
            eenumber[i] = int(current[current.find(pt) + 2:current.find(pt) +
                                      3])
            current = current[current.find(pt) + 4:]
            if i != 0:
                eestorage[i] += eestorage[i - 1] + 4
        return eestorage, eenumber


class wcls():
    def __init__(self):
        self.name = 'w'

    # # @njit
    @staticmethod
    def stor(foo):
        global wstorage, wnumber
        current = str(foo)
        wstorage = []
        wnumber = []
        regex = re.compile('[w]\d{2}')
        pattern = regex.findall(current)
        for i in range(len(pattern)):
            pt = pattern[i]
            wstorage.append(0)
            wnumber.append(0)
            wstorage[i] = current.find(pt)
            wnumber[i] = int(current[current.find(pt) + 1:current.find(pt) +
                                     3])
            current = current[current.find(pt) + 4:]
            if i != 0:
                wstorage[i] += wstorage[i - 1] + 4
        return wstorage, wnumber


class qcls():
    def __init__(self):
        self.name = 'q'

    # # @njit
    @staticmethod
    def stor(foo):
        # global wstorage, wnumber
        current = str(foo)
        wstorage = []
        wnumber = []
        regex = re.compile('[q]\d{2}')
        pattern = regex.findall(current)
        for i in range(len(pattern)):
            pt = pattern[i]
            wstorage.append(0)
            wnumber.append(0)
            wstorage[i] = current.find(pt)
            wnumber[i] = int(current[current.find(pt) + 1:current.find(pt) +
                                     3])
            current = current[current.find(pt) + 4:]
            if i != 0:
                wstorage[i] += wstorage[i - 1] + 4
        return wstorage, wnumber


class XXX(NC):
    def __init__(self, N, K):
        self.N = N
        self.K = K
        w = Symbol('w', commutative=False)
        edx = Symbol('edx', commutative=False)
        xx0 = Symbol('xx0', commutative=False)
        M = [[0 for j in range(self.N)] for i in range(self.N)]
        for i in range(self.N):
            for j in range(self.N):
                M[i][j] = Symbol('M' + str(i + 1) + str(j + 1),
                                 commutative=False)
        MM = M
        M = Matrix(M)

        Q = [0 for i in range(self.K)]
        for i in range(self.K):
            Q[i] = Symbol('q' + str(i + 1) + str(i + 1), commutative=False)
        Q = Matrix(Q)

        X = [0 for j in range(N)]
        for i in range(N):
            X[i] = Symbol('x' + str(i + 1) + str(i + 1), commutative=False)
        XX = X
        # X = Matrix(X)

        A = [[0 for j in range(self.K)] for i in range(self.N)]
        for i in range(self.N):
            for j in range(self.K):
                A[i][j] = Symbol('a' + str(i + 1) + str(j + 1),
                                 commutative=False)
        AA = A
        # A = Matrix(A)

        B = [[0 for j in range(self.K)] for i in range(self.N)]
        for i in range(self.N):
            for j in range(self.K):
                B[i][j] = Symbol('b' + str(i + 1) + str(j + 1),
                                 commutative=False)
        BB = B
        # B = Matrix(B)

        self.A = A
        self.M = M
        self.B = B
        self.M = M
        self.X = X
        self.edx = edx
        self.w = w
        self.xx0 = xx0
        self.Q = Q

    # @njit
    def sigma2(self):
        return list(permutations([x for x in range(1, self.N + 1)]))

    # @njit
    def Ematrix(self, N, a, b):
        Matr = [[0 for x in range(self.N)] for x in range(self.N)]
        for i in range(self.N):
            for j in range(self.N):
                if i == b and j == a:
                    Matr[i][j] = 1
        return Matrix(Matr)

    # @njit
    def LtG(self):
        L = [[0 for _ in range(self.N)] for _ in range(self.N)]
        for j in range(self.N):
            # L[j][j] += self.X[j]
            for i in range(self.N):
                for a in range(self.K):
                    if i < j:
                        L[i][j] += self.A[i][a] * self.B[j][a]
                    L[i][j] += self.A[i][a] * self.B[j][a] * Symbol(
                        'ee' + str(1), commutative=False)
        return L

    # @njit
    def Lrat_true(self, foo, i):
        result = zeros(self.N, self.N)
        for b in range(self.N):
            bib = Symbol('b' + str(i) + str(b + 1), commutative=False)
            for a in range(self.N):
                aia = Symbol('a' + str(i) + str(a + 1), commutative=False)
                result += TensorProduct(
                    self.Ematrix(N, a, b),
                    Matrix([KroneckerDelta(a, b) + (foo) * bib * aia]))
        return result

    # @njit
    def Lrat(self, foo, i):
        one = ones(self.N, self.N) * (1 / (foo) * b_b * a_a)
        diag = eye(self.N, self.N)
        return one + diag

    def inv(self, n):
        result = 0
        sigma = self.sigma()[n]
        for b in range(len(sigma)):
            for a in range(b):
                if sigma[a] > sigma[b]: result += 1
        return result

    # @njit
    def qdet(self, foo):
        result = 0
        sigma = self.sigma()
        c = self.c
        for j in range(len(sigma)):
            current = 1
            sm = 0
            for i in range(self.N):
                current = current * foo[sigma[j][i], i]
                sm += c(i, j)
            if sm == 0:
                sgn = 1
            elif (sm % 2) == 0:
                sgn = -1
            else:
                sgn = 1
            if sgn == 1:
                result += (-q)**inv(i) * current
            else:
                result += -(-q)**inv(i) * current
        return result

    # @njit
    def j_delta(self, m):
        N = self.N
        JJ = []
        for i in combinations(range(1, N + 1), m):
            JJ.append(i)
        return JJ

    # @njit
    def delta_sum(self, x, sigm):
        counter = 0
        for j in range(x):
            if self.sigma2()[sigm][j] > x:
                counter += 1
        return counter

    # @njit
    def tilda_cdet(self, foo):
        # sigma = [(0,1,2)]
        # sigma2 = [(1,2,3)]
        sigma = self.sigma()
        sigma2 = self.sigma2()
        result = 0
        for m in range(self.N - 1):
            if m % 2 == 0:
                mult1 = 1
            else:
                mult1 = -1
            for jtuple in self.j_delta(m):
                for j in range(len(sigma)):
                    current = 1
                    for i in range(self.N):
                        if i + 1 in jtuple:
                            if sigma[j][i] != i:
                                current = 0

                        else:
                            current *= foo[sigma[j][i], i]
                        # print(j, jtuple, sigma2[j], sigma[j][i]+1, i+1, i+1 in jtuple)
                    current *= (-1)**self.inv(j)
                    for alpha in range(m):
                        current *= self.delta_sum(jtuple[alpha], j)
                    current *= mult1
                    result += current
        return result

    # @njit
    def TXXX(self):
        result = 1
        for i in range(self.K):
            result *= self.Lrat_true(self.xx0, i + 1)
        return result

    # @njit
    def fifteen(self):
        return self.cdet(eye(self.N) - self.edx * self.TXXX())

    # @njit
    def eighteen_right(self):
        ltg = self.LtG()
        for i in range(self.N):
            ltg[i][i] = expand(self.X[i] - ltg[i][i])
        ltg = Matrix(ltg)
        current = self.cdet(ltg)
        for a in range(self.K):
            current = (1 - self.edx) * current
        left = expand(current)
        return left

    # @njit
    def eighteen_left(self):
        ltg = self.LtG()
        for i in range(self.N):
            ltg[i][i] = expand(self.X[i] - ltg[i][i])
        # ltg = Matrix(ltg)
        # current = self.tilda_cdet(ltg)
        # for a in range(self.K):
        # current = (1 - self.edx)*current
        # left = expand(current)
        return ltg

    # @njit
    @staticmethod
    def is_normalxx0(foo):
        flag = True
        bstorage = bcls.stor(foo)[0]
        astorage = acls.stor(foo)[0]
        pustorage = edxcls.stor(foo)
        eestorage = eecls.stor(foo)[0]
        uzstorage = xx0cls.stor(foo)[0]
        xstorage = xcls.stor(foo)[0]
        wstorage = wcls.stor(foo)[0]
        qstorage = qcls.stor(foo)[0]
        # template q11*w11*x11*xx0*b11*a11*edx*ee1
        for q in qstorage:
            for x in xstorage:
                if x < q: flag = False
            for a in astorage:
                if a < q: flag = False
            for xx0 in uzstorage:
                if xx0 < q: flag = False
            for edx in pustorage:
                if edx < q: flag = False
            for ee in eestorage:
                if ee < q: flag = False
            for b in bstorage:
                if b < q: flag = False
            for w in wstorage:
                if w < q: flag = False

        for w in wstorage:
            for x in xstorage:
                if x < w: flag = False
            for a in astorage:
                if a < w: flag = False
            for xx0 in uzstorage:
                if xx0 < w: flag = False
            for edx in pustorage:
                if edx < w: flag = False
            for ee in eestorage:
                if ee < w: flag = False
            for b in bstorage:
                if b < w: flag = False

        for x in xstorage:
            for a in astorage:
                if a < x: flag = False
            for xx0 in uzstorage:
                if xx0 < x: flag = False
            for edx in pustorage:
                if edx < x: flag = False
            for ee in eestorage:
                if ee < x: flag = False
            for b in bstorage:
                if b < x: flag = False

        for xx0 in uzstorage:
            for b in bstorage:
                if b < xx0: flag = False
            for a in astorage:
                if a < xx0: flag = False
            for edx in edxstorage:
                if edx < xx0: flag = False
            for ee in eestorage:
                if ee < xx0: flag = False

        for b in bstorage:
            for a in astorage:
                if a < b: flag = False
            for edx in edxstorage:
                if edx < b: flag = False
            for ee in eestorage:
                if ee < b: flag = False

        for a in astorage:
            for edx in edxstorage:
                if edx < a: flag = False
            for ee in eestorage:
                if ee < a: flag = False

        for edx in edxstorage:
            for ee in eestorage:
                if ee < edx: flag = False

        return flag

    # @njit
    @staticmethod
    def is_normalorderxx0(foo):
        flag = True
        xnumber = xcls.stor(foo)[1]
        bnumber = bcls.stor(foo)[1]
        anumber = acls.stor(foo)[1]
        uznumber = xx0cls.stor(foo)[1]
        eenumber = eecls.stor(foo)[1]
        wnumber = wcls.stor(foo)[1]
        qnumber = qcls.stor(foo)[1]
        if uznumber != sorted(uznumber):
            return False
        if anumber != sorted(anumber):
            return False
        if xnumber != sorted(xnumber):
            return False
        if bnumber != sorted(bnumber):
            return False
        if eenumber != sorted(eenumber):
            return False
        if wnumber != sorted(wnumber):
            return False
        if qnumber != sorted(qnumber):
            return False
        return XXX.is_normalxx0(foo)

    # @njit
    @staticmethod
    def abc(current, ans, only, notedx=True):
        onlystor = only.stor(current)
        onlyname = only().name
        oonlystor = ''
        oonlycurrent = ''
        if notedx:
            if onlystor[0] != []:
                onlysort = sorted(onlystor[1])
                onlycurrent = current[onlystor[0][0]:onlystor[0][-1] + 3]
                for i in range(len(onlysort)):
                    oonlycurrent += onlycurrent[:onlycurrent.find(onlyname) +
                                                4].replace(
                                                    str(onlystor[1][i]),
                                                    str(onlysort[i]))
                    onlycurrent = onlycurrent[onlycurrent.find(onlyname) + 4:]
                ans += oonlycurrent + '*'
        else:
            if onlystor != []:
                onlycurrent = current[onlystor[0]:onlystor[-1] + 3]
                ans += onlycurrent + '*'

        return ans

    # @njit
    @staticmethod
    def commutators(current, left, right, leftnumber, rightnumber, obv=False):

        leftname = left().name
        rightname = right().name
        sign = '+'
        if left == acls and right == bcls:
            if obv:
                sspar = '1-'
            else:
                sspar = ''
                sign = '-'
        if left == edxcls and right == xx0cls:
            sspar = rightname + str(rightnumber) + '*' + rightname + str(
                rightnumber + 1) + '*' + leftname
            sign = '-'
        if left == eecls and right == xcls:
            sspar = 'edx' + '*' + leftname + str(
                leftnumber) + '*' + leftname + str(leftnumber)
        if left == edxcls and right == xcls:
            sspar = 'edx'

        return sspar, sign

    # @njit
    @staticmethod
    def commute(current,
                result,
                left,
                right,
                index,
                lnumber=True,
                rnumber=True):
        # left*right -> right*left
        if lnumber:
            leftnumber = left.stor(current)[1]
            leftitem = left.stor(current)[0]
        else:
            leftitem = left.stor(current)
            leftnumber = ['' for _ in leftitem]
        if rnumber:
            rightnumber = right.stor(current)[1]
            rightitem = right.stor(current)[0]
        else:
            rightitem = right.stor(current)
            rightnumber = ['' for _ in rightitem]

        leftname = left.name
        rightname = right.name
        for i in range(len(rightitem)):
            for j in range(len(leftitem)):
                if leftitem[j] + 4 == rightitem[i]:
                    temp = current[leftitem[j]:leftitem[j] + 7].replace(
                        str(leftname) + str(leftnumber[j]) + '*' +
                        str(rightname) + str(rightnumber[i]),
                        str(rightname) + str(rightnumber[i]) + '*' +
                        str(leftname) + str(leftnumber[j]), 1)
                    temp = current[:leftitem[j]] + temp + current[leftitem[j] +
                                                                  7:]
                    result.remove(current)
                    result.insert(index, temp)
                    current = temp
        return current, result

    # @njit
    def cheat(current, result, left, right, index, signstorage):
        # edx x -> x edx - ee + q edx ee
        rightnumber = right.stor(current)[1]
        rightitem = right.stor(current)[0]
        leftitem = left.stor(current)
        leftnumber = ['' for _ in rightitem]
        leftname = left().name
        rightname = right().name

        for ii in range(len(rightitem)):
            for jj in range(len(leftitem)):
                if current.rstrip('-') == leftname + '*' + rightname + str(
                        rightnumber[ii]):
                    if current.find('-') != -1:
                        signstorage = current[current.find('-'):]
                        current = current.rstrip('-')

                    sspar = []
                    sspar.append(eecls().name + str(1))
                    sspar.append(leftname + '*' + eecls().name + str(1) + '-')
                    temp = current.replace(
                        leftname + '*' + rightname + str(rightnumber[ii]),
                        rightname + str(rightnumber[ii]) + '*' + leftname +
                        str(leftnumber[jj]))

                    current += signstorage
                    temp += signstorage
                    for ss in sspar:
                        ss += signstorage

                    result.remove(current)
                    result.insert(index, temp)
                    current = temp
                    for ss in sspar:
                        result.append(ss)

                else:
                    if jj != len(leftitem) - 1:
                        if leftitem[jj] + 4 == rightitem[ii] and leftitem[
                                jj + 1] > rightitem[ii]:
                            if current.find('-') != -1:
                                signstorage = current[current.find('-'):]
                                current = current.rstrip('-')

                            sspar = []
                            ss = eecls().name + str(1)
                            sspar.append(current[:leftitem[jj]] + ss +
                                         current[leftitem[jj] + 7:])
                            ss = leftname + '*' + eecls().name + str(1)
                            sspar.append(current[:leftitem[jj]] + ss +
                                         current[leftitem[jj] + 7:] + '-')

                            temp = current[leftitem[jj]:leftitem[jj] +
                                           7].replace(
                                               leftname + '*' + rightname +
                                               str(rightnumber[ii]),
                                               rightname +
                                               str(rightnumber[ii]) + '*' +
                                               leftname, 1)
                            temp = current[:leftitem[jj]] + temp + current[
                                leftitem[jj] + 7:]

                            current += signstorage
                            temp += signstorage

                            result.remove(current)
                            result.insert(index, temp)
                            current = temp

                            for ss in sspar:
                                ss = ss.replace('**', '*')
                                if ss[0] == '*':
                                    ss = sspar[1:]
                                elif ss[-1] == '*':
                                    ss = ss[:-1]
                                if ss != '':
                                    ss += signstorage
                                    result.append(ss)
                    else:
                        if leftitem[jj] + 4 == rightitem[ii]:
                            if current.find('-') != -1:
                                signstorage = current[current.find('-'):]
                                current = current.rstrip('-')

                            sspar = []
                            ss = eecls().name + str(1)
                            sspar.append(current[:leftitem[jj]] + ss +
                                         current[leftitem[jj] + 7:])
                            ss = leftname + '*' + eecls().name + str(1)
                            sspar.append(current[:leftitem[jj]] + ss +
                                         current[leftitem[jj] + 7:] + '-')

                            temp = current[leftitem[jj]:leftitem[jj] +
                                           7].replace(
                                               leftname + '*' + rightname +
                                               str(rightnumber[ii]),
                                               rightname +
                                               str(rightnumber[ii]) + '*' +
                                               leftname, 1)
                            temp = current[:leftitem[jj]] + temp + current[
                                leftitem[jj] + 7:]

                            current += signstorage
                            temp += signstorage

                            result.remove(current)
                            result.insert(index, temp)
                            current = temp

                            for ss in sspar:
                                ss = ss.replace('**', '*')
                                if ss[0] == '*':
                                    ss = sspar[1:]
                                elif ss[-1] == '*':
                                    ss = ss[:-1]
                                if ss != '':
                                    ss += signstorage
                                    result.append(ss)
        return current, result

    # @njit
    def wick(current,
             result,
             left,
             right,
             index,
             signstorage,
             lnumber=True,
             ee=False):
        # left*right -> right*left + spar
        if lnumber:
            leftnumber = left.stor(current)[1]
            leftitem = left.stor(current)[0]
        else:
            leftitem = left.stor(current)
            leftnumber = ['' for _ in leftitem]
        rightnumber = right.stor(current)[1]
        rightitem = right.stor(current)[0]
        # else:
        #     rightitem = right.stor(current)
        #     rightnumber = ['' for _ in rightitem]
        leftname = left().name
        rightname = right().name

        for ii in range(len(rightitem)):
            for jj in range(len(leftitem)):
                commutators = XXX.commutators(current, left, right,
                                              leftnumber[jj],
                                              rightnumber[ii])[0]
                sign = XXX.commutators(current, left, right, leftnumber[jj],
                                       rightnumber[ii])[1]
                if current.rstrip('-') == leftname + str(
                        leftnumber[jj]) + '*' + rightname + str(
                            rightnumber[ii]):
                    if current.find('-') != -1:
                        signstorage = current[current.find('-'):]
                        current = current.rstrip('-')
                    if leftnumber[jj] != rightnumber[ii] and lnumber and not ee:
                        temp = current[leftitem[jj]:leftitem[jj] + 7].replace(
                            leftname + str(leftnumber[jj]) + '*' + rightname +
                            str(rightnumber[ii]),
                            rightname + str(rightnumber[ii]) + '*' + leftname +
                            str(leftnumber[jj]), 1)
                        temp = current[:leftitem[jj]] + temp + current[
                            leftitem[jj] + 7:]

                        current += signstorage
                        temp += signstorage
                        result.remove(current)
                        result.insert(index, temp)
                        current = temp
                    else:
                        sspar = XXX.commutators(current, left, right,
                                                leftnumber[jj],
                                                rightnumber[ii], True)[0]
                        temp = current.replace(
                            leftname + str(leftnumber[jj]) + '*' + rightname +
                            str(rightnumber[ii]),
                            rightname + str(rightnumber[ii]) + '*' + leftname +
                            str(leftnumber[jj]))

                        current += signstorage
                        temp += signstorage
                        sspar += signstorage

                        result.remove(current)
                        result.insert(index, temp)
                        current = temp
                        result.append(sspar)

                else:
                    if jj != len(leftitem) - 1:
                        if leftitem[jj] + 4 == rightitem[ii] and leftitem[
                                jj + 1] > rightitem[ii]:
                            if current.find('-') != -1:
                                signstorage = current[current.find('-'):]
                                current = current.rstrip('-')
                            if leftnumber[jj] != rightnumber[
                                    ii] and lnumber and not ee:

                                temp = current[leftitem[jj]:leftitem[jj] +
                                               7].replace(
                                                   leftname +
                                                   str(leftnumber[jj]) + '*' +
                                                   rightname +
                                                   str(rightnumber[ii]),
                                                   rightname +
                                                   str(rightnumber[ii]) + '*' +
                                                   leftname +
                                                   str(leftnumber[jj]), 1)
                                temp = current[:leftitem[jj]] + temp + current[
                                    leftitem[jj] + 7:]

                                current += signstorage
                                temp += signstorage
                                result.remove(current)
                                result.insert(index, temp)
                                current = temp
                            else:

                                sspar = current[leftitem[jj]:leftitem[jj] +
                                                7].replace(
                                                    leftname +
                                                    str(leftnumber[jj]) + '*' +
                                                    rightname +
                                                    str(rightnumber[ii]),
                                                    commutators, 1)
                                sspar = current[:leftitem[
                                    jj]] + sspar + current[leftitem[jj] + 7:]

                                temp = current[leftitem[jj]:leftitem[jj] +
                                               7].replace(
                                                   leftname +
                                                   str(leftnumber[jj]) + '*' +
                                                   rightname +
                                                   str(rightnumber[ii]),
                                                   rightname +
                                                   str(rightnumber[ii]) + '*' +
                                                   leftname +
                                                   str(leftnumber[jj]), 1)
                                temp = current[:leftitem[jj]] + temp + current[
                                    leftitem[jj] + 7:]

                                current += signstorage
                                temp += signstorage

                                result.remove(current)
                                result.insert(index, temp)
                                current = temp
                                sspar = sspar.replace('**', '*')
                                if sspar[0] == '*':
                                    sspar = sspar[1:]
                                elif sspar[-1] == '*':
                                    sspar = sspar[:-1]
                                if sspar != '':
                                    if sign == '-':
                                        sspar += '-'
                                    sspar += signstorage
                                    result.append(sspar)
                    else:
                        if leftitem[jj] + 4 == rightitem[ii]:
                            if current.find('-') != -1:
                                signstorage = current[current.find('-'):]
                                current = current.rstrip('-')
                            if leftnumber[jj] != rightnumber[
                                    ii] and lnumber and not ee:

                                temp = current[leftitem[jj]:leftitem[jj] +
                                               7].replace(
                                                   leftname +
                                                   str(leftnumber[jj]) + '*' +
                                                   rightname +
                                                   str(rightnumber[ii]),
                                                   rightname +
                                                   str(rightnumber[ii]) + '*' +
                                                   leftname +
                                                   str(leftnumber[jj]))
                                temp = current[:leftitem[jj]] + temp + current[
                                    leftitem[jj] + 7:]

                                current += signstorage
                                temp += signstorage

                                result.remove(current)
                                result.insert(index, temp)
                                current = temp
                            else:

                                sspar = current[leftitem[jj]:leftitem[jj] +
                                                7].replace(
                                                    leftname +
                                                    str(leftnumber[jj]) + '*' +
                                                    rightname +
                                                    str(rightnumber[ii]),
                                                    commutators, 1)
                                sspar = current[:leftitem[
                                    jj]] + sspar + current[leftitem[jj] + 7:]

                                temp = current[leftitem[jj]:leftitem[jj] +
                                               7].replace(
                                                   leftname +
                                                   str(leftnumber[jj]) + '*' +
                                                   rightname +
                                                   str(rightnumber[ii]),
                                                   rightname +
                                                   str(rightnumber[ii]) + '*' +
                                                   leftname +
                                                   str(leftnumber[jj]))
                                temp = current[:leftitem[jj]] + temp + current[
                                    leftitem[jj] + 7:]

                                current += signstorage
                                temp += signstorage

                                result.remove(current)
                                result.insert(index, temp)
                                current = temp
                                sspar = sspar.replace('**', '*')
                                if sspar != '':
                                    if sspar[0] == '*':
                                        sspar = sspar[1:]
                                    elif sspar[-1] == '*':
                                        sspar = sspar[:-1]

                                    if sign == '-':
                                        sspar += '-'
                                    sspar += signstorage
                                    result.append(sspar)
        return current, result

    # @njit
    @staticmethod
    def FNFab(foo, index=0):
        result = []
        if isinstance(foo, str):
            result.append(foo)
        elif isinstance(foo, list):
            for item in foo:
                result.append(item)
        # if len(foo[0]) == 1:
        #     result.append(foo)
        # else:
        #     for item in foo:
        #         result.append(item)
        current = result[index]
        while not XXX.is_normalxx0(current):
            # print(result)
            signstorage = ''

            # WICK

            # ee x
            noncomm = XXX.wick(current, result, eecls, xcls, index,
                               signstorage, True, True)
            current, result = noncomm[0], noncomm[1]

            # edx x
            noncomm = XXX.cheat(current, result, edxcls, xcls, index,
                                signstorage)
            # noncomm = XXX.wick(current, result, edxcls, xcls, index, signstorage, False)
            current, result = noncomm[0], noncomm[1]

            # edx xx0
            noncomm = XXX.wick(current, result, edxcls, xx0cls, index,
                               signstorage, False)
            current, result = noncomm[0], noncomm[1]

            # a b
            noncomm = XXX.wick(current, result, acls, bcls, index, signstorage)
            current, result = noncomm[0], noncomm[1]
            # print(result)
            # COMMUTATORS

            # q
            comm = XXX.commute(current, result, wcls(), qcls(), index)
            current, result = comm[0], comm[1]

            comm = XXX.commute(current, result, xcls(), qcls(), index)
            current, result = comm[0], comm[1]

            comm = XXX.commute(current, result, xx0cls(), qcls(), index)
            current, result = comm[0], comm[1]

            comm = XXX.commute(current, result, bcls(), qcls(), index)
            current, result = comm[0], comm[1]

            comm = XXX.commute(current, result, acls(), qcls(), index)
            current, result = comm[0], comm[1]

            comm = XXX.commute(current, result, edxcls(), qcls(), index, False)
            current, result = comm[0], comm[1]

            comm = XXX.commute(current, result, eecls(), qcls(), index)
            current, result = comm[0], comm[1]

            # w
            comm = XXX.commute(current, result, xcls(), wcls(), index)
            current, result = comm[0], comm[1]

            comm = XXX.commute(current, result, xx0cls(), wcls(), index)
            current, result = comm[0], comm[1]

            comm = XXX.commute(current, result, bcls(), wcls(), index)
            current, result = comm[0], comm[1]

            comm = XXX.commute(current, result, acls(), wcls(), index)
            current, result = comm[0], comm[1]

            comm = XXX.commute(current, result, edxcls(), wcls(), index, False)
            current, result = comm[0], comm[1]

            comm = XXX.commute(current, result, eecls(), wcls(), index)
            current, result = comm[0], comm[1]

            # x
            comm = XXX.commute(current, result, xx0cls(), xcls(), index)
            current, result = comm[0], comm[1]

            comm = XXX.commute(current, result, bcls(), xcls(), index)
            current, result = comm[0], comm[1]

            comm = XXX.commute(current, result, acls(), xcls(), index)
            current, result = comm[0], comm[1]

            comm = XXX.commute(current, result, acls(), xcls(), index)
            current, result = comm[0], comm[1]

            # xx0
            comm = XXX.commute(current, result, bcls(), xx0cls(), index)
            current, result = comm[0], comm[1]

            comm = XXX.commute(current, result, acls(), xx0cls(), index)
            current, result = comm[0], comm[1]

            # b
            comm = XXX.commute(current, result, edxcls(), bcls(), index, False)
            current, result = comm[0], comm[1]

            comm = XXX.commute(current, result, eecls(), bcls(), index)
            current, result = comm[0], comm[1]

            # a
            comm = XXX.commute(current, result, edxcls(), acls(), index, False)
            current, result = comm[0], comm[1]

            comm = XXX.commute(current, result, eecls(), acls(), index)
            current, result = comm[0], comm[1]

            # ee
            comm = XXX.commute(current, result, eecls(), edxcls(), index, True,
                               False)
            current, result = comm[0], comm[1]

        index += 1
        if index == len(result):
            answer = []
            for row in result:
                count = 0
                while row.find('-') != -1:
                    count += 1
                    row = row[:-1]
                if count % 2 == 1:
                    row += '-'
                answer.append(row)
            return answer
        return XXX.FNFab(result, index)

    # @njit
    @staticmethod
    def FNFabc(foo):
        if foo[-1] == '-':
            minus = True
        else:
            minus = False
        ans = ''
        current = foo.rstrip('-')
        if current.isdigit():
            if minus:
                return current + '-'
            else:
                return current
        # template q11*w11*x11*xx0*b11*a11*edx*ee1
        ans = XXX.abc(current, ans, qcls)
        ans = XXX.abc(current, ans, wcls)
        ans = XXX.abc(current, ans, xcls)
        ans = XXX.abc(current, ans, xx0cls)
        ans = XXX.abc(current, ans, bcls)
        ans = XXX.abc(current, ans, acls)
        ans = XXX.abc(current, ans, edxcls, False)
        ans = XXX.abc(current, ans, eecls)

        ans = ans.rstrip('*')
        current = ans
        if minus:
            ans += '-'
        return ans

    # @njit
    @staticmethod
    def FNF2(foo):
        result = []
        for item in XXX.FNFab(foo):
            result.append(XXX.FNFabc(item))
        return result

    # @njit
    def answerab(self, foo):
        global plus, minus
        ans = True
        sym = self.symsplit(foo)[0]
        wr = []
        base = []
        for row in sym:
            wr += XXX.FNF2(self.decode_multiples(row))[1:]
            # base += XXX.FNF2(self.decode_multiples(row))[:1]
        plus = []
        minus = []

        for i in tnrange(len(wr)):
            if wr[i][-1] == '-':
                minus.append(wr[i].rstrip('-'))
            else:
                plus.append(wr[i])
        ans = XXX.is_equal(plus, minus)
        setp = set(plus)
        setm = set(minus)
        diff = list(setm - setp) + list(setp - setm)
        # setb = set(base)
        # if setp != setm:
        # ans = False
        return ans, len(plus), plus, len(minus), minus

    def answer_presentation(self, foo):
        global plus, minus
        ans = True
        sym = self.symsplit(foo)
        wr = []
        base = []
        for i in tnrange(len(sym[0])):
            #         print(i)
            wr += XXX.FNF2(self.decode_multiples(sym[0][i]))[1:]
            for _ in range(len(XXX.FNF2(self.decode_multiples(
                    sym[0][i]))[1:])):
                base += sym[1][i]
        plus = []
        minus = []

        for i in tnrange(len(wr)):
            if base[i] == '+':
                if wr[i][-1] == '-':
                    minus.append(wr[i].rstrip('-'))
                else:
                    plus.append(wr[i])
            else:
                if wr[i][-1] == '-':
                    plus.append(wr[i].rstrip('-'))
                else:
                    minus.append(wr[i])
        ans = self.is_equal(plus, minus)
        setp = set(plus)
        setm = set(minus)
        diff = []
        diff.append(setp - setm)
        diff.append(setm - setp)

        # if ans:
        #     anss = "Успех!. Возникло поправок со знаком плюс — {0}, поправок со знаком минус — {1} ".format(len(plus), len(minus))
        # else:
        #     anss = "Возникает '{0}' поправок со знаком плюс и '{1}' поправок со знаком минус.".format(len(plus), len(minus))
        return ans, len(plus), sorted(plus), len(minus), sorted(minus), sorted(
            diff)


if __name__ == '__main__':
    test = XXX(2, 2)
    print(
        "Успех!. Возникло '{0}' поправок со знаком плюс и '{1}' поправок с минусом. Все поправки сокращаются!"
        .format(12, 12))
