"""
Various helper functions
"""
import numpy as np


def print_visual_mul(W):
    """
    create a visual result of the multiplikation with m
    matrix-multiplication:
    row * column
    """
    for row in W:
        #print row
        expression = ''
        for par in range(0, row.size):
            if(row[par] != 0):
                if(row[par] < 0):
                    sign = ' - '
                else:
                    sign = ' + '

                nr = int(np.abs(row[par]))
                if(nr == 1):
                    nr = ''
                expression += '{0}{1}m{2}'.format(sign, nr, par + 1)
        expression = expression.strip()
        if(expression == ''):
            expression = '0'
        if(expression[0] == '+'):
            expression = expression[1:]
        print(expression)


def print_reg_mat(W):
    Wsquared = W.T.dot(W)
    print W
    print Wsquared
    print_visual_mul(Wsquared)
