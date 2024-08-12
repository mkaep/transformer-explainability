

def binary_generator(lst, start=1):
    for i in range(start, 2 ** len(lst)):
        # cut of '0b' at the start
        binary = bin(i)[2:]
        # make binary string long enough for indexing
        binary = '0' * (len(lst) - len(binary)) + binary
        yield [True if digit == '1' else False for digit in binary]
