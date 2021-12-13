if __name__ == '__main__':
    list = [1, 6.2, 3, None,9,2,6.1]
    list = sorted(list, key=lambda x: (x is None, x))
    # list.sort()
    print(list)
