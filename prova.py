if __name__ == '__main__':
    list = [1, 6.2, 3, None,9,2,6.1]
    list = sorted(list, key=lambda x: (x is None, x))
    # list.sort()
    print(list)
    a = 17.7
    print(a // 1)
    print (1+5.0)

    nodes = 5
    for i in range(0,10):
        for j in range(0, nodes):
            print("ciao")
        print("\n")
        nodes = 2 +i