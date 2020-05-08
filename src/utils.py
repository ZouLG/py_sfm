
def binary_search(g, k):
    start = 0
    end = len(g) - 1
    while start <= end:
        mid = start + (end - start) // 2
        if g[mid] == k:
            return mid
        elif g[mid] < k:
            start = mid + 1
        else:
            end = mid - 1
    return -1

