def save_list_to_txt(arr, f):
    with open(f, 'w') as file:
        for i in arr:
            file.writelines(str(i) + "\n")


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


def swap(arr, m, n):
    tmp = arr[m]
    arr[m] = arr[n]
    arr[n] = tmp


def save_k_most(k, arr, element, cmp=lambda x, y: x > y):
    if len(arr) < k:
        arr.append(element)
    elif cmp(element, arr[-1]):
        arr[-1] = element
    else:
        return

    cur = -1
    for i in range(-2, -len(arr) - 1, -1):
        if cmp(arr[i], element):
            return
        swap(arr, i, cur)
        cur = i
