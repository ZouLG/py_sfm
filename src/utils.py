

def set_axis_limit(ax, low, high, zlow=-10, zhigh=10):
    ax.set_xlim([low, high])
    ax.set_ylim([low, high])
    ax.set_zlim([zlow, zhigh])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")


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


def save_points_to_ply(pw):
    pass


def plot_points(pcd):
    pass