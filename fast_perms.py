#!/usr/bin/env python3

import itertools

from copy import deepcopy


def edit_dist_permutations(a: list, max_edit_dist: int):
    def edit_dist_helper(a: list, l: int, r: int, results: list = []):
        if l == r:
            results.append(deepcopy(a))
        else:
            for i in range(l, r):
                a[l], a[i] = a[i], a[l]
                if not (abs(a[i] - i) > max_edit_dist or abs(a[l] - l) > max_edit_dist):
                    edit_dist_helper(a, l+1, r, results)
                a[i], a[l] = a[l], a[i]

    results = []
    edit_dist_helper(a, 0, len(a), results)
    return results

def reduced_edit_dist_permutations(data: list, a: list, max_edit_dist: int):
    def edit_dist_helper(data: list, a: list, l: int, r: int, results: set, orders: set):
        # results.append(deepcopy(a))
        # results.append(deepcopy(data))
        if l == r:
            # results.add(deepcopy(data))
            results.add(str(data))
            orders.add(str(a))
        else:
            for i in range(l, r):
                a[l], a[i] = a[i], a[l]
                data[l], data[i] = data[i], data[l]
                if str(data) not in results and not (abs(a[i] - i) > max_edit_dist or abs(a[l] - l) > max_edit_dist):
                    edit_dist_helper(data, a, l+1, r, results, orders)
                a[l], a[i] = a[i], a[l]
                data[i], data[l] = data[l], data[i]

    results, orders = set(), set()
    edit_dist_helper(data, a, 0, len(a), results, orders)
    orders = [[int(i) for i in s.strip('[]').split(',')] for s in orders]
    return orders


def permute_top(a: list):
    def permute(a: list, l: int, r: int, results: list = []):
        if l == r:
            results.append(deepcopy(a))
        else:
            for i in range(l, r):
                a[l], a[i] = a[i], a[l]
                permute(a, l+1, r, results)
                a[l], a[i] = a[i], a[l]

    results = []
    permute(a, 0, len(a), results)
    return results


if __name__=='__main__':

    # pruned_orders = edit_dist_permutations(list(range(7)), 4)
    # orders = permute_top(list(range(7)))

    # for o in pruned_orders:
    #     print(o)

    # # for o in orders:
    # #     print(o)

    # print(f"eliminated {len(orders) - len(pruned_orders)} orders")

    data = [0, 0, 2]
    orders = reduced_edit_dist_permutations(data, list(range(len(data))), 3)
    print(len(orders))
    print(orders)
