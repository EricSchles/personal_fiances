import numpy as np
import dask
import json
from numba import jit
    
@jit
def equalizing_card_one_two(card_one, card_two):
    return 1.189*card_one >= 1.129*card_two

@jit
def find_optimal_card_two(card_one, card_two, budget):
    for _ in np.arange(0.01, budget, 0.01):
        card_two -= 0.01
        if equalizing_card_one_two(card_one, card_two):
            return [card_one, card_two]

@jit
def equation(card_one, card_two):
    return 1.189*card_one + 1.129*card_two

@jit
def budget_above_zero(orig_card_one, orig_card_two, card_one_two, orig_budget):
    card_one_diff = orig_card_one - card_one_two[0]
    card_two_diff = orig_card_one - card_one_two[1]
    if (card_one_diff + card_two_diff) > orig_budget:
        return False
    return True

@jit
def update_budget(budget, card_two, card_one_two):
    card_two_diff = card_two - card_one_two[1]
    budget -= card_two_diff
    return budget

@jit
def update_card_one_two(card_one_two, card_one):
    card_two = card_one_two[1]
    card_one -= 0.01
    return card_two, card_one

@jit
def possible_optimal_values(card_one, card_two, orig_card_one,
                            orig_card_two, budget,
                            orig_budget, max_guesses):
    values = []
    for _ in range(max_guesses):
        card_one_two = find_optimal_card_two(
            card_one, card_two, budget
        )
        if not card_one_two:
            break
        have_enough_money = budget_above_zero(
            orig_card_one,
            orig_card_two,
            card_one_two,
            orig_budget
        )
        if not have_enough_money:
            break
        budget = update_budget(
            budget,
            card_two,
            card_one_two
        )
        values.append(card_one_two)
        card_two, card_one = update_card_one_two(
            card_one_two, card_one
        )
        if budget == 0 or card_two == 0 or card_one == 0:
            break
    return values

def get_all_optimal_card_one_two(values):
    results = []
    for value in values:
        card_one, card_two = value[0], value[1]
        result = dask.delayed(equation)(card_one, card_two)
        results.append(result)
    return dask.compute(*results)

@jit
def get_min_card_one_two(values, results):
    values_index = results.index(min(results))
    return values[values_index]

def main():
    data = json.load(
        open("solve_min_interest.json","r")
    )
    card_one = data["card_one"][0]
    card_two = data["card_two"][0]
    orig_card_one = card_one
    orig_card_two = card_two
    budget = 7884.78
    orig_budget = budget

    max_guesses = len(list(np.arange(0.01, budget, 0.01)))
    values = possible_optimal_values(
        card_one, card_two, orig_card_one,
        orig_card_two, budget,
        orig_budget, max_guesses
    )
    results = get_all_optimal_card_one_two(values)
    
    values = get_min_card_one_two(
        values, results
    )
    new_card_one = values[0]
    new_card_two = values[1]
    print("pay", card_one - new_card_one,"of off card_one")
    print("pay", card_two - new_card_two, "of off card_two")

if __name__ == '__main__':
    main()
