import numpy as np
import dask
import json
from numba import jit

@jit
def equalizing_cards(cards):
    card_inequality = []
    for index in range(1, len(cards)):
        card_one = cards[index-1]
        card_two = cards[index]
        interest_amount_one = card_one[0]*card_one[1]
        interest_amount_two = card_two[0]*card_two[1]
        card_inequality.append(
            interest_amount_one >= interest_amount_two
        )
    return all(card_inequality)

@jit
def find_optimal_cards(cards, budget):
    tmp_cards = cards[:]
    for index in range(1, len(cards)): 
        for _ in np.arange(0.01, budget, 0.01):
            tmp_cards[index][1] -= 0.01
            if equalizing_cards(tmp_cards):
                return tmp_cards

@jit
def equation(cards):
    return sum(
        [data[key][0] * data[key][1]
         for key in data
         if "card" in key]
    )

def budget_above_zero(orig_cards,
                      guess, orig_budget):
    card_diffs = 0
    for index, orig_card in enumerate(orig_cards):
        card_diffs += orig_card[0] - guess[index][0]
    if card_diffs > orig_budget:
        return False
    return True

@jit
def update_budget(budget, card_two, guess):
    card_two_diff = card_two[0] - guess[1][0]
    budget -= card_two_diff
    return budget

@jit
def update_card_guess(guess, card_one):
    card_two = guess[1]
    card_one[0] -= 0.01
    return card_two, card_one

@jit
def possible_optimal_values(cards, orig_cards,
                            budget, orig_budget,
                            max_guesses):
    card_one = orig_cards[0]
    card_two = orig_cards[1]
    values = []
    for _ in range(max_guesses):
        guess = find_optimal_cards(
            cards, budget
        )
        if not guess:
            break
        have_enough_money = budget_above_zero(
            orig_cards,
            guess,
            orig_budget
        )
        if not have_enough_money:
            break
        budget = update_budget(
            budget,
            card_two,
            guess
        )
        values.append(guess)
        card_two, card_one = update_card_guess(
            guess, card_one
        )
        if budget == 0 or card_two == 0 or card_one == 0:
            break
    return values

def get_all_optimal_guess(values):
    results = []
    for value in values:
        cards = value
        result = dask.delayed(equation)(cards)
        results.append(result)
    return dask.compute(*results)

@jit
def get_min_guess(values, results):
    values_index = results.index(min(results))
    return values[values_index]

@jit
def adjust_percentages(cards):
    return [(card[1] * 0.01) + 1 for card in cards]
    
def main():
    data = json.load(
        open("solve_min_interest.json","r")
    )
    card_one = data["card_one"]
    card_two = data["card_two"]

    cards = [data[key]
             for key in data
             if "card" in key]
    cards = sorted(cards, key=lambda x: x[1])
    orig_cards = cards[:]
    budget = data["budget"]
    orig_budget = budget

    max_guesses = len(
        list(np.arange(0.01, budget, 0.01))
    )
    values = possible_optimal_values(
        cards, orig_cards,
        budget, orig_budget,
        max_guesses
    )
    results = get_all_optimal_guess(values)
    
    values = get_min_guess(
        values, results
    )
    new_card_one = values[0]
    new_card_two = values[1]
    print("pay", card_one - new_card_one,"of off card_one")
    print("pay", card_two - new_card_two, "of off card_two")

if __name__ == '__main__':
    main()
