import random

# Metodo che ritorna un valore casule da una distribuzione di probabilita' discreta (item_prob_dist)
def select_from_dist(item_prob_dist):
    ranreal = random.random()
    for (it, prob) in item_prob_dist.items():
        if ranreal < prob:
            return it
        else:
            ranreal -= prob
    raise RuntimeError(f"{item_prob_dist} is not a probability distribution")
