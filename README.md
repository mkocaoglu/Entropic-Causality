# Entropic-Causality
Test causal direction using a lower bound on the entropy of the exogenous variable in the causal model. The model with smaller total input entropy is chosen as the true model. 
Requires numpy, pandas, sklearn packages.

INPUT: 
For a text file with two columns, with no header, first column is X and second column is Y. Algorithm outputs either X->Y or Y->X with a score that indicates how confident it is in its decision.

pair0001.txt is taken from CauseEffectPairs repository at https://webdav.tuebingen.mpg.de/cause-effect/

HOW TO RUN:
To test on this cause effect pair, download entropicCausalPair.py and pair0001.txt into the same folder and run 

	python entropicCausalPair.py pair0001.txt

To test it on every (scalar) causal pair in the Tuebingen dataset, download every pair from https://webdav.tuebingen.mpg.de/cause-effect/ into the same folder and run

	python Tuebingen_loop.py

You can either call the function on an arbitrary file input.txt by importing the code as follows from your script

	import entropicCausalPair
	entropicCausalPair.main("input.txt")

or by simply running 

	python entropicCausalPair.py input.txt

from the terminal.