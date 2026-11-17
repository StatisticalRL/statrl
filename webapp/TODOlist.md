
1.
Add agents.yaml in each setting, with (partial) instanciations of algos: we keep the ones from the environment:
we keep the ones from the environemtn untouched, and instanciate the others.
For instance,UCB has parameters nbArms and delta(default is None). 
We can  set delta=None or to some value, and keep UCB(nbArms) constructor.
TS or KLUCB require kullback param, we can let KLUCB with bernoulli for instance.
We should know which come from the the environment though:
 For instance, explicit mapping: {"nbarms": env.nA} given somewhere, so that 
we known the mapping.

2. Add visualization of env, and of score.

