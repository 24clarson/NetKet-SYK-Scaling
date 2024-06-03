from netket.operator._abstract_observable import AbstractObservable

# Import Renyi2EntanglementEntropy without the Hilbert restrictions
from newrenyi import Renyi2EntanglementEntropy as NewRenyi

# Average bipartite entanglement entropy over many subsystems
entropies = []
for i in range(1, N//2+1):
    print(i)
    lst = []
    for j in range(1000):
        entanglement = NewRenyi(hi, np.random.choice(np.arange(0,N), size=i, replace=False))
        entropy = vstate.expect(entanglement)
        lst.append(entropy.mean)
    entropies.append(np.mean(lst))
print(entropies)

# Plot it
plt.plot(np.arange(1,1+len(entropies)), entropies, label="Entropy")
plt.xlabel("Subsystem Size")
plt.ylabel("Entanglement Entropy")
plt.show()
