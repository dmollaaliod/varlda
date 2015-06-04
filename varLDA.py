"""Implementation of mean-field variational LDA.
By Diego Molla (dmollaaliod@gmail.com)

Based on the blogpost ...

Created: 28 May 2015
"""
import numpy as np
from scipy.special import psi

def generate_docs(words, phi, docs, nwords):
    "Generate documents randomly"
    dataset = []
    for d in docs:
        doc = []
        for i in range(0,nwords):
            topic = np.random.choice((0,1),p=d)
            doc.append(np.random.choice(range(len(words)),p=phi[topic])) # sample from the
                                                             # topic
        dataset.append(doc)
    return dataset

def varlda(w,K,V,alpha,beta,iterations=10):
    "Perform variational LDA"
    M = len(w)
    
    # Initialisation
    pz = []
    for m in w:
        pzm = []
        for n in m:
            pzm.append([1/K]*K)
        pz.append(pzm)
    #print("After initialisation: Pz=")
    #print(pz)

    # Iterations
    for iter in range(iterations):
        
        # Update new beta
        newbeta = beta*np.ones((K,V))
        for k in range(K):
            for m in range(M):
                for n in range(len(w[m])):
                    v = w[m][n]
                    newbeta[k,v] += pz[m][n][k]
                    
        # Update new alpha
        newalpha = []
        for m in range(M):
            row = []
            for k in range(K):
                row.append(alpha+sum([pz[m][n][k] for n in range(len(w[m]))]))
            newalpha.append(row)

        # Update new pz
        sumbeta = newbeta.sum(0)
        for m in range(M):
            sumalpha = sum([newalpha[m][k] for k in range(K)])
            for n in range(len(w[m])):
                for k in range(K):
                    v = w[m][n]
                    pz[m][n][k] = np.exp(psi(newalpha[m][k])-psi(sumalpha) +
                                         psi(newbeta[k,v])-psi(sumbeta[v]))
        #print("Estimated Theta Priors (newalpha):")
        #for m in newalpha:
        #    print(m)
        #print()
        #print("Estimated Phi Priors (newbeta):")
        #for k in newbeta:
        #    print(k)


    # Return the results
    newalpha = np.array(newalpha)
    theta = newalpha/newalpha.sum(1,keepdims=True)
    phi = newbeta/newbeta.sum(1,keepdims=True)
    return (theta,phi)
        
if __name__ == "__main__":
    import doctest
    doctest.testmod()

    words = ("river","stream","bank","money","loan") # All words
    phi = ([0,0,1/3,1/3,1/3],    # word distr per topic
           [1/3,1/3,1/3,0,0])
    docs = ((1,0),(1,0),(1,0),(1,0),(1,0),(1,0),
            (0.5,0.5),(0.5,0.5),(0.5,0.5),(0.5,0.5),(0.5,0.5),
            (0,1),(0,1),(0,1),(0,1))       # topic distr per document
    nwords = 16 # words per document
    data = generate_docs(words,phi,docs,nwords)
    print("Generated %i documents" % len(data))
    print()
    (newtheta,newphi) = varlda(data,K=2,V=len(words),alpha=50/2,beta=0.01)
    print("Estimated Theta:")
    for m in newtheta:
        print(m)
    print()
    print("Estimated Phi:")
    for k in newphi:
        print(k)
