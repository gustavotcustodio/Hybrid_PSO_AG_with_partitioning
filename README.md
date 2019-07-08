# Comparison of hybrid approaches combining Particle Swarm Optimizators and Genetic Algorithms

This project compares three different proposals of PSOs: the classic PSO and two variations combining GAs: HGAPSO and LOGAPSO. The latter is a novel technique developed by me.

## Classic PSO

This approach has a group of particles moving through a search-space of candidate solutions by updating their velocities and positions in each iteration. 

Kennedy, James. "Particle swarm optimization." Encyclopedia of machine learning (2010): 760-766.

## HGAPSO

This hybrid technique combines PSO and GA concepts to create new candidate by crossover and mutations while applying PSO operators to increase the diversity of solutions.

Kao, Yi-Tung, and Erwie Zahara. "A hybrid genetic algorithm and particle swarm optimization for multimodal functions." Applied soft computing 8.2 (2008): 849-857.

## LOGAPSO

The proposed approach is a modified version of the PSO algorithm. It has the objective of using the PSO to explore potential solutions on the search-space and then using the GA to refine solutions.  

This method has a group of particles updating their solutions similarly to a classic PSO. However, every time a solution is updated, a procedure similar to Local Search has a chance of being executed.  In this procedure, a GA tries to find a direction in the search-space that improves the fitness of the current solution.  This combination of techniques was named Local GA incremented PSO (LOGAPSO).

The algorithm is detailed in the figure below:

![](flow_chart_proposal.pdf)

Below is provided a description of each variable:
* _x_ <sub>_i_</sub> : position of particle _i_.
* _v_ <sub>_i_</sub> : velocity of particle _i_.
* _p_ <sub>_i_</sub> : best solution found so far by particle _i_.
* _g_: best solution found by all particles.
