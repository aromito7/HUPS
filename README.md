
Heads Up Poker Simulator

Use python2.7 to run hups/hups.py

![poker](https://user-images.githubusercontent.com/5891073/35404558-6495ae4a-01d1-11e8-8b47-b35f24e31fb2.jpg)

My goal is to create an AI that uses a neural network and reinforcement learning to play heads up poker.  Currently it only simulates the preflop round of betting and then deals out and evaluates the hand if there isn't a fold.

What the program currently does is it allows you to play against my control AI as I'm still working on the learning algorithm for the other AI.

The Control:
My control AI that I'm using for training utilizes an ordered dictionary of starting hands and their respective win percentages against a random hand in Texas Holdem.  The control AI folds a certain percentage of hands at the bottom of it's rangee, calls with a middle percentage, and re-raises with the top percentage of their range.  

The Reinforcement Learning:
The other AI being trained must learn how often certain hands win, how it's opponent is likely to react, and what it's optimal decision should be.  It currently only makes decisions based on the number of raises that have occurred so far and the rank of each card in it's hand.  This results in over 1,000 different states which would require millions of training hands being played to have a decent sample size of results for each state.  By applying a neural network we can get the optimal decision process to converge much faster.

What I've learned so far:
My control AI uses a reasonably good algorithm for heads up poker.  However my current simulation only works for preflop betting so I had hoped that my learning AI would be able to exploit a huge difference in play strategy.  It currently raises about 85% of the time because there's only one round of betting so playing with worse cards doesn't punish you as it would in a normal game of poker.  On the other hand, there's so much more variance that folding any significant percentage of the time is much more punishing, and my AI was quick to catch onto that.
