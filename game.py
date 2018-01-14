from hups_utilities import *
import random
class Game:
    players = []
    hands = []
    startingHands = []
    output = True
    handDict = {}
    
    bigBlind = 10
    smallBlind = bigBlind / 2
    button = 1
    pot = 0
    currentBet = 0
    deck = []
    community = []
    
    handType = {1: 'Straight Flush', 2: 'Quads', 3: 'Full House', 4: 'Flush', 5: 'Straight', 
                6: 'Trips', 7: 'Two Pair', 8: 'Pair', 9: 'High Card'}
    
    handsPlayed = 0.0
    

    def __init__(self, players):
        self.players = players
        for x in range(len(players)):
            players[x].playerNumber = x
        self.deck = Deck()
        self.testHands()
        self.resetStatisticalData()
        
    def printChips(self):
        for n in range(len(self.players)):
            print("Player " + str(n+1) + ": " + str(self.players[n].chips) + " chips.")
    
    
    def testHands(self):
        deck = Deck()
        startingHands = []
        hands = []
        expectedResults = []
        
        testResults = 0
        
        #1.  Testing a straight flush with a higher straight and higher flush on board
        hands.append(deck.generateCards("askcqsjsts9s8s"))
        expectedResults.append(['Straight Flush', deck.generateCards("qsjsts9s8s")])
        
        #2.  Testing Quads with high card being part of a pair
        hands.append(deck.generateCards("asahadackskhqs"))
        expectedResults.append(['Quads', deck.generateCards("asahadacks")])
        
        #3.  Testing Full House with pair as part of another set
        hands.append(deck.generateCards("asahadkskhkdqs"))
        expectedResults.append(['Full House', deck.generateCards("asahadkskh")])
        
        #4.  Testing a 6 Flush with a straight to make sure that a straight flush isn't returned
        hands.append(deck.generateCards("qsjsth9s8s5s2s"))
        expectedResults.append(['Flush', deck.generateCards("qsjs9s8s5s")])
        
        #5.  Testing a 6 Straight with a pair and a 4 flush
        hands.append(deck.generateCards("jdtd9c9s8h7d6d"))
        expectedResults.append(['Straight', deck.generateCards("jdtd9c8h7d")])
        
        #6.  Testing trips with a 4 straight and 4 flush
        hands.append(deck.generateCards("jststhtd9s8s5d"))
        expectedResults.append(['Trips', deck.generateCards("tsthtdjs9s")])
        
        #7.  Testing two pair with high card sandwiched between them
        hands.append(deck.generateCards("qsqhjststd8s5h"))
        expectedResults.append(['Two Pair', deck.generateCards("qsqhtstdjs")])
        
        #8.  Testing one pair with two high cards below it
        hands.append(deck.generateCards("js9s9h8d7d6d4h"))
        expectedResults.append(['Pair', deck.generateCards("9s9hjs8d7d")])
        
        #9.  Testing only high cards with 4 straight and 4 flush
        hands.append(deck.generateCards("ksqsjsts8h7h6h"))
        expectedResults.append(['High Card', deck.generateCards("ksqsjsts8h")])
        
        #10. Testing a straight flush using the final card
        hands.append(deck.generateCards("asks9h8h7h6h5h"))
        expectedResults.append(['Straight Flush', deck.generateCards("9h8h7h6h5h")])
        
        #11. Testing quads using the final card
        hands.append(deck.generateCards("asksqs2s2h2d2c"))
        expectedResults.append(['Quads', deck.generateCards("2s2h2d2cas")])
        
        #12.Testing straight using the final card as a wheel
        hands.append(deck.generateCards("asksqs5h4h3h2h"))
        expectedResults.append(['Straight', deck.generateCards("5h4h3h2has")])
        
        #13. Testing straightflush with a wheel
        hands.append(deck.generateCards("ahksqs5h4h3h2h"))
        expectedResults.append(['Straight Flush', deck.generateCards("5h4h3h2hah")])
        
        #14. Testing wheel straight with a pair inside it
        hands.append(deck.generateCards("ah5s5c4c3s2h2s"))
        expectedResults.append(['Straight', deck.generateCards("5s4c3s2hah")])
        
        #15. Testing to make sure a 4-king wrap around doesn't register a straight
        hands.append(deck.generateCards("ahksthtc4c3h2s"))
        expectedResults.append(['Pair', deck.generateCards("thtcahks4c")])
        
        for x in range(len(hands)):
            #self.printOutput("Test Hand: " + str(x+1))
            hand = self.determineHand(hands[x])
            #self.printOutput([str(card) for card in hand[1]])
            #self.printOutput([str(card) for card in expectedResults[x][1]])
            result = (self.handType[hand[0]] == expectedResults[x][0] and hand[1] == expectedResults[x][1])
            if result:
                testResults += testResults
            else:
                print "Error on test: " + str(x+1)
                print "We started with: " + str([str(hands[x][y]) for y in range(len(hands[x]))])
                print "Expected a " + expectedResults[x][0] + " with: " + str([str(expectedResults[x][1][y]) for y in range(len(expectedResults[x][1]))])
                print "We got a " + self.handType[hand[0]] + " with: " + str([str(hand[1][y]) for y in range(len(hand[1]))])
                
    def printOutput(self, output):
        self.handDict['Text'] += str(output) + "\n"
        if self.output :
            print output
    
    def sortHands(self):
        self.printOutput("\nSorting Hands")
        for x in range(len(self.players)):
            self.hands[x] = self.hands[x] + self.community
            self.hands[x] = sorted(self.hands[x], reverse=True)

    def countSuits(self, hand):
        flushCount = [0] * 4
        flushSuit = ""
        maxSuit = 0
        for y in range(len(self.deck.suit)):
            for x in range(len(hand)):
                if hand[x].suit == self.deck.suit[y]:
                    flushCount[y] += 1
            if flushCount[y] > maxSuit:
                maxSuit = flushCount[y]
                flushSuit = self.deck.suit[y]
        return [flushSuit, maxSuit]
    
    def determineHand(self, hand): 
        
        hasPair, hasTrips, hasQuads, hasStraight, hasWheel, hasFlush, hasStraightFlush = [False]*7
        bestHand, flush, high, pairs, trips, quads, straight = [], [], [], [], [], [], []
        
        temp = self.countSuits(hand)
        flushSuit = temp[0]
        flushCount = temp[1]
        delta = [None] * (len(hand))
        pairCount = 1
        straightCount = 0
        
        if flushCount >= 5:
            hasFlush = True
            for card in hand:
                if card.suit == flushSuit:
                    flush.append(card)
        
        #Adding a buffer to the end of the hand so I can check card rank differences and for the wheel
        hand.append(hand[0])
        
        for x in range(len(hand)-1):
            delta[x]= hand[x].value - hand[x+1].value
            if delta[x] == 0:
                pairCount+=1
            else:
                if pairCount > 1:
                    if pairCount == 2: 
                        pairs.extend(hand[x-1: x+1])
                        hasPair = True
                    if pairCount == 3: 
                        trips.extend(hand[x-2: x+1])
                        hasTrips = True
                    if pairCount == 4: 
                        quads.extend(hand[x-3:x+1])
                        hasQuads = True
                    pairCount = 1
                if (delta[x] % 13) == 1:
                    straightCount += 1
                    if straightCount == 4: 
                        straightEnd = hand[x+1]
                        hasStraight = True 
                        if straightEnd.value == 14: 
                            hasWheel == True
                else: straightCount = 0   
                    
        #self.printOutput(delta[:-1])
        if hasFlush and hasStraight:
            flush.append(flush[0])
            straightFlushCount = 1
            for x in range(len(flush)-1):
                if (flush[x].value - flush[x+1].value) % 13 == 1:
                    straightFlushCount += 1
                    if straightFlushCount == 5:
                        bestHand = flush[x-3:x+2]
                        hasStraightFlush = True
                else:
                    straightFlushCount = 1
                #if x == len(flush)-2:

        if hasStraightFlush: bestHand = [1, bestHand]
        elif hasQuads: bestHand = [2, quads + sorted(list((set(hand) - set(quads))), reverse=True)[:1]]
        elif hasTrips and hasPair: bestHand = [3, trips[:3] + pairs[:2]]
        elif len(trips) > 3: bestHand = [3, trips[:5]]
        elif hasFlush: bestHand = [4, flush[:5]]
        elif hasStraight:
            bestHand = []
            endValue = straightEnd.value
            if endValue == 14: 
                hasWheel = True
                endValue = 1
            for x in range(4, -1, -1):
                for card in hand:
                    if card.value == endValue + x:
                        bestHand.append(card)
                        break
            if hasWheel: bestHand.append(hand[0])
            bestHand = [5, bestHand]
        elif hasTrips: bestHand = [6, trips[:3] + sorted(list((set(hand) - set(trips))), reverse=True)[:2]]
        elif hasPair and len(pairs) > 2: bestHand = [7, pairs[:4] + sorted(list((set(hand) - set(pairs))), reverse=True)[:1]]
        elif hasPair: bestHand = [8, pairs[:2] + sorted(list((set(hand) - set(pairs))), reverse=True)[:3]]
        else:
            bestHand = [9, hand[:5]]
        return bestHand
    
    def postBlinds(self):
        self.players[self.button].chips -= self.smallBlind
        self.players[(self.button + 1)%2].chips -= self.bigBlind
        self.pot = self.bigBlind + self.smallBlind
        self.currentBet = self.bigBlind - self.smallBlind
        
        self.printOutput("Player " + str(self.button) + " posts " + str(self.smallBlind) + " chips")
        self.printOutput("Player " + str((self.button + 1) % 2) + " posts " + str(self.bigBlind) + " chips")
        self.printOutput("Pot: " + str(self.pot))
    

    def playerWinsChips(self, player):
        self.printOutput("\nPlayer " + str(player) + " Wins " + str(self.pot) + " chips\n")
        self.players[player].chips += self.pot
        self.pot = 0
        
        for i in range(len(self.players)):
            self.printOutput("Player " + str(i) + ": " + str(self.players[i].chips) + " chips")
    
    def endHand(self, winningPlayer):                
        for x in self.players:
            winner = 0
            if winningPlayer == x:
                winner = 1
            elif winningPlayer == .5:
                winner = .5
            x.handOver(self.handDict, winner)

    def playerWins(self, winningPlayer):
        self.playerWinsChips(winningPlayer)
        self.handDict['Winner'] = winningPlayer
        self.endHand(winningPlayer)
        #The preflop starting matrix stores unsuited hands below the diagonal for readability
        
    def draw(self):
        self.handDict['Winner'] = .5
        self.printOutput("It's a Draw!")
        for x in range(len(self.players)):
            self.players[x].chips += self.pot/2
        self.pot = 0
        self.endHand(.5)
    
    def clearHandData(self):
        self.hands = []
        self.startingHands = []
        self.community = []
        
    def resetStatisticalData(self):
        self.handsPlayed = 0
        self.preflopHandsWon = []
        self.preflopHandsPlayed = []
        for x in range(13):
            self.preflopHandsWon.append([])
            self.preflopHandsPlayed.append([])
            for y in range(13):
                self.preflopHandsWon[x].append(1.0 + (random.random()/100))
                self.preflopHandsPlayed[x].append(2.0)

        
    def newHandCleanup(self):
        self.handDict = {'HandNumber': self.handsPlayed, 'Text' : '', 'Pot': 0, 'States': [], 'Player Bets': {0: [], 1: []},
        'RevealedCards': {0: [], 1: []}, 'CommunityCards': [], 'Betting': [], 'Bets': 1, 'Winner': None}
        self.printOutput("Hand: " + str(self.handsPlayed) + "\n")
        self.button = (self.button + 1) % 2
        self.betAmount = 0
        self.pot = 0
        self.postBlinds()
        self.clearHandData()
        self.deck.shuffle()
        self.handsPlayed += 1
    
        
    def nextHand(self):
        self.newHandCleanup()
        for x in range(len(self.players)):
            cards = self.deck.deal(2)
            cards = sorted(cards, reverse= True)
            self.hands.append(cards)
            self.players[x].hand = cards
            cards = sorted(cards, reverse= (cards[0].suit == cards[1].suit))
            self.startingHands.append(cards)

        self.printOutput("")
            
        for x in range(len(self.hands)):
            self.printOutput('Player ' + str(x) + ": " + str([str(self.hands[x][0]), str(self.hands[x][1])]))
        
        self.betOpportunity(self.button, None)
        if self.handDict['Winner'] != None:
            return
        #if 'Fold' in self.handDict['Betting']: return
        self.printOutput("Pot: " + str(self.pot))

        self.community = self.deck.deal(3)
        #self.printOutput("Flop: ")
        #self.printOutput([str(self.community[x]) for x in range(3)])
        
        self.community = self.community + self.deck.deal(1)
        #self.printOutput("Turn: ")
        #self.printOutput([str(self.community[x]) for x in range(4)])
        
        self.community = self.community + self.deck.deal(1)
        self.printOutput('\nRiver: ')
        self.printOutput([str(self.community[x]) for x in range(5)])
        
        '''display(community[0].image, community[1].image)'''
        self.sortHands()
    
        for x in range(len(self.hands)):
            self.hands[x] = self.determineHand(self.hands[x])
            self.printOutput("Player " + str(x) + " has " + self.handType[self.hands[x][0]] + ": " + str([str(card) for card in self.hands[x][1]]))

            
        if self.hands[0][0] < self.hands[1][0]:
            self.playerWins(0)
        elif self.hands[0][0] > self.hands[1][0]:
            self.playerWins(1)
        else:
            for x in range(len(self.hands[0][1])):
                if self.hands[0][1][x] > self.hands[1][1][x]:
                    self.playerWins(0)
                    break;
                elif self.hands[0][1][x] < self.hands[1][1][x]:
                    self.playerWins(1)
                    break;
                elif x == len(self.hands[0][1]) - 1:
                    self.draw()
        
        self.printOutput("----------------------------------------------------------------\n")
        
    #Recursive function that keeps calling itself until both players have a chance to bet and one of them chooses not to
    def storeState(self, playerNumber, action):
        self.handDict['Player Bets'][playerNumber].append([self.handDict['Bets'], action])

    def betOpportunity(self, playerNumber, previousAction = None):
        action = self.players[playerNumber].decide(previousAction, self.handDict['Bets'], self.currentBet)
        self.handDict['States'].append({'Player': playerNumber, })  
        self.handDict['Betting'].append(action)
        otherPlayerNumber = (playerNumber+1)%2
        self.storeState(playerNumber, action)
        if action == "Fold":
            self.printOutput("Player " + str(playerNumber) + " Folds")
            self.playerWins(otherPlayerNumber) 
        elif action == "Raise" or action == 'Bet':
            self.handDict['Bets'] += 1
            self.printOutput("Player " + str(playerNumber) + " " + action + "s " + str(self.pot + self.currentBet) + " chips")
            self.players[playerNumber].chips -= self.currentBet
            self.pot += self.currentBet
            
            self.currentBet = self.pot
            self.pot += self.currentBet
            self.players[playerNumber].chips -= self.currentBet
            
            self.betOpportunity(otherPlayerNumber, action)
        elif action == "Check":
            self.printOutput("Player " + str(playerNumber) + " Checks")
            if previousAction == None:
                self.betOpportunity(otherPlayerNumber, action)
        elif action == "Call":
            self.printOutput("Player " + str(playerNumber) + " Calls " + str(self.currentBet) + " chips")
            self.players[playerNumber].chips -= self.currentBet
            self.pot += self.currentBet
            self.currentBet = 0
            if previousAction == None:
                self.betOpportunity(otherPlayerNumber, action)
                
