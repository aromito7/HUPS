import PIL.Image
from random import shuffle
import numpy as np
from collections import OrderedDict

class Card:
    rank = ''
    suit = ''
    value = ''
    image = None
    
    
    def __init__(self, value, suit, image):
        self.value = value
        if value == 14:
            self.rank = 'A'
        elif value == 13:
            self.rank = 'K'
        elif value == 12:
            self.rank = 'Q'
        elif value == 11:
            self.rank = 'J'
        elif value == 10:
            self.rank = 'T'
        else: self.rank = str(value)
            
        self.suit = suit
        self.image = image

    def __str__(self):
        return str(self.rank+self.suit[0])
    
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return (self.value == other.value and self.suit == other.suit)
        else:
            return False
    
    def __lt__(self, other):
        return self.value < other.value
    
    def __hash__(self):
        return 1
    
class Deck:
    cards = []
    value = []
    rank = []
    suit = []
    image = []
    count = 0
    
    def __init__(self):
        self.value = range(1, 14)
        self.rank = ["A", "K", "Q", "J", "T"]
        self.rank.extend([str(x) for x in range(9, 1, -1)])
        self.value = [(i+11) % 13 + 2 for i in self.value]
        self.suit = ['clubs', 'spades', 'hearts', 'diamonds']
        
        im = PIL.Image.open('cards.png')
        w,h = im.size
        imc = im.crop((12*w/13,3*h/4,w,h))
        
        self.image = [im.crop((x*w/13, y*h/4, (x+1)*w/13, (y+1)*h/4)) for x in range(13) for y in range(4)]
        
        self.cards = [Card(self.value[x], self.suit[y], im.crop((x*w/13, y*h/4, (x+1)*w/13, (y+1)*h/4))) 
                      for x in range(13) for y in range(4)]
        
        for x in range(len(self.cards)):
            if self.cards[x].value==1:
                self.cards[x].value=14
        
        if len(self.cards) < 52:
            print "Deck creation error"
        
                
    def __str__(self):
        return ' '.join([str(x) for x in self.cards])
    
    def shuffle(self):
        shuffle(self.cards)
        self.count = 0
        
    def deal(self, number):
        deal = []
        for x in range(number):
            deal = deal + [self.cards[self.count]]
            self.count+=1
        return deal
    
    def getCards(self):
        return self.cards

    def generateCards(self, cardString):
        cards = []
        convertValue = {'a': 14, 'k': 13, 'q': 12, 'j': 11, 't': 10, 
                '9': 9, '8': 8, '7': 7, '6': 6, '5': 5, '4': 4, '3': 3, '2': 2}
        convertSuit = {'s': 'spades', 'h': 'hearts', 'd': 'diamonds', 'c': 'clubs'}

        if len(cardString)%2 == 1: print "Bad Card Input"
        else:
            for x in range(len(cardString)/2):
                cards.append(Card(convertValue[cardString[2*x]], convertSuit[cardString[2*x+1]], []))
        
        return cards
    
    def test(self):
        
        pocket = 0.0
        suited = 0.0
        hands = 100000
        for i in range(hands):
            self.shuffle()
            hand = self.deal(2)
            if hand[0].value == hand[1].value: pocket+=1
            if hand[0].suit == hand[1].suit: suited+=1

        print pocket/hands
        print suited/hands
        
    def test2(self):
        print deck.cards[0] == deck.cards[0]
        print deck.cards[0] == deck.cards[4]
        print deck.cards[0] == deck.cards[1]

def createPreflopOrderedRange(temp):
    startingHands = OrderedDict()
    for y in range(13):
        for x in range(13):
            cards = ['A', 'K', 'Q', 'J', 'T'] + [str(i) for i in range(9,1,-1)]
            y1 = cards[y]
            x1 = cards[x]

            if x > y:
                n1 = y1
                n2 = x1
                n3 = 's'
            else:
                n1 = x1
                n2 = y1
                n3 = 'o'

            startingHands[n1 + n2 + n3] = float(str(temp[x][y]))
    return OrderedDict(sorted(startingHands.items(), key=lambda t: -t[1]))