import sys
sys.path.append('../hups')
import unittest
from hups_utilities import *
from game import *
from hups import *

class basic_tests(unittest.TestCase):
    deck = Deck()
    game = Game([Player(True, 1), Player(True,1)])
    def test_001(self):
        hand = self.game.determineHand(self.deck.generateCards("askcqsjsts9s8s"))
        expected = ['Straight Flush', self.deck.generateCards("qsjsts9s8s")]
        result = self.game.handType[hand[0]] == expected[0] and hand[1] == expected[1]
        self.assertTrue(result)
        
    def test_002(self):
        hand = self.game.determineHand(self.deck.generateCards("askcqsjsts9s8s"))
        expected = ['Straight Flush', self.deck.generateCards("qsjsts9s8s")]
        result = self.game.handType[hand[0]] == expected[0] and hand[1] == expected[1]
        self.assertTrue(result)

    def test_003(self):        
        hand = self.game.determineHand(self.deck.generateCards("asahadackskhqs"))
        expected = ['Quads', self.deck.generateCards("asahadacks")]
        result = self.game.handType[hand[0]] == expected[0] and hand[1] == expected[1]
        self.assertTrue(result)

    def test_004(self):        
        hand = self.game.determineHand(self.deck.generateCards("asahadkskhkdqs"))
        expected = ['Full House', self.deck.generateCards("asahadkskh")]
        result = self.game.handType[hand[0]] == expected[0] and hand[1] == expected[1]
        self.assertTrue(result)

    def test_005(self):        
        hand = self.game.determineHand(self.deck.generateCards("qsjsth9s8s5s2s"))
        expected = ['Flush', self.deck.generateCards("qsjs9s8s5s")]
        result = self.game.handType[hand[0]] == expected[0] and hand[1] == expected[1]
        self.assertTrue(result)

    def test_006(self):        
        hand = self.game.determineHand(self.deck.generateCards("jdtd9c9s8h7d6d"))
        expected = ['Straight', self.deck.generateCards("jdtd9c8h7d")]
        result = self.game.handType[hand[0]] == expected[0] and hand[1] == expected[1]
        self.assertTrue(result)

    def test_007(self):        
        hand = self.game.determineHand(self.deck.generateCards("jststhtd9s8s5d"))
        expected = ['Trips', self.deck.generateCards("tsthtdjs9s")]
        result = self.game.handType[hand[0]] == expected[0] and hand[1] == expected[1]
        self.assertTrue(result)

    def test_008(self):        
        hand = self.game.determineHand(self.deck.generateCards("qsqhjststd8s5h"))
        expected = ['Two Pair', self.deck.generateCards("qsqhtstdjs")]
        result = self.game.handType[hand[0]] == expected[0] and hand[1] == expected[1]
        self.assertTrue(result)

    def test_009(self):        
        hand = self.game.determineHand(self.deck.generateCards("js9s9h8d7d6d4h"))
        expected = ['Pair', self.deck.generateCards("9s9hjs8d7d")]
        result = self.game.handType[hand[0]] == expected[0] and hand[1] == expected[1]
        self.assertTrue(result)

    def test_010(self):        
        hand = self.game.determineHand(self.deck.generateCards("ksqsjsts8h7h6h"))
        expected = ['High Card', self.deck.generateCards("ksqsjsts8h")]
        result = self.game.handType[hand[0]] == expected[0] and hand[1] == expected[1]
        self.assertTrue(result)

    def test_011(self):        
        hand = self.game.determineHand(self.deck.generateCards("asks9h8h7h6h5h"))
        expected = ['Straight Flush', self.deck.generateCards("9h8h7h6h5h")]
        result = self.game.handType[hand[0]] == expected[0] and hand[1] == expected[1]
        self.assertTrue(result)

    def test_012(self):        
        hand = self.game.determineHand(self.deck.generateCards("asksqs2s2h2d2c"))
        expected = ['Quads', self.deck.generateCards("2s2h2d2cas")]
        result = self.game.handType[hand[0]] == expected[0] and hand[1] == expected[1]
        self.assertTrue(result)

    def test_013(self):        
        hand = self.game.determineHand(self.deck.generateCards("asksqs5h4h3h2h"))
        expected = ['Straight', self.deck.generateCards("5h4h3h2has")]
        result = self.game.handType[hand[0]] == expected[0] and hand[1] == expected[1]
        self.assertTrue(result)

    def test_014(self):        
        hand = self.game.determineHand(self.deck.generateCards("ahksqs5h4h3h2h"))
        expected = ['Straight Flush', self.deck.generateCards("5h4h3h2hah")]
        result = self.game.handType[hand[0]] == expected[0] and hand[1] == expected[1]
        self.assertTrue(result)

    def test_015(self):        
        hand = self.game.determineHand(self.deck.generateCards("ah5s5c4c3s2h2s"))
        expected = ['Straight', self.deck.generateCards("5s4c3s2hah")]
        result = self.game.handType[hand[0]] == expected[0] and hand[1] == expected[1]
        self.assertTrue(result)

    def test_016(self):        
        hand = self.game.determineHand(self.deck.generateCards("ahksthtc4c3h2s"))
        expected = ['Pair', self.deck.generateCards("thtcahks4c")]
        result = self.game.handType[hand[0]] == expected[0] and hand[1] == expected[1]
        self.assertTrue(result)

if __name__ == '__main__':
    unittest.main()