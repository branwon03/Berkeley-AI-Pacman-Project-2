# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        from util import manhattanDistance
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        '''Create a list for distances between food and ghosts. Keep track of
        the value of each state for the agent. Check for the state of each food in the
        list of foods. If food is closer, more value is added. Do the same for the ghosts'''

        food = newFood.asList()
        foodDistance = []
        ghostDistance = []
        value = 0 #Value of state for agent

        for i in food:
            foodDistance.append(manhattanDistance(newPos, i)) #distance from each food

        #calculate values for finding nearby food
        for food in foodDistance:
            if food <= 4:
                value += 1
            elif food > 4 and food <= 15:
                value += 0.2
            else:
                value += 0.15

        for ghost in successorGameState.getGhostPositions():
            ghostDistance.append(manhattanDistance(ghost,newPos)) #distance from every ghost

        #calculate negative values for nearby ghosts
        for ghost in successorGameState.getGhostPositions():
            if ghost == newPos:
                value = 2 - value

            elif manhattanDistance(ghost,newPos) <= 3.5:
                value = 1 - value

        return successorGameState.getScore() + value

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"

        def maxValue(gameState, depth):
            
            Actions = gameState.getLegalActions(0)

            if len(Actions) == 0 or gameState.isWin() or gameState.isLose() or depth == self.depth:             ###The trvial situations(state)
                return(self.evaluationFunction(gameState), None)
            
            w = -(float("inf"))                                                                               ###We are trying to implement the 2 sides of the minimax algorithm the max and the min
            Act = None
            for action in Actions:                                                                          ###In that way that the 2 functions are calling each other is like building the tree(diagrams from tha class)
                sucsValue = minValue(gameState.generateSuccessor(0,action), 1, depth)                          #We have the available moves and we are seeking for the "best" one
                sucsValue = sucsValue[0]                                                                      #It is working exactly as the theory of minimax algorithm commands
                if sucsValue > w:                                                                            #Here we have as start -infinite
                    w, Act = sucsValue,action
            return(w, Act)

        def minValue(gameState, agentID, depth):
            
            Actions=gameState.getLegalActions(agentID)

            if len(Actions) == 0:
                return(self.evaluationFunction(gameState), None)
            
            l = float("inf")                                                                                  ###As we see in contrast with max we begin from +infinte
            Act  = None
            for action in Actions:
                if agentID == gameState.getNumAgents() -1:
                    sucsValue = maxValue(gameState.generateSuccessor(agentID, action), depth + 1)
                else:
                    sucsValue = minValue(gameState.generateSuccessor(agentID, action), agentID + 1, depth)        ###We are doing exactly the opposite from the max "function"
                sucsValue = sucsValue[0]
                if(sucsValue < l):
                    l, Act = sucsValue, action
            return(l, Act)
        
        max = maxValue(gameState, 0)[1]
        return max  
        #util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        actions = gameState.getLegalActions(0)
        maxResult = float('-inf')
        a = float('-inf')
        b = float('inf')
        for action in actions:
            successor = gameState.generateSuccessor(0, action)
            currentResult = self.minValue( successor, 0, 1, a, b )
            if currentResult > maxResult:
                maxResult = currentResult
                maxAction = action
                a = max( (a, currentResult) )
        return maxAction
    
    def maxValue(self, gameState, currDepth, a, b):
        if gameState.isWin() or gameState.isLose() or currDepth == self.depth:
            return self.evaluationFunction(gameState)
        actions = gameState.getLegalActions(0)
        maxValue = float('-inf')
        for action in actions:
            successor = gameState.generateSuccessor(0, action)
            maxValue = max( (maxValue, self.minValue(successor, currDepth, 1, a, b)) )
            if maxValue > b:
                return maxValue
            a = max( (a,maxValue) )
        return maxValue

    def minValue(self, gameState, currDepth, currAgent, a, b):
        if gameState.isWin() or gameState.isLose() or currDepth == self.depth:
            return self.evaluationFunction(gameState)
        actions = gameState.getLegalActions(currAgent)
        minValue = float('inf')
        agents = gameState.getNumAgents()
        for action in actions:
            successor = gameState.generateSuccessor(currAgent, action)
            if currAgent < agents - 1:
                minValue = min( (minValue, self.minValue(successor, currDepth, currAgent + 1, a, b)) )
            else:
                minValue = min( (minValue, self.maxValue(successor, currDepth + 1, a, b)) )
            if minValue < a:
                return minValue
            b = min( (b,minValue) )
        return minValue
        #util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        actions = gameState.getLegalActions(0)
        maxAction = 'Stop'
        maxResult = float('-inf')
        for a in actions:
            successor = gameState.generateSuccessor(0,a)
            currentResult = self.chanceExpect(successor, 0, 1)
            if currentResult > maxResult:
                maxResult = currentResult
                maxAction = a
        return maxAction
    
    def maxExpect(self, gameState, currDepth):
        if gameState.isWin() or gameState.isLose() or currDepth == self.depth:
            return self.evaluationFunction(gameState)
        actions = gameState.getLegalActions(0)
        successors = []
        for a in actions:
            successors.append( gameState.generateSuccessor(0, a) )
        return max( [self.chanceExpect(s, currDepth, 1) for s in successors] )
    def chanceExpect(self, gameState, currDepth, currAgent):
        if gameState.isWin() or gameState.isLose() or currDepth == self.depth:
            return self.evaluationFunction(gameState)
        actions = gameState.getLegalActions(currAgent)
        successors = []
        for a in actions:
            successors.append( gameState.generateSuccessor(currAgent, a) )
        if currAgent < gameState.getNumAgents() - 1:
            return sum( [self.chanceExpect(s, currDepth, currAgent + 1 ) for s in successors ] )/len(successors)
        else:
            return sum( [self.maxExpect(s, currDepth + 1 ) for s in successors ] )/len(successors)
        #util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    "*** YOUR CODE HERE ***"

    '''Create a list for distances between food and ghosts. Keep track of
    the value of each state for the agent. Check for the state of each food in the
    list of foods. If food is closer, more value is added. Do the same for the ghosts'''

    food = newFood.asList()
    foodDistance = []
    ghostDistance = []
    value = 0 #Value of state for agent

    for i in food:
        foodDistance.append(manhattanDistance(newPos, i)) #distance from each food

    #calculate values for finding nearby food
    for food in foodDistance:
        if food <= 4:
            value += 1
        elif food > 4 and food <= 15:
            value += 0.2
        else:
            value += 0.15

    for ghost in currentGameState.getGhostPositions():
        ghostDistance.append(manhattanDistance(ghost,newPos)) #distance from every ghost

    #calculate negative values for nearby ghosts
    for ghost in currentGameState.getGhostPositions():
        if ghost == newPos:
            value = 2 - value

        elif manhattanDistance(ghost,newPos) <= 3.5:
            value = 1 - value

    return currentGameState.getScore() + value
#util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

