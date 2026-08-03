# Classi astratte da implementare in GOFAI1:
# - Agent
# - Environment
# - Simulate

from display import Displayable


class Agent(Displayable):

    def initial_action(self, percept):
        return self.select_action(percept)

    # Ritorna la prossima azione dato il percept attuale
    def select_action(self, percept):
        raise NotImplementedError("select_action")


class Environment(Displayable):

    def initial_percept(self):
        raise NotImplementedError("initial_percept")

    # Effettua l'azione sull'ambiente e ritorna il prossimo percept
    def do(self, action):
        raise NotImplementedError("Environment.do")

# Simula l'interazione tra agente ed ambiente per n steps
class Simulate(Displayable):
    def __init__(self, agent, environment):
        self.agent = agent
        self.env = environment
        self.percept = self.env.initial_percept()
        self.percept_history = [self.percept]
        self.action_history = []

    # n sarà il numero di giorni da simulare, quindi quante volte simulare l'interazione agente-ambiente
    def go(self, n):
        for i in range(n):
            action = self.agent.select_action(self.percept)
            self.action_history.append(action)
            self.display(2, f"i={i} action={action}")
            self.percept = self.env.do(action)
            self.percept_history.append(self.percept)
            self.display(2, f"        percept={self.percept}")
