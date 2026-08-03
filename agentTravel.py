# GOFAI1
#
# Percept:
#   - 'weather'    : condizione atmosferica del giorno ('sereno', 'rischio_pioggia', 'rischio_neve')
#   - 'fuel_price' : prezzo del carburante (EUR/litro)
#   - 'distance'   : distanza (km, andata) da percorrere per il tragitto
#
# Azione:
#   - 'mezzo' : 'auto' | 'mezzi_pubblici'
#
# Misura di prestazione (performance measure) che l'agente cerca di ottimizzare ogni giorno:
#   costo_totale(mezzo) = w_cost * costo_economico(mezzo) + w_safety * costo_di_sicurezza(mezzo, weather)

import random
from agents import Agent, Environment, Simulate
from utilities import select_from_dist

# Genera giorno per giorno i percept dall'ambiente:
# - le condizioni meteo; 
# - il prezzo del carburante; 
# - la distanza da percorrere per il tragitto. 
class TravelEnv(Environment):
    # Distribuzione di probabilita' delle condizioni meteo giornaliere
    # 'sereno' = 65%, 'rischio_pioggia' = 25%, 'rischio_neve' = 10%
    weather_dist = {'sereno': 0.65, 'rischio_pioggia': 0.25, 'rischio_neve': 0.10}

    # Distanza da percorrere
    # distanza fissa = 15 km; deviazione standard = 2 km
    base_distance = 15.0
    distance_sd = 2.0

    # Prezzo del carburante
    # prezzo fisso = 1.80 EUR/litro; deviazione standard = 0.03 EUR/litro
    base_fuel_price = 1.80
    fuel_price_sd = 0.03

    # Inizializzazione dello stato interno dell'ambiente (giorno 0)
    def __init__(self):
        self.time = 0
        self.fuel_price = self.base_fuel_price
        self.weather_history = []    # memoria delle condizioni meteo
        self.fuel_price_history = []  # memoria del prezzo del carburante
        self.distance_history = []   # memoria delle distanze

    # Ritorna il percept iniziale dall'ambiente (giorno 1)
    def initial_percept(self):
        # Pesca un meteo a caso
        self.weather = select_from_dist(self.weather_dist)
        # Pesca un prezzo del carburante a caso (gaussiano attorno al prezzo base)
        self.distance = round(max(1.0, random.gauss(self.base_distance, self.distance_sd)), 1)
        # Aggiorna le storie
        self.weather_history.append(self.weather)
        self.distance_history.append(self.distance)
        # Consideriamo il prezzo del carburante iniziale come quello fissato (1.80)
        self.fuel_price_history.append(self.fuel_price)
        return {'weather': self.weather,
                'fuel_price': self.fuel_price,
                'distance': self.distance}

    # Fa l'azione dell'agente (cioè "memorizza" il mezzo con cui si è spostato il giorno precedente) e genera il percept del giorno successivo
    # NB: l'azione dell'agente non influenza l'ambiente (meteo e prezzo carburante)
    def do(self, action):
        # ------------ Effetto azione dell'agente (nessuno) --------------

        # Avanza di 1 giorno
        self.time += 1

        # ------------ Creazione nuovo percept --------------

        # Nuove condizioni meteo del giorno successivo (in modo casuale)
        self.weather = select_from_dist(self.weather_dist)
        self.weather_history.append(self.weather)

        # Nuova distanza del giorno successivo (in modo "rumoroso" gaussiano)
        self.distance = round(max(1.0, random.gauss(self.base_distance, self.distance_sd)), 1)
        self.distance_history.append(self.distance)

        # Nuovo prezzo del carburante del giorno successivo (in modo "rumoroso" gaussiano)
        self.fuel_price = round(max(0.5, self.fuel_price + random.gauss(0, self.fuel_price_sd)), 3)
        self.fuel_price_history.append(self.fuel_price)

        return {'weather': self.weather,
                'fuel_price': self.fuel_price,
                'distance': self.distance}


# Agente che, dato il percept giornaliero (meteo, prezzo carburante, distanza), consiglia se effettuare il tragitto in auto o con i mezzi pubblici.
# La decisione deriva dal confronto tra il costo per l'auto e il costo per i mezzi pubblici, somme pesate di:
# - un costo economico (carburante per l'auto, biglietto per i mezzi pubblici)
# - un costo di sicurezza (rischio stradale) dalle condizioni atmosferiche (molto piu' alto per l'auto in caso di pioggia/neve, quasi costante e 
#   basso per i mezzi pubblici).
# -> OBIETTIVO: dato un percept, ricavare i costi nei 2 casi (auto/mezzi) e scegliero quello migliore
class TravelAgent(Agent):
    # Consumo medio dell'autovettura (litri/km)
    fuel_usage_per_km = 0.07
    round_trip_factor = 2 # fattore per calcolare il consumo totale per andata e ritorno

    # Costo di un biglietto per i mezzi pubblici
    public_transport_cost = 5.0

    # Costo di sicurezza associato all'uso dell'auto, in base al meteo:
    # rappresenta il rischio aggiuntivo di incidente/danno dovuto alle condizioni della strada
    # Sereno: 0% di incidente, Rischio pioggia: 6% di incidente, Rischio neve: 25% di incidente
    car_risk_cost = {'sereno': 0.0, 'rischio_pioggia': 6.0, 'rischio_neve': 25.0}

    # Costo di sicurezza per i mezzi pubblici: basso e quasi indipendente dal meteo
    # (piccola componente dovuta al tragitto a piedi verso la fermata/stazione)
    # Sereno: 0.5% di incidente, Rischio pioggia: 1% di incidente, Rischio neve: 1.5% di incidente
    public_transport_risk_cost = {'sereno': 0.5, 'rischio_pioggia': 1.0, 'rischio_neve': 1.5}

    # Inizializzazione dell'agente con i pesi per costo economico (w_cost) e costo di sicurezza (w_safety)
    # Dò più peso ai coti di sicurezza, potrei rischiare di farmi male
    def __init__(self, w_cost=1.0, w_safety=5.0):
        self.w_cost = w_cost
        self.w_safety = w_safety
        self.spent = 0.0          # totale spesa economica sostenuta
        self.safety_cost_tot = 0.0  # totale "costo di sicurezza" accumulato (indicatore di rischio corso)
        self.choice_history = []  # memoria delle azioni scelte

    # Ritorna il costo dei viaggi andata/ritorno con l'auto sul percept giornaliero
    # = distanza * consumo carburante * prezzo carburante * 2
    def economic_cost_car(self, percept):
        return self.round_trip_factor * percept['distance'] * self.fuel_usage_per_km * percept['fuel_price']

    # Ritorna il costo dei viaggi andata/ritorno con i mezzi pubblici sul percept giornaliero (solo costo biglietto)
    def economic_cost_public(self, percept):
        return self.public_transport_cost

    # Ritorna il costo totale (economico + sicurezza) per un mezzo dato il percept giornaliero
    # Ritorna anche il costo economico e il costo di sicurezza separatamente
    def total_cost(self, mezzo, percept):
        # Ricavo il meteo giornaliero
        weather = percept['weather']
        # Calcolo economic e safety in base al mezzo scelto dall'agente
        if mezzo == 'auto':
            economic = self.economic_cost_car(percept)
            safety = self.car_risk_cost[weather]
        else:  # 'mezzi_pubblici'
            economic = self.economic_cost_public(percept)
            safety = self.public_transport_risk_cost[weather]
            # Somma pesata tra costi economici e di sicurezza
        return self.w_cost * economic + self.w_safety * safety, economic, safety

    # Dato il percept giornaliero, ritorna il mezzo consigliato dall'agente e aggiorna il suo stato interno
    def select_action(self, percept):
        # Recupero il costo totale, economiuco e di sicurezza per auto e mezzi pubblici
        cost_car, econ_car, safety_car = self.total_cost('auto', percept)
        cost_pub, econ_pub, safety_pub = self.total_cost('mezzi_pubblici', percept)

        # Scelgo il mezzo con il costo totale minore (cioè con l'utilita' maggiore)
        if cost_car <= cost_pub:
            mezzo = 'auto'
            economic, safety = econ_car, safety_car
        else:
            mezzo = 'mezzi_pubblici'
            economic, safety = econ_pub, safety_pub

        # Aggiorno lo stato interno dell'agente
        self.spent += economic
        self.safety_cost_tot += safety
        self.choice_history.append(mezzo)

        return {'mezzo': mezzo}

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    random.seed(0)

    # Simulazione di 30 giorni
    n_days = 30
    env = TravelEnv()
    ag = TravelAgent(w_cost=1.0, w_safety=5.0)
    sim = Simulate(ag, env)
    sim.go(n_days)

    n_auto = ag.choice_history.count('auto')
    n_pub = ag.choice_history.count('mezzi_pubblici')

    # ------------ Tabella riassuntiva: itero sulle history di ambiente e agente --------------
    # NB: env.*_history ha un elemento in piu' (il percept del giorno n_days+1, generato dall'ultima
    # chiamata a do()), che non ha ancora una corrispondente scelta dell'agente: lo escludo con [:n_days]
    print("-" * 100)
    print(f"{'Giorno':>6} | {'Meteo':<16} | {'Prezzo (EUR/l)':>14} | {'Distanza (km)':>13} | {'Consiglio':<16}")
    print("-" * 100)
    for day, (weather, fuel_price, distance, mezzo) in enumerate(
        zip(env.weather_history[:n_days],
            env.fuel_price_history[:n_days],
            env.distance_history[:n_days],
            ag.choice_history),
        start=1
    ):
        print(f"{day:>6} | {weather:<16} | {fuel_price:>14.3f} | {distance:>13.1f} | {mezzo:<16}")

    print("-" * 100)
    print()
    print(f"Giorni simulati: {n_days}")
    print(f"Giorni in cui l'agente ha consigliato l'auto: {n_auto}")
    print(f"Giorni in cui l'agente ha consigliato i mezzi pubblici: {n_pub}")
    print(f"Spesa economica totale stimata: {ag.spent:.2f} EUR")
    print(f"Spesa media giornaliera: {ag.spent/len(ag.choice_history):.2f} EUR")
