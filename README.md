# Task-AI: GOFAI + ML
La seguente repository contiene il progetto per l'esame di Fondamento e Applicazioni dell'AI, in cui lo scopo era quello di realizzare in Python un **Agente Intelligente** (GOFAI = "Good Old-Fashioned Artificial Intelligence") e un modello di **Machine Learning** (ML) portando a termine delle task specifiche.
## GOFAI
Implementare (in Python) un agente che riceve i seguenti percept dall'ambiente ed effettua le seguenti azioni:
 - Percept: giorno per giorno, (i) indicazione sulle condizioni atmosferiche (sereno, rischio pioggia, rischio neve), (ii) indicazione sul prezzo del carburante, (iii) distanza da percorrere (ad esempio, per andare a lavoro).
 - Azione: dato il percept giornaliero corrente, consigliare all'utente se effettuare il tragitto previsto giornaliero (ad esempio, per andare a lavoro) con la propria autovettura o mediante mezzi pubblici.
Le azioni dell'agente devono derivare dall'ottimizzazione di una misura di performance che tiene in considerazione la sicurezza stradale derivante dalle condizioni atmosferiche e il costo totale dello spostamento.
## ML
Implementare in PyTorch:
 - Un'architettura di neural network con almeno un layer RNN per il task di sentiment analysis sul dataset 'Airlines-Tweets-Sentiments', disponibile qui: https://www.openml.org/search?type=data&status=active&id=43397.
 - Fase di training (addestramento) del modello sottostante la suddetta architettura.
 - Fase di test del modello addestrato.
Gli iperparametri in gioco possono essere settati a valori di default (non è necessario implementare una validation phase).

