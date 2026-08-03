# È una classe di utilità per fare debug/logging controllabile, non per la logica dell'agente in sé. Serve a stampare messaggi diagnostici durante l'esecuzione,
# ma con la possibilità di decidere quanti dettagli vedere, senza dover riscrivere il codice.
class Displayable(object):
    max_display_level = 1

    def display(self, level, *args, **nargs):
       if level <= self.max_display_level:
            print(*args, **nargs)
