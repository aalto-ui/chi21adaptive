class Adaptation():
    # move item in slot i to slot j
    def __init__(self, move):
        self.i = move[0]
        self.j = move[1]
        self.type = move[2]
        self.expose = move[3]

    def __str__(self):
        return str((self.i, self.j, self.type.value, int(self.expose)))

    def __repr__(self):
        return str((self.i, self.j, self.type.value, int(self.expose)))

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.i == other.i and self.j == other.j and self.type == other.type and self.expose == other.expose

    def __hash__(self):
        return hash((self.i, self.j, self.type, self.expose))
