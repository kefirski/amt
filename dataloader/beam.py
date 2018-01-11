class Beam:
    def __init__(self, prob, data):
        self.prob = prob
        self.data = data

    @staticmethod
    def start_search(probs, n_beams):
        probs = [[i, p] for i, p in enumerate(probs)]
        probs = sorted(probs, key=lambda pair: pair[1])[-n_beams:]

        return [Beam(prob=p, data=[idx]) for idx, p in probs]

    @staticmethod
    def update(beams: list, probs):
        n_beams = len(beams)

        for i in range(n_beams):
            probs[i] *= beams[i].prob

        probs = [[beam, idx, p] for i, beam in enumerate(beams) for idx, p in enumerate(probs[i])]
        probs = sorted(probs, key=lambda triple: triple[2])[-n_beams:]

        return [Beam(prob, old_beam.data + [idx]) for old_beam, idx, prob in probs]
