

class DataIO:
    def __init__(self, input_file=None, output_file=None):
        self.input_file = input_file
        self.output_file = output_file

    def read(self):
        data = []
        with open(self.input_file) as in_put:
            for line in in_put:
                if line.strip():
                    inline = line.split('\t')
                    if len(inline) == 1:
                        raise TypeError("Error: invalid data!")
                    data.append((int(inline[0]), int(inline[1]), int(inline[2])))
        return data





