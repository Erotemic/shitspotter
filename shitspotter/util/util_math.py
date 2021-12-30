import fractions


class Rational(fractions.Fraction):
    """
    Extension of the Fraction class, mostly to make printing nicer

    >>> 3 * -(Rational(3) / 2)
    """
    def __str__(self):
        if self.denominator == 1:
            return str(self.numerator)
        else:
            return '{}'.format(self.numerator / self.denominator)
            # return '({}/{})'.format(self.numerator, self.denominator)

    def __json__(self):
        return {
            'type': 'rational',
            'numerator': self.numerator,
            'denominator': self.denominator,
        }

    def __smalljson__(self):
        return '{:d}/{:d}'.format(self.numerator, self.denominator)

    @classmethod
    def coerce(cls, data):
        from PIL.TiffImagePlugin import IFDRational
        if isinstance(data, dict):
            return cls.from_json(data)
        elif isinstance(data, IFDRational):
            return cls(data.numerator, data.denominator)
        elif isinstance(data, int):
            return cls(data, 1)
        elif isinstance(data, str):
            return cls(*map(int, data.split('/')))
        else:
            raise TypeError

    @classmethod
    def from_json(cls, data):
        return cls(data['numerator'], data['denominator'])

    def __repr__(self):
        return str(self)

    def __neg__(self):
        return Rational(super().__neg__())

    def __add__(self, other):
        return Rational(super().__add__(other))

    def __radd__(self, other):
        return Rational(super().__radd__(other))

    def __sub__(self, other):
        return Rational(super().__sub__(other))

    def __mul__(self, other):
        return Rational(super().__mul__(other))

    def __rmul__(self, other):
        return Rational(super().__rmul__(other))

    def __truediv__(self, other):
        return Rational(super().__truediv__(other))

    def __floordiv__(self, other):
        return Rational(super().__floordiv__(other))
