import numpy as np


class Stock:
    all_elements = np.arange(2)

    @classmethod
    def initiate_stock(cls, size):
        cls.all_elements = np.arange(size)

    def __init__(self):
        self.elements_used = []

    def fetch_element(self, index):
        if index >= self.all_elements.size:
            raise IndexError
        
        if index in self.elements_used:
            return 0
        
        self.elements_used.append(index)
        return self.all_elements[index]


if __name__ == "__main__":
    Stock.initiate_stock(5)

    stock_1 = Stock()
    stock_2 = Stock()

    print(stock_1.fetch_element(2))
    print(stock_2.fetch_element(2))

    print(stock_1.fetch_element(2))
    print(stock_2.fetch_element(4))

    print(stock_1.elements_used)
    print(stock_2.elements_used)
