# -*- coding: UTF-8-*-
""" Zadanie 2

Sklep internetowy (use case) Wyobraź sobie, że jesteś właścicielem sklepu internetowego, oraz magazynu, który przechowuje Twoje produkty.
Zdefiniuj jakie produkty chcesz sprzedawać. Zastanów się jak możesz skorzystać z poznanych metod programowania obiektowego w swoim biznesie.
Przemyśl jakie cechy charakteryzują Twoje obiekty i jakie metody mogą być dla Ciebie użyteczne.

Rozważ funkcje produktów - nie zapomnij też o cenie, czy ilości na stanie
rozważ funkcję zakupu/sprzedaży produktu (zarobek i zmiana na stanie)
 Dodaj klika produktów
Zademonstruj działanie funkcji
"""

# Tworze klasę Product dla kilku produktów w Sklepie
class Sklep:
    def __init__(self, nazwa, cena, ilosc):
# inicjuję nazwę prod., cenę prod., jakość prod.
        self.nazwa = nazwa
        self.cena = cena
        self.ilosc = ilosc

    def get_info(self):
        return f"Sklep: {self.nazwa}, Cena: {self.cena}, Ilość prod: {self.ilosc}"

    def update_quantity(self, quantity_sold):
        if self.ilosc >= quantity_sold:
            self.ilosc -= quantity_sold
            return True
        else:
            return False
def purchase_product(product, ilosc):
    product.ilosc += ilosc

def sell_product(product, quantity_sold):
    if product.update_quantity(quantity_sold):
        total_price = product.cena * quantity_sold
        return total_price
    else:
        return "Insufficient quantity in stock."
product1 = Sklep("Laptop", 1500, 10)
product2 = Sklep("Telefon komórkowy", 800, 15 )
product3 = Sklep("Słuchawki", 100, 20)

def main():
    # Zakup produktu
    purchase_product(product1, 5)

    # Sprzedaż produktu
    total_price = sell_product(product1, 3)
    if isinstance(total_price, str):
        print(total_price)
    else:
        print(f"Product sold. Total price: {total_price}")

    # Wyświetlanie informacji o produkcie
    print(product1.get_info())
    print(product2.get_info())
    print(product3.get_info())

if __name__ == '__main__':
    main()
