# -*- coding: UTF-8-*-
""" Zadanie 1

Otrzymaliśmy dane w następującej postaci: "{dzień tygodnia};{cztery znaki zawierające wskazanie czujnika}" (zawsze są to 4 znaki!). Napisz skrypt, który wygeneruje raport z konkretnego dnia na prośbę użytkownika:
Podaj dzień dla którego ma zostać wyliczony raport (wielkość liter nie powinna mieć znaczenia,; dzień mozna wybierać dowolnie)
 Pobierz odpowiednią wartość z danych
 Przelicz wskazanie czujnika na temperaturę dzieląc przez wartość z tabeli
Wypisz temperaturę z dokładnością do trzeciego miejsca po przecinku (tak jak na przykładzie) i ze znakiem specjalnym ℃ (\u2103)
BONUS. Do punktu 1 dodaj walidację - upewnij się, że użytkownik podał jeden z dni który możemy znaleźć w danych
BONUS. Do punktu 4 dodaj wizualizację temperatury w zakresie 0 do 100 stopni na pasku złożonym z 20 znaków (np. dla 50 stopni, zapełnione jest 10/20 znaków)
"""
# dane do zadania 1 przedsyawione w słowniku
data = { 'Monday' : 1250,
         'Tuesday' : 1405,
         'Wednesday': 1750,
         'Thursday' :1100,
         'Friday' : 800,
         'Saturday': 1225,
         'Sunday' : 1355}

print(data)

# Tworzę klasę do generowania raportu

class TemperatureReport:
# inicjalizuję self orad dane do classy
    def __init__(self, data):
        self.data = data

    def generate_report(self):
# funkcja generująca raport
        while True:
# wielkość liter nie ma znaczenia, jeśli taki dzień jest w data go on, jesli nie prosze o poprawne wprowadzenie nazwy dnia tygodnia
            day = input("Podaj dzień, dla którego chcesz stworzyć raport:\t").capitalize()
            if day in self.data:
                break
            else:
                print("Nieprawidłowy dzień. Spróbuj ponownie!")

# wartości dla danych na wyjściu programu
        value = self.data[day]
        temperature = value / 10
        temperature_formatted = f"{temperature:.3f} St. Celcjusza"
        temperature_wskaz = "#" * int(temperature / 5) + "-" * (20 - int(temperature / 5))

# wyjście na ekran komputera
        print(f"\nRaport dla dnia {day}:")
        print(f"Wartość / wskazanie czujnika: \t{value}")
        print(f"Temperatura: {temperature_formatted}")
        print(f"Temperatura na pasku: {temperature_wskaz} (0 - 100)")

def main():
    raport = TemperatureReport(data)
    raport.generate_report()

# zmienna name przyjmuje wartość z main()
if __name__ == '__main__':
    main()