# Instrukcja Blender
## Tworzenie pomieszczenia z zwykłego obiektu Cube w Blenderze
1. Ustaw wymiary Cube do wymiarów pomieszczenia do testowania
2. Zatwierdź zmiany skali **CTRL+A** (żeby wszędzie była skala 1.0)
3. Utwórz ściany za pomocą zakładki **Modifiers**
	- Dodaj modifier **Solidify**
	- Ustaw grubość ściany w **Thickness** na `0.2 m`
	- Zaznacz pole **Even Thickness**
	- **Offset** `1.0`, ściany utworzą się na zewnątrz, żeby zachować poprawne wymiary pomieszczenia.
	- Zatwierdzić modifier **CTRL-A**
4. Dodaj **Material** do obiektu pomieszczenia stosując nazewnictwo materiałów z listy materiałów
5. Odpowiednio dla materiału nazwij obiekt pomieszczenia

## Nazewnictwo obiektów w projekcie Blender
1. Każdy obiekt musi mieć w nazwie podany materiał z listy materiałów
2. Siatka obiektu musi mieć dodany materiał w Blenderze z nazwą z listy materiałów

### Przykład:
**Room walls**
```bash
object: smooth_concrete_walls
	└── mesh: walls_mesh
		└── material: smooth_concrete
```

**Wooden Cube**
```bash
object: wood_50mm_thick_cube
	└── mesh: cube_mesh
		└── material: wood_50mm_thick
```
> Tak spreparowane obiekty w modelu w Blenderze zapewnią poprawne działanie wokselizera

## Ustawienia eksportu obiektu do pliku .obj
Zaznacz opcję **Triangulated Mesh** \
Zaznacz opcję **Materials**
	
> Wyeksportowany model w tym formacie utworzy dwa pliki.
> 1. model.obj - mesh
> 2. model.mtl - materials

## Materials
| id | name                     | alpha | density | color   |
|----|--------------------------|-------|---------|---------|
| 0  | air                      | 0.0   | 1.225   | #ffffff |
| 1  | smooth_concrete          | 0.02  | 2400.0  | #a0a0a0 |
| 2  | rough_concrete           | 0.03  | 2300.0  | #7a7a7a |
| 3  | brickwork_unglazed       | 0.03  | 1900.0  | #b24d3d |
| 4  | marble_or_glazed_tile    | 0.01  | 2700.0  | #e0f7fa |
| 5  | wood_50mm_thick          | 0.05  | 600.0   | #5d4037 |
| 6  | plywood_panelling        | 0.13  | 550.0   | #d7a26c |
| 7  | heavy_carpet_on_concrete | 0.25  | 200.0   | #2e7d32 |
| 8  | heavy_curtains_draped    | 0.66  | 1.5     | #6a1b9a |
| 9  | plaster_surface          | 0.02  | 1200.0  | #fff9c4 |
| 10 | solid_wooden_door        | 0.07  | 700.0   | #8d6e63 |
| 11 | water_surface_pool       | 0.01  | 1000.0  | #0288d1 |
| 12 | upholstered_seating      | 0.76  | 150.0   | #c62828 |
| 13 | audience_on_timber_seats | 0.62  | 500.0   | #ff8f00 |
| 14 | glass_window             | 0.1   | 2500.0  | #b3e5fc |
| 15 | metal                    | 0.02  | 7800.0  | #b0bec5 |
| 16 | acoustic_foam            | 0.85  | 40.0    | #37474f |

## Rooms (X, Y, Z) [m]
1. Cube (3.0, 3.0, 3.0)
	- fale będą na siebie bardzo nakładać, ekstremalne wzmocnienia i wygaszenia
2. Golden Ratio (4.0, 6.4, 2.5)
	- fale rezonansowe rozłożą się bardzo równomiernie
3. Long Corridor (2.0, 15.0, 2.5)
	- dźwięk wędruje w osi Y, wysokie częstotliwości będą się szybciej wytracać
	- flutter-echo od wielu odbić, bardzo wąski korytarz
4. Small Cabin (1.0, 1.0, 1.5)
	- testowanie wysokich częstotliwości przy "małej" liczbie wokseli
5. Deep Well (2.0, 2.0, 15.0)
	- dźwięk wędruje w osi Z, wysokie częstotliwości będą się szybciej wytracać
	- test warstwy pml przy otwartym otworze studni
