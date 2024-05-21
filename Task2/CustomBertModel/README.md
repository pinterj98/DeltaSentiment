# Task 2

A saját modellünk a GPT-4o által felcímkézett adathalmazon transfer-learning technikával betanított FinBert modell.

Az eljárás során a következő lépéseket tettük:
- Az adathalmazt felbontottuk három részre (test1: tsla23, ups23, test2: zbh22, zbh23, train: összes többi)
- Az tanítón belül véletlenszerű felosztással ~ 10-15% validáció, többi tanító.
- A FinBert embedder részét befagyasztottuk, a klasszifikációs réteget lecseréltük egy egyszerű kétrétegű előrecsatolt neurális hálóra (64 -> 1).
- Tanítottuk a hálót, amíg nem stagnált a javulás a validációs halmazon.
- Kifagyasztottuk az embedder réteget és az egészet tanítottuk (csak 1 epochig, mert nagyon hamar túltanul).
- Újra lefagyasztottuk és folytattuk a klasszifikációs rétegek tanítását, amig nem stagnált újra a teljesítmény. 

Technikai információk:
A tanítás NVIDIA RTX 4060 GPUn történt. Lefagyasztott embedderrel körülbelül ~3 perc egy epoch (512 batch size), embeddert is tanítva körülbelül ~12 perc egy epoch (32 batch size).