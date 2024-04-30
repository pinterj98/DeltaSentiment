# Task 2

A Task 2 a dokumentumokból a feladat szempontjából releváns szövegrészletek automatizált kinyerésével foglalkozik.

## Eszközök felderítése

Semantic search eljárással szeretnénk megoldani a problémát, melyhez szükség van valamilyen vektorbeágyazásra:

- FinBert (?) (A szerző szerint a finbertből mondatokhoz tartozó szemantikus vektor nem biztos, hogy értelmes jelentéssel bír. Nem javasolja szemantikus kereséshez.)
- Mondat beágyazó módszerek: [SRoberta Finance Finetuned](https://huggingface.co/yseop/roberta-base-finance-hypernym-identification), [all Mini LM](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), [all mpnet](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)