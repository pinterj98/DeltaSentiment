# Task 1

A Task 1 a dokumentumok sentiment scorejainak meghatározása, illetve 2022-ről 2023-ra a delta sentiment scoreok meghatározása. 

## Eszközök felderítése

Rövid irodalmi áttekintés és konzultáció után a következő modellek használata mellett döntöttünk:

- [FinBert](https://huggingface.co/ProsusAI/finbert)
- [FinVader](https://github.com/PetrKorab/FinVADER)
- [Gemini](https://ai.google.dev/)
- [Gemma](https://huggingface.co/google/gemma-2b), [Gemma-Instruct](https://huggingface.co/google/gemma-2b-it)

Mindegyik modellhez minimális működő példa található *Minimal Working Examples* mappában. 

## Eszközök tesztelése

Fontos kérdés az eszközökre nézve többek között, hogy mekkora inputot fogad el az eljárás, illetve mennyire zavarja az outputot, ha nem releváns információkkal van higítva az input.

Ezzel kapcsolatban a következő észrevételeket kaptuk:

- FinBERT: 512 token max (kb. fél oldal). Szöveg hosszától erősen függ a kapott score, táblázatok is jelentősen befolyásolják.
 
- FinVADER: hosszabb (néhány oldalas) szövegeket is megeszi, de ezekre a score 1-hez tart. Táblázatokra kevésbé érzékeny.


## Megjegyzések

- Gemini csak VPN-en (pl.: USA) keresztül érhető el.
- Gemma csak huggingface tokennel működik.


## Tartalma a notebookoknak

- sentiment_score_test_cases.ipynb: Hogyan változik a prediktált score, amennyiben az adatot felhigítjuk nem releváns információval.
- FinBert_vs_FinVader.ipynb: Hogyan teljesít a FinVader és a FinBert a szemantikus kereséssel szűrt adatokon?
