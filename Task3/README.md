# Task 3

A Task 3 a pdf-ek automatizált feldolgozásával foglalkozik.

## Eszközök felderítése

Rengeteg PDF miner eszköz létezik, mellyel megoldható a probléma:
- PDFMiner (Python)
- Tika (Java szerver)
- Adobe Services API (Adobe Acrobat Pro program)

## Eszközök tesztelése

Az eszközök tesztelése során a következőeket tapasztaltuk:

- PDFMiner használata bonyolult, ha tényleg szép eredményt akarunk elérni. Az alap config nem működik túlságosan jól.
- Tika mérsekelten jól működik. HTML fájllá konvertálja a PDF-et, de például table-öket, ulokat nem detektál. (Lehet hogy jobban is konfigurálható?)
- Adobe Acrobat Pro kifejezetten jól működik. Figureöket kiszűri, table-öket, ul-okat is detektál. -> Szép tiszta szöveg. (Az API fizetős, bár van ingyenes teszt. Egyelőre nem sikerült működésre bírni.)

## Eljárásunk

A következőt tervezzük végrehajtani:

- Tika / Adobe (?) átkonvertálja a PDF-et HTML fájllá.
- HTML fájlt feldolgozzuk BeautifulSouppal: Tiszta paragraphok kinyerése. -> Szűrés kellően hosszú paragraphokra. 
- Paragraphok felbontása mondatokra. (?)
- További tisztítások, amennyiben szükséges (pl.: számértékek helyettesítése numberrel, stb...)

