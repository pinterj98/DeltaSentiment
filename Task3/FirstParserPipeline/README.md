# First Parser

Az első parsolónk a következő eljárás alapján készül: 
- PDF -> HTML Adobe Acrobat Pro (Már ezek a fileok vannak a mappában)
- HTML -> Paragraphok, melyek legalább 8 szóból állnak, BeautifulSoup (parser.py megcsinálja  *name.html* -> *name_paragraphs.json*)
- Paragraphok -> Mondatok, NLTK (parser.py megcsinálja *name.html* -> *name_sentences.json*)

combined_sentences -> Végső fájl, amiben a mondatlisták hozzá vannak rendelve a cégek reportjaihoz cégnév+évszám_sentences.json kulcsokkal (pl.: tsla22_sentences).