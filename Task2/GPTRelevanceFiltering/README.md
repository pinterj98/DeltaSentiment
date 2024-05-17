# Task 2

A GPTRelevanceLabeling notebookban GPT-4o-val címkézzük fel a dokumentumok mondatait az alapján, hogy relevánsak-e vagy sem számunkra. 
A kódot nem lehet 100%-ra automatizáltan futtatni, mert néhol eltér a megadott output struktúrától az llm (~ 200 promptonként egyszer elrontja), ekkor manuálisan újra kell futtatni.

A RelevanceBert notebookban egy saját Bert modellt (yiyangkhust/Finbert tranfer learning) tanítunk relevancia szűrésre a felcímkézett adathalmaz alapján.