# SMask-for-VK

Программа, позволяющая музыкантам, играющим на инструментах симфонического оркестра, делать записи в соцсетях и редактировать высоту звучания в режиме онлайн. 

Для ее создания я использовала датасет TINYSO, авторы:
An audio dataset of isolated musical notes’
Carmine Emanuele; Daniele Ghisi; 
Vincent Lostanlen; 
Fabien Lévy; 
Joshua Fineberg; 
Yan Maresz
TinySOL ======= Version 6.0, February 2020.  
 
Created By --------------
Carmine-Emanuele Cella 
(1), Daniele Ghisi 
(1), Vincent Lostanlen 
(2), Fabien Lévy 
(3), Joshua Fineberg 
(4), Yan Maresz 
(5)  (1): UC Berkeley 
(2): New York University
(3): Columbia University
(4): Boston University 
(5): Conservatoire de Paris

С помощью датасета я обучила модель предсказывать предполагаемые исполнителем ноты, которые на самом деле могли быть сыграны выше или ниже в Hz. 

Далее, с помощью модели crepe я извлекла среднее значение частоты каждой ноты и сдвинула каждую ноту на нужную разницу частоты вверх или вниз, в зависимости от ситуации.

На выходе исполнитель получает готовый, "очищенный" от фальши аудиофайл, который можно сразу запускать в рилз)
