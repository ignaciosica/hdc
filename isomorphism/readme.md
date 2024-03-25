# Isomorphism

**Traducir**

Es isomorfismo en grafos significa que ambas son topológicamente equivalentes. Esto significa que se puede establecer una relación uno a uno entre sus nodos y las aristas entre ellos corresponderían también uno a uno. Los algoritmos actuales capaces de determinar si dos grafos son isomorfos son computacionalmente muy demandantes. Se probará un algoritmo basado en hdc con una complejidad de tiempo O(n * e) donde n es la cantidad de nodos y e la cantidad de aristas.  

**Hipótesis:** 
Se podría utilizar las herramientas que propone hdc para resolver el problema de isomorphismo de grafos. 

**Solución 1:** 
Se me ocurrió un algoritmo con complejidad polinomial que en principio pensé que podría resolver el problema. El algoritmo se basa en la centralidad de los nodos y en las propiedades de las operaciones de HDC como la conmutatividad del bundle. El algoritmo es el siguiente. Primero se le asigna a cada nodo un hypervector determinado segun el grado del mismo. Luego, se representa cada arista como el bind de los nodos si es que estos tienen distinto grado, en caso de tener el mismo, la arista se representa solamente con el hypervector que representa el grado. Luego, el grafo se construye haciendo el bundle de todos los hypervectores que representan las aristas. Se hace esta codificación con dos grafos y si la resta de sus hypervectores da 0 o el bind da 1 o la distancia por coseno da 1, se podría decir que estos grafos son isomorfos.

**Contra ejemplos:**

|          counter-example A           |          counter-example B           |
|:------------------------------------:|:------------------------------------:|
| ![[counterexample_isomorphism.png\|300]] | ![[counterexample_isomorphism_2.png\|300]] |

**Solución 2:** 
La base del algoritmo es el mismo pero cada nodo se codifica como una recorrida breath-first de todo el grafo donde se codifica cada uno de los niveles con una permutación. El grafo se codifica como el bundle de todos los nodos. Creo que esto solucionaría en principio los dos contraejemplo manteniendo un orden de complejidad polinomial.									 
									 