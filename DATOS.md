# Documentacion de Recogida de Datos

**Alumno:** Lucas Pérez García

## Formato del CSV

Tu archivo `data/partidas.csv` debe tener **minimo** estas columnas:

| Columna | Descripcion | Ejemplo |
|---------|-------------|---------|
| `numero_ronda` | Numero de la ronda (1, 2, 3...) | 1 |
| `jugada_j1` | Jugada del jugador 1 | piedra |
| `jugada_j2` | Jugada del jugador 2 (oponente) | papel |

### Ejemplo de CSV minimo:

```csv
numero_ronda,jugada_j1,jugada_j2
1,piedra,papel
2,tijera,piedra
3,papel,papel
4,piedra,tijera
...
```

---

## Como recogiste los datos?

Marca con [x] el metodo usado y describe brevemente:

### Metodo de recogida:

- [ ] **Programa propio**: Cree un programa para jugar y guardar datos
- [ ] **Manual**: Jugue partidas y apunte los resultados a mano
- [ ] **Aplicacion/Web externa**: Use una app y exporte los datos
- [X] **Otro**: De unos compañeros de clase

### Descripcion del proceso:

```
(Explica aqui como recogiste los datos. Si usaste un programa,
describe brevemente como funciona. Si fue manual, explica el proceso.)

Inicialmente los compañeros (Al igual que los datos que tenía antes de pedírselos) usaron un programa propio para la recolección de datos pero dicho método tiene varios fallos. Desde el hecho de que se tiende a usar el dedo que menos esfuerzo te consume de forma inconsciente hasta que el hecho de no estar jugando de forma directa contra el oponente puede quitarle seriedad y disminuir la concentración. 

Tras ello hicieron una recogida de datos manual lo que otorgó datos de mayor calidad.

```
---

## Datos adicionales capturados

Si capturaste datos extra ademas de los basicos, marcalos aqui:

- [ ] `tiempo_reaccion_ms` - Tiempo que tardo el jugador en responder
- [ ] `timestamp` - Fecha/hora de cada jugada
- [X] `sesion` - ID de sesion de juego
- [ ] `resultado` - victoria/derrota/empate
- [X] Otro: Rondas de cada partida

### Descripcion de datos adicionales:

```
Existe una columna partida y una ronda. La partida es la serie de rondas hasta que se gana por lo que si hay empates no se acaba la partida hasta que uno gana.


```

---

## Estadisticas del dataset

- **Total de rondas:** 308
- **Numero de sesiones/partidas:** 200
- **Contra cuantas personas diferentes:** 1 recogido en 4-5 días distintos y haciendo descanso cada 20 partidas.

### Tipo de IA:

- [X] **IA Especifica**: Entrenada para ganar a UNA persona concreta
  - Nombre/identificador del oponente: Victor Lacruz Sancho
- [ ] **IA General**: Entrenada para ganar a cualquier oponente
